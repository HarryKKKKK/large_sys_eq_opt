#include "cpu/grid_cpu.hpp"
#include "cpu/solver_mpi.hpp"
#include "init.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

struct TimingStats {
    double output_dir_time = 0.0;
    double config_and_init_time = 0.0;
    double snapshot_schedule_time = 0.0;

    double main_loop_time = 0.0;
    double boundary_time = 0.0;      // In this pure MPI version, this should remain ~0.
    double compute_dt_time = 0.0;
    double advance_time = 0.0;
    double snapshot_write_time = 0.0;

    double total_program_time = 0.0;
};

struct RunOptions {
    int resolution_scale = 1;
    bool write_output = false;
    int num_snapshots = 5;

    std::string case_name = "shock_bubble";

    RiemannSolver solver = RiemannSolver::HLL;
    std::string solver_name = "hll";
};

RiemannSolver parse_riemann_solver(const std::string& name) {
    if (name == "hll" || name == "HLL") {
        return RiemannSolver::HLL;
    }

    if (name == "hllc" || name == "HLLC") {
        return RiemannSolver::HLLC;
    }

    if (name == "exact" || name == "Exact" || name == "EXACT") {
        return RiemannSolver::Exact;
    }

    if (name == "force" || name == "Force" || name == "FORCE") {
        return RiemannSolver::FORCE;
    }

    throw std::runtime_error(
        "Unknown Riemann solver: " + name +
        ". Supported solvers are: hll, hllc, exact, force."
    );
}

std::string riemann_solver_to_string(RiemannSolver solver) {
    switch (solver) {
        case RiemannSolver::HLL:
            return "hll";
        case RiemannSolver::HLLC:
            return "hllc";
        case RiemannSolver::Exact:
            return "exact";
        case RiemannSolver::FORCE:
            return "force";
        default:
            return "unknown";
    }
}

RunOptions parse_run_options(int argc, char** argv) {
    RunOptions opts;

    if (argc >= 2 && argv[1][0] != '-') {
        opts.resolution_scale = std::stoi(argv[1]);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--output") {
            opts.write_output = true;
            opts.num_snapshots = 5;
        } else if (arg == "--timing-only") {
            opts.write_output = true;
            opts.num_snapshots = 0;
        } else if (arg == "--case") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--case requires a value, e.g. shock_bubble or blast_wave.");
            }

            opts.case_name = argv[++i];
            (void)parse_case_id(opts.case_name);
        } else if (arg == "--solver") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--solver requires a value: hll, hllc, exact, or force.");
            }

            opts.solver_name = argv[++i];
            opts.solver = parse_riemann_solver(opts.solver_name);
            opts.solver_name = riemann_solver_to_string(opts.solver);
        }
    }

    if (opts.resolution_scale <= 0) {
        throw std::runtime_error("resolution_scale must be positive.");
    }

    return opts;
}

CaseConfig scale_case_config(CaseConfig cfg, int scale) {
    cfg.nx *= scale;
    cfg.ny *= scale;
    return cfg;
}

void ensure_output_dir() {
    std::filesystem::create_directories("outputs");
}

std::string make_snapshot_name(
    const std::string& case_name,
    const std::string& solver_name,
    int scale,
    int rank,
    int snap_id
) {
    std::ostringstream oss;
    oss << "outputs/mpi_"
        << case_name
        << "_"
        << solver_name
        << "_n" << scale
        << "_rank" << std::setw(5) << std::setfill('0') << rank
        << "_snapshot_" << snap_id << ".csv";
    return oss.str();
}

std::string make_timing_name(
    const std::string& case_name,
    const std::string& solver_name,
    int scale
) {
    std::ostringstream oss;
    oss << "outputs/mpi_"
        << case_name
        << "_"
        << solver_name
        << "_n" << scale
        << "_timing.txt";
    return oss.str();
}

void write_local_grid_csv(
    const Grid2D& grid,
    const MpiDecomp2D& mp,
    const std::string& filename
) {
    std::ofstream out(filename);
    out << "i,j,x,y,rho,rhou,rhov,E\n";

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        const int global_j = mp.ng + mp.y0_global + (j - grid.j_begin());

        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Conserved& U = grid(i, j);

            out << i << ","
                << global_j << ","
                << grid.x_center(i) << ","
                << grid.y_center(j) << ","
                << U.rho << ","
                << U.rhou << ","
                << U.rhov << ","
                << U.E << "\n";
        }
    }
}

void reduce_timing_max(
    const TimingStats& local,
    TimingStats& global,
    MPI_Comm comm
) {
    MPI_Reduce(&local.output_dir_time,       &global.output_dir_time,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.config_and_init_time,  &global.config_and_init_time,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.snapshot_schedule_time,&global.snapshot_schedule_time,1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.main_loop_time,        &global.main_loop_time,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.boundary_time,         &global.boundary_time,         1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.compute_dt_time,       &global.compute_dt_time,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.advance_time,          &global.advance_time,          1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.snapshot_write_time,   &global.snapshot_write_time,   1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.total_program_time,    &global.total_program_time,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
}

void write_timing_report(
    const std::string& filename,
    const RunOptions& opts,
    const CaseConfig& cfg,
    const MpiDecomp2D& mp,
    int step,
    double final_time,
    int snapshots_written,
    const TimingStats& ts
) {
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(6);

    out << "=== MPI Timing Report ===\n";
    out << "case_name                 : " << opts.case_name << "\n";
    out << "riemann_solver            : " << opts.solver_name << "\n";
    out << "resolution_scale          : " << opts.resolution_scale << "\n";
    out << "mpi_ranks                 : " << mp.size << "\n";
    out << "nx                        : " << cfg.nx << "\n";
    out << "ny                        : " << cfg.ny << "\n";
    out << "total_cells               : "
        << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny) << "\n";
    out << "ng                        : " << cfg.ng << "\n";
    out << "cfl                       : " << cfg.cfl << "\n";
    out << "t_end                     : " << cfg.t_end << "\n";
    out << "final_time                : " << final_time << "\n";
    out << "total_steps               : " << step << "\n";
    out << "snapshots_written         : " << snapshots_written << "\n\n";

    out << "output_dir_time           : " << ts.output_dir_time << " s\n";
    out << "config_and_init_time      : " << ts.config_and_init_time << " s\n";
    out << "snapshot_schedule_time    : " << ts.snapshot_schedule_time << " s\n";
    out << "main_loop_time            : " << ts.main_loop_time << " s\n";
    out << "  boundary_time           : " << ts.boundary_time << " s\n";
    out << "  compute_dt_time         : " << ts.compute_dt_time << " s\n";
    out << "  advance_time            : " << ts.advance_time << " s\n";
    out << "  snapshot_write_time     : " << ts.snapshot_write_time << " s\n";
    out << "total_program_time        : " << ts.total_program_time << " s\n\n";

    if (step > 0) {
        out << "avg_time_per_step         : " << ts.main_loop_time / step << " s\n";
        out << "avg_boundary_per_step     : " << ts.boundary_time / step << " s\n";
        out << "avg_compute_dt_per_step   : " << ts.compute_dt_time / step << " s\n";
        out << "avg_advance_per_step      : " << ts.advance_time / step << " s\n";
    }
}

void print_timing_report(
    const TimingStats& ts,
    int step
) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== MPI Timing Summary ===\n";
    std::cout << "output_dir_time           : " << ts.output_dir_time << " s\n";
    std::cout << "config_and_init_time      : " << ts.config_and_init_time << " s\n";
    std::cout << "snapshot_schedule_time    : " << ts.snapshot_schedule_time << " s\n";
    std::cout << "main_loop_time            : " << ts.main_loop_time << " s\n";
    std::cout << "  boundary_time           : " << ts.boundary_time << " s\n";
    std::cout << "  compute_dt_time         : " << ts.compute_dt_time << " s\n";
    std::cout << "  advance_time            : " << ts.advance_time << " s\n";
    std::cout << "  snapshot_write_time     : " << ts.snapshot_write_time << " s\n";
    std::cout << "total_program_time        : " << ts.total_program_time << " s\n";

    if (step > 0) {
        std::cout << "avg_time_per_step         : " << ts.main_loop_time / step << " s\n";
        std::cout << "avg_boundary_per_step     : " << ts.boundary_time / step << " s\n";
        std::cout << "avg_compute_dt_per_step   : " << ts.compute_dt_time / step << " s\n";
        std::cout << "avg_advance_per_step      : " << ts.advance_time / step << " s\n";
    }

    std::cout << "==========================\n";
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto program_start = Clock::now();
    TimingStats timings;

    try {
        const RunOptions opts = parse_run_options(argc, argv);

        if (opts.write_output) {
            const auto t0 = Clock::now();

            if (rank == 0) {
                ensure_output_dir();
            }

            MPI_Barrier(MPI_COMM_WORLD);

            const auto t1 = Clock::now();
            timings.output_dir_time = Seconds(t1 - t0).count();
        }

        CaseConfig cfg;
        Grid2D U;
        Grid2D U_mid;
        Grid2D U_next;
        MpiDecomp2D mp;
        MpiCpuWorkspace ws;

        {
            const auto t0 = Clock::now();

            cfg = scale_case_config(
                get_case_config(opts.case_name),
                opts.resolution_scale
            );

            mp = make_y_slab_decomp(cfg, MPI_COMM_WORLD);

            U = make_local_mpi_grid(opts.case_name, cfg, mp);

            U_mid = Grid2D(
                cfg.nx,
                mp.ny_local,
                cfg.ng,
                cfg.x_min,
                cfg.x_max,
                U.y_min(),
                U.y_max()
            );

            U_next = Grid2D(
                cfg.nx,
                mp.ny_local,
                cfg.ng,
                cfg.x_min,
                cfg.x_max,
                U.y_min(),
                U.y_max()
            );

            // Allocate the MPI face-flux workspace once.
            ws.init(U.nx(), U.ny());

            MPI_Barrier(MPI_COMM_WORLD);

            const auto t1 = Clock::now();
            timings.config_and_init_time = Seconds(t1 - t0).count();
        }

        if (rank == 0) {
            std::cout << "[MPI] case_name          : " << opts.case_name << "\n";
            std::cout << "[MPI] riemann_solver     : " << opts.solver_name << "\n";
            std::cout << "[MPI] resolution_scale   : " << opts.resolution_scale << "\n";
            std::cout << "[MPI] mpi_ranks          : " << size << "\n";
            std::cout << "[MPI] nx                 : " << cfg.nx << "\n";
            std::cout << "[MPI] ny                 : " << cfg.ny << "\n";
            std::cout << "[MPI] total_cells        : "
                      << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny)
                      << "\n";
            std::cout << "[MPI] write_output       : "
                      << (opts.write_output ? "true" : "false") << "\n";
            std::cout << "[MPI] num_snapshots      : "
                      << (opts.write_output ? opts.num_snapshots : 0) << "\n";
        }

        double t = 0.0;
        int step = 0;

        const int num_snapshots = opts.write_output ? opts.num_snapshots : 0;
        std::vector<double> snapshot_times;
        int next_snapshot = 0;

        if (opts.write_output && num_snapshots > 0) {
            const auto t0 = Clock::now();

            snapshot_times.reserve(num_snapshots);

            for (int k = 1; k <= num_snapshots; ++k) {
                snapshot_times.push_back(
                    cfg.t_end * static_cast<double>(k) /
                    static_cast<double>(num_snapshots)
                );
            }

            const auto t1 = Clock::now();
            timings.snapshot_schedule_time = Seconds(t1 - t0).count();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const auto loop_start = Clock::now();

        while (t < cfg.t_end) {
            double dt = 0.0;

            {
                const auto t0 = Clock::now();

                // compute_dt_mpi only reads interior cells, so no halo exchange is
                // required here.
                dt = compute_dt_mpi(U, mp, cfg.cfl);

                const auto t1 = Clock::now();
                timings.compute_dt_time += Seconds(t1 - t0).count();
            }

            if (t + dt > cfg.t_end) {
                dt = cfg.t_end - t;
            }

            {
                const auto t0 = Clock::now();

                // Pure MPI second-order update. The solver performs the only
                // necessary halo exchange: after x-sweep, before y-sweep.
                advance_second_order_mpi(
                    U,
                    U_mid,
                    U_next,
                    mp,
                    dt,
                    opts.solver,
                    ws
                );

                std::swap(U, U_next);

                const auto t1 = Clock::now();
                timings.advance_time += Seconds(t1 - t0).count();
            }

            t += dt;
            ++step;

            while (opts.write_output &&
                   next_snapshot < num_snapshots &&
                   t >= snapshot_times[next_snapshot]) {
                const auto t0 = Clock::now();

                const std::string filename = make_snapshot_name(
                    opts.case_name,
                    opts.solver_name,
                    opts.resolution_scale,
                    rank,
                    next_snapshot + 1
                );

                write_local_grid_csv(U, mp, filename);

                const auto t1 = Clock::now();
                timings.snapshot_write_time += Seconds(t1 - t0).count();

                ++next_snapshot;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        const auto loop_end = Clock::now();
        timings.main_loop_time = Seconds(loop_end - loop_start).count();

        const auto program_end = Clock::now();
        timings.total_program_time = Seconds(program_end - program_start).count();

        TimingStats global_timings;
        reduce_timing_max(timings, global_timings, MPI_COMM_WORLD);

        int global_steps = 0;
        MPI_Reduce(
            &step,
            &global_steps,
            1,
            MPI_INT,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::cout << "[MPI] Finished.\n";
            std::cout << "[MPI] Final time = " << t << "\n";
            std::cout << "[MPI] Total steps = " << global_steps << "\n";

            print_timing_report(global_timings, global_steps);

            if (opts.write_output) {
                const std::string timing_file = make_timing_name(
                    opts.case_name,
                    opts.solver_name,
                    opts.resolution_scale
                );

                write_timing_report(
                    timing_file,
                    opts,
                    cfg,
                    mp,
                    global_steps,
                    t,
                    next_snapshot,
                    global_timings
                );

                std::cout << "[MPI] Timing report written to "
                          << timing_file << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] ERROR: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
