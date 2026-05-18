#include "cpu/boundary_cpu.hpp"
#include "cpu/grid_cpu.hpp"
#include "cpu/solver_cpu.hpp"
#include "init.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include "riemann.hpp"

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

using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

struct TimingStats {
    double output_dir_time        = 0.0;
    double config_and_init_time   = 0.0;
    double workspace_alloc_time   = 0.0;
    double snapshot_schedule_time = 0.0;

    double main_loop_time         = 0.0;
    double boundary_time          = 0.0;
    double compute_dt_time        = 0.0;
    double advance_time           = 0.0;
    double snapshot_write_time    = 0.0;

    double total_program_time     = 0.0;
};

struct RunOptions {
    int resolution_scale = 1;
    bool write_output = false;
    int num_snapshots = 5;

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

    throw std::runtime_error(
        "Unknown Riemann solver: " + name +
        ". Supported solvers are: hll, hllc."
    );
}

std::string riemann_solver_to_string(RiemannSolver solver) {
    switch (solver) {
        case RiemannSolver::HLL:
            return "hll";
        case RiemannSolver::HLLC:
            return "hllc";
        default:
            return "unknown";
    }
}

RunOptions parse_run_options(int argc, char** argv) {
    RunOptions opts;

    if (argc >= 2) {
        opts.resolution_scale = std::stoi(argv[1]);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--output") {
            opts.write_output = true;
            opts.num_snapshots = 5;
        } else if (arg == "--timing-only") {
            opts.write_output = true;
            opts.num_snapshots = 0;
        } else if (arg == "--solver") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--solver requires a value: hll or hllc.");
            }

            opts.solver_name = argv[++i];
            opts.solver = parse_riemann_solver(opts.solver_name);
            opts.solver_name = riemann_solver_to_string(opts.solver);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.resolution_scale <= 0) {
        throw std::runtime_error("resolution_scale must be positive.");
    }

    return opts;
}

void ensure_output_dir() {
    std::filesystem::create_directories("outputs");
}

std::string make_snapshot_name(const std::string& prefix, int snap_id) {
    std::ostringstream oss;
    oss << "outputs/" << prefix << "_snapshot_" << snap_id << ".csv";
    return oss.str();
}

std::string make_timing_name(const std::string& prefix) {
    std::ostringstream oss;
    oss << "outputs/" << prefix << "_timing.txt";
    return oss.str();
}

void write_grid_csv(const Grid2D& grid, const std::string& filename) {
    std::ofstream out(filename);
    out << "i,j,x,y,rho,rhou,rhov,E\n";

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Conserved& U = grid(i, j);

            out << i               << ","
                << j               << ","
                << grid.x_center(i) << ","
                << grid.y_center(j) << ","
                << U.rho  << ","
                << U.rhou << ","
                << U.rhov << ","
                << U.E    << "\n";
        }
    }
}

void write_timing_report(const std::string& filename,
                         const std::string& case_name,
                         const CaseConfig&  cfg,
                         const RunOptions&  opts,
                         int    step,
                         double final_time,
                         int    num_snapshots_written,
                         const TimingStats& ts) {
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(6);

    out << "=== CPU Timing Report ===\n";
    out << "case_name                 : " << case_name << "\n";
    out << "riemann_solver            : " << opts.solver_name << "\n";
    out << "resolution_scale          : " << opts.resolution_scale << "\n";
    out << "nx                        : " << cfg.nx    << "\n";
    out << "ny                        : " << cfg.ny    << "\n";
    out << "total_cells               : "
        << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny) << "\n";
    out << "ng                        : " << cfg.ng    << "\n";
    out << "x_min                     : " << cfg.x_min << "\n";
    out << "x_max                     : " << cfg.x_max << "\n";
    out << "y_min                     : " << cfg.y_min << "\n";
    out << "y_max                     : " << cfg.y_max << "\n";
    out << "cfl                       : " << cfg.cfl   << "\n";
    out << "t_end                     : " << cfg.t_end << "\n";
    out << "final_time                : " << final_time << "\n";
    out << "total_steps               : " << step       << "\n";
    out << "snapshots_written         : " << num_snapshots_written << "\n";
    out << "\n";

    out << "output_dir_time           : " << ts.output_dir_time        << " s\n";
    out << "config_and_init_time      : " << ts.config_and_init_time   << " s\n";
    out << "workspace_alloc_time      : " << ts.workspace_alloc_time   << " s\n";
    out << "snapshot_schedule_time    : " << ts.snapshot_schedule_time << " s\n";
    out << "main_loop_time            : " << ts.main_loop_time         << " s\n";
    out << "  boundary_time           : " << ts.boundary_time          << " s\n";
    out << "  compute_dt_time         : " << ts.compute_dt_time        << " s\n";
    out << "  advance_time            : " << ts.advance_time           << " s\n";
    out << "  snapshot_write_time     : " << ts.snapshot_write_time    << " s\n";
    out << "total_program_time        : " << ts.total_program_time     << " s\n";
    out << "\n";

    if (step > 0) {
        out << "avg_time_per_step         : " << (ts.main_loop_time  / step) << " s\n";
        out << "avg_boundary_per_step     : " << (ts.boundary_time   / step) << " s\n";
        out << "avg_compute_dt_per_step   : " << (ts.compute_dt_time / step) << " s\n";
        out << "avg_advance_per_step      : " << (ts.advance_time    / step) << " s\n";
    } else {
        out << "avg_time_per_step         : 0.000000 s\n";
        out << "avg_boundary_per_step     : 0.000000 s\n";
        out << "avg_compute_dt_per_step   : 0.000000 s\n";
        out << "avg_advance_per_step      : 0.000000 s\n";
    }

    const double measured =
        ts.boundary_time +
        ts.compute_dt_time +
        ts.advance_time +
        ts.snapshot_write_time;

    out << "other_loop_time           : "
        << std::max(0.0, ts.main_loop_time - measured) << " s\n";
}

void print_timing_report(const TimingStats& ts, int step) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== CPU Timing Summary ===\n";
    std::cout << "output_dir_time           : " << ts.output_dir_time        << " s\n";
    std::cout << "config_and_init_time      : " << ts.config_and_init_time   << " s\n";
    std::cout << "workspace_alloc_time      : " << ts.workspace_alloc_time   << " s\n";
    std::cout << "snapshot_schedule_time    : " << ts.snapshot_schedule_time << " s\n";
    std::cout << "main_loop_time            : " << ts.main_loop_time         << " s\n";
    std::cout << "  boundary_time           : " << ts.boundary_time          << " s\n";
    std::cout << "  compute_dt_time         : " << ts.compute_dt_time        << " s\n";
    std::cout << "  advance_time            : " << ts.advance_time           << " s\n";
    std::cout << "  snapshot_write_time     : " << ts.snapshot_write_time    << " s\n";
    std::cout << "total_program_time        : " << ts.total_program_time     << " s\n";

    if (step > 0) {
        std::cout << "avg_time_per_step         : " << (ts.main_loop_time  / step) << " s\n";
        std::cout << "avg_boundary_per_step     : " << (ts.boundary_time   / step) << " s\n";
        std::cout << "avg_compute_dt_per_step   : " << (ts.compute_dt_time / step) << " s\n";
        std::cout << "avg_advance_per_step      : " << (ts.advance_time    / step) << " s\n";
    }

    std::cout << "==========================\n";
}

} // namespace

int main(int argc, char** argv) {
    const auto program_start = Clock::now();
    TimingStats timings;

    const RunOptions opts = parse_run_options(argc, argv);

    if (opts.write_output) {
        const auto t0 = Clock::now();
        ensure_output_dir();
        const auto t1 = Clock::now();
        timings.output_dir_time = Seconds(t1 - t0).count();
    }

    const std::string case_name = "shock_bubble";

    CaseConfig cfg;
    Grid2D U;
    Grid2D U_mid;
    Grid2D U_next;

    // ----------------------------------------------------------
    // Config + initial condition
    // ----------------------------------------------------------
    {
        const auto t0 = Clock::now();

        cfg = get_case_config(case_name);
        cfg.nx *= opts.resolution_scale;
        cfg.ny *= opts.resolution_scale;

        U = make_n_grid(case_name, opts.resolution_scale);

        U_mid = Grid2D(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        U_next = Grid2D(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        timings.config_and_init_time = Seconds(Clock::now() - t0).count();
    }

    std::cout << "[CPU] case_name          : " << case_name << "\n";
    std::cout << "[CPU] riemann_solver     : " << opts.solver_name << "\n";
    std::cout << "[CPU] resolution_scale   : " << opts.resolution_scale << "\n";
    std::cout << "[CPU] nx                 : " << cfg.nx << "\n";
    std::cout << "[CPU] ny                 : " << cfg.ny << "\n";
    std::cout << "[CPU] total_cells        : "
              << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny)
              << "\n";
    std::cout << "[CPU] write_output       : "
              << (opts.write_output ? "true" : "false") << "\n";
    std::cout << "[CPU] num_snapshots      : "
              << (opts.write_output ? opts.num_snapshots : 0) << "\n";

    // ----------------------------------------------------------
    // Workspace allocation
    // ----------------------------------------------------------
    CpuWorkspace cpu_ws;

    {
        const auto t0 = Clock::now();

        cpu_ws.init(cfg.nx, cfg.ny);

        const auto t1 = Clock::now();
        timings.workspace_alloc_time = Seconds(t1 - t0).count();
    }

    // ----------------------------------------------------------
    // Snapshot schedule
    // ----------------------------------------------------------
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
                cfg.t_end * static_cast<double>(k) / static_cast<double>(num_snapshots)
            );
        }

        const auto t1 = Clock::now();
        timings.snapshot_schedule_time = Seconds(t1 - t0).count();
    }

    // ----------------------------------------------------------
    // Main time loop
    // ----------------------------------------------------------
    const auto loop_start = Clock::now();

    while (t < cfg.t_end) {
        {
            const auto t0 = Clock::now();

            apply_transmissive_boundary(U);

            const auto t1 = Clock::now();
            timings.boundary_time += Seconds(t1 - t0).count();
        }

        double dt = 0.0;

        {
            const auto t0 = Clock::now();

            dt = compute_dt(U, cfg.cfl);

            const auto t1 = Clock::now();
            timings.compute_dt_time += Seconds(t1 - t0).count();
        }

        if (t + dt > cfg.t_end) {
            dt = cfg.t_end - t;
        }

        {
            const auto t0 = Clock::now();

            advance_second_order(
                U,
                U_mid,
                U_next,
                dt,
                cpu_ws,
                opts.solver
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

            std::ostringstream snapshot_prefix;
            snapshot_prefix << "cpu_"
                            << opts.solver_name
                            << "_n"
                            << opts.resolution_scale;

            const std::string filename =
                make_snapshot_name(snapshot_prefix.str(), next_snapshot + 1);

            write_grid_csv(U, filename);

            const auto t1 = Clock::now();
            timings.snapshot_write_time += Seconds(t1 - t0).count();

            std::cout << "[CPU] Wrote snapshot " << (next_snapshot + 1)
                      << " at t = " << t
                      << " -> " << filename << "\n";

            ++next_snapshot;
        }
    }

    const auto loop_end = Clock::now();
    timings.main_loop_time = Seconds(loop_end - loop_start).count();

    const auto program_end = Clock::now();
    timings.total_program_time = Seconds(program_end - program_start).count();

    // ----------------------------------------------------------
    // Summary
    // ----------------------------------------------------------
    std::cout << "[CPU] Finished.\n";
    std::cout << "[CPU] Final time = " << t << "\n";
    std::cout << "[CPU] Total steps = " << step << "\n";

    print_timing_report(timings, step);

    if (opts.write_output) {
        std::ostringstream timing_prefix;
        timing_prefix << "cpu_"
                      << opts.solver_name
                      << "_n"
                      << opts.resolution_scale;

        const std::string timing_file = make_timing_name(timing_prefix.str());

        write_timing_report(
            timing_file,
            case_name,
            cfg,
            opts,
            step,
            t,
            next_snapshot,
            timings
        );

        std::cout << "[CPU] Timing report written to " << timing_file << "\n";
    } else {
        std::cout << "[CPU] Output disabled. No files were written.\n";
    }

    return 0;
}