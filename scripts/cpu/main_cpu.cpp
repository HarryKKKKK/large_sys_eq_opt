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
    if (name == "force" || name == "FORCE" || name == "Force") {
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

    if (argc >= 2) {
        opts.resolution_scale = std::stoi(argv[1]);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--output") {
            opts.write_output = true;
            opts.num_snapshots = 5;
        } else if (arg == "--timing-only") {
            // Write only a timing report; do not write solution snapshots.
            opts.write_output = true;
            opts.num_snapshots = 0;
        } else if (arg == "--case") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--case requires a value, e.g. shock_bubble or blast_wave.");
            }
            opts.case_name = argv[++i];

            // Validate case name early.
            (void)parse_case_id(opts.case_name);
        } else if (arg == "--solver") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--solver requires a value: hll, hllc, exact, or force.");
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

std::string make_output_prefix(const RunOptions& opts) {
    std::ostringstream oss;
    oss << "cpu_"
        << opts.case_name
        << "_"
        << opts.solver_name
        << "_n"
        << opts.resolution_scale;
    return oss.str();
}

void write_grid_csv(const Grid2D& grid, const CaseConfig& cfg, const std::string& filename) {
    std::ofstream out(filename);
    out << "i,j,x,y,rho,rhou,rhov,E\n";

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Conserved& U = grid(i, j);

            const double x = cfg.x_min +
                (static_cast<double>(i - cfg.ng) + 0.5) *
                ((cfg.x_max - cfg.x_min) / static_cast<double>(cfg.nx));

            const double y = cfg.y_min +
                (static_cast<double>(j - cfg.ng) + 0.5) *
                ((cfg.y_max - cfg.y_min) / static_cast<double>(cfg.ny));

            out << i << ","
                << j << ","
                << x << ","
                << y << ","
                << U.rho << ","
                << U.rhou << ","
                << U.rhov << ","
                << U.E << "\n";
        }
    }
}

void write_timing_report(const std::string& filename,
                         const RunOptions& opts,
                         const CaseConfig& cfg,
                         int step,
                         double final_time,
                         int num_snapshots_written,
                         double main_loop_time,
                         double total_program_time) {
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(6);

    out << "=== CPU Timing Report ===\n";
    out << "case_name                 : " << opts.case_name << "\n";
    out << "riemann_solver            : " << opts.solver_name << "\n";
    out << "resolution_scale          : " << opts.resolution_scale << "\n";
    out << "nx                        : " << cfg.nx << "\n";
    out << "ny                        : " << cfg.ny << "\n";
    out << "total_cells               : "
        << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny) << "\n";
    out << "ng                        : " << cfg.ng << "\n";
    out << "x_min                     : " << cfg.x_min << "\n";
    out << "x_max                     : " << cfg.x_max << "\n";
    out << "y_min                     : " << cfg.y_min << "\n";
    out << "y_max                     : " << cfg.y_max << "\n";
    out << "cfl                       : " << cfg.cfl << "\n";
    out << "t_end                     : " << cfg.t_end << "\n";
    out << "final_time                : " << final_time << "\n";
    out << "total_steps               : " << step << "\n";
    out << "snapshots_written         : " << num_snapshots_written << "\n";
    out << "\n";

    out << "main_loop_time            : " << main_loop_time << " s\n";
    out << "total_program_time        : " << total_program_time << " s\n";

    if (step > 0) {
        out << "avg_time_per_step         : " << (main_loop_time / step) << " s\n";
    } else {
        out << "avg_time_per_step         : 0.000000 s\n";
    }
}

void print_timing_summary(double main_loop_time,
                          double total_program_time,
                          int step) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== CPU Timing Summary ===\n";
    std::cout << "main_loop_time            : " << main_loop_time << " s\n";
    std::cout << "total_program_time        : " << total_program_time << " s\n";

    if (step > 0) {
        std::cout << "avg_time_per_step         : "
                  << (main_loop_time / step) << " s\n";
    }

    std::cout << "==========================\n";
}

} // namespace

int main(int argc, char** argv) {
    const auto program_start = Clock::now();

    const RunOptions opts = parse_run_options(argc, argv);

    if (opts.write_output) {
        ensure_output_dir();
    }

    CaseConfig cfg = get_n_case_config(opts.case_name, opts.resolution_scale);

    Grid2D U = make_n_grid(opts.case_name, opts.resolution_scale);

    Grid2D U_mid(
        cfg.nx, cfg.ny, cfg.ng,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max
    );

    Grid2D U_next(
        cfg.nx, cfg.ny, cfg.ng,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max
    );

    CpuWorkspace cpu_ws;
    cpu_ws.init(cfg.nx, cfg.ny);

    std::cout << "[CPU] case_name          : " << opts.case_name << "\n";
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

    const int num_snapshots = opts.write_output ? opts.num_snapshots : 0;
    std::vector<double> snapshot_times;
    snapshot_times.reserve(num_snapshots);

    for (int k = 1; k <= num_snapshots; ++k) {
        snapshot_times.push_back(
            cfg.t_end * static_cast<double>(k) /
            static_cast<double>(num_snapshots)
        );
    }

    double t = 0.0;
    int step = 0;
    int next_snapshot = 0;

    const std::string output_prefix = make_output_prefix(opts);

    const auto loop_start = Clock::now();

    while (t < cfg.t_end) {
        apply_transmissive_boundary(U);

        double dt = compute_dt(U, cfg.cfl);

        if (t + dt > cfg.t_end) {
            dt = cfg.t_end - t;
        }

        advance_second_order(
            U,
            U_mid,
            U_next,
            dt,
            cpu_ws,
            opts.solver
        );

        std::swap(U, U_next);

        t += dt;
        ++step;

        while (opts.write_output &&
               next_snapshot < num_snapshots &&
               t >= snapshot_times[next_snapshot]) {
            const std::string filename =
                make_snapshot_name(output_prefix, next_snapshot + 1);

            write_grid_csv(U, cfg, filename);

            std::cout << "[CPU] Wrote snapshot " << (next_snapshot + 1)
                      << " at t = " << t
                      << " -> " << filename << "\n";

            ++next_snapshot;
        }
    }

    const auto loop_end = Clock::now();
    const double main_loop_time = Seconds(loop_end - loop_start).count();

    const auto program_end = Clock::now();
    const double total_program_time = Seconds(program_end - program_start).count();

    std::cout << "[CPU] Finished.\n";
    std::cout << "[CPU] Final time = " << t << "\n";
    std::cout << "[CPU] Total steps = " << step << "\n";

    print_timing_summary(main_loop_time, total_program_time, step);

    if (opts.write_output) {
        const std::string timing_file = make_timing_name(output_prefix);

        write_timing_report(
            timing_file,
            opts,
            cfg,
            step,
            t,
            next_snapshot,
            main_loop_time,
            total_program_time
        );

        std::cout << "[CPU] Timing report written to " << timing_file << "\n";
    } else {
        std::cout << "[CPU] Output disabled. No files were written.\n";
    }

    return 0;
}
