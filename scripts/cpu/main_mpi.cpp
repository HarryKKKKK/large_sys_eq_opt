#include "cpu/grid_cpu.hpp"
#include "cpu/solver_mpi.hpp"
#include "init.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include <mpi.h>

#include <algorithm>
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
        } else if (arg == "--no-output") {
            opts.write_output = false;
            opts.num_snapshots = 0;
        } else if (arg == "--timing-only") {
            // Kept for compatibility with old scripts.
            // In this clean version, it simply disables snapshot output.
            opts.write_output = false;
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

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
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

        if (rank == 0) {
            std::cout << "[MPI] case_name          : " << opts.case_name << "\n";
            std::cout << "[MPI] riemann_solver     : " << opts.solver_name << "\n";
            std::cout << "[MPI] resolution_scale   : " << opts.resolution_scale << "\n";
            std::cout << "[MPI] nx                 : " << cfg.nx << "\n";
            std::cout << "[MPI] ny                 : " << cfg.ny << "\n";
            std::cout << "[MPI] total_cells        : "
                      << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny)
                      << "\n";
            std::cout << "[MPI] ranks              : " << size << "\n";
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
            snapshot_times.reserve(num_snapshots);
            for (int k = 1; k <= num_snapshots; ++k) {
                snapshot_times.push_back(
                    cfg.t_end * static_cast<double>(k) /
                    static_cast<double>(num_snapshots)
                );
            }
        }

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
                const std::string filename = make_snapshot_name(
                    opts.case_name,
                    opts.solver_name,
                    opts.resolution_scale,
                    rank,
                    next_snapshot + 1
                );

                write_grid_csv(U, cfg, filename);

                if (rank == 0) {
                    std::cout << "[MPI] Wrote snapshot " << (next_snapshot + 1)
                              << " at t = " << t << "\n";
                }

                ++next_snapshot;
            }
        }

        if (rank == 0) {
            std::cout << "[MPI] Finished.\n";
            std::cout << "[MPI] Final time = " << t << "\n";
            std::cout << "[MPI] Total steps = " << step << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[MPI rank " << rank << "] ERROR: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
