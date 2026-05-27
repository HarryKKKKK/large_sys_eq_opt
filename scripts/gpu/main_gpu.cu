#include "cpu/grid_cpu.hpp"
#include "init.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include "gpu/grid_gpu.cuh"
#include "gpu/solver_gpu.cuh"

#include "riemann.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

inline void check_cuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " in " << call << " : " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

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

    if (argc >= 2) {
        opts.resolution_scale = std::stoi(argv[1]);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--output") {
            opts.write_output = true;
            opts.num_snapshots = 5;
        } else if (arg == "--no-output") {
            opts.write_output = false;
            opts.num_snapshots = 0;
        } else if (arg == "--timing-only") {
            // Kept only for compatibility with old scripts.
            // This clean version does not perform or print any timing.
            opts.write_output = false;
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

void write_aos_csv(const std::vector<Conserved>& data,
                   int nx, int ny, int ng,
                   double x_min, double y_min,
                   double dx, double dy,
                   const std::string& filename) {
    std::ofstream out(filename);
    out << "i,j,x,y,rho,rhou,rhov,E\n";

    const int total_nx = nx + 2 * ng;

    for (int j = ng; j < ng + ny; ++j) {
        for (int i = ng; i < ng + nx; ++i) {
            const std::size_t idx = static_cast<std::size_t>(j * total_nx + i);
            const double x = x_min + (static_cast<double>(i - ng) + 0.5) * dx;
            const double y = y_min + (static_cast<double>(j - ng) + 0.5) * dy;

            out << i << ","
                << j << ","
                << x << ","
                << y << ","
                << data[idx].rho << ","
                << data[idx].rhou << ","
                << data[idx].rhov << ","
                << data[idx].E << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const RunOptions opts = parse_run_options(argc, argv);

        if (opts.write_output) {
            ensure_output_dir();
        }

        CaseConfig cfg = get_n_case_config(opts.case_name, opts.resolution_scale);
        Grid2D host_init = make_n_grid(opts.case_name, opts.resolution_scale);

        std::cout << "[GPU] case_name        : " << opts.case_name << "\n";
        std::cout << "[GPU] riemann_solver   : " << opts.solver_name << "\n";
        std::cout << "[GPU] resolution_scale : " << opts.resolution_scale << "\n";
        std::cout << "[GPU] nx               : " << cfg.nx << "\n";
        std::cout << "[GPU] ny               : " << cfg.ny << "\n";
        std::cout << "[GPU] total_cells      : "
                  << static_cast<long long>(cfg.nx) * static_cast<long long>(cfg.ny)
                  << "\n";
        std::cout << "[GPU] write_output     : "
                  << (opts.write_output ? "true" : "false") << "\n";
        std::cout << "[GPU] num_snapshots    : "
                  << (opts.write_output ? opts.num_snapshots : 0) << "\n";

        Grid2DGPU d_U(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        Grid2DGPU d_U_mid(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        Grid2DGPU d_U_next(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        GpuWorkspace ws;
        init_gpu_workspace(ws, d_U);

        d_U.upload_from_aos(host_init.data());

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

        const double dx = (cfg.x_max - cfg.x_min) / static_cast<double>(cfg.nx);
        const double dy = (cfg.y_max - cfg.y_min) / static_cast<double>(cfg.ny);

        while (t < cfg.t_end) {
            // In the integrated shared-memory + clamp-boundary GPU solver,
            // transmissive boundary conditions are handled inside the stencil
            // load functions used by the sweep kernels. Therefore, there is no
            // explicit boundary kernel in this main loop.
            double dt = compute_dt_gpu(d_U, ws, cfg.cfl);

            if (t + dt > cfg.t_end) {
                dt = cfg.t_end - t;
            }

            advance_second_order_gpu(
                d_U,
                d_U_mid,
                d_U_next,
                ws,
                dt,
                opts.solver
            );

            d_U.swap(d_U_next);

            t += dt;
            ++step;

            while (opts.write_output &&
                   next_snapshot < num_snapshots &&
                   t >= snapshot_times[next_snapshot]) {
                std::vector<Conserved> host_snapshot;
                d_U.download_to_aos(host_snapshot);

                std::ostringstream snapshot_prefix;
                snapshot_prefix << "gpu_"
                                << opts.case_name
                                << "_"
                                << opts.solver_name
                                << "_n"
                                << opts.resolution_scale;

                const std::string filename =
                    make_snapshot_name(snapshot_prefix.str(), next_snapshot + 1);

                write_aos_csv(
                    host_snapshot,
                    cfg.nx, cfg.ny, cfg.ng,
                    cfg.x_min, cfg.y_min,
                    dx, dy,
                    filename
                );

                std::cout << "[GPU] Wrote snapshot " << (next_snapshot + 1)
                          << " at t = " << t
                          << " -> " << filename << "\n";

                ++next_snapshot;
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        free_gpu_workspace(ws);

        std::cout << "[GPU] Finished.\n";
        std::cout << "[GPU] Final time = " << t << "\n";
        std::cout << "[GPU] Total steps = " << step << "\n";

        if (!opts.write_output) {
            std::cout << "[GPU] Output disabled. No files were written.\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[GPU] Error: " << e.what() << "\n";
        return 1;
    }
}
