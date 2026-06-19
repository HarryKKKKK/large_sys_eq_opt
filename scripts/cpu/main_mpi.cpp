#include "cpu/grid_cpu.hpp"
#include "cpu/solver_mpi.hpp"
#include "init.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include <mpi.h>

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
    bool write_output    = false;
    int num_snapshots    = 5;

    std::string case_name   = "shock_bubble";
    RiemannSolver solver    = RiemannSolver::HLL;
    std::string solver_name = "hll";

    // 2D Cartesian process grid. 0 = let MPI_Dims_create choose automatically.
    int px = 0;
    int py = 0;

    // dt recompute interval (1 = every step, N = every N steps).
    int dt_interval = 4;
};

RiemannSolver parse_riemann_solver(const std::string& name) {
    if (name == "hll"  || name == "HLL")                       return RiemannSolver::HLL;
    if (name == "hllc" || name == "HLLC")                      return RiemannSolver::HLLC;
    if (name == "exact"|| name == "Exact"|| name == "EXACT")   return RiemannSolver::Exact;
    if (name == "force"|| name == "Force"|| name == "FORCE")   return RiemannSolver::FORCE;
    throw std::runtime_error(
        "Unknown Riemann solver: " + name +
        ". Supported: hll, hllc, exact, force.");
}

std::string riemann_solver_to_string(RiemannSolver solver) {
    switch (solver) {
        case RiemannSolver::HLL:   return "hll";
        case RiemannSolver::HLLC:  return "hllc";
        case RiemannSolver::Exact: return "exact";
        case RiemannSolver::FORCE: return "force";
        default:                   return "unknown";
    }
}

RunOptions parse_run_options(int argc, char** argv) {
    RunOptions opts;

    if (argc >= 2 && argv[1][0] != '-')
        opts.resolution_scale = std::stoi(argv[1]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--output") {
            opts.write_output  = true;
            opts.num_snapshots = 5;
        } else if (arg == "--timing-only") {
            opts.write_output  = false;
            opts.num_snapshots = 0;
        } else if (arg == "--case") {
            if (i + 1 >= argc) throw std::runtime_error("--case requires a value.");
            opts.case_name = argv[++i];
            (void)parse_case_id(opts.case_name);
        } else if (arg == "--solver") {
            if (i + 1 >= argc) throw std::runtime_error("--solver requires a value.");
            opts.solver_name = argv[++i];
            opts.solver      = parse_riemann_solver(opts.solver_name);
            opts.solver_name = riemann_solver_to_string(opts.solver);
        } else if (arg == "--px") {
            if (i + 1 >= argc) throw std::runtime_error("--px requires a value.");
            opts.px = std::stoi(argv[++i]);
        } else if (arg == "--py") {
            if (i + 1 >= argc) throw std::runtime_error("--py requires a value.");
            opts.py = std::stoi(argv[++i]);
        } else if (arg == "--dt-interval") {
            if (i + 1 >= argc) throw std::runtime_error("--dt-interval requires a value.");
            opts.dt_interval = std::stoi(argv[++i]);
            if (opts.dt_interval < 1)
                throw std::runtime_error("--dt-interval must be >= 1.");
        } else if (arg[0] == '-') {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.resolution_scale <= 0)
        throw std::runtime_error("resolution_scale must be positive.");

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
    const std::string& case_name, const std::string& solver_name,
    int scale, int rank, int snap_id
) {
    std::ostringstream oss;
    oss << "outputs/mpi_" << case_name << "_" << solver_name
        << "_n" << scale
        << "_rank" << std::setw(5) << std::setfill('0') << rank
        << "_snapshot_" << snap_id << ".csv";
    return oss.str();
}

void write_local_grid_csv(
    const Grid2D& grid, const MpiDecomp2D& mp, const std::string& filename
) {
    std::ofstream out(filename);
    out << "i,j,x,y,rho,rhou,rhov,E\n";

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        const int global_j = mp.ng + mp.y0_global + (j - grid.j_begin());
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Conserved& U = grid(i, j);
            out << i                  << ","
                << global_j           << ","
                << grid.x_center(i)  << ","
                << grid.y_center(j)  << ","
                << U.rho  << ","
                << U.rhou << ","
                << U.rhov << ","
                << U.E    << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        const RunOptions opts = parse_run_options(argc, argv);

        if (opts.write_output && rank == 0)
            ensure_output_dir();
        MPI_Barrier(MPI_COMM_WORLD);

        CaseConfig cfg = scale_case_config(
            get_case_config(opts.case_name), opts.resolution_scale);

        MpiDecomp2D mp = make_cartesian_decomp(
            cfg, MPI_COMM_WORLD, opts.px, opts.py);

        Grid2D U = make_local_mpi_grid(opts.case_name, cfg, mp);

        Grid2D U_mid(
            mp.nx_local, mp.ny_local, cfg.ng,
            U.x_min(), U.x_max(),
            U.y_min(), U.y_max()
        );
        Grid2D U_next(
            mp.nx_local, mp.ny_local, cfg.ng,
            U.x_min(), U.x_max(),
            U.y_min(), U.y_max()
        );

        MpiCpuWorkspace ws;
        ws.init(mp.nx_local, mp.ny_local, cfg.ng);
        ws.dt_interval = opts.dt_interval;

        if (rank == 0) {
            std::cout << "[MPI] case_name          : " << opts.case_name    << "\n";
            std::cout << "[MPI] riemann_solver     : " << opts.solver_name  << "\n";
            std::cout << "[MPI] resolution_scale   : " << opts.resolution_scale << "\n";
            std::cout << "[MPI] mpi_ranks          : " << mp.size           << "\n";
            std::cout << "[MPI] process_grid       : " << mp.px << "x" << mp.py << "\n";
            std::cout << "[MPI] nx                 : " << cfg.nx            << "\n";
            std::cout << "[MPI] ny                 : " << cfg.ny            << "\n";
            std::cout << "[MPI] total_cells        : "
                      << static_cast<long long>(cfg.nx) *
                         static_cast<long long>(cfg.ny) << "\n";
            std::cout << "[MPI] dt_interval        : " << opts.dt_interval  << "\n";
            std::cout << "[MPI] write_output       : "
                      << (opts.write_output ? "true" : "false") << "\n";
        }

        double t    = 0.0;
        int    step = 0;

        const int num_snapshots = opts.write_output ? opts.num_snapshots : 0;
        std::vector<double> snapshot_times;
        int next_snapshot = 0;

        if (opts.write_output && num_snapshots > 0) {
            snapshot_times.reserve(num_snapshots);
            for (int k = 1; k <= num_snapshots; ++k)
                snapshot_times.push_back(
                    cfg.t_end * static_cast<double>(k) /
                    static_cast<double>(num_snapshots));
        }

        // Ensure ghost cells of the initial U are valid before the first x-sweep.
        apply_transmissive_boundary_mpi(U, mp, ws);

        MPI_Barrier(MPI_COMM_WORLD);

        while (t < cfg.t_end) {
            double dt = compute_dt_mpi_cached(U, mp, cfg.cfl, ws);

            if (t + dt > cfg.t_end)
                dt = cfg.t_end - t;

            advance_second_order_mpi(U, U_mid, U_next, mp, dt, opts.solver, ws);
            std::swap(U, U_next);

            // Fill ghost cells of the new U so the next x-sweep can use them.
            // This is the only boundary call per timestep (no redundant calls
            // inside advance_second_order_mpi).
            apply_transmissive_boundary_mpi(U, mp, ws);

            t += dt;
            ++step;

            while (opts.write_output &&
                   next_snapshot < num_snapshots &&
                   t >= snapshot_times[next_snapshot]) {
                write_local_grid_csv(
                    U, mp,
                    make_snapshot_name(
                        opts.case_name, opts.solver_name,
                        opts.resolution_scale, rank, next_snapshot + 1));
                ++next_snapshot;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int global_steps = 0;
        MPI_Reduce(&step, &global_steps, 1, MPI_INT, MPI_MAX, 0, mp.comm);

        if (rank == 0) {
            std::cout << "[MPI] Finished.\n";
            std::cout << "[MPI] Final time  = " << t            << "\n";
            std::cout << "[MPI] Total steps = " << global_steps << "\n";
            if (!opts.write_output)
                std::cout << "[MPI] Output disabled.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] ERROR: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}