#include "cpu/grid_cpu.hpp"
#include "init.hpp"
#include "test_cases.hpp"
#include "types.hpp"

#include "gpu/boundary_gpu.cuh"
#include "gpu/grid_gpu.cuh"
#include "gpu/solver_gpu.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

} 

int main() {
    ensure_output_dir();

    const std::string case_name = "shock_bubble";
    const CaseConfig cfg = get_case_config(case_name);

    Grid2D host_init = make_initial_grid(case_name);

    Grid2DGPU d_U(
        cfg.nx, cfg.ny, cfg.ng,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max
    );
    Grid2DGPU d_U_next(
        cfg.nx, cfg.ny, cfg.ng,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max
    );

    d_U.upload_from_aos(host_init.data());
    d_U_next.upload_from_aos(host_init.data());

    double t = 0.0;
    int step = 0;

    const int num_snapshots = 5;
    std::vector<double> snapshot_times;
    snapshot_times.reserve(num_snapshots);
    for (int k = 1; k <= num_snapshots; ++k) {
        snapshot_times.push_back(cfg.t_end * static_cast<double>(k) / num_snapshots);
    }
    int next_snapshot = 0;

    const double dx = (cfg.x_max - cfg.x_min) / static_cast<double>(cfg.nx);
    const double dy = (cfg.y_max - cfg.y_min) / static_cast<double>(cfg.ny);

    while (t < cfg.t_end) {
        apply_transmissive_boundary_gpu(d_U);

        double dt = compute_dt_gpu(d_U, cfg.cfl);
        if (t + dt > cfg.t_end) {
            dt = cfg.t_end - t;
        }

        advance_first_order_gpu(d_U, d_U_next, dt);
        d_U.swap(d_U_next);

        t += dt;
        ++step;

        while (next_snapshot < num_snapshots && t >= snapshot_times[next_snapshot]) {
            std::vector<Conserved> host_snapshot;
            d_U.download_to_aos(host_snapshot);

            const std::string filename = make_snapshot_name("gpu", next_snapshot + 1);
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

    std::cout << "[GPU] Finished.\n";
    std::cout << "[GPU] Final time = " << t << "\n";
    std::cout << "[GPU] Total steps = " << step << "\n";

    return 0;
}