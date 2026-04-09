#include "cpu/boundary_cpu.hpp"
#include "cpu/grid_cpu.hpp"
#include "init.hpp"
#include "cpu/solver_cpu.hpp"
#include "test_cases.hpp"
#include "types.hpp"

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

void write_grid_csv(const Grid2D& grid, const std::string& filename) {
    std::ofstream out(filename);
    out << "i,j,x,y,rho,rhou,rhov,E\n";

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Conserved& U = grid(i, j);
            out << i << ","
                << j << ","
                << grid.x_center(i) << ","
                << grid.y_center(j) << ","
                << U.rho << ","
                << U.rhou << ","
                << U.rhov << ","
                << U.E << "\n";
        }
    }
}

} 

int main() {
    ensure_output_dir();

    const std::string case_name = "shock_bubble";
    const CaseConfig cfg = get_case_config(case_name);

    Grid2D U = make_initial_grid(case_name);
    Grid2D U_next(
        cfg.nx, cfg.ny, cfg.ng,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max
    );
    initialise_grid(U_next, case_name);

    double t = 0.0;
    int step = 0;

    const int num_snapshots = 5;
    std::vector<double> snapshot_times;
    snapshot_times.reserve(num_snapshots);
    for (int k = 1; k <= num_snapshots; ++k) {
        snapshot_times.push_back(cfg.t_end * static_cast<double>(k) / num_snapshots);
    }
    int next_snapshot = 0;

    while (t < cfg.t_end) {
        apply_transmissive_boundary(U);

        double dt = compute_dt(U, cfg.cfl);
        if (t + dt > cfg.t_end) {
            dt = cfg.t_end - t;
        }

        advance_first_order(U, U_next, dt);
        std::swap(U, U_next);

        t += dt;
        ++step;

        while (next_snapshot < num_snapshots && t >= snapshot_times[next_snapshot]) {
            const std::string filename = make_snapshot_name("cpu", next_snapshot + 1);
            write_grid_csv(U, filename);

            std::cout << "[CPU] Wrote snapshot " << (next_snapshot + 1)
                      << " at t = " << t
                      << " -> " << filename << "\n";

            ++next_snapshot;
        }
    }

    std::cout << "[CPU] Finished.\n";
    std::cout << "[CPU] Final time = " << t << "\n";
    std::cout << "[CPU] Total steps = " << step << "\n";

    return 0;
}