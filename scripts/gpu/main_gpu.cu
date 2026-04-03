#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "gpu/grid_gpu.cuh"
#include "gpu/boundary_gpu.cuh"
#include "gpu/solver_gpu.cuh"

#include "cpu/grid_cpu.hpp"  
#include "physics.hpp"
#include "test_cases.hpp"
#include "types.hpp"

void write_density_csv(const Grid2D& grid, const std::string& filename);

int main() {
    try {
        const std::string case_name = "shock_bubble";
        const CaseConfig cfg = get_case_config(case_name);

        Grid2D host_init(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );
        initialise_case(host_init, case_name);

        Grid2DGPU d_old(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );
        Grid2DGPU d_new(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        d_old.copy_from_host(host_init);
        d_new.copy_from_host(host_init);

        double t = 0.0;
        int step = 0;

        auto t0 = std::chrono::steady_clock::now();

        while (t < cfg.t_end) {
            apply_transmissive_boundary_gpu(d_old);

            double dt = compute_dt_gpu(d_old, cfg.cfl);
            if (t + dt > cfg.t_end) {
                dt = cfg.t_end - t;
            }

            advance_first_order_gpu(d_old, d_new, dt);

            std::swap(d_old, d_new);

            t += dt;
            ++step;

            if (step % 100 == 0) {
                std::cout << "Step " << step
                          << ", t = " << t
                          << ", dt = " << dt << '\n';
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        const double wall_seconds =
            std::chrono::duration<double>(t1 - t0).count();

        Grid2D host_final(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );
        d_old.copy_to_host(host_final);

        write_density_csv(host_final, "shock_bubble_gpu_final.csv");

        std::cout << std::setprecision(16);
        std::cout << "\nGPU run finished.\n";
        std::cout << "Steps       : " << step << '\n';
        std::cout << "Final time  : " << t << '\n';
        std::cout << "Wall time(s): " << wall_seconds << '\n';

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}