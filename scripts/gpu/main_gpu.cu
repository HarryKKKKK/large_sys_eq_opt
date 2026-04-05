#include "cpu/grid_cpu.hpp"
#include "init.hpp"
#include "test_cases.hpp"

#include "gpu/boundary_gpu.cuh"
#include "gpu/grid_gpu.cuh"
#include "gpu/solver_gpu.cuh"

int main() {
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

    d_U.copy_from_host(host_init);
    d_U_next.copy_from_host(host_init);

    double t = 0.0;

    while (t < cfg.t_end) {
        apply_transmissive_boundary_gpu(d_U);
        double dt = compute_dt_gpu(d_U, cfg.cfl);
        if (t + dt > cfg.t_end) {
            dt = cfg.t_end - t;
        }

        advance_first_order_gpu(d_U, d_U_next, dt);
        std::swap(d_U, d_U_next);
        t += dt;
    }

    return 0;
}