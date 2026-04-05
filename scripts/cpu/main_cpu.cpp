#include "cpu/boundary_cpu.hpp"
#include "cpu/grid_cpu.hpp"
#include "init.hpp"
#include "cpu/solver_cpu.hpp"
#include "test_cases.hpp"
#include "types.hpp"

int main() {
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

    while (t < cfg.t_end) {
        apply_transmissive_boundary(U);
        double dt = compute_dt(U, cfg.cfl);
        if (t + dt > cfg.t_end) {
            dt = cfg.t_end - t;
        }

        advance_first_order(U, U_next, dt);
        std::swap(U, U_next);
        t += dt;
    }

    return 0;
}