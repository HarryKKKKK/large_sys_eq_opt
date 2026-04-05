#include "cpu/solver_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cpu/boundary_cpu.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

double compute_dt(const Grid2D& grid, double cfl) {
    double max_speed = 0.0;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));
            const double a = phys::sound_speed(V);
            const double sx = std::abs(V.u) + a;
            const double sy = std::abs(V.v) + a;

            max_speed = std::max(max_speed, std::max(sx, sy));
        }
    }
    const double dt_x = grid.dx() / max_speed;
    const double dt_y = grid.dy() / max_speed;

    return cfl * std::min(dt_x, dt_y);
}

void advance_first_order(const Grid2D& Uold, Grid2D& Unew, double dt) {
    const int ib = Uold.i_begin();
    const int ie = Uold.i_end();
    const int jb = Uold.j_begin();
    const int je = Uold.j_end();

    const double dx = Uold.dx();
    const double dy = Uold.dy();

    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const Conserved Fx_p = hll_flux(Uold(i, j),     Uold(i + 1, j), Direction::X);
            const Conserved Fx_m = hll_flux(Uold(i - 1, j), Uold(i, j),     Direction::X);

            const Conserved Fy_p = hll_flux(Uold(i, j),     Uold(i, j + 1), Direction::Y);
            const Conserved Fy_m = hll_flux(Uold(i, j - 1), Uold(i, j),     Direction::Y);

            Unew(i, j) = Uold(i, j)
                       - (dt / dx) * (Fx_p - Fx_m)
                       - (dt / dy) * (Fy_p - Fy_m);
        }
    }

    apply_transmissive_boundary(Unew);
}