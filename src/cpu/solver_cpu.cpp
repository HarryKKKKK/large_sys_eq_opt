#include "cpu/solver_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpu/boundary_cpu.hpp"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

inline double minmod_scalar(double a, double b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (a > 0.0) ? std::min(a, b) : std::max(a, b);
}

inline Primitive minmod_primitive(const Primitive& a, const Primitive& b) {
    return Primitive(
        minmod_scalar(a.rho, b.rho),
        minmod_scalar(a.u,   b.u),
        minmod_scalar(a.v,   b.v),
        minmod_scalar(a.p,   b.p)
    );
}

inline bool is_physical(const Primitive& V) {
    return std::isfinite(V.rho) && std::isfinite(V.u) &&
           std::isfinite(V.v)   && std::isfinite(V.p) &&
           (V.rho > kRhoFloor)  && (V.p > kPFloor);
}

inline Primitive enforce_physical_primitive(const Primitive& candidate,
                                            const Primitive& fallback) {
    return is_physical(candidate) ? candidate : fallback;
}

inline Conserved enforce_physical_conserved(const Conserved& candidate,
                                            const Conserved& fallback) {
    const Primitive Vcand = phys::cons_to_prim(candidate);
    return is_physical(Vcand) ? candidate : fallback;
}

inline Primitive limited_slope(const Primitive& Wm,
                               const Primitive& Wc,
                               const Primitive& Wp) {
    return minmod_primitive(Wc - Wm, Wp - Wc);
}

inline void reconstruct_cell_muscl_hancock(
    const Conserved& Um,
    const Conserved& Uc,
    const Conserved& Up,
    double dt_over_d,
    Direction dir,
    Conserved& U_left_star,
    Conserved& U_right_star
) {
    const Primitive Wm = phys::cons_to_prim(Um);
    const Primitive Wc = phys::cons_to_prim(Uc);
    const Primitive Wp = phys::cons_to_prim(Up);

    const Primitive slope = limited_slope(Wm, Wc, Wp);

    Primitive W_left  = Wc - 0.5 * slope;
    Primitive W_right = Wc + 0.5 * slope;

    W_left  = enforce_physical_primitive(W_left,  Wc);
    W_right = enforce_physical_primitive(W_right, Wc);

    const Conserved U_left  = phys::prim_to_cons(W_left);
    const Conserved U_right = phys::prim_to_cons(W_right);

    const Conserved F_left  = physical_flux(U_left,  dir);
    const Conserved F_right = physical_flux(U_right, dir);

    const Conserved half_update = 0.5 * dt_over_d * (F_right - F_left);

    U_left_star  = enforce_physical_conserved(U_left  - half_update, U_left);
    U_right_star = enforce_physical_conserved(U_right - half_update, U_right);
}

inline Conserved muscl_hancock_flux_x(const Grid2D& U, int i, int j, double dt_over_dx) {
    Conserved Ui_L_star, Ui_R_star;
    Conserved Uip1_L_star, Uip1_R_star;

    reconstruct_cell_muscl_hancock(
        U(i - 1, j), U(i, j), U(i + 1, j),
        dt_over_dx, Direction::X,
        Ui_L_star, Ui_R_star
    );

    reconstruct_cell_muscl_hancock(
        U(i, j), U(i + 1, j), U(i + 2, j),
        dt_over_dx, Direction::X,
        Uip1_L_star, Uip1_R_star
    );

    return hll_flux(Ui_R_star, Uip1_L_star, Direction::X);
}

inline Conserved muscl_hancock_flux_y(const Grid2D& U, int i, int j, double dt_over_dy) {
    Conserved Uj_L_star, Uj_R_star;
    Conserved Ujp1_L_star, Ujp1_R_star;

    reconstruct_cell_muscl_hancock(
        U(i, j - 1), U(i, j), U(i, j + 1),
        dt_over_dy, Direction::Y,
        Uj_L_star, Uj_R_star
    );

    reconstruct_cell_muscl_hancock(
        U(i, j), U(i, j + 1), U(i, j + 2),
        dt_over_dy, Direction::Y,
        Ujp1_L_star, Ujp1_R_star
    );

    return hll_flux(Uj_R_star, Ujp1_L_star, Direction::Y);
}

void sweep_x_second_order(const Grid2D& Uin, Grid2D& Uout, double dt) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const double dt_over_dx = dt / Uin.dx();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const Conserved Fx_p = muscl_hancock_flux_x(Uin, i,     j, dt_over_dx);
            const Conserved Fx_m = muscl_hancock_flux_x(Uin, i - 1, j, dt_over_dx);

            Uout(i, j) = Uin(i, j) - dt_over_dx * (Fx_p - Fx_m);
        }
    }
}

void sweep_y_second_order(const Grid2D& Uin, Grid2D& Uout, double dt) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const double dt_over_dy = dt / Uin.dy();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const Conserved Fy_p = muscl_hancock_flux_y(Uin, i, j,     dt_over_dy);
            const Conserved Fy_m = muscl_hancock_flux_y(Uin, i, j - 1, dt_over_dy);

            Uout(i, j) = Uin(i, j) - dt_over_dy * (Fy_p - Fy_m);
        }
    }
}

} // namespace

double compute_dt(const Grid2D& grid, double cfl) {
    double max_speed = 0.0;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(max:max_speed) schedule(static)
#endif
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));
            const double a = phys::sound_speed(V);
            const double sx = std::abs(V.u) + a;
            const double sy = std::abs(V.v) + a;
            max_speed = std::max(max_speed, std::max(sx, sy));
        }
    }

    if (max_speed <= 0.0) {
        throw std::runtime_error("compute_dt: non-positive maximum wave speed.");
    }

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed;
}

void advance_first_order(const Grid2D& Uold, Grid2D& Unew, double dt) {
    const int ib = Uold.i_begin();
    const int ie = Uold.i_end();
    const int jb = Uold.j_begin();
    const int je = Uold.j_end();

    const double dt_over_dx = dt / Uold.dx();
    const double dt_over_dy = dt / Uold.dy();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const Conserved Fx_p = hll_flux(Uold(i, j),     Uold(i + 1, j), Direction::X);
            const Conserved Fx_m = hll_flux(Uold(i - 1, j), Uold(i,     j), Direction::X);
            const Conserved Fy_p = hll_flux(Uold(i,     j), Uold(i, j + 1), Direction::Y);
            const Conserved Fy_m = hll_flux(Uold(i, j - 1), Uold(i,     j), Direction::Y);

            Unew(i, j) = Uold(i, j)
                       - dt_over_dx * (Fx_p - Fx_m)
                       - dt_over_dy * (Fy_p - Fy_m);
        }
    }

    apply_transmissive_boundary(Unew);
}

void advance_second_order(const Grid2D& Uold, Grid2D& Utmp, Grid2D& Unew, double dt) {
    sweep_x_second_order(Uold, Utmp, dt);
    apply_transmissive_boundary(Utmp);

    sweep_y_second_order(Utmp, Unew, dt);
    apply_transmissive_boundary(Unew);
}