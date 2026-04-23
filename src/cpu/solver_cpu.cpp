#include "cpu/solver_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

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
    if (a * b <= 0.0) return 0.0;
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

// Flux at x-interface i+1/2, j
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

// Flux at y-interface i, j+1/2
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

// Cache index helpers — identical to GPU version for easy cross-referencing
inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

// ============================================================
// Fill fx_cache with all x-face fluxes for one time level.
// Cache layout: (nx+1) faces per row, ny rows.
// Face local_i_face = 0 corresponds to the left ghost interface
// at global i = ib-1, so cell i maps to local_i_face = i-(ib-1).
// ============================================================
void fill_x_face_cache(
    const Grid2D& Uin,
    double dt,
    std::vector<Conserved>& fx_cache   // pre-sized by CpuWorkspace
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int nx_faces = (ie - ib) + 1;
    const double dt_over_dx = dt / Uin.dx();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib - 1; i < ie; ++i) {
            const int local_j      = j - jb;
            const int local_i_face = i - (ib - 1);
            fx_cache[xface_idx(local_j, local_i_face, nx_faces)] =
                muscl_hancock_flux_x(Uin, i, j, dt_over_dx);
        }
    }
}

// ============================================================
// Fill fy_cache with all y-face fluxes for one time level.
// Cache layout: nx cells per row, (ny+1) face-rows.
// Face local_j_face = 0 corresponds to the bottom ghost interface
// at global j = jb-1, so cell j maps to local_j_face = j-(jb-1).
// ============================================================
void fill_y_face_cache(
    const Grid2D& Uin,
    double dt,
    std::vector<Conserved>& fy_cache   // pre-sized by CpuWorkspace
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int nx_cells = ie - ib;
    const double dt_over_dy = dt / Uin.dy();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb - 1; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j_face = j - (jb - 1);
            const int local_i      = i - ib;
            fy_cache[yface_idx(local_j_face, local_i, nx_cells)] =
                muscl_hancock_flux_y(Uin, i, j, dt_over_dy);
        }
    }
}

} // namespace

// ============================================================
// Public API
// ============================================================

double compute_dt(const Grid2D& grid, double cfl) {
    double max_speed = 0.0;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(max:max_speed) schedule(static)
#endif
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));
            const double a  = phys::sound_speed(V);
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
            const Conserved Fx_p = hll_flux(Uold(i,     j), Uold(i + 1, j), Direction::X);
            const Conserved Fx_m = hll_flux(Uold(i - 1, j), Uold(i,     j), Direction::X);
            const Conserved Fy_p = hll_flux(Uold(i, j    ), Uold(i, j + 1), Direction::Y);
            const Conserved Fy_m = hll_flux(Uold(i, j - 1), Uold(i,     j), Direction::Y);

            Unew(i, j) = Uold(i, j)
                       - dt_over_dx * (Fx_p - Fx_m)
                       - dt_over_dy * (Fy_p - Fy_m);
        }
    }

    apply_transmissive_boundary(Unew);
}

// ============================================================
// Second-order MUSCL-Hancock, x-then-y dimensional splitting.
//
// Mirrors advance_second_order_gpu exactly:
//   1. fill x-face flux cache from Uold
//   2. apply x update -> Utmp
//   3. apply BC to Utmp
//   4. fill y-face flux cache from Utmp
//   5. apply y update -> Unew
//   6. apply BC to Unew
//
// No heap allocation inside this function: both caches live in ws.
// ============================================================
void advance_second_order(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    double dt,
    CpuWorkspace& ws
) {
    if (!ws.is_initialized()) {
        throw std::runtime_error(
            "advance_second_order: CpuWorkspace not initialised. "
            "Call ws.init(cfg.nx, cfg.ny) before the time loop.");
    }

    const int ib = Uold.i_begin();
    const int ie = Uold.i_end();
    const int jb = Uold.j_begin();
    const int je = Uold.j_end();

    const int nx_faces = (ie - ib) + 1;
    const int nx_cells =  ie - ib;
    const double dt_over_dx = dt / Uold.dx();
    const double dt_over_dy = dt / Uold.dy();

    // ----------------------------------------------------------
    // Step 1 & 2: x-sweep  (Uold -> Utmp)
    // ----------------------------------------------------------
    fill_x_face_cache(Uold, dt, ws.fx_cache);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j        = j - jb;
            const int local_i_face_m = (i - 1) - (ib - 1);
            const int local_i_face_p =  i      - (ib - 1);

            const Conserved& Fx_m = ws.fx_cache[xface_idx(local_j, local_i_face_m, nx_faces)];
            const Conserved& Fx_p = ws.fx_cache[xface_idx(local_j, local_i_face_p, nx_faces)];

            Utmp(i, j) = Uold(i, j) - dt_over_dx * (Fx_p - Fx_m);
        }
    }

    // ----------------------------------------------------------
    // Step 3: BC on Utmp  (fills ghost cells for y-sweep)
    // ----------------------------------------------------------
    apply_transmissive_boundary(Utmp);

    // ----------------------------------------------------------
    // Step 4 & 5: y-sweep  (Utmp -> Unew)
    // ----------------------------------------------------------
    fill_y_face_cache(Utmp, dt, ws.fy_cache);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_i        = i - ib;
            const int local_j_face_m = (j - 1) - (jb - 1);
            const int local_j_face_p =  j      - (jb - 1);

            const Conserved& Fy_m = ws.fy_cache[yface_idx(local_j_face_m, local_i, nx_cells)];
            const Conserved& Fy_p = ws.fy_cache[yface_idx(local_j_face_p, local_i, nx_cells)];

            Unew(i, j) = Utmp(i, j) - dt_over_dy * (Fy_p - Fy_m);
        }
    }

    // ----------------------------------------------------------
    // Step 6: BC on Unew
    // ----------------------------------------------------------
    apply_transmissive_boundary(Unew);
}