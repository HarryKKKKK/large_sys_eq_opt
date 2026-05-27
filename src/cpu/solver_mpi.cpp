#include "cpu/solver_mpi.hpp"

#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

// -----------------------------------------------------------------------------
// MPI datatype for Conserved = four doubles: rho, rhou, rhov, E.
// -----------------------------------------------------------------------------
MPI_Datatype mpi_conserved_type() {
    static MPI_Datatype dtype = MPI_DATATYPE_NULL;
    static bool initialised = false;

    if (!initialised) {
        MPI_Type_contiguous(4, MPI_DOUBLE, &dtype);
        MPI_Type_commit(&dtype);
        initialised = true;
    }

    return dtype;
}

inline std::size_t flat_index(const Grid2D& grid, int i, int j) {
    return static_cast<std::size_t>(j * grid.total_nx() + i);
}

// -----------------------------------------------------------------------------
// MUSCL-Hancock reconstruction helpers.
// -----------------------------------------------------------------------------
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
           (V.rho > kRhoFloor) && (V.p > kPFloor);
}

inline Primitive enforce_physical_primitive(
    const Primitive& candidate,
    const Primitive& fallback
) {
    return is_physical(candidate) ? candidate : fallback;
}

inline Conserved enforce_physical_conserved(
    const Conserved& candidate,
    const Conserved& fallback
) {
    const Primitive Vcand = phys::cons_to_prim(candidate);
    return is_physical(Vcand) ? candidate : fallback;
}

inline Primitive limited_slope(
    const Primitive& Wm,
    const Primitive& Wc,
    const Primitive& Wp
) {
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

inline Conserved muscl_hancock_flux_x(
    const Grid2D& U,
    int i,
    int j,
    double dt_over_dx,
    RiemannSolver solver
) {
    Conserved Ui_L_star;
    Conserved Ui_R_star;
    Conserved Uip1_L_star;
    Conserved Uip1_R_star;

    reconstruct_cell_muscl_hancock(
        U(i - 1, j),
        U(i,     j),
        U(i + 1, j),
        dt_over_dx,
        Direction::X,
        Ui_L_star,
        Ui_R_star
    );

    reconstruct_cell_muscl_hancock(
        U(i,     j),
        U(i + 1, j),
        U(i + 2, j),
        dt_over_dx,
        Direction::X,
        Uip1_L_star,
        Uip1_R_star
    );

    return riemann_flux(Ui_R_star, Uip1_L_star, Direction::X, solver);
}

inline Conserved muscl_hancock_flux_y(
    const Grid2D& U,
    int i,
    int j,
    double dt_over_dy,
    RiemannSolver solver
) {
    Conserved Uj_L_star;
    Conserved Uj_R_star;
    Conserved Ujp1_L_star;
    Conserved Ujp1_R_star;

    reconstruct_cell_muscl_hancock(
        U(i, j - 1),
        U(i, j),
        U(i, j + 1),
        dt_over_dy,
        Direction::Y,
        Uj_L_star,
        Uj_R_star
    );

    reconstruct_cell_muscl_hancock(
        U(i, j),
        U(i, j + 1),
        U(i, j + 2),
        dt_over_dy,
        Direction::Y,
        Ujp1_L_star,
        Ujp1_R_star
    );

    return riemann_flux(Uj_R_star, Ujp1_L_star, Direction::Y, solver);
}

inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

// -----------------------------------------------------------------------------
// Boundary handling.
// In y-slab MPI:
//   - x boundaries are physical on every rank.
//   - y boundaries are physical only on the first/last rank.
//   - internal y boundaries are filled by halo exchange.
// -----------------------------------------------------------------------------
void apply_x_transmissive_boundary(Grid2D& grid) {
    const int ng = grid.ng();
    const int nx_tot = grid.total_nx();
    const int ny_tot = grid.total_ny();

    for (int j = 0; j < ny_tot; ++j) {
        for (int g = 0; g < ng; ++g) {
            grid(g, j) = grid(ng, j);
            grid(nx_tot - 1 - g, j) = grid(nx_tot - 1 - ng, j);
        }
    }
}

void apply_y_physical_boundary(Grid2D& grid, const MpiDecomp2D& mp) {
    const int ng = grid.ng();
    const int nx_tot = grid.total_nx();
    const int ny_tot = grid.total_ny();

    if (mp.nbr_down == MPI_PROC_NULL) {
        for (int i = 0; i < nx_tot; ++i) {
            for (int g = 0; g < ng; ++g) {
                grid(i, g) = grid(i, ng);
            }
        }
    }

    if (mp.nbr_up == MPI_PROC_NULL) {
        for (int i = 0; i < nx_tot; ++i) {
            for (int g = 0; g < ng; ++g) {
                grid(i, ny_tot - 1 - g) = grid(i, ny_tot - 1 - ng);
            }
        }
    }
}

void fill_x_face_cache_mpi(
    const Grid2D& Uin,
    double dt,
    RiemannSolver solver,
    MpiCpuWorkspace& ws
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int nx_faces = (ie - ib) + 1;
    const double dt_over_dx = dt / Uin.dx();

    for (int j = jb; j < je; ++j) {
        for (int i = ib - 1; i < ie; ++i) {
            const int local_j = j - jb;
            const int local_i_face = i - (ib - 1);

            ws.fx_cache[xface_idx(local_j, local_i_face, nx_faces)] =
                muscl_hancock_flux_x(Uin, i, j, dt_over_dx, solver);
        }
    }
}

void fill_y_face_cache_mpi(
    const Grid2D& Uin,
    double dt,
    RiemannSolver solver,
    MpiCpuWorkspace& ws
) {
    const int ib = Uin.i_begin();
    const int ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int je = Uin.j_end();

    const int nx_cells = ie - ib;
    const double dt_over_dy = dt / Uin.dy();

    for (int j = jb - 1; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j_face = j - (jb - 1);
            const int local_i = i - ib;

            ws.fy_cache[yface_idx(local_j_face, local_i, nx_cells)] =
                muscl_hancock_flux_y(Uin, i, j, dt_over_dy, solver);
        }
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Decomposition and local initialisation.
// -----------------------------------------------------------------------------
MpiDecomp2D make_y_slab_decomp(const CaseConfig& global_cfg, MPI_Comm comm) {
    MpiDecomp2D mp;
    mp.comm = comm;
    MPI_Comm_rank(comm, &mp.rank);
    MPI_Comm_size(comm, &mp.size);

    mp.nx_global = global_cfg.nx;
    mp.ny_global = global_cfg.ny;
    mp.ng = global_cfg.ng;

    if (mp.size > global_cfg.ny) {
        if (mp.rank == 0) {
            throw std::runtime_error(
                "MPI size must not exceed ny_global for y-slab decomposition."
            );
        }
    }

    const int base = global_cfg.ny / mp.size;
    const int rem  = global_cfg.ny % mp.size;

    mp.ny_local = base + ((mp.rank < rem) ? 1 : 0);
    mp.y0_global = mp.rank * base + std::min(mp.rank, rem);

    mp.nbr_down = (mp.rank == 0) ? MPI_PROC_NULL : mp.rank - 1;
    mp.nbr_up   = (mp.rank == mp.size - 1) ? MPI_PROC_NULL : mp.rank + 1;

    return mp;
}

Grid2D make_local_mpi_grid(
    CaseId case_id,
    const CaseConfig& cfg,
    const MpiDecomp2D& mp
) {
    const double dy = (cfg.y_max - cfg.y_min) / static_cast<double>(cfg.ny);
    const double local_y_min = cfg.y_min + static_cast<double>(mp.y0_global) * dy;
    const double local_y_max = local_y_min + static_cast<double>(mp.ny_local) * dy;

    Grid2D grid(
        cfg.nx,
        mp.ny_local,
        cfg.ng,
        cfg.x_min,
        cfg.x_max,
        local_y_min,
        local_y_max
    );

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            grid(i, j) = initial_state_at(
                case_id,
                grid.x_center(i),
                grid.y_center(j)
            );
        }
    }

    // Initial halos: full boundary is fine here because it is done once.
    apply_transmissive_boundary_mpi(grid, mp);

    return grid;
}

Grid2D make_local_mpi_grid(
    const std::string& case_name,
    const CaseConfig& cfg,
    const MpiDecomp2D& mp
) {
    return make_local_mpi_grid(parse_case_id(case_name), cfg, mp);
}

// -----------------------------------------------------------------------------
// Halo exchange.
// -----------------------------------------------------------------------------
void exchange_halo_y(Grid2D& grid, const MpiDecomp2D& mp) {
    const int ng = grid.ng();

    if (ng != mp.ng) {
        throw std::runtime_error(
            "exchange_halo_y: grid.ng() does not match decomposition ng."
        );
    }

    const int nx_tot = grid.total_nx();
    const int count = ng * nx_tot;
    MPI_Datatype T = mpi_conserved_type();

    std::vector<Conserved>& data = grid.data();

    Conserved* send_down = data.data() + flat_index(grid, 0, grid.j_begin());
    Conserved* recv_down = data.data() + flat_index(grid, 0, 0);

    Conserved* send_up = data.data() + flat_index(grid, 0, grid.j_end() - ng);
    Conserved* recv_up = data.data() + flat_index(grid, 0, grid.j_end());

    MPI_Sendrecv(
        send_down, count, T, mp.nbr_down, 100,
        recv_down, count, T, mp.nbr_down, 101,
        mp.comm, MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        send_up, count, T, mp.nbr_up, 101,
        recv_up, count, T, mp.nbr_up, 100,
        mp.comm, MPI_STATUS_IGNORE
    );
}

void apply_transmissive_boundary_mpi(Grid2D& grid, const MpiDecomp2D& mp) {
    apply_x_transmissive_boundary(grid);
    exchange_halo_y(grid, mp);
    apply_y_physical_boundary(grid, mp);
}

// -----------------------------------------------------------------------------
// Global CFL timestep.
// Pure MPI version: no OpenMP reduction.
// -----------------------------------------------------------------------------
double compute_dt_mpi(const Grid2D& grid, const MpiDecomp2D& mp, double cfl) {
    double max_speed_local = 0.0;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));
            const double a = phys::sound_speed(V);

            const double sx = std::abs(V.u) + a;
            const double sy = std::abs(V.v) + a;

            max_speed_local = std::max(max_speed_local, std::max(sx, sy));
        }
    }

    double max_speed_global = 0.0;
    MPI_Allreduce(
        &max_speed_local,
        &max_speed_global,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        mp.comm
    );

    if (max_speed_global <= 0.0) {
        throw std::runtime_error(
            "compute_dt_mpi: non-positive global maximum wave speed."
        );
    }

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed_global;
}

// -----------------------------------------------------------------------------
// Directionally split second-order update.
// Pure MPI version:
//   - no OpenMP pragmas;
//   - supports the selected Riemann solver;
//   - only one MPI halo exchange is needed inside each timestep, after x-sweep;
//   - final Unew only refreshes x ghost cells, because y halos are not needed until
//     the next y-sweep and will be produced from Utmp at that time.
// -----------------------------------------------------------------------------
void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    RiemannSolver solver,
    MpiCpuWorkspace& ws
) {
    if (!ws.is_initialized_for(Uold.nx(), Uold.ny())) {
        ws.init(Uold.nx(), Uold.ny());
    }

    const int ib = Uold.i_begin();
    const int ie = Uold.i_end();
    const int jb = Uold.j_begin();
    const int je = Uold.j_end();

    const int nx_faces = (ie - ib) + 1;
    const int nx_cells = ie - ib;

    const double dt_over_dx = dt / Uold.dx();
    const double dt_over_dy = dt / Uold.dy();

    // x-sweep: Uold -> Utmp.
    // Uold only needs valid x ghost cells for this sweep.
    fill_x_face_cache_mpi(Uold, dt, solver, ws);

    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_j = j - jb;
            const int local_i_face_m = (i - 1) - (ib - 1);
            const int local_i_face_p = i - (ib - 1);

            const Conserved& Fx_m =
                ws.fx_cache[xface_idx(local_j, local_i_face_m, nx_faces)];
            const Conserved& Fx_p =
                ws.fx_cache[xface_idx(local_j, local_i_face_p, nx_faces)];

            Utmp(i, j) = Uold(i, j) - dt_over_dx * (Fx_p - Fx_m);
        }
    }

    // y-sweep needs y halos of Utmp.
    apply_transmissive_boundary_mpi(Utmp, mp);
    fill_y_face_cache_mpi(Utmp, dt, solver, ws);

    for (int j = jb; j < je; ++j) {
        for (int i = ib; i < ie; ++i) {
            const int local_i = i - ib;
            const int local_j_face_m = (j - 1) - (jb - 1);
            const int local_j_face_p = j - (jb - 1);

            const Conserved& Fy_m =
                ws.fy_cache[yface_idx(local_j_face_m, local_i, nx_cells)];
            const Conserved& Fy_p =
                ws.fy_cache[yface_idx(local_j_face_p, local_i, nx_cells)];

            Unew(i, j) = Utmp(i, j) - dt_over_dy * (Fy_p - Fy_m);
        }
    }

    // Only x ghost cells are required for the next x-sweep.
    // Do not exchange y halos here; it is unnecessary and expensive.
    apply_x_transmissive_boundary(Unew);
}

// Backward-compatible overload.
// If main_mpi.cpp has not yet been updated to pass the selected solver,
// this preserves compilation but defaults to HLL.
void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    MpiCpuWorkspace& ws
) {
    advance_second_order_mpi(
        Uold,
        Utmp,
        Unew,
        mp,
        dt,
        RiemannSolver::HLL,
        ws
    );
}
