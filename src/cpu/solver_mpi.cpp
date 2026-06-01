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
// MPI datatype for Conserved (4 doubles).
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

// -----------------------------------------------------------------------------
// MUSCL-Hancock helpers
// -----------------------------------------------------------------------------
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

inline Primitive enforce_physical_primitive(
    const Primitive& candidate, const Primitive& fallback
) {
    return is_physical(candidate) ? candidate : fallback;
}

inline Conserved enforce_physical_conserved(
    const Conserved& candidate, const Conserved& fallback
) {
    return is_physical(phys::cons_to_prim(candidate)) ? candidate : fallback;
}

inline Primitive limited_slope(
    const Primitive& Wm, const Primitive& Wc, const Primitive& Wp
) {
    return minmod_primitive(Wc - Wm, Wp - Wc);
}

inline void reconstruct_cell_muscl_hancock(
    const Conserved& Um, const Conserved& Uc, const Conserved& Up,
    double dt_over_d, Direction dir,
    Conserved& U_left_star, Conserved& U_right_star
) {
    const Primitive Wm = phys::cons_to_prim(Um);
    const Primitive Wc = phys::cons_to_prim(Uc);
    const Primitive Wp = phys::cons_to_prim(Up);

    const Primitive slope = limited_slope(Wm, Wc, Wp);
    Primitive W_left  = enforce_physical_primitive(Wc - 0.5 * slope, Wc);
    Primitive W_right = enforce_physical_primitive(Wc + 0.5 * slope, Wc);

    const Conserved U_left  = phys::prim_to_cons(W_left);
    const Conserved U_right = phys::prim_to_cons(W_right);

    const Conserved F_left  = physical_flux(U_left,  dir);
    const Conserved F_right = physical_flux(U_right, dir);
    const Conserved half_update = 0.5 * dt_over_d * (F_right - F_left);

    U_left_star  = enforce_physical_conserved(U_left  - half_update, U_left);
    U_right_star = enforce_physical_conserved(U_right - half_update, U_right);
}

inline Conserved muscl_hancock_flux_x(
    const Grid2D& U, int i, int j, double dt_over_dx, RiemannSolver solver
) {
    Conserved Ui_L, Ui_R, Uip1_L, Uip1_R;
    reconstruct_cell_muscl_hancock(
        U(i-1,j), U(i,  j), U(i+1,j), dt_over_dx, Direction::X, Ui_L,   Ui_R);
    reconstruct_cell_muscl_hancock(
        U(i,  j), U(i+1,j), U(i+2,j), dt_over_dx, Direction::X, Uip1_L, Uip1_R);
    return riemann_flux(Ui_R, Uip1_L, Direction::X, solver);
}

inline Conserved muscl_hancock_flux_y(
    const Grid2D& U, int i, int j, double dt_over_dy, RiemannSolver solver
) {
    Conserved Uj_L, Uj_R, Ujp1_L, Ujp1_R;
    reconstruct_cell_muscl_hancock(
        U(i,j-1), U(i,j  ), U(i,j+1), dt_over_dy, Direction::Y, Uj_L,   Uj_R);
    reconstruct_cell_muscl_hancock(
        U(i,j  ), U(i,j+1), U(i,j+2), dt_over_dy, Direction::Y, Ujp1_L, Ujp1_R);
    return riemann_flux(Uj_R, Ujp1_L, Direction::Y, solver);
}

// -----------------------------------------------------------------------------
// Physical boundary helpers
// -----------------------------------------------------------------------------
void apply_y_physical_boundary(Grid2D& grid, const MpiDecomp2D& mp) {
    const int ng = grid.ng();
    const int nx_tot = grid.total_nx();
    const int ny_tot = grid.total_ny();
    if (mp.nbr_down == MPI_PROC_NULL) {
        for (int i = 0; i < nx_tot; ++i)
            for (int g = 0; g < ng; ++g)
                grid(i, g) = grid(i, ng);
    }
    if (mp.nbr_up == MPI_PROC_NULL) {
        for (int i = 0; i < nx_tot; ++i)
            for (int g = 0; g < ng; ++g)
                grid(i, ny_tot-1-g) = grid(i, ny_tot-1-ng);
    }
}

void apply_x_physical_boundary_cartesian(Grid2D& grid, const MpiDecomp2D& mp) {
    const int ng = grid.ng();
    const int nx_tot = grid.total_nx();
    const int ny_tot = grid.total_ny();
    if (mp.nbr_left == MPI_PROC_NULL) {
        for (int j = 0; j < ny_tot; ++j)
            for (int g = 0; g < ng; ++g)
                grid(g, j) = grid(ng, j);
    }
    if (mp.nbr_right == MPI_PROC_NULL) {
        for (int j = 0; j < ny_tot; ++j)
            for (int g = 0; g < ng; ++g)
                grid(nx_tot-1-g, j) = grid(nx_tot-1-ng, j);
    }
}

// -----------------------------------------------------------------------------
// Face flux cache filling (interior range variant for overlap)
// -----------------------------------------------------------------------------
void fill_x_face_cache_interior(
    const Grid2D& Uin, double dt, RiemannSolver solver, MpiCpuWorkspace& ws,
    int i_face_begin, int i_face_end
) {
    const int ib = Uin.i_begin(), ie = Uin.i_end();
    const int jb = Uin.j_begin(), je = Uin.j_end();
    const int nx_faces = (ie - ib) + 1;
    const double dt_over_dx = dt / Uin.dx();

    for (int j = jb; j < je; ++j) {
        const int row = (j - jb) * nx_faces;
        for (int lf = i_face_begin; lf < i_face_end; ++lf) {
            const int i = (ib - 1) + lf;
            ws.fx_cache[row + lf] =
                muscl_hancock_flux_x(Uin, i, j, dt_over_dx, solver);
        }
    }
}

void fill_y_face_cache_interior(
    const Grid2D& Uin, double dt, RiemannSolver solver, MpiCpuWorkspace& ws,
    int j_face_begin, int j_face_end
) {
    const int ib = Uin.i_begin(), ie = Uin.i_end();
    const int jb = Uin.j_begin();
    const int nx_cells = ie - ib;
    const double dt_over_dy = dt / Uin.dy();

    for (int lf = j_face_begin; lf < j_face_end; ++lf) {
        const int j = (jb - 1) + lf;
        const int row = lf * nx_cells;
        for (int i = ib; i < ie; ++i)
            ws.fy_cache[row + (i - ib)] =
                muscl_hancock_flux_y(Uin, i, j, dt_over_dy, solver);
    }
}

// -----------------------------------------------------------------------------
// Halo pack/unpack helpers
// -----------------------------------------------------------------------------
static void pack_left_send(
    const Grid2D& grid, std::vector<Conserved>& buf, int ng
) {
    const int jb = grid.j_begin(), je = grid.j_end();
    const int ib = grid.i_begin();
    int k = 0;
    for (int j = jb; j < je; ++j)
        for (int g = 0; g < ng; ++g)
            buf[k++] = grid(ib + g, j);
}

static void pack_right_send(
    const Grid2D& grid, std::vector<Conserved>& buf, int ng
) {
    const int jb = grid.j_begin(), je = grid.j_end();
    const int ie = grid.i_end();
    int k = 0;
    for (int j = jb; j < je; ++j)
        for (int g = 0; g < ng; ++g)
            buf[k++] = grid(ie - ng + g, j);
}

static void unpack_left_recv(
    Grid2D& grid, const std::vector<Conserved>& buf, int ng
) {
    const int jb = grid.j_begin(), je = grid.j_end();
    int k = 0;
    for (int j = jb; j < je; ++j)
        for (int g = 0; g < ng; ++g)
            grid(g, j) = buf[k++];
}

static void unpack_right_recv(
    Grid2D& grid, const std::vector<Conserved>& buf, int ng
) {
    const int jb = grid.j_begin(), je = grid.j_end();
    const int nx_tot = grid.total_nx();
    const int ng_ = grid.ng();
    int k = 0;
    for (int j = jb; j < je; ++j)
        for (int g = 0; g < ng; ++g)
            grid(nx_tot - ng_ + g, j) = buf[k++];
}

} // namespace

// =============================================================================
// Decomposition
// =============================================================================

MpiDecomp2D make_cartesian_decomp(
    const CaseConfig& cfg, MPI_Comm comm, int px, int py
) {
    MpiDecomp2D mp;
    mp.comm = comm;
    MPI_Comm_rank(comm, &mp.rank);
    MPI_Comm_size(comm, &mp.size);

    mp.nx_global = cfg.nx;
    mp.ny_global = cfg.ny;
    mp.ng        = cfg.ng;

    int dims[2] = {px, py};
    MPI_Dims_create(mp.size, 2, dims);
    mp.px = dims[0];
    mp.py = dims[1];

    if (mp.px > cfg.nx || mp.py > cfg.ny)
        throw std::runtime_error(
            "make_cartesian_decomp: process grid exceeds domain dimensions.");

    int periods[2] = {0, 0};
    // reorder=0: preserve rank numbering consistent with MPI_COMM_WORLD.
    // This ensures mp.rank is valid for both mp.comm and mp.cart_comm.
    MPI_Cart_create(comm, 2, dims, periods, /*reorder=*/0, &mp.cart_comm);

    int coords[2];
    MPI_Cart_coords(mp.cart_comm, mp.rank, 2, coords);
    mp.rank_x = coords[0];
    mp.rank_y = coords[1];

    auto distribute = [](int total, int nproc, int r, int& local, int& offset) {
        const int base = total / nproc;
        const int rem  = total % nproc;
        local  = base + (r < rem ? 1 : 0);
        offset = r * base + std::min(r, rem);
    };

    distribute(cfg.nx, mp.px, mp.rank_x, mp.nx_local, mp.x0_global);
    distribute(cfg.ny, mp.py, mp.rank_y, mp.ny_local, mp.y0_global);

    MPI_Cart_shift(mp.cart_comm, 0, 1, &mp.nbr_left,  &mp.nbr_right);
    MPI_Cart_shift(mp.cart_comm, 1, 1, &mp.nbr_down,  &mp.nbr_up);

    return mp;
}

MpiDecomp2D make_y_slab_decomp(const CaseConfig& cfg, MPI_Comm comm) {
    return make_cartesian_decomp(cfg, comm, 1, 0);
}

// =============================================================================
// Grid initialisation
// =============================================================================

Grid2D make_local_mpi_grid(
    CaseId case_id, const CaseConfig& cfg, const MpiDecomp2D& mp
) {
    const double dx = (cfg.x_max - cfg.x_min) / static_cast<double>(cfg.nx);
    const double dy = (cfg.y_max - cfg.y_min) / static_cast<double>(cfg.ny);

    const double local_x_min = cfg.x_min + static_cast<double>(mp.x0_global) * dx;
    const double local_x_max = local_x_min + static_cast<double>(mp.nx_local) * dx;
    const double local_y_min = cfg.y_min + static_cast<double>(mp.y0_global) * dy;
    const double local_y_max = local_y_min + static_cast<double>(mp.ny_local) * dy;

    Grid2D grid(
        mp.nx_local, mp.ny_local, cfg.ng,
        local_x_min, local_x_max,
        local_y_min, local_y_max
    );

    for (int j = grid.j_begin(); j < grid.j_end(); ++j)
        for (int i = grid.i_begin(); i < grid.i_end(); ++i)
            grid(i, j) = initial_state_at(
                case_id, grid.x_center(i), grid.y_center(j));

    MpiCpuWorkspace ws_tmp;
    ws_tmp.init(mp.nx_local, mp.ny_local, cfg.ng);
    apply_transmissive_boundary_mpi(grid, mp, ws_tmp);

    return grid;
}

Grid2D make_local_mpi_grid(
    const std::string& case_name, const CaseConfig& cfg, const MpiDecomp2D& mp
) {
    return make_local_mpi_grid(parse_case_id(case_name), cfg, mp);
}

// =============================================================================
// Halo exchange: full 2D (x and y)
// =============================================================================

void exchange_halo(Grid2D& grid, const MpiDecomp2D& mp, MpiCpuWorkspace& ws) {
    const int ng     = grid.ng();
    const int nx_tot = grid.total_nx();
    const int ny_loc = grid.j_end() - grid.j_begin();
    MPI_Datatype T   = mpi_conserved_type();

    // Y-direction: contiguous rows, no packing needed.
    {
        std::vector<Conserved>& data = grid.data();
        auto flat = [&](int i, int j) -> std::size_t {
            return static_cast<std::size_t>(j * nx_tot + i);
        };

        const int count_y = ng * nx_tot;
        MPI_Request reqs_y[4];
        MPI_Irecv(data.data() + flat(0, 0),                count_y, T,
                  mp.nbr_down, 101, mp.cart_comm, &reqs_y[0]);
        MPI_Irecv(data.data() + flat(0, grid.j_end()),     count_y, T,
                  mp.nbr_up,   100, mp.cart_comm, &reqs_y[1]);
        MPI_Isend(data.data() + flat(0, grid.j_begin()),   count_y, T,
                  mp.nbr_down, 100, mp.cart_comm, &reqs_y[2]);
        MPI_Isend(data.data() + flat(0, grid.j_end()-ng),  count_y, T,
                  mp.nbr_up,   101, mp.cart_comm, &reqs_y[3]);
        MPI_Waitall(4, reqs_y, MPI_STATUSES_IGNORE);
    }

    // X-direction: strided columns, must pack.
    {
        const int count_x = ng * ny_loc;
        pack_left_send(grid,  ws.send_left,  ng);
        pack_right_send(grid, ws.send_right, ng);

        MPI_Request reqs_x[4];
        MPI_Irecv(ws.recv_left.data(),  count_x, T, mp.nbr_left,  201, mp.cart_comm, &reqs_x[0]);
        MPI_Irecv(ws.recv_right.data(), count_x, T, mp.nbr_right, 200, mp.cart_comm, &reqs_x[1]);
        MPI_Isend(ws.send_left.data(),  count_x, T, mp.nbr_left,  200, mp.cart_comm, &reqs_x[2]);
        MPI_Isend(ws.send_right.data(), count_x, T, mp.nbr_right, 201, mp.cart_comm, &reqs_x[3]);
        MPI_Waitall(4, reqs_x, MPI_STATUSES_IGNORE);

        if (mp.nbr_left  != MPI_PROC_NULL) unpack_left_recv(grid,  ws.recv_left,  ng);
        if (mp.nbr_right != MPI_PROC_NULL) unpack_right_recv(grid, ws.recv_right, ng);
    }
}

// Legacy y-only (kept for compatibility).
void exchange_halo_y(Grid2D& grid, const MpiDecomp2D& mp) {
    const int ng     = grid.ng();
    const int nx_tot = grid.total_nx();
    MPI_Datatype T   = mpi_conserved_type();

    std::vector<Conserved>& data = grid.data();
    auto flat = [&](int i, int j) -> std::size_t {
        return static_cast<std::size_t>(j * nx_tot + i);
    };

    const int count = ng * nx_tot;
    MPI_Request reqs[4];
    MPI_Irecv(data.data() + flat(0, 0),               count, T, mp.nbr_down, 101, mp.comm, &reqs[0]);
    MPI_Irecv(data.data() + flat(0, grid.j_end()),     count, T, mp.nbr_up,   100, mp.comm, &reqs[1]);
    MPI_Isend(data.data() + flat(0, grid.j_begin()),   count, T, mp.nbr_down, 100, mp.comm, &reqs[2]);
    MPI_Isend(data.data() + flat(0, grid.j_end()-ng),  count, T, mp.nbr_up,   101, mp.comm, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

void apply_transmissive_boundary_mpi(
    Grid2D& grid, const MpiDecomp2D& mp, MpiCpuWorkspace& ws
) {
    exchange_halo(grid, mp, ws);
    apply_y_physical_boundary(grid, mp);
    apply_x_physical_boundary_cartesian(grid, mp);
}

// =============================================================================
// Timestep
// =============================================================================

double compute_dt_mpi(const Grid2D& grid, const MpiDecomp2D& mp, double cfl) {
    double max_speed_local = 0.0;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));
            const double a  = phys::sound_speed(V);
            const double sx = std::abs(V.u) + a;
            const double sy = std::abs(V.v) + a;
            const double s  = sx > sy ? sx : sy;
            if (s > max_speed_local) max_speed_local = s;
        }
    }

    double max_speed_global = 0.0;
    // Use mp.comm (MPI_COMM_WORLD) — avoids any cart_comm ordering issues.
    MPI_Allreduce(
        &max_speed_local, &max_speed_global, 1,
        MPI_DOUBLE, MPI_MAX, mp.comm
    );

    if (max_speed_global <= 0.0)
        throw std::runtime_error(
            "compute_dt_mpi: non-positive global maximum wave speed.");

    const double min_spacing = grid.dx() < grid.dy() ? grid.dx() : grid.dy();
    return cfl * min_spacing / max_speed_global;
}

// Cached variant: recomputes dt every dt_interval steps.
// FIX: capture recompute decision before incrementing dt_step so the return
// value matches the step that actually recomputed.
double compute_dt_mpi_cached(
    const Grid2D& grid, const MpiDecomp2D& mp, double cfl, MpiCpuWorkspace& ws
) {
    // Decide whether to recompute BEFORE touching dt_step.
    const bool recompute = (ws.cached_dt < 0.0 || ws.dt_step % ws.dt_interval == 0);

    if (recompute)
        ws.cached_dt = compute_dt_mpi(grid, mp, cfl);

    ++ws.dt_step;

    // Return the exact freshly-computed value on recompute steps,
    // and a conservative 0.9x on cached steps.
    return recompute ? ws.cached_dt : 0.9 * ws.cached_dt;
}

// =============================================================================
// Second-order advance with communication/computation overlap.
//
// Structure per timestep:
//   X-sweep (Uold -> Utmp):
//     Ghost cells of Uold are already valid (filled by caller after previous step).
//     Compute all x-face fluxes and apply update — no communication needed.
//
//   Y-sweep (Utmp -> Unew):
//     Phase 1: Post non-blocking halo exchange for Utmp (all directions).
//     Phase 2: Compute interior y-face fluxes (no ghost cells needed) — overlaps comms.
//     Phase 3: Wait for halo exchange, unpack, apply physical boundaries.
//     Phase 4: Compute boundary y-face fluxes (need ghost cells).
//     Phase 5: Apply y update to get Unew.
//
// After advance returns, the caller must call apply_transmissive_boundary_mpi
// on the result (Unew after swap) to prepare ghost cells for the next x-sweep.
// =============================================================================

void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    RiemannSolver solver,
    MpiCpuWorkspace& ws
) {
    if (!ws.is_initialized_for(Uold.nx(), Uold.ny()))
        ws.init(Uold.nx(), Uold.ny(), Uold.ng());

    const int ib = Uold.i_begin(), ie = Uold.i_end();
    const int jb = Uold.j_begin(), je = Uold.j_end();
    const int ng = Uold.ng();

    const int nx_faces = (ie - ib) + 1;
    const int nx_cells = ie - ib;
    const int ny_cells = je - jb;

    const double dt_over_dx = dt / Uold.dx();
    const double dt_over_dy = dt / Uold.dy();

    // =========================================================================
    // X-SWEEP: Uold -> Utmp
    // Ghost cells of Uold already valid from caller's boundary fill.
    // =========================================================================
    const int x_inner_begin = ng;
    const int x_inner_end   = nx_faces - ng;

    // Interior faces (no ghost cells needed)
    fill_x_face_cache_interior(Uold, dt, solver, ws, x_inner_begin, x_inner_end);
    // Boundary faces (use already-valid ghost cells)
    fill_x_face_cache_interior(Uold, dt, solver, ws, 0,            x_inner_begin);
    fill_x_face_cache_interior(Uold, dt, solver, ws, x_inner_end,  nx_faces);

    for (int j = jb; j < je; ++j) {
        const int row = (j - jb) * nx_faces;
        for (int i = ib; i < ie; ++i) {
            const int lf = i - ib;
            Utmp(i, j) = Uold(i, j)
                - dt_over_dx * (ws.fx_cache[row + lf + 1] - ws.fx_cache[row + lf]);
        }
    }

    // =========================================================================
    // Y-SWEEP: Utmp -> Unew  (communication / computation overlap)
    // =========================================================================

    // --- Phase 1: Post non-blocking halo exchange for Utmp. ------------------
    const int nx_tot  = Utmp.total_nx();
    MPI_Datatype T    = mpi_conserved_type();
    std::vector<Conserved>& utmp_data = Utmp.data();

    auto flat = [&](int i, int j) -> std::size_t {
        return static_cast<std::size_t>(j * nx_tot + i);
    };

    const int count_y = ng * nx_tot;
    const int count_x = ng * ny_cells;

    MPI_Request reqs[8];
    int nreq = 0;

    // Y-direction (contiguous)
    MPI_Irecv(utmp_data.data() + flat(0, 0),        count_y, T,
              mp.nbr_down, 301, mp.cart_comm, &reqs[nreq++]);
    MPI_Irecv(utmp_data.data() + flat(0, je),       count_y, T,
              mp.nbr_up,   300, mp.cart_comm, &reqs[nreq++]);
    MPI_Isend(utmp_data.data() + flat(0, jb),       count_y, T,
              mp.nbr_down, 300, mp.cart_comm, &reqs[nreq++]);
    MPI_Isend(utmp_data.data() + flat(0, je - ng),  count_y, T,
              mp.nbr_up,   301, mp.cart_comm, &reqs[nreq++]);

    // X-direction (pack then post)
    pack_left_send(Utmp,  ws.send_left,  ng);
    pack_right_send(Utmp, ws.send_right, ng);
    MPI_Irecv(ws.recv_left.data(),  count_x, T, mp.nbr_left,  401, mp.cart_comm, &reqs[nreq++]);
    MPI_Irecv(ws.recv_right.data(), count_x, T, mp.nbr_right, 400, mp.cart_comm, &reqs[nreq++]);
    MPI_Isend(ws.send_left.data(),  count_x, T, mp.nbr_left,  400, mp.cart_comm, &reqs[nreq++]);
    MPI_Isend(ws.send_right.data(), count_x, T, mp.nbr_right, 401, mp.cart_comm, &reqs[nreq++]);

    // --- Phase 2: Compute interior y-faces while halo is in flight. ----------
    // Interior y-faces do not touch any ghost row, so they are safe to compute now.
    const int y_inner_begin = ng;
    const int y_inner_end   = ny_cells + 1 - ng;

    fill_y_face_cache_interior(Utmp, dt, solver, ws, y_inner_begin, y_inner_end);

    // --- Phase 3: Wait for halo, unpack, apply physical boundaries. ----------
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (mp.nbr_left  != MPI_PROC_NULL) unpack_left_recv(Utmp,  ws.recv_left,  ng);
    if (mp.nbr_right != MPI_PROC_NULL) unpack_right_recv(Utmp, ws.recv_right, ng);

    apply_y_physical_boundary(Utmp, mp);
    apply_x_physical_boundary_cartesian(Utmp, mp);

    // --- Phase 4: Compute boundary y-faces (need ghost cells). ---------------
    fill_y_face_cache_interior(Utmp, dt, solver, ws, 0,           y_inner_begin);
    fill_y_face_cache_interior(Utmp, dt, solver, ws, y_inner_end, ny_cells + 1);

    // --- Phase 5: Apply y update to get Unew. --------------------------------
    for (int j = jb; j < je; ++j) {
        const int lf_m  = j - jb;
        const int lf_p  = lf_m + 1;
        const int row_m = lf_m * nx_cells;
        const int row_p = lf_p * nx_cells;
        for (int i = ib; i < ie; ++i) {
            const int li = i - ib;
            Unew(i, j) = Utmp(i, j)
                - dt_over_dy * (ws.fy_cache[row_p + li] - ws.fy_cache[row_m + li]);
        }
    }

    // Unew ghost cells are stale. The caller (main loop) must call
    // apply_transmissive_boundary_mpi(Unew, mp, ws) after swap before the
    // next timestep. No communication is started here.
}

// Backward-compatible HLL overload.
void advance_second_order_mpi(
    const Grid2D& Uold, Grid2D& Utmp, Grid2D& Unew,
    const MpiDecomp2D& mp, double dt, MpiCpuWorkspace& ws
) {
    advance_second_order_mpi(Uold, Utmp, Unew, mp, dt, RiemannSolver::HLL, ws);
}