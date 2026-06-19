#pragma once

#include "cpu/grid_cpu.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// 2D Cartesian MPI decomposition.
//
// The global grid is split into a px × py process grid.
// Each rank owns nx_local × ny_local interior cells.
//
// y-slab is the special case px=1, py=size (previous behaviour).
// x-slab is the special case px=size, py=1.
// Square decomposition minimises halo perimeter for square domains.
//
// Neighbours:
//   nbr_left  / nbr_right  : ±x direction
//   nbr_down  / nbr_up     : ±y direction
// MPI_PROC_NULL is used at physical boundaries.
// -----------------------------------------------------------------------------
struct MpiDecomp2D {
    MPI_Comm comm      = MPI_COMM_WORLD;
    MPI_Comm cart_comm = MPI_COMM_NULL;   // Cartesian communicator
    int rank  = 0;
    int size  = 1;

    // Process grid dimensions
    int px = 1;   // number of ranks in x
    int py = 1;   // number of ranks in y

    // This rank's position in the process grid
    int rank_x = 0;
    int rank_y = 0;

    int nx_global = 0;
    int ny_global = 0;
    int ng = 2;

    // Local interior dimensions and offsets
    int nx_local  = 0;
    int ny_local  = 0;
    int x0_global = 0;   // zero-based global interior column of first local column
    int y0_global = 0;   // zero-based global interior row    of first local row

    // Neighbours (MPI_PROC_NULL at physical boundary)
    int nbr_left  = MPI_PROC_NULL;
    int nbr_right = MPI_PROC_NULL;
    int nbr_down  = MPI_PROC_NULL;
    int nbr_up    = MPI_PROC_NULL;
};

struct MpiCpuWorkspace {
    int nx    = 0;
    int ny    = 0;
    bool ready = false;

    // Face flux caches
    std::vector<Conserved> fx_cache;   // (nx+1) * ny
    std::vector<Conserved> fy_cache;   // nx * (ny+1)

    // Halo send/recv buffers for x direction (left/right neighbours)
    // Each buffer holds ng * ny_local Conserved values.
    std::vector<Conserved> send_left;
    std::vector<Conserved> recv_left;
    std::vector<Conserved> send_right;
    std::vector<Conserved> recv_right;

    // dt caching: recompute every dt_interval steps
    int    dt_interval  = 4;
    int    dt_step      = 0;
    double cached_dt    = -1.0;

    void init(int nx_, int ny_, int ng_) {
        nx = nx_;
        ny = ny_;

        fx_cache.assign(
            static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny),
            Conserved(0, 0, 0, 0)
        );
        fy_cache.assign(
            static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1),
            Conserved(0, 0, 0, 0)
        );

        // x-halo buffers: ng columns × ny rows
        const std::size_t halo_x = static_cast<std::size_t>(ng_) *
                                   static_cast<std::size_t>(ny);
        send_left.assign(halo_x,  Conserved(0, 0, 0, 0));
        recv_left.assign(halo_x,  Conserved(0, 0, 0, 0));
        send_right.assign(halo_x, Conserved(0, 0, 0, 0));
        recv_right.assign(halo_x, Conserved(0, 0, 0, 0));

        ready = true;
    }

    bool is_initialized_for(int nx_, int ny_) const {
        return ready && nx == nx_ && ny == ny_;
    }
};

// -----------------------------------------------------------------------------
// Decomposition factories.
//
// make_cartesian_decomp: full 2D Cartesian.
//   Pass px=0, py=0 to let MPI_Dims_create choose the best split automatically.
//
// make_y_slab_decomp: backward-compatible wrapper (px=1).
// -----------------------------------------------------------------------------
MpiDecomp2D make_cartesian_decomp(
    const CaseConfig& global_cfg,
    MPI_Comm comm = MPI_COMM_WORLD,
    int px = 0,
    int py = 0
);

MpiDecomp2D make_y_slab_decomp(
    const CaseConfig& global_cfg,
    MPI_Comm comm = MPI_COMM_WORLD
);

// -----------------------------------------------------------------------------
// Grid initialisation
// -----------------------------------------------------------------------------
Grid2D make_local_mpi_grid(
    CaseId case_id,
    const CaseConfig& global_cfg,
    const MpiDecomp2D& mp
);

Grid2D make_local_mpi_grid(
    const std::string& case_name,
    const CaseConfig& global_cfg,
    const MpiDecomp2D& mp
);

// -----------------------------------------------------------------------------
// Halo exchange (both x and y directions)
// -----------------------------------------------------------------------------
void exchange_halo(Grid2D& grid, const MpiDecomp2D& mp, MpiCpuWorkspace& ws);

// Legacy y-only exchange (kept for compatibility)
void exchange_halo_y(Grid2D& grid, const MpiDecomp2D& mp);

void apply_transmissive_boundary_mpi(
    Grid2D& grid,
    const MpiDecomp2D& mp,
    MpiCpuWorkspace& ws
);

// -----------------------------------------------------------------------------
// Timestep (with optional caching)
// -----------------------------------------------------------------------------
double compute_dt_mpi(
    const Grid2D& grid,
    const MpiDecomp2D& mp,
    double cfl
);

double compute_dt_mpi_cached(
    const Grid2D& grid,
    const MpiDecomp2D& mp,
    double cfl,
    MpiCpuWorkspace& ws
);

// -----------------------------------------------------------------------------
// Second-order advance with communication/computation overlap
// -----------------------------------------------------------------------------
void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    RiemannSolver solver,
    MpiCpuWorkspace& ws
);

// Backward-compatible HLL overload
void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    MpiCpuWorkspace& ws
);