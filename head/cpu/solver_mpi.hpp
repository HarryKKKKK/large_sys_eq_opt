#pragma once

#include "cpu/grid_cpu.hpp"
#include "riemann.hpp"
#include "test_cases.hpp"

// Avoid OpenMPI deprecated C++ binding warnings.
// The code uses the C MPI API only, e.g. MPI_Init, MPI_Comm_rank, MPI_Allreduce.
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

struct MpiDecomp2D {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;

    int nx_global = 0;
    int ny_global = 0;
    int ng = 2;

    // y-slab decomposition. Each rank owns all x cells and a contiguous block of y rows.
    int ny_local = 0;
    int y0_global = 0;   // zero-based global interior row index of this rank's first row

    int nbr_down = MPI_PROC_NULL;
    int nbr_up   = MPI_PROC_NULL;
};

struct MpiCpuWorkspace {
    int nx = 0;
    int ny = 0;
    bool ready = false;

    // x-face flux cache: (nx + 1) * ny
    std::vector<Conserved> fx_cache;

    // y-face flux cache: nx * (ny + 1)
    std::vector<Conserved> fy_cache;

    void init(int nx_, int ny_) {
        nx = nx_;
        ny = ny_;

        const std::size_t n_fx =
            static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny);

        const std::size_t n_fy =
            static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1);

        const Conserved zero_flux(0.0, 0.0, 0.0, 0.0);

        fx_cache.assign(n_fx, zero_flux);
        fy_cache.assign(n_fy, zero_flux);

        ready = true;
    }

    bool is_initialized_for(int nx_, int ny_) const {
        return ready && nx == nx_ && ny == ny_;
    }
};

MpiDecomp2D make_y_slab_decomp(
    const CaseConfig& global_cfg,
    MPI_Comm comm = MPI_COMM_WORLD
);

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

void exchange_halo_y(
    Grid2D& grid,
    const MpiDecomp2D& mp
);

void apply_transmissive_boundary_mpi(
    Grid2D& grid,
    const MpiDecomp2D& mp
);

double compute_dt_mpi(
    const Grid2D& grid,
    const MpiDecomp2D& mp,
    double cfl
);

void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    RiemannSolver solver,
    MpiCpuWorkspace& ws
);

// Backward-compatible HLL overload.
void advance_second_order_mpi(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    const MpiDecomp2D& mp,
    double dt,
    MpiCpuWorkspace& ws
);
