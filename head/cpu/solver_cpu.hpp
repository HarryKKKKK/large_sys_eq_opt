#pragma once

#include "cpu/grid_cpu.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>

// ============================================================
// Pre-allocated workspace for second-order advance.
// Mirrors GpuWorkspace on the CPU side: allocate once before
// the time loop, then pass into every advance_second_order call
// to avoid per-step heap allocation.
// ============================================================
struct CpuWorkspace {
    int nx = 0;
    int ny = 0;

    // x-face flux cache: (nx+1) * ny entries
    // indexed by xface_idx(local_j, local_i_face, nx+1)
    std::vector<Conserved> fx_cache;

    // y-face flux cache: nx * (ny+1) entries
    // indexed by yface_idx(local_j_face, local_i, nx)
    std::vector<Conserved> fy_cache;

    void init(int nx_, int ny_) {
        nx = nx_;
        ny = ny_;
        fx_cache.resize(static_cast<std::size_t>(nx + 1) * ny);
        fy_cache.resize(static_cast<std::size_t>(nx) * (ny + 1));
    }

    bool is_initialized() const {
        return nx > 0 && ny > 0 &&
               fx_cache.size() == static_cast<std::size_t>(nx + 1) * ny &&
               fy_cache.size() == static_cast<std::size_t>(nx) * (ny + 1);
    }
};

// CFL timestep
double compute_dt(const Grid2D& grid, double cfl);

// First-order Godunov (kept for reference / debugging)
void advance_first_order(
    const Grid2D& Uold,
    Grid2D& Unew,
    double dt
);

// Second-order MUSCL-Hancock, dimensional splitting.
// ws must be initialised with ws.init(cfg.nx, cfg.ny) before the time loop.
void advance_second_order(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    double dt,
    CpuWorkspace& ws
);