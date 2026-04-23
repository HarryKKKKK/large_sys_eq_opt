#pragma once

#include "gpu/grid_gpu.cuh"

#include <cstddef>

struct GpuWorkspace {
    int nx = 0;
    int ny = 0;

    double* speed_d     = nullptr;
    double* max_speed_d = nullptr;

    void*       reduce_tmp       = nullptr;
    std::size_t reduce_tmp_bytes = 0;

    double* fx_mass   = nullptr;
    double* fx_momx   = nullptr;
    double* fx_momy   = nullptr;
    double* fx_energy = nullptr;

    double* fy_mass   = nullptr;
    double* fy_momx   = nullptr;
    double* fy_momy   = nullptr;
    double* fy_energy = nullptr;
};

void init_gpu_workspace(GpuWorkspace& ws, const Grid2DGPU& grid);
void free_gpu_workspace(GpuWorkspace& ws);

double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl);

void advance_first_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU&       Unew,
    double           dt
);

void advance_second_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU&       Utmp,
    Grid2DGPU&       Unew,
    GpuWorkspace&    ws,
    double           dt
);