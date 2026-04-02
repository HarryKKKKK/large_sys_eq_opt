#pragma once

#include "grid_gpu.cuh"

double compute_dt_gpu(const Grid2DGPU& grid, double cfl);

void advance_first_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU& Unew,
    double dt
);

inline double minmod_gpu_host(double a, double b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (a > 0.0) ? std::min(a, b) : std::max(a, b);
}