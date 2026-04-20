#pragma once

#include "grid_gpu.cuh"

double compute_dt_gpu(const Grid2DGPU& grid, double cfl);

void advance_first_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU& Unew,
    double dt
);

void advance_second_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU& Utmp,
    Grid2DGPU& Unew,
    double dt
);