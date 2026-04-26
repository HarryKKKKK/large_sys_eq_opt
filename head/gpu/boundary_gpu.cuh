#pragma once

#include "grid_gpu.cuh"

void apply_transmissive_boundary_x_gpu(Grid2DGPU& grid);
void apply_transmissive_boundary_y_gpu(Grid2DGPU& grid);

void apply_transmissive_boundary_gpu(Grid2DGPU& grid);