#include "gpu/boundary_gpu.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void apply_transmissive_boundary_x_kernel(Grid2DGPUView grid) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + grid.j_begin();

    if (j >= grid.j_end()) {
        return;
    }

    const int ib = grid.i_begin();
    const int ie = grid.i_end();

    for (int g = 1; g <= grid.ng; ++g) {
        const int left_ghost  = ib - g;
        const int right_ghost = ie - 1 + g;

        const int left_src  = ib;
        const int right_src = ie - 1;

        const int idx_lg = grid.flat_index(left_ghost, j);
        const int idx_ls = grid.flat_index(left_src, j);

        grid.rho[idx_lg]  = grid.rho[idx_ls];
        grid.rhou[idx_lg] = grid.rhou[idx_ls];
        grid.rhov[idx_lg] = grid.rhov[idx_ls];
        grid.E[idx_lg]    = grid.E[idx_ls];

        const int idx_rg = grid.flat_index(right_ghost, j);
        const int idx_rs = grid.flat_index(right_src, j);

        grid.rho[idx_rg]  = grid.rho[idx_rs];
        grid.rhou[idx_rg] = grid.rhou[idx_rs];
        grid.rhov[idx_rg] = grid.rhov[idx_rs];
        grid.E[idx_rg]    = grid.E[idx_rs];
    }
}

__global__ void apply_transmissive_boundary_y_kernel(Grid2DGPUView grid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= grid.total_nx()) {
        return;
    }

    const int jb = grid.j_begin();
    const int je = grid.j_end();

    for (int g = 1; g <= grid.ng; ++g) {
        const int bottom_ghost = jb - g;
        const int top_ghost    = je - 1 + g;

        const int bottom_src = jb;
        const int top_src    = je - 1;

        const int idx_bg = grid.flat_index(i, bottom_ghost);
        const int idx_bs = grid.flat_index(i, bottom_src);

        grid.rho[idx_bg]  = grid.rho[idx_bs];
        grid.rhou[idx_bg] = grid.rhou[idx_bs];
        grid.rhov[idx_bg] = grid.rhov[idx_bs];
        grid.E[idx_bg]    = grid.E[idx_bs];

        const int idx_tg = grid.flat_index(i, top_ghost);
        const int idx_ts = grid.flat_index(i, top_src);

        grid.rho[idx_tg]  = grid.rho[idx_ts];
        grid.rhou[idx_tg] = grid.rhou[idx_ts];
        grid.rhov[idx_tg] = grid.rhov[idx_ts];
        grid.E[idx_tg]    = grid.E[idx_ts];
    }
}

} 

void apply_transmissive_boundary_x_gpu(Grid2DGPU& grid) {
    const int threads = 256;
    const int blocks = (grid.ny() + threads - 1) / threads;

    apply_transmissive_boundary_x_kernel<<<blocks, threads>>>(make_view(grid));
    cudaGetLastError();
}

void apply_transmissive_boundary_y_gpu(Grid2DGPU& grid) {
    const int threads = 256;
    const int total_nx = grid.total_nx();
    const int blocks = (total_nx + threads - 1) / threads;

    apply_transmissive_boundary_y_kernel<<<blocks, threads>>>(make_view(grid));
    cudaGetLastError();
}

void apply_transmissive_boundary_gpu(Grid2DGPU& grid) {
    apply_transmissive_boundary_x_gpu(grid);
    apply_transmissive_boundary_y_gpu(grid);
}