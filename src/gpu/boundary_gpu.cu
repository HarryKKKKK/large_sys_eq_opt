#include "gpu/boundary_gpu.cuh"

namespace {

__global__ void apply_transmissive_boundary_x_kernel(Grid2DGPUView grid) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + grid.j_begin();

    if (j >= grid.j_end()) {
        return;
    }

    const int ng = grid.ng;
    const int ib = grid.i_begin();
    const int ie = grid.i_end();

    for (int g = 0; g < ng; ++g) {
        const int iL = ib - 1 - g;
        const int iR = ie + g;

        const int idx_left_ghost  = grid.flat_index(iL, j);
        const int idx_left_inner  = grid.flat_index(ib, j);
        const int idx_right_ghost = grid.flat_index(iR, j);
        const int idx_right_inner = grid.flat_index(ie - 1, j);

        grid.rho[idx_left_ghost]  = grid.rho[idx_left_inner];
        grid.rhou[idx_left_ghost] = grid.rhou[idx_left_inner];
        grid.rhov[idx_left_ghost] = grid.rhov[idx_left_inner];
        grid.E[idx_left_ghost]    = grid.E[idx_left_inner];

        grid.rho[idx_right_ghost]  = grid.rho[idx_right_inner];
        grid.rhou[idx_right_ghost] = grid.rhou[idx_right_inner];
        grid.rhov[idx_right_ghost] = grid.rhov[idx_right_inner];
        grid.E[idx_right_ghost]    = grid.E[idx_right_inner];
    }
}

__global__ void apply_transmissive_boundary_y_kernel(Grid2DGPUView grid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= grid.total_nx()) {
        return;
    }

    const int ng = grid.ng;
    const int jb = grid.j_begin();
    const int je = grid.j_end();

    for (int g = 0; g < ng; ++g) {
        const int jB = jb - 1 - g;
        const int jT = je + g;

        const int idx_bottom_ghost = grid.flat_index(i, jB);
        const int idx_bottom_inner = grid.flat_index(i, jb);
        const int idx_top_ghost    = grid.flat_index(i, jT);
        const int idx_top_inner    = grid.flat_index(i, je - 1);

        grid.rho[idx_bottom_ghost]  = grid.rho[idx_bottom_inner];
        grid.rhou[idx_bottom_ghost] = grid.rhou[idx_bottom_inner];
        grid.rhov[idx_bottom_ghost] = grid.rhov[idx_bottom_inner];
        grid.E[idx_bottom_ghost]    = grid.E[idx_bottom_inner];

        grid.rho[idx_top_ghost]  = grid.rho[idx_top_inner];
        grid.rhou[idx_top_ghost] = grid.rhou[idx_top_inner];
        grid.rhov[idx_top_ghost] = grid.rhov[idx_top_inner];
        grid.E[idx_top_ghost]    = grid.E[idx_top_inner];
    }
}

} 

void apply_transmissive_boundary_gpu(Grid2DGPU& grid) {
    Grid2DGPUView view = make_view(grid);

    {
        const int threads = 256;
        const int blocks = (grid.ny() + threads - 1) / threads;
        apply_transmissive_boundary_x_kernel<<<blocks, threads>>>(view);
        cudaGetLastError();
    }

    {
        const int threads = 256;
        const int blocks = (grid.total_nx() + threads - 1) / threads;
        apply_transmissive_boundary_y_kernel<<<blocks, threads>>>(view);
        cudaGetLastError();
    }

    cudaDeviceSynchronize();
}