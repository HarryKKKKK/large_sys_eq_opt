#include "solver_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr double gamma_gas = 1.4;
constexpr double small_p = 1e-12;
constexpr double small_rho = 1e-12;

struct PrimitiveGPU {
    double rho;
    double u;
    double v;
    double p;
};

struct FluxGPU {
    double rho;
    double rhou;
    double rhov;
    double E;
};

__device__ inline PrimitiveGPU cons_to_prim(double rho, double rhou, double rhov, double E) {
    PrimitiveGPU W{};
    W.rho = fmax(rho, small_rho);

    const double inv_rho = 1.0 / W.rho;
    W.u = rhou * inv_rho;
    W.v = rhov * inv_rho;

    const double kinetic = 0.5 * W.rho * (W.u * W.u + W.v * W.v);
    W.p = fmax((gamma_gas - 1.0) * (E - kinetic), small_p);

    return W;
}

__device__ inline FluxGPU flux_x(double rho, double rhou, double rhov, double E) {
    const PrimitiveGPU W = cons_to_prim(rho, rhou, rhov, E);

    FluxGPU F{};
    F.rho  = rhou;
    F.rhou = rhou * W.u + W.p;
    F.rhov = rhov * W.u;
    F.E    = (E + W.p) * W.u;
    return F;
}

__device__ inline FluxGPU flux_y(double rho, double rhou, double rhov, double E) {
    const PrimitiveGPU W = cons_to_prim(rho, rhou, rhov, E);

    FluxGPU F{};
    F.rho  = rhov;
    F.rhou = rhou * W.v;
    F.rhov = rhov * W.v + W.p;
    F.E    = (E + W.p) * W.v;
    return F;
}

__device__ inline FluxGPU hll_flux_x(
    double rhoL, double rhouL, double rhovL, double EL,
    double rhoR, double rhouR, double rhovR, double ER) {

    const PrimitiveGPU WL = cons_to_prim(rhoL, rhouL, rhovL, EL);
    const PrimitiveGPU WR = cons_to_prim(rhoR, rhouR, rhovR, ER);

    const double aL = sqrt(gamma_gas * WL.p / WL.rho);
    const double aR = sqrt(gamma_gas * WR.p / WR.rho);

    const double SL = fmin(WL.u - aL, WR.u - aR);
    const double SR = fmax(WL.u + aL, WR.u + aR);

    const FluxGPU FL = flux_x(rhoL, rhouL, rhovL, EL);
    const FluxGPU FR = flux_x(rhoR, rhouR, rhovR, ER);

    if (SL >= 0.0) {
        return FL;
    }
    if (SR <= 0.0) {
        return FR;
    }

    const double inv = 1.0 / (SR - SL);

    FluxGPU F{};
    F.rho  = (SR * FL.rho  - SL * FR.rho  + SL * SR * (rhoR  - rhoL )) * inv;
    F.rhou = (SR * FL.rhou - SL * FR.rhou + SL * SR * (rhouR - rhouL)) * inv;
    F.rhov = (SR * FL.rhov - SL * FR.rhov + SL * SR * (rhovR - rhovL)) * inv;
    F.E    = (SR * FL.E    - SL * FR.E    + SL * SR * (ER    - EL   )) * inv;
    return F;
}

__device__ inline FluxGPU hll_flux_y(
    double rhoL, double rhouL, double rhovL, double EL,
    double rhoR, double rhouR, double rhovR, double ER) {

    const PrimitiveGPU WL = cons_to_prim(rhoL, rhouL, rhovL, EL);
    const PrimitiveGPU WR = cons_to_prim(rhoR, rhouR, rhovR, ER);

    const double aL = sqrt(gamma_gas * WL.p / WL.rho);
    const double aR = sqrt(gamma_gas * WR.p / WR.rho);

    const double SL = fmin(WL.v - aL, WR.v - aR);
    const double SR = fmax(WL.v + aL, WR.v + aR);

    const FluxGPU FL = flux_y(rhoL, rhouL, rhovL, EL);
    const FluxGPU FR = flux_y(rhoR, rhouR, rhovR, ER);

    if (SL >= 0.0) {
        return FL;
    }
    if (SR <= 0.0) {
        return FR;
    }

    const double inv = 1.0 / (SR - SL);

    FluxGPU F{};
    F.rho  = (SR * FL.rho  - SL * FR.rho  + SL * SR * (rhoR  - rhoL )) * inv;
    F.rhou = (SR * FL.rhou - SL * FR.rhou + SL * SR * (rhouR - rhouL)) * inv;
    F.rhov = (SR * FL.rhov - SL * FR.rhov + SL * SR * (rhovR - rhovL)) * inv;
    F.E    = (SR * FL.E    - SL * FR.E    + SL * SR * (ER    - EL   )) * inv;
    return F;
}

__global__ void compute_local_speed_kernel(Grid2DGPUView grid, double* speed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_begin();

    if (i >= grid.i_end() || j >= grid.j_end()) {
        return;
    }

    const int idx = grid.flat_index(i, j);
    const PrimitiveGPU W = cons_to_prim(
        grid.rho[idx], grid.rhou[idx], grid.rhov[idx], grid.E[idx]);

    const double a = sqrt(gamma_gas * W.p / W.rho);
    const double sx = fabs(W.u) + a;
    const double sy = fabs(W.v) + a;
    speed[idx] = fmax(sx, sy);
}

__global__ void advance_first_order_kernel(
    Grid2DGPUView Uold,
    Grid2DGPUView Unew,
    double dt) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x + Uold.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + Uold.j_begin();

    if (i >= Uold.i_end() || j >= Uold.j_end()) {
        return;
    }

    const int c  = Uold.flat_index(i, j);
    const int im = Uold.flat_index(i - 1, j);
    const int ip = Uold.flat_index(i + 1, j);
    const int jm = Uold.flat_index(i, j - 1);
    const int jp = Uold.flat_index(i, j + 1);

    const FluxGPU FxL = hll_flux_x(
        Uold.rho[im], Uold.rhou[im], Uold.rhov[im], Uold.E[im],
        Uold.rho[c],  Uold.rhou[c],  Uold.rhov[c],  Uold.E[c]);

    const FluxGPU FxR = hll_flux_x(
        Uold.rho[c],  Uold.rhou[c],  Uold.rhov[c],  Uold.E[c],
        Uold.rho[ip], Uold.rhou[ip], Uold.rhov[ip], Uold.E[ip]);

    const FluxGPU FyB = hll_flux_y(
        Uold.rho[jm], Uold.rhou[jm], Uold.rhov[jm], Uold.E[jm],
        Uold.rho[c],  Uold.rhou[c],  Uold.rhov[c],  Uold.E[c]);

    const FluxGPU FyT = hll_flux_y(
        Uold.rho[c],  Uold.rhou[c],  Uold.rhov[c],  Uold.E[c],
        Uold.rho[jp], Uold.rhou[jp], Uold.rhov[jp], Uold.E[jp]);

    const double dtdx = dt / Uold.dx;
    const double dtdy = dt / Uold.dy;

    Unew.rho[c] =
        Uold.rho[c]
        - dtdx * (FxR.rho  - FxL.rho)
        - dtdy * (FyT.rho  - FyB.rho);

    Unew.rhou[c] =
        Uold.rhou[c]
        - dtdx * (FxR.rhou - FxL.rhou)
        - dtdy * (FyT.rhou - FyB.rhou);

    Unew.rhov[c] =
        Uold.rhov[c]
        - dtdx * (FxR.rhov - FxL.rhov)
        - dtdy * (FyT.rhov - FyB.rhov);

    Unew.E[c] =
        Uold.E[c]
        - dtdx * (FxR.E    - FxL.E)
        - dtdy * (FyT.E    - FyB.E);
}

}

double compute_dt_gpu(const Grid2DGPU& grid, double cfl) {
    const std::size_t n = grid.num_cells();
    double* speed_d = nullptr;
    CUDA_CHECK(cudaMalloc(&speed_d, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(speed_d, 0, n * sizeof(double)));

    const dim3 threads(16, 16);
    const dim3 blocks(
        (grid.nx() + threads.x - 1) / threads.x,
        (grid.ny() + threads.y - 1) / threads.y);

    compute_local_speed_kernel<<<blocks, threads>>>(make_view(grid), speed_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> speed_h(n, 0.0);
    CUDA_CHECK(cudaMemcpy(speed_h.data(), speed_d, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(speed_d));

    double smax = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            smax = std::max(smax, speed_h[grid.flat_index(i, j)]);
        }
    }

    if (smax <= 0.0) {
        return std::numeric_limits<double>::max();
    }

    const double h = std::min(grid.dx(), grid.dy());
    return cfl * h / smax;
}

void advance_first_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU& Unew,
    double dt
) {
    const dim3 threads(16, 16);
    const dim3 blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y);

    advance_first_order_kernel<<<blocks, threads>>>(
        make_view(Uold),
        make_view(Unew),
        dt
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}