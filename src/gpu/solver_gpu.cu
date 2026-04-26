#include "gpu/solver_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "gpu/boundary_gpu.cuh"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

__device__ inline double minmod_gpu(double a, double b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (a > 0.0) ? fmin(a, b) : fmax(a, b);
}

__device__ inline Primitive minmod_gpu(const Primitive& a, const Primitive& b) {
    return Primitive(
        minmod_gpu(a.rho, b.rho),
        minmod_gpu(a.u,   b.u),
        minmod_gpu(a.v,   b.v),
        minmod_gpu(a.p,   b.p)
    );
}

__device__ inline bool is_physical_gpu(const Primitive& V) {
    return isfinite(V.rho) && isfinite(V.u) &&
           isfinite(V.v)   && isfinite(V.p) &&
           (V.rho > kRhoFloor) && (V.p > kPFloor);
}

__device__ inline Primitive enforce_physical_primitive_gpu(const Primitive& candidate,
                                                           const Primitive& fallback) {
    return is_physical_gpu(candidate) ? candidate : fallback;
}

__device__ inline Conserved enforce_physical_conserved_gpu(const Conserved& candidate,
                                                           const Conserved& fallback) {
    const Primitive Vcand = phys::cons_to_prim(candidate);
    return is_physical_gpu(Vcand) ? candidate : fallback;
}

__device__ inline Conserved load_state(const Grid2DGPUView& U, int i, int j) {
    const int idx = U.flat_index(i, j);
    return Conserved(U.rho[idx], U.rhou[idx], U.rhov[idx], U.E[idx]);
}

__device__ inline Primitive limited_slope_gpu(const Primitive& Wm,
                                              const Primitive& Wc,
                                              const Primitive& Wp) {
    return minmod_gpu(Wc - Wm, Wp - Wc);
}

__device__ inline void reconstruct_cell_muscl_hancock_gpu(
    const Conserved& Um,
    const Conserved& Uc,
    const Conserved& Up,
    double dt_over_d,
    Direction dir,
    Conserved& U_left_star,
    Conserved& U_right_star
) {
    const Primitive Wm = phys::cons_to_prim(Um);
    const Primitive Wc = phys::cons_to_prim(Uc);
    const Primitive Wp = phys::cons_to_prim(Up);

    const Primitive slope = limited_slope_gpu(Wm, Wc, Wp);

    Primitive W_left  = Wc - 0.5 * slope;
    Primitive W_right = Wc + 0.5 * slope;

    W_left  = enforce_physical_primitive_gpu(W_left,  Wc);
    W_right = enforce_physical_primitive_gpu(W_right, Wc);

    const Conserved U_left  = phys::prim_to_cons(W_left);
    const Conserved U_right = phys::prim_to_cons(W_right);

    const Conserved F_left  = physical_flux(U_left,  dir);
    const Conserved F_right = physical_flux(U_right, dir);

    const Conserved half_update = 0.5 * dt_over_d * (F_right - F_left);

    U_left_star  = U_left  - half_update;
    U_right_star = U_right - half_update;

    U_left_star  = enforce_physical_conserved_gpu(U_left_star,  U_left);
    U_right_star = enforce_physical_conserved_gpu(U_right_star, U_right);
}

__device__ inline Conserved muscl_hancock_flux_x_gpu(const Grid2DGPUView& U, int i, int j, double dt) {
    const double dt_over_dx = dt / U.dx;

    Conserved Ui_L_star, Ui_R_star;
    Conserved Uip1_L_star, Uip1_R_star;

    reconstruct_cell_muscl_hancock_gpu(
        load_state(U, i - 1, j),
        load_state(U, i,     j),
        load_state(U, i + 1, j),
        dt_over_dx,
        Direction::X,
        Ui_L_star, Ui_R_star
    );

    reconstruct_cell_muscl_hancock_gpu(
        load_state(U, i,     j),
        load_state(U, i + 1, j),
        load_state(U, i + 2, j),
        dt_over_dx,
        Direction::X,
        Uip1_L_star, Uip1_R_star
    );

    return hll_flux(Ui_R_star, Uip1_L_star, Direction::X);
}

__device__ inline Conserved muscl_hancock_flux_y_gpu(const Grid2DGPUView& U, int i, int j, double dt) {
    const double dt_over_dy = dt / U.dy;

    Conserved Uj_L_star, Uj_R_star;
    Conserved Ujp1_L_star, Ujp1_R_star;

    reconstruct_cell_muscl_hancock_gpu(
        load_state(U, i, j - 1),
        load_state(U, i, j),
        load_state(U, i, j + 1),
        dt_over_dy,
        Direction::Y,
        Uj_L_star, Uj_R_star
    );

    reconstruct_cell_muscl_hancock_gpu(
        load_state(U, i, j),
        load_state(U, i, j + 1),
        load_state(U, i, j + 2),
        dt_over_dy,
        Direction::Y,
        Ujp1_L_star, Ujp1_R_star
    );

    return hll_flux(Uj_R_star, Ujp1_L_star, Direction::Y);
}

__global__ void compute_local_speed_kernel(Grid2DGPUView grid, double* speed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_begin();

    if (i >= grid.i_end() || j >= grid.j_end()) {
        return;
    }

    const int idx = grid.flat_index(i, j);
    const Conserved U(grid.rho[idx], grid.rhou[idx], grid.rhov[idx], grid.E[idx]);
    const Primitive V = phys::cons_to_prim(U);
    const double a = phys::sound_speed(V);
    const double sx = fabs(V.u) + a;
    const double sy = fabs(V.v) + a;
    speed[idx] = fmax(sx, sy);
}

__global__ void advance_first_order_kernel(Grid2DGPUView Uold,
                                           Grid2DGPUView Unew,
                                           double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + Uold.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + Uold.j_begin();

    if (i >= Uold.i_end() || j >= Uold.j_end()) {
        return;
    }

    const Conserved Uc  = load_state(Uold, i,     j);
    const Conserved Uim = load_state(Uold, i - 1, j);
    const Conserved Uip = load_state(Uold, i + 1, j);
    const Conserved Ujm = load_state(Uold, i, j - 1);
    const Conserved Ujp = load_state(Uold, i, j + 1);

    const Conserved FxL = hll_flux(Uim, Uc,  Direction::X);
    const Conserved FxR = hll_flux(Uc,  Uip, Direction::X);
    const Conserved FyB = hll_flux(Ujm, Uc,  Direction::Y);
    const Conserved FyT = hll_flux(Uc,  Ujp, Direction::Y);

    const double dtdx = dt / Uold.dx;
    const double dtdy = dt / Uold.dy;
    const Conserved Unew_cell = Uc
                              - dtdx * (FxR - FxL)
                              - dtdy * (FyT - FyB);

    const int c = Uold.flat_index(i, j);
    Unew.rho[c]  = Unew_cell.rho;
    Unew.rhou[c] = Unew_cell.rhou;
    Unew.rhov[c] = Unew_cell.rhov;
    Unew.E[c]    = Unew_cell.E;
}

__global__ void sweep_x_second_order_kernel(Grid2DGPUView Uin,
                                            Grid2DGPUView Uout,
                                            double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + Uin.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + Uin.j_begin();

    if (i >= Uin.i_end() || j >= Uin.j_end()) {
        return;
    }

    const Conserved Fx_p = muscl_hancock_flux_x_gpu(Uin, i,     j, dt);
    const Conserved Fx_m = muscl_hancock_flux_x_gpu(Uin, i - 1, j, dt);

    const Conserved Uc = load_state(Uin, i, j);
    const double dtdx = dt / Uin.dx;
    const Conserved Unew_cell = Uc - dtdx * (Fx_p - Fx_m);

    const int c = Uin.flat_index(i, j);
    Uout.rho[c]  = Unew_cell.rho;
    Uout.rhou[c] = Unew_cell.rhou;
    Uout.rhov[c] = Unew_cell.rhov;
    Uout.E[c]    = Unew_cell.E;
}

__global__ void sweep_y_second_order_kernel(Grid2DGPUView Uin, Grid2DGPUView Uout, double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + Uin.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + Uin.j_begin();

    if (i >= Uin.i_end() || j >= Uin.j_end()) {
        return;
    }

    const Conserved Fy_p = muscl_hancock_flux_y_gpu(Uin, i, j,     dt);
    const Conserved Fy_m = muscl_hancock_flux_y_gpu(Uin, i, j - 1, dt);

    const Conserved Uc = load_state(Uin, i, j);
    const double dtdy = dt / Uin.dy;
    const Conserved Unew_cell = Uc - dtdy * (Fy_p - Fy_m);

    const int c = Uin.flat_index(i, j);
    Uout.rho[c]  = Unew_cell.rho;
    Uout.rhou[c] = Unew_cell.rhou;
    Uout.rhov[c] = Unew_cell.rhov;
    Uout.E[c]    = Unew_cell.E;
}

}

double compute_dt_gpu(const Grid2DGPU& grid, double cfl) {
    const std::size_t n = grid.num_cells();

    double* speed_d = nullptr;
    cudaMalloc(&speed_d, n * sizeof(double));
    cudaMemset(speed_d, 0, n * sizeof(double));

    const dim3 threads(16, 16);
    const dim3 blocks(
        (grid.nx() + threads.x - 1) / threads.x,
        (grid.ny() + threads.y - 1) / threads.y
    );

    compute_local_speed_kernel<<<blocks, threads>>>(make_view(grid), speed_d);
    cudaGetLastError();
    cudaDeviceSynchronize();

    std::vector<double> speed_h(n, 0.0);
    cudaMemcpy(speed_h.data(), speed_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(speed_d);

    double max_speed = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            max_speed = std::max(max_speed, speed_h[grid.flat_index(i, j)]);
        }
    }

    if (max_speed <= 0.0) {
        return std::numeric_limits<double>::max();
    }

    const double dt_x = grid.dx() / max_speed;
    const double dt_y = grid.dy() / max_speed;
    return cfl * std::min(dt_x, dt_y);
}

void advance_first_order_gpu(const Grid2DGPU& Uold, Grid2DGPU& Unew, double dt) {
    const dim3 threads(16, 16);
    const dim3 blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y
    );

    advance_first_order_kernel<<<blocks, threads>>>(make_view(Uold), make_view(Unew), dt);
    cudaGetLastError();
    cudaDeviceSynchronize();
    apply_transmissive_boundary_gpu(Unew);
}

void advance_second_order_gpu(const Grid2DGPU& Uold,
                              Grid2DGPU& Utmp,
                              Grid2DGPU& Unew,
                              double dt) {
    const dim3 threads(16, 16);
    const dim3 blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y
    );

    sweep_x_second_order_kernel<<<blocks, threads>>>(make_view(Uold), make_view(Utmp), dt);
    cudaGetLastError();
    apply_transmissive_boundary_y_gpu(Utmp);
    sweep_y_second_order_kernel<<<blocks, threads>>>(make_view(Utmp), make_view(Unew), dt);
    cudaGetLastError();
    // TODO: could be replace by boundry_x ?
    // apply_transmissive_boundary_x_gpu(Unew);
    apply_transmissive_boundary_gpu(Unew);
}
