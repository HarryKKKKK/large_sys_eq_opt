#include "gpu/solver_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/boundary_gpu.cuh"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

inline void check_cuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " in " + call + " : " + cudaGetErrorString(err)
        );
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

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

__device__ inline Conserved load_state(const ConstGrid2DGPUView& U, int i, int j) {
    const int idx = U.flat_index(i, j);
    return Conserved(U.rho[idx], U.rhou[idx], U.rhov[idx], U.E[idx]);
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

__device__ inline Conserved muscl_hancock_flux_x_gpu(const ConstGrid2DGPUView& U, int i, int j, double dt) {
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

__device__ inline Conserved muscl_hancock_flux_y_gpu(const ConstGrid2DGPUView& U, int i, int j, double dt) {
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

__device__ inline int xface_idx(int local_j, int local_i_face, int nx_faces) {
    return local_j * nx_faces + local_i_face;
}

__device__ inline int yface_idx(int local_j_face, int local_i, int nx_cells) {
    return local_j_face * nx_cells + local_i;
}

__global__ void compute_local_speed_kernel(ConstGrid2DGPUView grid, double* speed) {
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

__global__ void advance_first_order_kernel(ConstGrid2DGPUView Uold,
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

__global__ void compute_x_face_fluxes_kernel(
    ConstGrid2DGPUView Uin,
    double dt,
    double* fx_mass,
    double* fx_momx,
    double* fx_momy,
    double* fx_energy
) {
    const int nx_faces = Uin.nx + 1;
    const int local_i_face = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i_face >= nx_faces || local_j >= Uin.ny) {
        return;
    }

    const int i = (Uin.i_begin() - 1) + local_i_face;  // face between i and i+1
    const int j = Uin.j_begin() + local_j;

    const Conserved F = muscl_hancock_flux_x_gpu(Uin, i, j, dt);
    const int idx = xface_idx(local_j, local_i_face, nx_faces);

    fx_mass[idx]   = F.rho;
    fx_momx[idx]   = F.rhou;
    fx_momy[idx]   = F.rhov;
    fx_energy[idx] = F.E;
}

__global__ void update_from_x_face_fluxes_kernel(
    ConstGrid2DGPUView Uin,
    Grid2DGPUView Uout,
    double dt,
    const double* fx_mass,
    const double* fx_momx,
    const double* fx_momy,
    const double* fx_energy
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i >= Uin.nx || local_j >= Uin.ny) {
        return;
    }

    const int i = Uin.i_begin() + local_i;
    const int j = Uin.j_begin() + local_j;

    const int nx_faces = Uin.nx + 1;
    const int idx_m = xface_idx(local_j, local_i,     nx_faces); // i-1/2
    const int idx_p = xface_idx(local_j, local_i + 1, nx_faces); // i+1/2

    const Conserved Fx_m(fx_mass[idx_m], fx_momx[idx_m], fx_momy[idx_m], fx_energy[idx_m]);
    const Conserved Fx_p(fx_mass[idx_p], fx_momx[idx_p], fx_momy[idx_p], fx_energy[idx_p]);

    const Conserved Uc = load_state(Uin, i, j);
    const double dtdx = dt / Uin.dx;
    const Conserved Unew_cell = Uc - dtdx * (Fx_p - Fx_m);

    const int c = Uin.flat_index(i, j);
    Uout.rho[c]  = Unew_cell.rho;
    Uout.rhou[c] = Unew_cell.rhou;
    Uout.rhov[c] = Unew_cell.rhov;
    Uout.E[c]    = Unew_cell.E;
}

__global__ void compute_y_face_fluxes_kernel(
    ConstGrid2DGPUView Uin,
    double dt,
    double* fy_mass,
    double* fy_momx,
    double* fy_momy,
    double* fy_energy
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j_face = blockIdx.y * blockDim.y + threadIdx.y;

    const int ny_faces = Uin.ny + 1;

    if (local_i >= Uin.nx || local_j_face >= ny_faces) {
        return;
    }

    const int i = Uin.i_begin() + local_i;
    const int j = (Uin.j_begin() - 1) + local_j_face; // face between j and j+1

    const Conserved F = muscl_hancock_flux_y_gpu(Uin, i, j, dt);
    const int idx = yface_idx(local_j_face, local_i, Uin.nx);

    fy_mass[idx]   = F.rho;
    fy_momx[idx]   = F.rhou;
    fy_momy[idx]   = F.rhov;
    fy_energy[idx] = F.E;
}

__global__ void update_from_y_face_fluxes_kernel(
    ConstGrid2DGPUView Uin,
    Grid2DGPUView Uout,
    double dt,
    const double* fy_mass,
    const double* fy_momx,
    const double* fy_momy,
    const double* fy_energy
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i >= Uin.nx || local_j >= Uin.ny) {
        return;
    }

    const int i = Uin.i_begin() + local_i;
    const int j = Uin.j_begin() + local_j;

    const int idx_m = yface_idx(local_j,     local_i, Uin.nx); // j-1/2
    const int idx_p = yface_idx(local_j + 1, local_i, Uin.nx); // j+1/2

    const Conserved Fy_m(fy_mass[idx_m], fy_momx[idx_m], fy_momy[idx_m], fy_energy[idx_m]);
    const Conserved Fy_p(fy_mass[idx_p], fy_momx[idx_p], fy_momy[idx_p], fy_energy[idx_p]);

    const Conserved Uc = load_state(Uin, i, j);
    const double dtdy = dt / Uin.dy;
    const Conserved Unew_cell = Uc - dtdy * (Fy_p - Fy_m);

    const int c = Uin.flat_index(i, j);
    Uout.rho[c]  = Unew_cell.rho;
    Uout.rhou[c] = Unew_cell.rhou;
    Uout.rhov[c] = Unew_cell.rhov;
    Uout.E[c]    = Unew_cell.E;
}

} // namespace

double compute_dt_gpu(const Grid2DGPU& grid, double cfl) {
    const std::size_t n = grid.num_cells();

    double* speed_d = nullptr;
    CUDA_CHECK(cudaMalloc(&speed_d, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(speed_d, 0, n * sizeof(double)));

    const dim3 threads(16, 16);
    const dim3 blocks(
        (grid.nx() + threads.x - 1) / threads.x,
        (grid.ny() + threads.y - 1) / threads.y
    );

    compute_local_speed_kernel<<<blocks, threads>>>(make_view(grid), speed_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> speed_h(n, 0.0);
    CUDA_CHECK(cudaMemcpy(speed_h.data(), speed_d, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(speed_d));

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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_transmissive_boundary_gpu(Unew);
}

void advance_second_order_gpu(const Grid2DGPU& Uold,
                              Grid2DGPU& Utmp,
                              Grid2DGPU& Unew,
                              double dt) {
    const dim3 threads(16, 16);
    const dim3 cell_blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y
    );

    const dim3 xface_blocks(
        ((Uold.nx() + 1) + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y
    );

    const dim3 yface_blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        ((Uold.ny() + 1) + threads.y - 1) / threads.y
    );

    const std::size_t num_xfaces = static_cast<std::size_t>(Uold.nx() + 1) * Uold.ny();
    const std::size_t num_yfaces = static_cast<std::size_t>(Uold.ny() + 1) * Uold.nx();

    double *fx_mass = nullptr, *fx_momx = nullptr, *fx_momy = nullptr, *fx_energy = nullptr;
    double *fy_mass = nullptr, *fy_momx = nullptr, *fy_momy = nullptr, *fy_energy = nullptr;

    CUDA_CHECK(cudaMalloc(&fx_mass,   num_xfaces * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fx_momx,   num_xfaces * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fx_momy,   num_xfaces * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fx_energy, num_xfaces * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&fy_mass,   num_yfaces * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fy_momx,   num_yfaces * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fy_momy,   num_yfaces * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fy_energy, num_yfaces * sizeof(double)));

    compute_x_face_fluxes_kernel<<<xface_blocks, threads>>>(
        make_view(Uold), dt, fx_mass, fx_momx, fx_momy, fx_energy
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    update_from_x_face_fluxes_kernel<<<cell_blocks, threads>>>(
        make_view(Uold), make_view(Utmp), dt, fx_mass, fx_momx, fx_momy, fx_energy
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_transmissive_boundary_gpu(Utmp);

    compute_y_face_fluxes_kernel<<<yface_blocks, threads>>>(
        make_view(Utmp), dt, fy_mass, fy_momx, fy_momy, fy_energy
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    update_from_y_face_fluxes_kernel<<<cell_blocks, threads>>>(
        make_view(Utmp), make_view(Unew), dt, fy_mass, fy_momx, fy_momy, fy_energy
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_transmissive_boundary_gpu(Unew);

    CUDA_CHECK(cudaFree(fx_mass));
    CUDA_CHECK(cudaFree(fx_momx));
    CUDA_CHECK(cudaFree(fx_momy));
    CUDA_CHECK(cudaFree(fx_energy));

    CUDA_CHECK(cudaFree(fy_mass));
    CUDA_CHECK(cudaFree(fy_momx));
    CUDA_CHECK(cudaFree(fy_momy));
    CUDA_CHECK(cudaFree(fy_energy));
}