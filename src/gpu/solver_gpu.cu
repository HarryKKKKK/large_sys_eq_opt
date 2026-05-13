#include "gpu/solver_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "gpu/boundary_gpu.cuh"
#include "physics.hpp"
#include "riemann.hpp"
#include "types.hpp"

namespace {

constexpr double kRhoFloor = 1.0e-12;
constexpr double kPFloor   = 1.0e-12;

inline void check_cuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " in " << call << " : " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

// -----------------------------------------------------------------------------
// Basic device helpers
// -----------------------------------------------------------------------------

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

__device__ inline Primitive enforce_physical_primitive_gpu(
    const Primitive& candidate,
    const Primitive& fallback
) {
    return is_physical_gpu(candidate) ? candidate : fallback;
}

__device__ inline Conserved enforce_physical_conserved_gpu(
    const Conserved& candidate,
    const Conserved& fallback
) {
    const Primitive Vcand = phys::cons_to_prim(candidate);
    return is_physical_gpu(Vcand) ? candidate : fallback;
}

__device__ inline Conserved load_state(const ConstGrid2DGPUView& U, int i, int j) {
    const int idx = U.flat_index(i, j);
    return Conserved(U.rho[idx], U.rhou[idx], U.rhov[idx], U.E[idx]);
}

__device__ inline Primitive limited_slope_gpu(
    const Primitive& Wm,
    const Primitive& Wc,
    const Primitive& Wp
) {
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

    U_left_star  = enforce_physical_conserved_gpu(U_left  - half_update, U_left);
    U_right_star = enforce_physical_conserved_gpu(U_right - half_update, U_right);
}

// -----------------------------------------------------------------------------
// Shared memory load/store helpers
// -----------------------------------------------------------------------------

__device__ inline Conserved load_conserved_smem(
    const double* __restrict__ s_rho,
    const double* __restrict__ s_rhou,
    const double* __restrict__ s_rhov,
    const double* __restrict__ s_E,
    int idx
) {
    return Conserved(
        s_rho[idx],
        s_rhou[idx],
        s_rhov[idx],
        s_E[idx]
    );
}

__device__ inline void store_conserved_smem(
    double* __restrict__ s_rho,
    double* __restrict__ s_rhou,
    double* __restrict__ s_rhov,
    double* __restrict__ s_E,
    int idx,
    const Conserved& U
) {
    s_rho[idx]  = U.rho;
    s_rhou[idx] = U.rhou;
    s_rhov[idx] = U.rhov;
    s_E[idx]    = U.E;
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__global__ void compute_local_speed_kernel(ConstGrid2DGPUView grid, double* speed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + grid.i_begin();
    const int j = blockIdx.y * blockDim.y + threadIdx.y + grid.j_begin();

    if (i >= grid.i_end() || j >= grid.j_end()) {
        return;
    }

    const int idx = grid.flat_index(i, j);

    const Conserved U(
        grid.rho[idx],
        grid.rhou[idx],
        grid.rhov[idx],
        grid.E[idx]
    );

    const Primitive V = phys::cons_to_prim(U);
    const double a = phys::sound_speed(V);

    const double sx = fabs(V.u) + a;
    const double sy = fabs(V.v) + a;

    speed[idx] = fmax(sx, sy);
}

__global__ void advance_first_order_kernel(
    ConstGrid2DGPUView Uold,
    Grid2DGPUView Unew,
    double dt
) {
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

// -----------------------------------------------------------------------------
// Ultimate x-sweep kernel:
// global U -> shared U tile -> shared reconstructed states -> shared fluxes
// -> global Uout
// -----------------------------------------------------------------------------

__global__ void advance_x_reconstruct_smem_fused_kernel(
    ConstGrid2DGPUView Uin,
    Grid2DGPUView Uout,
    double dt
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;

    const int block_i_start = blockIdx.x * blockDim.x;

    /*
      For updating blockDim.x cells, we need blockDim.x + 1 faces.

      Face f needs reconstructed states from cells:
          left cell  = f - 1
          right cell = f

      Therefore we reconstruct cells:
          block_i_start - 1 ... block_i_start + blockDim.x

      Reconstruction of a cell needs one extra neighbour on both sides.
      Therefore the conserved state tile is:
          block_i_start - 2 ... block_i_start + blockDim.x + 1
    */

    const int state_tile_w = blockDim.x + 4;
    const int state_tile_h = blockDim.y;
    const int state_tile_n = state_tile_w * state_tile_h;

    const int recon_tile_w = blockDim.x + 2;
    const int recon_tile_h = blockDim.y;
    const int recon_tile_n = recon_tile_w * recon_tile_h;

    const int flux_tile_w = blockDim.x + 1;
    const int flux_tile_h = blockDim.y;
    const int flux_tile_n = flux_tile_w * flux_tile_h;

    extern __shared__ double smem[];

    double* s_rho  = smem;
    double* s_rhou = s_rho  + state_tile_n;
    double* s_rhov = s_rhou + state_tile_n;
    double* s_E    = s_rhov + state_tile_n;

    double* s_L_rho  = s_E      + state_tile_n;
    double* s_L_rhou = s_L_rho  + recon_tile_n;
    double* s_L_rhov = s_L_rhou + recon_tile_n;
    double* s_L_E    = s_L_rhov + recon_tile_n;

    double* s_R_rho  = s_L_E    + recon_tile_n;
    double* s_R_rhou = s_R_rho  + recon_tile_n;
    double* s_R_rhov = s_R_rhou + recon_tile_n;
    double* s_R_E    = s_R_rhov + recon_tile_n;

    double* s_F_rho  = s_R_E    + recon_tile_n;
    double* s_F_rhou = s_F_rho  + flux_tile_n;
    double* s_F_rhov = s_F_rhou + flux_tile_n;
    double* s_F_E    = s_F_rhov + flux_tile_n;

    const int i_state_start = Uin.i_begin() + block_i_start - 2;
    const int valid_i_min = Uin.i_begin() - 2;
    const int valid_i_max = Uin.i_end() + 1;

    // -------------------------------------------------------------------------
    // 1. Load conserved state tile into shared memory.
    // -------------------------------------------------------------------------
    for (int linear = tid; linear < state_tile_n; linear += block_threads) {
        const int sj = linear / state_tile_w;
        const int si = linear - sj * state_tile_w;

        const int lj = blockIdx.y * blockDim.y + sj;
        const int gi = i_state_start + si;

        if (lj < Uin.ny && gi >= valid_i_min && gi <= valid_i_max) {
            const int gj = Uin.j_begin() + lj;
            const int gidx = Uin.flat_index(gi, gj);

            s_rho[linear]  = Uin.rho[gidx];
            s_rhou[linear] = Uin.rhou[gidx];
            s_rhov[linear] = Uin.rhov[gidx];
            s_E[linear]    = Uin.E[gidx];
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. Reconstruct each required cell once and cache left/right star states.
    //    recon local cell index = block_i_start - 1 + sr
    // -------------------------------------------------------------------------
    const double dt_over_dx = dt / Uin.dx;

    for (int linear = tid; linear < recon_tile_n; linear += block_threads) {
        const int sj = linear / recon_tile_w;
        const int sr = linear - sj * recon_tile_w;

        const int lj = blockIdx.y * blockDim.y + sj;
        const int local_recon_i = block_i_start - 1 + sr;

        if (lj < Uin.ny && local_recon_i >= -1 && local_recon_i <= Uin.nx) {
            const int state_center = sj * state_tile_w + (sr + 1);

            const Conserved Um = load_conserved_smem(
                s_rho, s_rhou, s_rhov, s_E,
                state_center - 1
            );

            const Conserved Uc = load_conserved_smem(
                s_rho, s_rhou, s_rhov, s_E,
                state_center
            );

            const Conserved Up = load_conserved_smem(
                s_rho, s_rhou, s_rhov, s_E,
                state_center + 1
            );

            Conserved U_L_star;
            Conserved U_R_star;

            reconstruct_cell_muscl_hancock_gpu(
                Um,
                Uc,
                Up,
                dt_over_dx,
                Direction::X,
                U_L_star,
                U_R_star
            );

            store_conserved_smem(
                s_L_rho, s_L_rhou, s_L_rhov, s_L_E,
                linear,
                U_L_star
            );

            store_conserved_smem(
                s_R_rho, s_R_rhou, s_R_rhov, s_R_E,
                linear,
                U_R_star
            );
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 3. Compute each x-face flux once inside shared memory.
    //    face local index = block_i_start + sf
    //
    //    face f uses:
    //        left  = R_star of cell f - 1
    //        right = L_star of cell f
    // -------------------------------------------------------------------------
    for (int linear = tid; linear < flux_tile_n; linear += block_threads) {
        const int sj = linear / flux_tile_w;
        const int sf = linear - sj * flux_tile_w;

        const int lj = blockIdx.y * blockDim.y + sj;
        const int local_face_i = block_i_start + sf;

        if (lj < Uin.ny && local_face_i >= 0 && local_face_i <= Uin.nx) {
            const int left_recon_idx  = sj * recon_tile_w + sf;
            const int right_recon_idx = sj * recon_tile_w + sf + 1;

            const Conserved UL = load_conserved_smem(
                s_R_rho, s_R_rhou, s_R_rhov, s_R_E,
                left_recon_idx
            );

            const Conserved UR = load_conserved_smem(
                s_L_rho, s_L_rhou, s_L_rhov, s_L_E,
                right_recon_idx
            );

            const Conserved F = hll_flux(UL, UR, Direction::X);

            s_F_rho[linear]  = F.rho;
            s_F_rhou[linear] = F.rhou;
            s_F_rhov[linear] = F.rhov;
            s_F_E[linear]    = F.E;
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 4. Update cells directly.
    // -------------------------------------------------------------------------
    if (local_i >= Uin.nx || local_j >= Uin.ny) {
        return;
    }

    const int sj = threadIdx.y;

    const int fidx_m = sj * flux_tile_w + threadIdx.x;
    const int fidx_p = sj * flux_tile_w + threadIdx.x + 1;

    const Conserved Fx_m(
        s_F_rho[fidx_m],
        s_F_rhou[fidx_m],
        s_F_rhov[fidx_m],
        s_F_E[fidx_m]
    );

    const Conserved Fx_p(
        s_F_rho[fidx_p],
        s_F_rhou[fidx_p],
        s_F_rhov[fidx_p],
        s_F_E[fidx_p]
    );

    const int state_cell_idx = sj * state_tile_w + threadIdx.x + 2;

    const Conserved Uc = load_conserved_smem(
        s_rho, s_rhou, s_rhov, s_E,
        state_cell_idx
    );

    const Conserved Unew_cell = Uc - dt_over_dx * (Fx_p - Fx_m);

    const int gi = Uin.i_begin() + local_i;
    const int gj = Uin.j_begin() + local_j;
    const int gidx = Uin.flat_index(gi, gj);

    Uout.rho[gidx]  = Unew_cell.rho;
    Uout.rhou[gidx] = Unew_cell.rhou;
    Uout.rhov[gidx] = Unew_cell.rhov;
    Uout.E[gidx]    = Unew_cell.E;
}

// -----------------------------------------------------------------------------
// Ultimate y-sweep kernel:
// global U -> shared U tile -> shared reconstructed states -> shared fluxes
// -> global Uout
// -----------------------------------------------------------------------------

__global__ void advance_y_reconstruct_smem_fused_kernel(
    ConstGrid2DGPUView Uin,
    Grid2DGPUView Uout,
    double dt
) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_j = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;

    const int block_j_start = blockIdx.y * blockDim.y;

    /*
      Same idea as x-sweep, but along y.

      For updating blockDim.y cells, we need blockDim.y + 1 y-faces.

      Face f needs reconstructed states from:
          lower cell = f - 1
          upper cell = f

      Therefore reconstruct:
          block_j_start - 1 ... block_j_start + blockDim.y

      State tile:
          block_j_start - 2 ... block_j_start + blockDim.y + 1
    */

    const int state_tile_w = blockDim.x;
    const int state_tile_h = blockDim.y + 4;
    const int state_tile_n = state_tile_w * state_tile_h;

    const int recon_tile_w = blockDim.x;
    const int recon_tile_h = blockDim.y + 2;
    const int recon_tile_n = recon_tile_w * recon_tile_h;

    const int flux_tile_w = blockDim.x;
    const int flux_tile_h = blockDim.y + 1;
    const int flux_tile_n = flux_tile_w * flux_tile_h;

    extern __shared__ double smem[];

    double* s_rho  = smem;
    double* s_rhou = s_rho  + state_tile_n;
    double* s_rhov = s_rhou + state_tile_n;
    double* s_E    = s_rhov + state_tile_n;

    double* s_L_rho  = s_E      + state_tile_n;
    double* s_L_rhou = s_L_rho  + recon_tile_n;
    double* s_L_rhov = s_L_rhou + recon_tile_n;
    double* s_L_E    = s_L_rhov + recon_tile_n;

    double* s_R_rho  = s_L_E    + recon_tile_n;
    double* s_R_rhou = s_R_rho  + recon_tile_n;
    double* s_R_rhov = s_R_rhou + recon_tile_n;
    double* s_R_E    = s_R_rhov + recon_tile_n;

    double* s_F_rho  = s_R_E    + recon_tile_n;
    double* s_F_rhou = s_F_rho  + flux_tile_n;
    double* s_F_rhov = s_F_rhou + flux_tile_n;
    double* s_F_E    = s_F_rhov + flux_tile_n;

    const int j_state_start = Uin.j_begin() + block_j_start - 2;
    const int valid_j_min = Uin.j_begin() - 2;
    const int valid_j_max = Uin.j_end() + 1;

    // -------------------------------------------------------------------------
    // 1. Load conserved state tile into shared memory.
    // -------------------------------------------------------------------------
    for (int linear = tid; linear < state_tile_n; linear += block_threads) {
        const int sj = linear / state_tile_w;
        const int si = linear - sj * state_tile_w;

        const int li = blockIdx.x * blockDim.x + si;
        const int gj = j_state_start + sj;

        if (li < Uin.nx && gj >= valid_j_min && gj <= valid_j_max) {
            const int gi = Uin.i_begin() + li;
            const int gidx = Uin.flat_index(gi, gj);

            s_rho[linear]  = Uin.rho[gidx];
            s_rhou[linear] = Uin.rhou[gidx];
            s_rhov[linear] = Uin.rhov[gidx];
            s_E[linear]    = Uin.E[gidx];
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. Reconstruct each required cell once and cache lower/upper star states.
    //
    //    For y direction:
    //        U_left_star  means lower-side state
    //        U_right_star means upper-side state
    // -------------------------------------------------------------------------
    const double dt_over_dy = dt / Uin.dy;

    for (int linear = tid; linear < recon_tile_n; linear += block_threads) {
        const int sr = linear / recon_tile_w;
        const int si = linear - sr * recon_tile_w;

        const int li = blockIdx.x * blockDim.x + si;
        const int local_recon_j = block_j_start - 1 + sr;

        if (li < Uin.nx && local_recon_j >= -1 && local_recon_j <= Uin.ny) {
            const int state_center = (sr + 1) * state_tile_w + si;

            const Conserved Um = load_conserved_smem(
                s_rho, s_rhou, s_rhov, s_E,
                state_center - state_tile_w
            );

            const Conserved Uc = load_conserved_smem(
                s_rho, s_rhou, s_rhov, s_E,
                state_center
            );

            const Conserved Up = load_conserved_smem(
                s_rho, s_rhou, s_rhov, s_E,
                state_center + state_tile_w
            );

            Conserved U_L_star;
            Conserved U_R_star;

            reconstruct_cell_muscl_hancock_gpu(
                Um,
                Uc,
                Up,
                dt_over_dy,
                Direction::Y,
                U_L_star,
                U_R_star
            );

            store_conserved_smem(
                s_L_rho, s_L_rhou, s_L_rhov, s_L_E,
                linear,
                U_L_star
            );

            store_conserved_smem(
                s_R_rho, s_R_rhou, s_R_rhov, s_R_E,
                linear,
                U_R_star
            );
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 3. Compute each y-face flux once inside shared memory.
    //
    //    face f uses:
    //        lower = R_star of cell f - 1
    //        upper = L_star of cell f
    // -------------------------------------------------------------------------
    for (int linear = tid; linear < flux_tile_n; linear += block_threads) {
        const int sf = linear / flux_tile_w;
        const int si = linear - sf * flux_tile_w;

        const int li = blockIdx.x * blockDim.x + si;
        const int local_face_j = block_j_start + sf;

        if (li < Uin.nx && local_face_j >= 0 && local_face_j <= Uin.ny) {
            const int lower_recon_idx = sf * recon_tile_w + si;
            const int upper_recon_idx = (sf + 1) * recon_tile_w + si;

            const Conserved UL = load_conserved_smem(
                s_R_rho, s_R_rhou, s_R_rhov, s_R_E,
                lower_recon_idx
            );

            const Conserved UR = load_conserved_smem(
                s_L_rho, s_L_rhou, s_L_rhov, s_L_E,
                upper_recon_idx
            );

            const Conserved F = hll_flux(UL, UR, Direction::Y);

            s_F_rho[linear]  = F.rho;
            s_F_rhou[linear] = F.rhou;
            s_F_rhov[linear] = F.rhov;
            s_F_E[linear]    = F.E;
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 4. Update cells directly.
    // -------------------------------------------------------------------------
    if (local_i >= Uin.nx || local_j >= Uin.ny) {
        return;
    }

    const int si = threadIdx.x;

    const int fidx_m = threadIdx.y * flux_tile_w + si;
    const int fidx_p = (threadIdx.y + 1) * flux_tile_w + si;

    const Conserved Fy_m(
        s_F_rho[fidx_m],
        s_F_rhou[fidx_m],
        s_F_rhov[fidx_m],
        s_F_E[fidx_m]
    );

    const Conserved Fy_p(
        s_F_rho[fidx_p],
        s_F_rhou[fidx_p],
        s_F_rhov[fidx_p],
        s_F_E[fidx_p]
    );

    const int state_cell_idx = (threadIdx.y + 2) * state_tile_w + si;

    const Conserved Uc = load_conserved_smem(
        s_rho, s_rhou, s_rhov, s_E,
        state_cell_idx
    );

    const Conserved Unew_cell = Uc - dt_over_dy * (Fy_p - Fy_m);

    const int gi = Uin.i_begin() + local_i;
    const int gj = Uin.j_begin() + local_j;
    const int gidx = Uin.flat_index(gi, gj);

    Uout.rho[gidx]  = Unew_cell.rho;
    Uout.rhou[gidx] = Unew_cell.rhou;
    Uout.rhov[gidx] = Unew_cell.rhov;
    Uout.E[gidx]    = Unew_cell.E;
}

} // namespace

// -----------------------------------------------------------------------------
// Workspace management
// -----------------------------------------------------------------------------

void init_gpu_workspace(GpuWorkspace& ws, const Grid2DGPU& grid) {
    free_gpu_workspace(ws);

    ws.nx = grid.nx();
    ws.ny = grid.ny();

    const std::size_t total_cells = grid.num_cells();

    CUDA_CHECK(cudaMalloc(&ws.speed_d, total_cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws.max_speed_d, sizeof(double)));

    ws.reduce_tmp_bytes = 0;

    cub::DeviceReduce::Max(
        nullptr,
        ws.reduce_tmp_bytes,
        ws.speed_d,
        ws.max_speed_d,
        total_cells
    );

    CUDA_CHECK(cudaMalloc(&ws.reduce_tmp, ws.reduce_tmp_bytes));
}

void free_gpu_workspace(GpuWorkspace& ws) {
    if (ws.speed_d) {
        CUDA_CHECK(cudaFree(ws.speed_d));
    }

    if (ws.max_speed_d) {
        CUDA_CHECK(cudaFree(ws.max_speed_d));
    }

    if (ws.reduce_tmp) {
        CUDA_CHECK(cudaFree(ws.reduce_tmp));
    }

    ws = GpuWorkspace{};
}

// -----------------------------------------------------------------------------
// Time step
// -----------------------------------------------------------------------------

double compute_dt_gpu(const Grid2DGPU& grid, GpuWorkspace& ws, double cfl) {
    if (ws.nx != grid.nx() || ws.ny != grid.ny() || ws.speed_d == nullptr) {
        throw std::runtime_error("compute_dt_gpu: workspace not initialized for this grid.");
    }

    const dim3 threads(16, 16);

    const dim3 blocks(
        (grid.nx() + threads.x - 1) / threads.x,
        (grid.ny() + threads.y - 1) / threads.y
    );

    compute_local_speed_kernel<<<blocks, threads>>>(make_view(grid), ws.speed_d);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceReduce::Max(
        ws.reduce_tmp,
        ws.reduce_tmp_bytes,
        ws.speed_d,
        ws.max_speed_d,
        grid.num_cells()
    );

    CUDA_CHECK(cudaGetLastError());

    double max_speed = 0.0;

    CUDA_CHECK(cudaMemcpy(
        &max_speed,
        ws.max_speed_d,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    if (max_speed <= 0.0) {
        return std::numeric_limits<double>::max();
    }

    return cfl * std::min(grid.dx(), grid.dy()) / max_speed;
}

// -----------------------------------------------------------------------------
// First-order advance
// -----------------------------------------------------------------------------

void advance_first_order_gpu(const Grid2DGPU& Uold, Grid2DGPU& Unew, double dt) {
    const dim3 threads(16, 16);

    const dim3 blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y
    );

    advance_first_order_kernel<<<blocks, threads>>>(
        make_view(Uold),
        make_view(Unew),
        dt
    );

    CUDA_CHECK(cudaGetLastError());

    apply_transmissive_boundary_gpu(Unew);
}

// -----------------------------------------------------------------------------
// Second-order advance: ultimate shared-memory reconstruction-cache version
// -----------------------------------------------------------------------------

void advance_second_order_gpu(
    const Grid2DGPU& Uold,
    Grid2DGPU& Utmp,
    Grid2DGPU& Unew,
    GpuWorkspace& ws,
    double dt
) {
    if (ws.nx != Uold.nx() || ws.ny != Uold.ny() || ws.speed_d == nullptr) {
        throw std::runtime_error("advance_second_order_gpu: workspace not initialized for this grid.");
    }

    /*
      Recommended initial block size.

      You should later benchmark:
          dim3 threads(16, 16);
          dim3 threads(32, 8);
          dim3 threads(32, 4);

      For now, 16x16 is safer because this kernel uses more shared memory.
    */
    const dim3 threads(16, 16);

    const dim3 blocks(
        (Uold.nx() + threads.x - 1) / threads.x,
        (Uold.ny() + threads.y - 1) / threads.y
    );

    // -------------------------------------------------------------------------
    // x sweep shared memory size
    // -------------------------------------------------------------------------
    const int x_state_tile_w = threads.x + 4;
    const int x_state_tile_h = threads.y;
    const int x_state_tile_n = x_state_tile_w * x_state_tile_h;

    const int x_recon_tile_w = threads.x + 2;
    const int x_recon_tile_h = threads.y;
    const int x_recon_tile_n = x_recon_tile_w * x_recon_tile_h;

    const int x_flux_tile_w = threads.x + 1;
    const int x_flux_tile_h = threads.y;
    const int x_flux_tile_n = x_flux_tile_w * x_flux_tile_h;

    const std::size_t x_smem_bytes =
        (
            4 * static_cast<std::size_t>(x_state_tile_n) +
            8 * static_cast<std::size_t>(x_recon_tile_n) +
            4 * static_cast<std::size_t>(x_flux_tile_n)
        ) * sizeof(double);

    advance_x_reconstruct_smem_fused_kernel<<<blocks, threads, x_smem_bytes>>>(
        make_view(static_cast<const Grid2DGPU&>(Uold)),
        make_view(Utmp),
        dt
    );

    CUDA_CHECK(cudaGetLastError());

    apply_transmissive_boundary_gpu(Utmp);

    // -------------------------------------------------------------------------
    // y sweep shared memory size
    // -------------------------------------------------------------------------
    const int y_state_tile_w = threads.x;
    const int y_state_tile_h = threads.y + 4;
    const int y_state_tile_n = y_state_tile_w * y_state_tile_h;

    const int y_recon_tile_w = threads.x;
    const int y_recon_tile_h = threads.y + 2;
    const int y_recon_tile_n = y_recon_tile_w * y_recon_tile_h;

    const int y_flux_tile_w = threads.x;
    const int y_flux_tile_h = threads.y + 1;
    const int y_flux_tile_n = y_flux_tile_w * y_flux_tile_h;

    const std::size_t y_smem_bytes =
        (
            4 * static_cast<std::size_t>(y_state_tile_n) +
            8 * static_cast<std::size_t>(y_recon_tile_n) +
            4 * static_cast<std::size_t>(y_flux_tile_n)
        ) * sizeof(double);

    advance_y_reconstruct_smem_fused_kernel<<<blocks, threads, y_smem_bytes>>>(
        make_view(static_cast<const Grid2DGPU&>(Utmp)),
        make_view(Unew),
        dt
    );

    CUDA_CHECK(cudaGetLastError());

    apply_transmissive_boundary_gpu(Unew);
}