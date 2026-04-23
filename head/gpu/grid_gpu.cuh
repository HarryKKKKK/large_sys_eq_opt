#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "types.hpp"

class Grid2DGPU {
public:
    Grid2DGPU() = default;

    Grid2DGPU(int nx_, int ny_, int ng_,
              double x_min_, double x_max_,
              double y_min_, double y_max_) {
        allocate(nx_, ny_, ng_, x_min_, x_max_, y_min_, y_max_);
    }

    Grid2DGPU(const Grid2DGPU&) = delete;
    Grid2DGPU& operator=(const Grid2DGPU&) = delete;

    Grid2DGPU(Grid2DGPU&& other) noexcept { move_from(std::move(other)); }

    Grid2DGPU& operator=(Grid2DGPU&& other) noexcept {
        if (this != &other) { release(); move_from(std::move(other)); }
        return *this;
    }

    ~Grid2DGPU() { release(); }

    void allocate(int nx_, int ny_, int ng_,
                  double x_min_, double x_max_,
                  double y_min_, double y_max_) {
        release();
        nx__ = nx_;  ny__ = ny_;  ng__ = ng_;
        x_min__ = x_min_;  x_max__ = x_max_;
        y_min__ = y_min_;  y_max__ = y_max_;
        dx__ = (x_max__ - x_min__) / static_cast<double>(nx__);
        dy__ = (y_max__ - y_min__) / static_cast<double>(ny__);
        const std::size_t n = num_cells();
        cudaMalloc(&rho_,  n * sizeof(double));
        cudaMalloc(&rhou_, n * sizeof(double));
        cudaMalloc(&rhov_, n * sizeof(double));
        cudaMalloc(&E_,    n * sizeof(double));
    }

    void release() {
        if (rho_)  cudaFree(rho_);
        if (rhou_) cudaFree(rhou_);
        if (rhov_) cudaFree(rhov_);
        if (E_)    cudaFree(E_);
        rho_ = nullptr;  rhou_ = nullptr;  rhov_ = nullptr;  E_ = nullptr;
        nx__ = 0;  ny__ = 0;  ng__ = 0;
        x_min__ = 0.0;  x_max__ = 1.0;
        y_min__ = 0.0;  y_max__ = 1.0;
        dx__ = 0.0;  dy__ = 0.0;
    }

    void fill_zero() {
        const std::size_t bytes = num_cells() * sizeof(double);
        cudaMemset(rho_,  0, bytes);
        cudaMemset(rhou_, 0, bytes);
        cudaMemset(rhov_, 0, bytes);
        cudaMemset(E_,    0, bytes);
    }

    int nx() const { return nx__; }
    int ny() const { return ny__; }
    int ng() const { return ng__; }
    int total_nx() const { return nx__ + 2 * ng__; }
    int total_ny() const { return ny__ + 2 * ng__; }
    double x_min() const { return x_min__; }
    double x_max() const { return x_max__; }
    double y_min() const { return y_min__; }
    double y_max() const { return y_max__; }
    double dx() const { return dx__; }
    double dy() const { return dy__; }
    int i_begin() const { return ng__; }
    int i_end()   const { return ng__ + nx__; }
    int j_begin() const { return ng__; }
    int j_end()   const { return ng__ + ny__; }

    double x_center(int i) const {
        return x_min__ + (static_cast<double>(i - ng__) + 0.5) * dx__;
    }
    double y_center(int j) const {
        return y_min__ + (static_cast<double>(j - ng__) + 0.5) * dy__;
    }

    std::size_t flat_index(int i, int j) const {
        return static_cast<std::size_t>(j * total_nx() + i);
    }
    std::size_t num_cells() const {
        return static_cast<std::size_t>(total_nx()) * total_ny();
    }

    double*       rho_ptr()        { return rho_; }
    double*       rhou_ptr()       { return rhou_; }
    double*       rhov_ptr()       { return rhov_; }
    double*       E_ptr()          { return E_; }
    const double* rho_ptr()  const { return rho_; }
    const double* rhou_ptr() const { return rhou_; }
    const double* rhov_ptr() const { return rhov_; }
    const double* E_ptr()    const { return E_; }

    void upload_from_aos(const std::vector<Conserved>& host_data) {
        std::vector<double> rho_h(num_cells()), rhou_h(num_cells()),
                            rhov_h(num_cells()), E_h(num_cells());
        for (std::size_t k = 0; k < num_cells(); ++k) {
            rho_h[k]  = host_data[k].rho;
            rhou_h[k] = host_data[k].rhou;
            rhov_h[k] = host_data[k].rhov;
            E_h[k]    = host_data[k].E;
        }
        const std::size_t bytes = num_cells() * sizeof(double);
        cudaMemcpy(rho_,  rho_h.data(),  bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(rhou_, rhou_h.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(rhov_, rhov_h.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(E_,    E_h.data(),    bytes, cudaMemcpyHostToDevice);
    }

    void download_to_aos(std::vector<Conserved>& host_data) const {
        host_data.resize(num_cells());
        std::vector<double> rho_h(num_cells()), rhou_h(num_cells()),
                            rhov_h(num_cells()), E_h(num_cells());
        const std::size_t bytes = num_cells() * sizeof(double);
        cudaMemcpy(rho_h.data(),  rho_,  bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(rhou_h.data(), rhou_, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(rhov_h.data(), rhov_, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(E_h.data(),    E_,    bytes, cudaMemcpyDeviceToHost);
        for (std::size_t k = 0; k < num_cells(); ++k) {
            host_data[k].rho  = rho_h[k];
            host_data[k].rhou = rhou_h[k];
            host_data[k].rhov = rhov_h[k];
            host_data[k].E    = E_h[k];
        }
    }

    void swap(Grid2DGPU& other) {
        std::swap(rho_,  other.rho_);
        std::swap(rhou_, other.rhou_);
        std::swap(rhov_, other.rhov_);
        std::swap(E_,    other.E_);
    }

private:
    int nx__ = 0, ny__ = 0, ng__ = 0;
    double x_min__ = 0.0, x_max__ = 1.0;
    double y_min__ = 0.0, y_max__ = 1.0;
    double dx__ = 0.0, dy__ = 0.0;
    double* rho_  = nullptr;
    double* rhou_ = nullptr;
    double* rhov_ = nullptr;
    double* E_    = nullptr;

    void move_from(Grid2DGPU&& other) {
        nx__ = other.nx__;  ny__ = other.ny__;  ng__ = other.ng__;
        x_min__ = other.x_min__;  x_max__ = other.x_max__;
        y_min__ = other.y_min__;  y_max__ = other.y_max__;
        dx__ = other.dx__;  dy__ = other.dy__;
        rho_  = other.rho_;   rhou_ = other.rhou_;
        rhov_ = other.rhov_;  E_    = other.E_;
        other.nx__ = 0;  other.ny__ = 0;  other.ng__ = 0;
        other.x_min__ = 0.0;  other.x_max__ = 1.0;
        other.y_min__ = 0.0;  other.y_max__ = 1.0;
        other.dx__ = 0.0;  other.dy__ = 0.0;
        other.rho_ = nullptr;  other.rhou_ = nullptr;
        other.rhov_ = nullptr; other.E_    = nullptr;
    }
};

// ============================================================
// Grid2DGPUView — mutable non-owning view (write kernels)
// ============================================================
struct Grid2DGPUView {
    int nx, ny, ng;
    double x_min, x_max, y_min, y_max, dx, dy;
    double* rho;
    double* rhou;
    double* rhov;
    double* E;

    __host__ __device__ int total_nx() const { return nx + 2 * ng; }
    __host__ __device__ int total_ny() const { return ny + 2 * ng; }
    __host__ __device__ int i_begin()  const { return ng; }
    __host__ __device__ int i_end()    const { return ng + nx; }
    __host__ __device__ int j_begin()  const { return ng; }
    __host__ __device__ int j_end()    const { return ng + ny; }
    __host__ __device__ int flat_index(int i, int j) const {
        return j * total_nx() + i;
    }
};

// ============================================================
// ConstGrid2DGPUView — read-only non-owning view (read kernels)
// ============================================================
struct ConstGrid2DGPUView {
    int nx, ny, ng;
    double x_min, x_max, y_min, y_max, dx, dy;
    const double* rho;
    const double* rhou;
    const double* rhov;
    const double* E;

    __host__ __device__ int total_nx() const { return nx + 2 * ng; }
    __host__ __device__ int total_ny() const { return ny + 2 * ng; }
    __host__ __device__ int i_begin()  const { return ng; }
    __host__ __device__ int i_end()    const { return ng + nx; }
    __host__ __device__ int j_begin()  const { return ng; }
    __host__ __device__ int j_end()    const { return ng + ny; }
    __host__ __device__ int flat_index(int i, int j) const {
        return j * total_nx() + i;
    }
};

// ============================================================
// make_view helpers
// ============================================================

inline Grid2DGPUView make_view(Grid2DGPU& grid) {
    return Grid2DGPUView{
        grid.nx(), grid.ny(), grid.ng(),
        grid.x_min(), grid.x_max(),
        grid.y_min(), grid.y_max(),
        grid.dx(), grid.dy(),
        grid.rho_ptr(), grid.rhou_ptr(), grid.rhov_ptr(), grid.E_ptr()
    };
}

inline ConstGrid2DGPUView make_view(const Grid2DGPU& grid) {
    return ConstGrid2DGPUView{
        grid.nx(), grid.ny(), grid.ng(),
        grid.x_min(), grid.x_max(),
        grid.y_min(), grid.y_max(),
        grid.dx(), grid.dy(),
        grid.rho_ptr(), grid.rhou_ptr(), grid.rhov_ptr(), grid.E_ptr()
    };
}