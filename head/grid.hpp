#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include "types.hpp"

class Grid2D {
public:
    Grid2D() = default;

    Grid2D(int nx_, int ny_, int ng_, double x_min_, double x_max_, double y_min_, double y_max_)
        : nx_(nx_), ny_(ny_), ng_(ng_), x_min_(x_min_), x_max_(x_max_), y_min_(y_min_), y_max_(y_max_) {

        dx_ = (x_max_ - x_min_) / static_cast<double>(nx_);
        dy_ = (y_max_ - y_min_) / static_cast<double>(ny_);

        const int total_x = nx_ + 2 * ng_;
        const int total_y = ny_ + 2 * ng_;
        U_.assign(static_cast<std::size_t>(total_x * total_y), Conserved{});
    }

    // ---------------------------
    // Getters
    // ---------------------------
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int ng() const { return ng_; }

    int total_nx() const { return nx_ + 2 * ng_; }
    int total_ny() const { return ny_ + 2 * ng_; }

    double x_min() const { return x_min_; }
    double x_max() const { return x_max_; }
    double y_min() const { return y_min_; }
    double y_max() const { return y_max_; }

    double dx() const { return dx_; }
    double dy() const { return dy_; }

    int i_begin() const { return ng_; }
    int i_end()   const { return ng_ + nx_; }  
    int j_begin() const { return ng_; }
    int j_end()   const { return ng_ + ny_; } 

    // ---------------------------
    // Data access
    // ---------------------------
    Conserved& operator()(int i, int j) {
        return U_[flat_index(i, j)];
    }

    const Conserved& operator()(int i, int j) const {
        return U_[flat_index(i, j)];
    }

    std::vector<Conserved>& data() { return U_; }
    const std::vector<Conserved>& data() const { return U_; }

    // ---------------------------
    // Helpers
    // ---------------------------
    void fill(const Conserved& value) {
        std::fill(U_.begin(), U_.end(), value);
    }

private:
    int nx_ = 0;
    int ny_ = 0;
    int ng_ = 0;

    double x_min_ = 0.0;
    double x_max_ = 1.0;
    double y_min_ = 0.0;
    double y_max_ = 1.0;

    double dx_ = 0.0;
    double dy_ = 0.0;

    std::vector<Conserved> U_;

    std::size_t flat_index(int i, int j) const {
        return static_cast<std::size_t>((j * total_nx()) + i);
    }
};