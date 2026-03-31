#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "boundary.hpp"
#include "grid.hpp"
#include "physics.hpp"
#include "solver.hpp"
#include "test_cases.hpp"
#include "types.hpp"

namespace {

void write_csv(const Grid2D& grid, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }

    out << "x,y,rho,u,v,p\n";

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const Primitive V = phys::cons_to_prim(grid(i, j));

            out << std::setprecision(16)
                << grid.x_center(i) << ","
                << grid.y_center(j) << ","
                << V.rho << ","
                << V.u << ","
                << V.v << ","
                << V.p << "\n";
        }
    }
}

double total_mass(const Grid2D& grid) {
    double sum = 0.0;
    const double dA = grid.dx() * grid.dy();

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            sum += grid(i, j).rho;
        }
    }

    return sum * dA;
}

double total_x_momentum(const Grid2D& grid) {
    double sum = 0.0;
    const double dA = grid.dx() * grid.dy();

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            sum += grid(i, j).rhou;
        }
    }

    return sum * dA;
}

double total_y_momentum(const Grid2D& grid) {
    double sum = 0.0;
    const double dA = grid.dx() * grid.dy();

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            sum += grid(i, j).rhov;
        }
    }

    return sum * dA;
}

double total_energy(const Grid2D& grid) {
    double sum = 0.0;
    const double dA = grid.dx() * grid.dy();

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            sum += grid(i, j).E;
        }
    }

    return sum * dA;
}

double min_density(const Grid2D& grid) {
    double value = grid(grid.i_begin(), grid.j_begin()).rho;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            value = std::min(value, grid(i, j).rho);
        }
    }

    return value;
}

double min_pressure(const Grid2D& grid) {
    double value = phys::pressure(grid(grid.i_begin(), grid.j_begin()));

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            value = std::min(value, phys::pressure(grid(i, j)));
        }
    }

    return value;
}

void print_diagnostics(const Grid2D& grid, const std::string& label) {
    std::cout << label
              << " mass=" << total_mass(grid)
              << ", mx=" << total_x_momentum(grid)
              << ", my=" << total_y_momentum(grid)
              << ", E=" << total_energy(grid)
              << ", min(rho)=" << min_density(grid)
              << ", min(p)=" << min_pressure(grid)
              << "\n";
}

} // namespace

int main() {
    try {
        const std::string case_name = "shock_bubble";

        const CaseConfig cfg = get_case_config(case_name);

        Grid2D Uold(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        Grid2D Unew(
            cfg.nx, cfg.ny, cfg.ng,
            cfg.x_min, cfg.x_max,
            cfg.y_min, cfg.y_max
        );

        initialise_case(Uold, case_name);

        std::cout << "Running case: " << case_name << "\n";
        std::cout << "Grid: " << cfg.nx << " x " << cfg.ny
                  << ", domain = ["
                  << cfg.x_min << ", " << cfg.x_max << "] x ["
                  << cfg.y_min << ", " << cfg.y_max << "]\n";
        std::cout << "CFL = " << cfg.cfl
                  << ", t_end = " << cfg.t_end << "\n";

        print_diagnostics(Uold, "Initial:");

        double t = 0.0;
        int step = 0;

        while (t < cfg.t_end) {
            apply_transmissive_boundary(Uold);

            double dt = compute_dt(Uold, cfg.cfl);
            if (t + dt > cfg.t_end) {
                dt = cfg.t_end - t;
            }

            advance_first_order(Uold, Unew, dt);
            std::swap(Uold, Unew);

            t += dt;
            ++step;

            if (step % 20 == 0 || t >= cfg.t_end) {
                std::cout << "step=" << step
                          << ", t=" << t
                          << ", dt=" << dt
                          << ", min(rho)=" << min_density(Uold)
                          << ", min(p)=" << min_pressure(Uold)
                          << "\n";
            }
        }

        print_diagnostics(Uold, "Final:");
        write_csv(Uold, cfg.output_name);

        std::cout << "Finished. Output written to " << cfg.output_name << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}