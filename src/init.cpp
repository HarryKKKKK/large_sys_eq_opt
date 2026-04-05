#include "init.hpp"

namespace {

inline double cell_center_x(const Grid2D& grid, int i) {
    return grid.x_min() + (static_cast<double>(i - grid.ng()) + 0.5) * grid.dx();
}

inline double cell_center_y(const Grid2D& grid, int j) {
    return grid.y_min() + (static_cast<double>(j - grid.ng()) + 0.5) * grid.dy();
}

}

void initialise_grid(Grid2D& grid, CaseId case_id) {
    for (int j = 0; j < grid.total_ny(); ++j) {
        for (int i = 0; i < grid.total_nx(); ++i) {
            const double x = cell_center_x(grid, i);
            const double y = cell_center_y(grid, j);
            grid(i, j) = initial_state_at(case_id, x, y);
        }
    }
}

void initialise_grid(Grid2D& grid, const std::string& case_name) {
    initialise_grid(grid, parse_case_id(case_name));
}

Grid2D make_initial_grid(CaseId case_id) {
    const CaseConfig cfg = get_case_config(case_id);

    Grid2D grid(
        cfg.nx, cfg.ny, cfg.ng,
        cfg.x_min, cfg.x_max,
        cfg.y_min, cfg.y_max
    );

    initialise_grid(grid, case_id);
    return grid;
}

Grid2D make_initial_grid(const std::string& case_name) {
    return make_initial_grid(parse_case_id(case_name));
}