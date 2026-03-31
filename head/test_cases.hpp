#pragma once

#include <string>

#include "grid.hpp"

struct CaseConfig {
    int nx = 0;
    int ny = 0;
    int ng = 1;

    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;

    double cfl = 0.4;
    double t_end = 0.0;

    std::string output_name;
};

CaseConfig make_shock_bubble_config();

CaseConfig get_case_config(const std::string& case_name);

void initialise_shock_bubble(Grid2D& grid);

void initialise_case(Grid2D& grid, const std::string& case_name);