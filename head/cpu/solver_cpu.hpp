#pragma once

#include "grid_cpu.hpp"

double compute_dt(const Grid2D& grid, double cfl);

void advance_first_order(
    const Grid2D& Uold,
    Grid2D& Unew,
    double dt
);

void advance_second_order(
    const Grid2D& Uold,
    Grid2D& Utmp,
    Grid2D& Unew,
    double dt
);
