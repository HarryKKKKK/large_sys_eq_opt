#pragma once

#include <string>

#include "cpu/grid_cpu.hpp"
#include "test_cases.hpp"

Grid2D make_initial_grid(CaseId case_id);
Grid2D make_initial_grid(const std::string& case_name);

void initialise_grid(Grid2D& grid, CaseId case_id);
void initialise_grid(Grid2D& grid, const std::string& case_name);