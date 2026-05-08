#pragma once

#include <string>

#include "cpu/grid_cpu.hpp"
#include "test_cases.hpp"

Grid2D make_initial_grid(CaseId case_id);
Grid2D make_initial_grid(const std::string& case_name);

Grid2D make_n_grid(CaseId case_id, int n);
Grid2D make_n_grid(const std::string& case_name, int n);

void initialise_grid(Grid2D& grid, CaseId case_id);
void initialise_grid(Grid2D& grid, const std::string& case_name);