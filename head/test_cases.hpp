#pragma once

#include <string>

#include "physics.hpp"
#include "types.hpp"

struct CaseConfig {
    int nx;
    int ny;
    int ng;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double cfl;
    double t_end;
};

enum class CaseId {
    ShockBubble
};

CaseId parse_case_id(const std::string& case_name);

CaseConfig get_case_config(CaseId case_id);
CaseConfig get_case_config(const std::string& case_name);

Conserved initial_state_at(CaseId case_id, double x, double y);
Conserved initial_state_at(const std::string& case_name, double x, double y);