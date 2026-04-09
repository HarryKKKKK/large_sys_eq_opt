#include "test_cases.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

inline Conserved make_conserved(double rho, double u, double v, double p) {
    Conserved U{};
    U.rho  = rho;
    U.rhou = rho * u;
    U.rhov = rho * v;
    U.E    = p / (phys::gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    return U;
}

Conserved shock_bubble_state(double x, double y) {
    constexpr double x_shock = 0.2;
    constexpr double bubble_cx = 0.5;
    constexpr double bubble_cy = 0.0;
    constexpr double bubble_r  = 0.2;

    const double dx = x - bubble_cx;
    const double dy = y - bubble_cy;
    const bool in_bubble = (dx * dx + dy * dy <= bubble_r * bubble_r);

    // shocked air
    if (x < x_shock) {
        return make_conserved(1.3764, 0.394, 0.0, 1.5698);
    }

    // helium bubble
    if (in_bubble) {
        return make_conserved(0.1819, 0.0, 0.0, 1.0);
    }

    // ambient air
    return make_conserved(1.0, 0.0, 0.0, 1.0);
}

}

CaseId parse_case_id(const std::string& case_name) {
    if (case_name == "shock_bubble") {
        return CaseId::ShockBubble;
    }

    throw std::runtime_error("Unknown case name: " + case_name);
}

CaseConfig get_case_config(CaseId case_id) {
    switch (case_id) {
        case CaseId::ShockBubble:
            return CaseConfig{
                500,        // nx
                197,        // ny
                2,          // ng
                0.0,        // x_min
                1.0,        // x_max
                -0.178,     // y_min
                0.178,      // y_max
                0.4,        // cfl
                0.3         // t_end
            };
    }

    throw std::runtime_error("Unhandled CaseId in get_case_config.");
}

CaseConfig get_case_config(const std::string& case_name) {
    return get_case_config(parse_case_id(case_name));
}

Conserved initial_state_at(CaseId case_id, double x, double y) {
    switch (case_id) {
        case CaseId::ShockBubble:
            return shock_bubble_state(x, y);
    }

    throw std::runtime_error("Unhandled CaseId in initial_state_at.");
}

Conserved initial_state_at(const std::string& case_name, double x, double y) {
    return initial_state_at(parse_case_id(case_name), x, y);
}