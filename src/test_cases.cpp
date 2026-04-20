#include "test_cases.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "physics.hpp"

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
    constexpr double rho_air    = 1.29;
    constexpr double rho_helium = 0.214;
    constexpr double p0         = 1.01325e5;
    constexpr double mach       = 1.22;

    constexpr double bubble_r  = 0.025;
    constexpr double bubble_cx = 0.035;
    constexpr double bubble_cy = 0.0445;
    constexpr double shock_x   = 0.005;

    const double a0 = std::sqrt(phys::gamma * p0 / rho_air);
    const double M2 = mach * mach;

    const double rho2 = rho_air * ((phys::gamma + 1.0) * M2)
                      / ((phys::gamma - 1.0) * M2 + 2.0);

    const double p2 = p0 * (1.0 + 2.0 * phys::gamma / (phys::gamma + 1.0) * (M2 - 1.0));

    const double u2 = 2.0 * a0 / (phys::gamma + 1.0)
                    * (mach - 1.0 / mach);

    double rho = rho_air;
    double u   = 0.0;
    double v   = 0.0;
    double p   = p0;
    if (x < shock_x) {
        rho = rho2;
        u   = u2;
        v   = 0.0;
        p   = p2;
    }

    const double dx = x - bubble_cx;
    const double dy = y - bubble_cy;
    const bool in_bubble = (dx * dx + dy * dy <= bubble_r * bubble_r);

    if (in_bubble) {
        rho = rho_helium;
    }

    return make_conserved(rho, u, v, p);
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
                0.225,      // x_max
                0.0,        // y_min
                0.089,      // y_max
                0.4,        // cfl
                0.0011741   // t_end
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