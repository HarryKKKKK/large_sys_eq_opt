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

// -----------------------------------------------------------------------------
// Test case 1: Shock-bubble interaction
// -----------------------------------------------------------------------------

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

    const double p2 = p0 * (
        1.0 + 2.0 * phys::gamma / (phys::gamma + 1.0) * (M2 - 1.0)
    );

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

// -----------------------------------------------------------------------------
// Test case 2: 2D blast wave / circular explosion
//
// A circular high-pressure region is placed at the centre of the domain.
// The strong pressure jump generates an outward-propagating shock wave.
// -----------------------------------------------------------------------------

Conserved blast_wave_state(double x, double y) {
    constexpr double x0 = 0.5;
    constexpr double y0 = 0.5;
    constexpr double r0 = 0.1;

    constexpr double rho_inside = 1.0;
    constexpr double rho_outside = 1.0;

    constexpr double p_inside = 100.0;
    constexpr double p_outside = 1.0;

    constexpr double u0 = 0.0;
    constexpr double v0 = 0.0;

    const double dx = x - x0;
    const double dy = y - y0;
    const double r2 = dx * dx + dy * dy;

    if (r2 <= r0 * r0) {
        return make_conserved(rho_inside, u0, v0, p_inside);
    }

    return make_conserved(rho_outside, u0, v0, p_outside);
}

} // namespace

CaseId parse_case_id(const std::string& case_name) {
    if (case_name == "shock_bubble") {
        return CaseId::ShockBubble;
    }

    if (case_name == "blast_wave") {
        return CaseId::BlastWave;
    }

    throw std::runtime_error(
        "Unknown case name: " + case_name +
        ". Supported cases are: shock_bubble, blast_wave."
    );
}

std::string case_id_to_string(CaseId case_id) {
    switch (case_id) {
        case CaseId::ShockBubble:
            return "shock_bubble";

        case CaseId::BlastWave:
            return "blast_wave";
    }

    throw std::runtime_error("Unhandled CaseId in case_id_to_string.");
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

        case CaseId::BlastWave:
            return CaseConfig{
                500,        // nx
                500,        // ny
                2,          // ng
                0.0,        // x_min
                1.0,        // x_max
                0.0,        // y_min
                1.0,        // y_max
                0.4,        // cfl
                0.2         // t_end
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

        case CaseId::BlastWave:
            return blast_wave_state(x, y);
    }

    throw std::runtime_error("Unhandled CaseId in initial_state_at.");
}

Conserved initial_state_at(const std::string& case_name, double x, double y) {
    return initial_state_at(parse_case_id(case_name), x, y);
}

CaseConfig get_n_case_config(CaseId case_id, int n) {
    if (n < 1) {
        throw std::runtime_error("get_n_case_config: n must be >= 1.");
    }

    CaseConfig cfg = get_case_config(case_id);

    cfg.nx *= n;
    cfg.ny *= n;

    return cfg;
}

CaseConfig get_n_case_config(const std::string& case_name, int n) {
    return get_n_case_config(parse_case_id(case_name), n);
}