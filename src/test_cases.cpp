#include "test_cases.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "boundary.hpp"
#include "physics.hpp"
#include "types.hpp"

namespace {

struct ShockBubbleSpec {
    double gamma = 1.4;

    double rho_air = 1.29;
    double rho_helium = 0.214;
    double p0 = 1.01325e5;

    double mach_shock = 1.22;

    double bubble_radius = 0.025;
    double bubble_cx = 0.035;
    double bubble_cy = 0.0445;

    double shock_x = 0.005;

    double x_min = 0.0;
    double x_max = 0.225;
    double y_min = 0.0;
    double y_max = 0.089;

    double t_end = 0.0011741;
};

ShockBubbleSpec make_shock_bubble_spec() {
    return ShockBubbleSpec{};
}

Primitive make_post_shock_air_state(const ShockBubbleSpec& spec) {
    const double g = spec.gamma;
    const double M1 = spec.mach_shock;

    const double rho1 = spec.rho_air;
    const double p1 = spec.p0;

    const double a1 = std::sqrt(g * p1 / rho1);
    const double Vs = M1 * a1;

    const double rho2_over_rho1 =
        ((g + 1.0) * M1 * M1) / ((g - 1.0) * M1 * M1 + 2.0);

    const double p2_over_p1 =
        1.0 + (2.0 * g / (g + 1.0)) * (M1 * M1 - 1.0);

    const double rho2 = rho2_over_rho1 * rho1;
    const double p2 = p2_over_p1 * p1;

    const double u1_shock = Vs;
    const double u2_shock = u1_shock * (rho1 / rho2);

    const double u2_lab = Vs - u2_shock;

    return Primitive(rho2, u2_lab, 0.0, p2);
}

Primitive make_pre_shock_air_state(const ShockBubbleSpec& spec) {
    return Primitive(spec.rho_air, 0.0, 0.0, spec.p0);
}

Primitive make_helium_bubble_state(const ShockBubbleSpec& spec) {
    return Primitive(spec.rho_helium, 0.0, 0.0, spec.p0);
}

}

CaseConfig make_shock_bubble_config() {
    const ShockBubbleSpec spec = make_shock_bubble_spec();

    CaseConfig cfg;
    cfg.nx = 500;
    cfg.ny = 197;
    cfg.ng = 1;

    cfg.x_min = spec.x_min;
    cfg.x_max = spec.x_max;
    cfg.y_min = spec.y_min;
    cfg.y_max = spec.y_max;

    cfg.cfl = 0.4;
    cfg.t_end = spec.t_end;

    cfg.output_name = "shock_bubble_cpu_final.csv";
    return cfg;
}

CaseConfig get_case_config(const std::string& case_name) {
    if (case_name == "shock_bubble") {
        return make_shock_bubble_config();
    }

    throw std::runtime_error("Unknown case name: " + case_name);
}

void initialise_shock_bubble(Grid2D& grid) {
    const ShockBubbleSpec spec = make_shock_bubble_spec();

    const Primitive pre_shock_air  = make_pre_shock_air_state(spec);
    const Primitive post_shock_air = make_post_shock_air_state(spec);
    const Primitive bubble_helium  = make_helium_bubble_state(spec);

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const double x = grid.x_center(i);
            const double y = grid.y_center(j);

            Primitive V = (x < spec.shock_x) ? post_shock_air : pre_shock_air;

            const double dx = x - spec.bubble_cx;
            const double dy = y - spec.bubble_cy;
            const double r2 = dx * dx + dy * dy;

            if (r2 <= spec.bubble_radius * spec.bubble_radius) {
                V = bubble_helium;
            }

            grid(i, j) = phys::prim_to_cons(V);
        }
    }

    apply_transmissive_boundary(grid);
}

void initialise_case(Grid2D& grid, const std::string& case_name) {
    if (case_name == "shock_bubble") {
        initialise_shock_bubble(grid);
        return;
    }

    throw std::runtime_error("Unknown case name: " + case_name);
}