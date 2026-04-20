#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "physics.hpp"
#include "types.hpp"

enum class Direction {
    X, Y
};

HD inline Conserved physical_flux(const Conserved& U, Direction dir) {
    if (dir == Direction::X) {
        return phys::flux_to_conserved(phys::flux_x(U));
    } else {
        return phys::flux_to_conserved(phys::flux_y(U));
    }
}

HD inline double normal_velocity(const Primitive& V, Direction dir) {
    return (dir == Direction::X) ? V.u : V.v;
}

HD inline Conserved hll_flux(const Conserved& UL, const Conserved& UR, Direction dir) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double SL = fmin(unL - aL, unR - aR);
    const double SR = fmax(unL + aL, unR + aR);

    const Conserved FL = physical_flux(UL, dir);
    const Conserved FR = physical_flux(UR, dir);

    if (SL >= 0.0) {
        return FL;
    }
    if (SR <= 0.0) {
        return FR;
    }

    return (SR * FL - SL * FR + (SL * SR) * (UR - UL)) / (SR - SL);
}