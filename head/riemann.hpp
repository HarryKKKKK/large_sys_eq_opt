#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "physics.hpp"
#include "types.hpp"

enum class Direction {
    X, Y
};

inline Conserved physical_flux(const Conserved& U, Direction dir) {
    if (dir == Direction::X) {
        return flux_to_conserved(phys::flux_x(U));
    } else {
        return flux_to_conserved(phys::flux_y(U));
    }
}


inline double normal_velocity(const Primitive& V, Direction dir) {
    return (dir == Direction::X) ? V.u : V.v;
}

// -----------------------------------
// HLL solver
// -----------------------------------
inline Conserved hll_flux(const Conserved& UL, const Conserved& UR, Direction dir) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    // FIXME: wavespeed estimation
    const double SL = std::min(unL - aL, unR - aR);
    const double SR = std::max(unL + aL, unR + aR);

    const Conserved FL = physical_flux(UL, dir);
    const Conserved FR = physical_flux(UR, dir);

    if (SL >= 0.0) {
        return FL;
    }
    if (SR <= 0.0) {
        return FR;
    }
    // Star region
    return (SR * FL - SL * FR + (SL * SR) * (UR - UL)) / (SR - SL);
}

// -----------------------------------
// FIXME: HLLC solver
// -----------------------------------

// -----------------------------------
// FIXME: Exact Riemann solver
// -----------------------------------