#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "types.hpp"

namespace phys {

constexpr double gamma = 1.4;

// ---------------------------------
// Physics Quantities
// ---------------------------------

inline Primitive cons_to_prim(const Conserved& U) {
    const double inv_rho = 1.0 / U.rho;
    const double u = U.rhou * inv_rho;
    const double v = U.rhov * inv_rho;
    const double kinetic = 0.5 * U.rho * (u * u + v * v);
    const double p = (gamma - 1.0) * (U.E - kinetic);
    return Primitive(U.rho, u, v, p);
}

inline Conserved prim_to_cons(const Primitive& V) {
    const double rhou = V.rho * V.u;
    const double rhov = V.rho * V.v;
    const double kinetic = 0.5 * V.rho * (V.u * V.u + V.v * V.v);
    const double E = V.p / (gamma - 1.0) + kinetic;
    return Conserved(V.rho, rhou, rhov, E);
}


inline double pressure(const Conserved& U) {
    const double inv_rho = 1.0 / U.rho;
    const double u = U.rhou * inv_rho;
    const double v = U.rhov * inv_rho;
    const double kinetic = 0.5 * U.rho * (u * u + v * v);
    const double p = (gamma - 1.0) * (U.E - kinetic);
    return p;
}

inline double sound_speed(const Primitive& V) {
    return std::sqrt(gamma * V.p / V.rho);
}

inline double sound_speed(const Conserved& U) {
    return sound_speed(cons_to_prim(U));
}

inline double specific_internal_energy(const Primitive& V) {
    return V.p / ((gamma - 1.0) * V.rho);
}

inline double total_enthalpy(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    return (U.E + V.p) / U.rho;
}

// ---------------------------
// Fluxes
// ---------------------------

inline Flux flux_x(const Conserved& U) {
    const Primitive V = cons_to_prim(U);

    return Flux(
        U.rhou,
        U.rhou * V.u + V.p,
        U.rhov * V.u,
        (U.E + V.p) * V.u
    );
}

inline Flux flux_y(const Conserved& U) {
    const Primitive V = cons_to_prim(U);

    return Flux(
        U.rhov,
        U.rhou * V.v,
        U.rhov * V.v + V.p,
        (U.E + V.p) * V.v
    );
}

inline Conserved flux_to_conserved(const Flux& F) {
    return Conserved(F.mass, F.momx, F.momy, F.energy);
}

// ---------------------------
// Signal-speeds
// ---------------------------

inline double max_signal_speed_x(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    const double a = sound_speed(V);
    return std::abs(V.u) + a;
}

inline double max_signal_speed_y(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    const double a = sound_speed(V);
    return std::abs(V.v) + a;
}

inline double max_signal_speed(const Conserved& U) {
    return std::max(max_signal_speed_x(U), max_signal_speed_y(U));
}

}