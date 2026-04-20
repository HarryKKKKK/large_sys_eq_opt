#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "types.hpp"

namespace phys {

constexpr double gamma = 1.4;

HD inline Primitive cons_to_prim(const Conserved& U) {
    const double inv_rho = 1.0 / U.rho;
    const double u = U.rhou * inv_rho;
    const double v = U.rhov * inv_rho;
    const double kinetic = 0.5 * U.rho * (u * u + v * v);
    const double p = (gamma - 1.0) * (U.E - kinetic);
    return Primitive(U.rho, u, v, p);
}

HD inline Conserved prim_to_cons(const Primitive& V) {
    const double rhou = V.rho * V.u;
    const double rhov = V.rho * V.v;
    const double kinetic = 0.5 * V.rho * (V.u * V.u + V.v * V.v);
    const double E = V.p / (gamma - 1.0) + kinetic;
    return Conserved(V.rho, rhou, rhov, E);
}

HD inline double pressure(const Conserved& U) {
    const double inv_rho = 1.0 / U.rho;
    const double u = U.rhou * inv_rho;
    const double v = U.rhov * inv_rho;
    const double kinetic = 0.5 * U.rho * (u * u + v * v);
    const double p = (gamma - 1.0) * (U.E - kinetic);
    return p;
}

HD inline double sound_speed(const Primitive& V) {
    return sqrt(gamma * V.p / V.rho);
}

HD inline double sound_speed(const Conserved& U) {
    return sound_speed(cons_to_prim(U));
}

HD inline double specific_internal_energy(const Primitive& V) {
    return V.p / ((gamma - 1.0) * V.rho);
}

HD inline double total_enthalpy(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    return (U.E + V.p) / U.rho;
}

HD inline Flux flux_x(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    return Flux(
        U.rhou,
        U.rhou * V.u + V.p,
        U.rhov * V.u,
        (U.E + V.p) * V.u
    );
}

HD inline Flux flux_y(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    return Flux(
        U.rhov,
        U.rhou * V.v,
        U.rhov * V.v + V.p,
        (U.E + V.p) * V.v
    );
}

HD inline Conserved flux_to_conserved(const Flux& F) {
    return Conserved(F.mass, F.momx, F.momy, F.energy);
}

HD inline double max_signal_speed_x(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    const double a = sound_speed(V);
    return fabs(V.u) + a;
}

HD inline double max_signal_speed_y(const Conserved& U) {
    const Primitive V = cons_to_prim(U);
    const double a = sound_speed(V);
    return fabs(V.v) + a;
}

HD inline double max_signal_speed(const Conserved& U) {
    return fmax(max_signal_speed_x(U), max_signal_speed_y(U));
}

}