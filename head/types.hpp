#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

struct Conserved {
    double rho  = 0.0; 
    double rhou = 0.0; 
    double rhov = 0.0;  
    double E    = 0.0; 

    Conserved() = default;

    Conserved(double rho_, double rhou_, double rhov_, double E_)
        : rho(rho_), rhou(rhou_), rhov(rhov_), E(E_) {}
};

struct Primitive {
    double rho = 0.0;
    double u   = 0.0; 
    double v   = 0.0;
    double p   = 0.0; 

    Primitive() = default;

    Primitive(double rho_, double u_, double v_, double p_)
        : rho(rho_), u(u_), v(v_), p(p_) {}
};

struct Flux {
    double mass   = 0.0;
    double momx   = 0.0;
    double momy   = 0.0;
    double energy = 0.0;

    Flux() = default;

    Flux(double mass_, double momx_, double momy_, double energy_)
        : mass(mass_), momx(momx_), momy(momy_), energy(energy_) {}
};

// ---------------------------
// Basic operations
// ---------------------------

inline Conserved operator+(const Conserved& a, const Conserved& b) {
    return Conserved(
        a.rho + b.rho,
        a.rhou + b.rhou,
        a.rhov + b.rhov,
        a.E + b.E
    );
}

inline Conserved operator-(const Conserved& a, const Conserved& b) {
    return Conserved(
        a.rho - b.rho,
        a.rhou - b.rhou,
        a.rhov - b.rhov,
        a.E - b.E
    );
}

inline Conserved operator*(double s, const Conserved& a) {
    return Conserved(
        s * a.rho,
        s * a.rhou,
        s * a.rhov,
        s * a.E
    );
}

inline Conserved operator*(const Conserved& a, double s) {
    return s * a;
}

inline Conserved operator/(const Conserved& a, double s) {
    return Conserved(
        a.rho / s,
        a.rhou / s,
        a.rhov / s,
        a.E / s
    );
}

inline Conserved& operator+=(Conserved& a, const Conserved& b) {
    a.rho  += b.rho;
    a.rhou += b.rhou;
    a.rhov += b.rhov;
    a.E    += b.E;
    return a;
}

inline Conserved& operator-=(Conserved& a, const Conserved& b) {
    a.rho  -= b.rho;
    a.rhou -= b.rhou;
    a.rhov -= b.rhov;
    a.E    -= b.E;
    return a;
}

inline Conserved& operator*=(Conserved& a, double s) {
    a.rho  *= s;
    a.rhou *= s;
    a.rhov *= s;
    a.E    *= s;
    return a;
}

inline Primitive operator+(const Primitive& a, const Primitive& b) {
    return Primitive(
        a.rho + b.rho,
        a.u   + b.u,
        a.v   + b.v,
        a.p   + b.p
    );
}

inline Primitive operator-(const Primitive& a, const Primitive& b) {
    return Primitive(
        a.rho - b.rho,
        a.u   - b.u,
        a.v   - b.v,
        a.p   - b.p
    );
}

inline Primitive operator*(double s, const Primitive& a) {
    return Primitive(
        s * a.rho,
        s * a.u,
        s * a.v,
        s * a.p
    );
}

inline Primitive operator*(const Primitive& a, double s) {
    return s * a;
}

inline Primitive operator/(const Primitive& a, double s) {
    return Primitive(
        a.rho / s,
        a.u   / s,
        a.v   / s,
        a.p   / s
    );
}