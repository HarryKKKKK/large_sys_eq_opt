#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "physics.hpp"
#include "types.hpp"

enum class Direction {
    X, Y
};

enum class RiemannSolver {
    HLL,
    HLLC
};

// -----------------------------------------------------------------------------
// Physical flux wrapper
// -----------------------------------------------------------------------------

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

HD inline double tangential_velocity(const Primitive& V, Direction dir) {
    return (dir == Direction::X) ? V.v : V.u;
}

// -----------------------------------------------------------------------------
// HLL approximate Riemann solver
// -----------------------------------------------------------------------------

HD inline Conserved hll_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction dir
) {
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

// -----------------------------------------------------------------------------
// HLLC helper: construct star-region conservative state.
// 
// The formula is written using normal/tangential velocities so that the same
// function supports both x- and y-direction fluxes.
// -----------------------------------------------------------------------------

HD inline Conserved hllc_star_state(
    const Conserved& U,
    const Primitive& V,
    double S,
    double Sstar,
    Direction dir
) {
    const double rho = V.rho;
    const double p   = V.p;

    const double un = normal_velocity(V, dir);
    const double ut = tangential_velocity(V, dir);

    const double denom = S - Sstar;

    /*
      If denom is extremely small, fall back to the original state. This is a
      defensive guard for near-degenerate wave-speed estimates.
    */
    if (fabs(denom) < 1.0e-14) {
        return U;
    }

    const double rho_star = rho * (S - un) / denom;

    const double E_per_rho = U.E / rho;

    const double E_star = rho_star * (
        E_per_rho +
        (Sstar - un) * (Sstar + p / (rho * (S - un)))
    );

    if (dir == Direction::X) {
        return Conserved(
            rho_star,
            rho_star * Sstar,
            rho_star * ut,
            E_star
        );
    } else {
        return Conserved(
            rho_star,
            rho_star * ut,
            rho_star * Sstar,
            E_star
        );
    }
}

// -----------------------------------------------------------------------------
// HLLC approximate Riemann solver
// 
// HLLC restores the contact wave missing in HLL. It is usually less diffusive
// around contact discontinuities and material interfaces, but it is more
// expensive than HLL because it estimates an additional middle wave speed.
// -----------------------------------------------------------------------------

HD inline Conserved hllc_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction dir
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double pL = VL.p;
    const double pR = VR.p;

    const double rhoL = VL.rho;
    const double rhoR = VR.rho;

    /*
      Simple Davis wave-speed estimates.

      These are robust and consistent with the HLL implementation above:
          SL = min(u_L - a_L, u_R - a_R)
          SR = max(u_L + a_L, u_R + a_R)
    */
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

    /*
      Contact-wave speed estimate:

          S* =
          [p_R - p_L + rho_L u_L (S_L - u_L)
                     - rho_R u_R (S_R - u_R)]
          ------------------------------------------------
          [rho_L (S_L - u_L) - rho_R (S_R - u_R)]
    */
    const double numerator =
        pR - pL +
        rhoL * unL * (SL - unL) -
        rhoR * unR * (SR - unR);

    const double denominator =
        rhoL * (SL - unL) -
        rhoR * (SR - unR);

    if (fabs(denominator) < 1.0e-14) {
        /*
          Degenerate case: fall back to HLL for robustness.
        */
        return hll_flux(UL, UR, dir);
    }

    const double Sstar = numerator / denominator;

    if (Sstar >= 0.0) {
        const Conserved UstarL = hllc_star_state(
            UL,
            VL,
            SL,
            Sstar,
            dir
        );

        return FL + SL * (UstarL - UL);
    } else {
        const Conserved UstarR = hllc_star_state(
            UR,
            VR,
            SR,
            Sstar,
            dir
        );

        return FR + SR * (UstarR - UR);
    }
}

// -----------------------------------------------------------------------------
// Unified Riemann solver interface
// -----------------------------------------------------------------------------

HD inline Conserved riemann_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction dir,
    RiemannSolver solver
) {
    switch (solver) {
        case RiemannSolver::HLL:
            return hll_flux(UL, UR, dir);

        case RiemannSolver::HLLC:
            return hllc_flux(UL, UR, dir);

        default:
            return hll_flux(UL, UR, dir);
    }
}