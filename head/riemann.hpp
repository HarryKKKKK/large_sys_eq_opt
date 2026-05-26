#pragma once

#include <algorithm>
#include <cmath>

#include "physics.hpp"
#include "types.hpp"

enum class Direction {
    X, Y
};

enum class RiemannSolver {
    HLL,
    HLLC,
    Exact,
    FORCE
};

// -----------------------------------------------------------------------------
// Basic helpers
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

HD inline Conserved make_conserved_from_normal_tangential(
    double rho,
    double un,
    double ut,
    double p,
    Direction dir
) {
    double u = 0.0;
    double v = 0.0;

    if (dir == Direction::X) {
        u = un;
        v = ut;
    } else {
        u = ut;
        v = un;
    }

    return Conserved(
        rho,
        rho * u,
        rho * v,
        p / (phys::gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    );
}

HD inline bool finite_number(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}

HD inline bool primitive_is_physical_riemann(const Primitive& V) {
    return finite_number(V.rho) &&
           finite_number(V.u) &&
           finite_number(V.v) &&
           finite_number(V.p) &&
           V.rho > 0.0 &&
           V.p > 0.0;
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

    if (!primitive_is_physical_riemann(VL) ||
        !primitive_is_physical_riemann(VR)) {
        const Conserved FL = physical_flux(UL, dir);
        const Conserved FR = physical_flux(UR, dir);
        return 0.5 * (FL + FR);
    }

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

    const double denom = SR - SL;

    if (fabs(denom) < 1.0e-14) {
        return 0.5 * (FL + FR);
    }

    return (SR * FL - SL * FR + (SL * SR) * (UR - UL)) / denom;
}

// -----------------------------------------------------------------------------
// HLLC helper
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

    if (fabs(denom) < 1.0e-14) {
        return U;
    }

    const double rho_star = rho * (S - un) / denom;

    if (!finite_number(rho_star) || rho_star <= 0.0) {
        return U;
    }

    const double E_per_rho = U.E / rho;

    const double E_star = rho_star * (
        E_per_rho +
        (Sstar - un) * (Sstar + p / (rho * (S - un)))
    );

    if (!finite_number(E_star)) {
        return U;
    }

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
// -----------------------------------------------------------------------------

HD inline Conserved hllc_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction dir
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    if (!primitive_is_physical_riemann(VL) ||
        !primitive_is_physical_riemann(VR)) {
        return hll_flux(UL, UR, dir);
    }

    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double pL = VL.p;
    const double pR = VR.p;

    const double rhoL = VL.rho;
    const double rhoR = VR.rho;

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

    const double numerator =
        pR - pL +
        rhoL * unL * (SL - unL) -
        rhoR * unR * (SR - unR);

    const double denominator =
        rhoL * (SL - unL) -
        rhoR * (SR - unR);

    if (fabs(denominator) < 1.0e-14) {
        return hll_flux(UL, UR, dir);
    }

    const double Sstar = numerator / denominator;

    if (!finite_number(Sstar)) {
        return hll_flux(UL, UR, dir);
    }

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
// FORCE-type approximate Riemann solver
//
// Standard FORCE is the average of Lax--Friedrichs and Richtmyer fluxes.
// The classical formula requires dt/dx. Since the current riemann_flux interface
// does not pass dt/dx, this implementation uses a local wave-speed estimate:
//     alpha = max(|u_L| + a_L, |u_R| + a_R)
// and uses dt/dx ~= 1 / alpha.
// This gives a local FORCE-type flux while preserving the existing solver API.
// -----------------------------------------------------------------------------

HD inline Conserved force_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction dir
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    if (!primitive_is_physical_riemann(VL) ||
        !primitive_is_physical_riemann(VR)) {
        return hll_flux(UL, UR, dir);
    }

    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double alpha = fmax(fabs(unL) + aL, fabs(unR) + aR);

    const Conserved FL = physical_flux(UL, dir);
    const Conserved FR = physical_flux(UR, dir);

    if (!finite_number(alpha) || alpha <= 1.0e-14) {
        return 0.5 * (FL + FR);
    }

    const Conserved F_lf = 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL);

    const Conserved U_ri = 0.5 * (UL + UR) - 0.5 * (1.0 / alpha) * (FR - FL);

    const Primitive V_ri = phys::cons_to_prim(U_ri);

    if (!primitive_is_physical_riemann(V_ri)) {
        return F_lf;
    }

    const Conserved F_ri = physical_flux(U_ri, dir);

    return 0.5 * (F_lf + F_ri);
}

// -----------------------------------------------------------------------------
// Exact Riemann solver helpers
// -----------------------------------------------------------------------------

HD inline void exact_pressure_function(
    double p,
    double rhoK,
    double pK,
    double aK,
    double& f,
    double& fd
) {
    const double gamma = phys::gamma;

    if (p > pK) {
        // Shock branch
        const double A = 2.0 / ((gamma + 1.0) * rhoK);
        const double B = (gamma - 1.0) / (gamma + 1.0) * pK;

        const double q = sqrt(A / (p + B));

        f = (p - pK) * q;
        fd = q * (1.0 - 0.5 * (p - pK) / (p + B));
    } else {
        // Rarefaction branch
        const double pratio = p / pK;
        const double expo = (gamma - 1.0) / (2.0 * gamma);

        f = 2.0 * aK / (gamma - 1.0) * (pow(pratio, expo) - 1.0);
        fd = (1.0 / (rhoK * aK)) *
             pow(pratio, -(gamma + 1.0) / (2.0 * gamma));
    }
}

HD inline double exact_guess_pressure(
    const Primitive& VL,
    const Primitive& VR,
    Direction dir
) {
    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    const double p_pvrs =
        0.5 * (VL.p + VR.p) -
        0.125 * (unR - unL) * (VL.rho + VR.rho) * (aL + aR);

    return fmax(1.0e-12, p_pvrs);
}

HD inline bool exact_star_pressure_velocity(
    const Primitive& VL,
    const Primitive& VR,
    Direction dir,
    double& p_star,
    double& u_star
) {
    const double rhoL = VL.rho;
    const double rhoR = VR.rho;

    const double pL = VL.p;
    const double pR = VR.p;

    const double aL = phys::sound_speed(VL);
    const double aR = phys::sound_speed(VR);

    const double unL = normal_velocity(VL, dir);
    const double unR = normal_velocity(VR, dir);

    p_star = exact_guess_pressure(VL, VR, dir);

    for (int iter = 0; iter < 20; ++iter) {
        double fL = 0.0;
        double fR = 0.0;
        double fdL = 0.0;
        double fdR = 0.0;

        exact_pressure_function(p_star, rhoL, pL, aL, fL, fdL);
        exact_pressure_function(p_star, rhoR, pR, aR, fR, fdR);

        const double f = fL + fR + (unR - unL);
        const double fd = fdL + fdR;

        if (fabs(fd) < 1.0e-14) {
            return false;
        }

        const double p_old = p_star;
        p_star = p_old - f / fd;

        if (!finite_number(p_star) || p_star <= 0.0) {
            p_star = 0.5 * p_old;
        }

        if (fabs(p_star - p_old) / (0.5 * (p_star + p_old) + 1.0e-14) < 1.0e-8) {
            break;
        }
    }

    if (!finite_number(p_star) || p_star <= 0.0) {
        return false;
    }

    double fL = 0.0;
    double fR = 0.0;
    double fd_dummy_L = 0.0;
    double fd_dummy_R = 0.0;

    exact_pressure_function(p_star, rhoL, pL, aL, fL, fd_dummy_L);
    exact_pressure_function(p_star, rhoR, pR, aR, fR, fd_dummy_R);

    u_star = 0.5 * (unL + unR + fR - fL);

    return finite_number(u_star);
}

HD inline Conserved exact_sample_left(
    const Primitive& VL,
    double p_star,
    double u_star,
    Direction dir
) {
    const double gamma = phys::gamma;

    const double rhoL = VL.rho;
    const double pL = VL.p;
    const double unL = normal_velocity(VL, dir);
    const double utL = tangential_velocity(VL, dir);
    const double aL = phys::sound_speed(VL);

    const double xi = 0.0;

    double rho = rhoL;
    double un = unL;
    double p = pL;

    if (p_star > pL) {
        // Left shock
        const double p_ratio = p_star / pL;
        const double SL = unL -
            aL * sqrt(
                (gamma + 1.0) / (2.0 * gamma) * p_ratio +
                (gamma - 1.0) / (2.0 * gamma)
            );

        if (xi <= SL) {
            rho = rhoL;
            un = unL;
            p = pL;
        } else {
            rho = rhoL *
                ((p_ratio + (gamma - 1.0) / (gamma + 1.0)) /
                 ((gamma - 1.0) / (gamma + 1.0) * p_ratio + 1.0));
            un = u_star;
            p = p_star;
        }
    } else {
        // Left rarefaction
        const double a_star = aL * pow(p_star / pL, (gamma - 1.0) / (2.0 * gamma));

        const double SHL = unL - aL;
        const double STL = u_star - a_star;

        if (xi <= SHL) {
            rho = rhoL;
            un = unL;
            p = pL;
        } else if (xi >= STL) {
            rho = rhoL * pow(p_star / pL, 1.0 / gamma);
            un = u_star;
            p = p_star;
        } else {
            const double a =
                2.0 / (gamma + 1.0) *
                (aL + 0.5 * (gamma - 1.0) * (unL - xi));

            un =
                2.0 / (gamma + 1.0) *
                (aL + 0.5 * (gamma - 1.0) * unL + xi);

            const double aratio = a / aL;

            rho = rhoL * pow(aratio, 2.0 / (gamma - 1.0));
            p   = pL   * pow(aratio, 2.0 * gamma / (gamma - 1.0));
        }
    }

    return make_conserved_from_normal_tangential(rho, un, utL, p, dir);
}

HD inline Conserved exact_sample_right(
    const Primitive& VR,
    double p_star,
    double u_star,
    Direction dir
) {
    const double gamma = phys::gamma;

    const double rhoR = VR.rho;
    const double pR = VR.p;
    const double unR = normal_velocity(VR, dir);
    const double utR = tangential_velocity(VR, dir);
    const double aR = phys::sound_speed(VR);

    const double xi = 0.0;

    double rho = rhoR;
    double un = unR;
    double p = pR;

    if (p_star > pR) {
        // Right shock
        const double p_ratio = p_star / pR;
        const double SR = unR +
            aR * sqrt(
                (gamma + 1.0) / (2.0 * gamma) * p_ratio +
                (gamma - 1.0) / (2.0 * gamma)
            );

        if (xi >= SR) {
            rho = rhoR;
            un = unR;
            p = pR;
        } else {
            rho = rhoR *
                ((p_ratio + (gamma - 1.0) / (gamma + 1.0)) /
                 ((gamma - 1.0) / (gamma + 1.0) * p_ratio + 1.0));
            un = u_star;
            p = p_star;
        }
    } else {
        // Right rarefaction
        const double a_star = aR * pow(p_star / pR, (gamma - 1.0) / (2.0 * gamma));

        const double SHR = unR + aR;
        const double STR = u_star + a_star;

        if (xi >= SHR) {
            rho = rhoR;
            un = unR;
            p = pR;
        } else if (xi <= STR) {
            rho = rhoR * pow(p_star / pR, 1.0 / gamma);
            un = u_star;
            p = p_star;
        } else {
            const double a =
                2.0 / (gamma + 1.0) *
                (aR - 0.5 * (gamma - 1.0) * (unR - xi));

            un =
                2.0 / (gamma + 1.0) *
                (-aR + 0.5 * (gamma - 1.0) * unR + xi);

            const double aratio = a / aR;

            rho = rhoR * pow(aratio, 2.0 / (gamma - 1.0));
            p   = pR   * pow(aratio, 2.0 * gamma / (gamma - 1.0));
        }
    }

    return make_conserved_from_normal_tangential(rho, un, utR, p, dir);
}

// -----------------------------------------------------------------------------
// Exact ideal-gas Euler Riemann solver.
//
// This solves the one-dimensional normal Riemann problem at each face. It is
// useful as a validation/reference solver, but is significantly more expensive
// than HLL/HLLC/FORCE.
// -----------------------------------------------------------------------------

HD inline Conserved exact_flux(
    const Conserved& UL,
    const Conserved& UR,
    Direction dir
) {
    const Primitive VL = phys::cons_to_prim(UL);
    const Primitive VR = phys::cons_to_prim(UR);

    if (!primitive_is_physical_riemann(VL) ||
        !primitive_is_physical_riemann(VR)) {
        return hll_flux(UL, UR, dir);
    }

    double p_star = 0.0;
    double u_star = 0.0;

    const bool ok = exact_star_pressure_velocity(
        VL,
        VR,
        dir,
        p_star,
        u_star
    );

    if (!ok || !finite_number(p_star) || !finite_number(u_star) || p_star <= 0.0) {
        return hll_flux(UL, UR, dir);
    }

    Conserved U_sample;

    if (u_star >= 0.0) {
        U_sample = exact_sample_left(VL, p_star, u_star, dir);
    } else {
        U_sample = exact_sample_right(VR, p_star, u_star, dir);
    }

    const Primitive V_sample = phys::cons_to_prim(U_sample);

    if (!primitive_is_physical_riemann(V_sample)) {
        return hll_flux(UL, UR, dir);
    }

    return physical_flux(U_sample, dir);
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

        case RiemannSolver::Exact:
            return exact_flux(UL, UR, dir);

        case RiemannSolver::FORCE:
            return force_flux(UL, UR, dir);

        default:
            return hll_flux(UL, UR, dir);
    }
}