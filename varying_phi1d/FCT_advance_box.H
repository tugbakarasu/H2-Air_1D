#ifndef FCT_ADVANCE_BOX_H
#define FCT_ADVANCE_BOX_H

#include "AMReX.H"
#include "cns_prob_parm.H"
#include <AMReX_REAL.H>

using namespace amrex;
using amrex::Real;

// Normalize mass fractions
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void normalizeMassFractions(Real& yF, Real& yO, Real& yP) {
    Real total_mass_fraction = yF + yO + yP;
    if (std::abs(total_mass_fraction - 1.0) > 1.e-10) {
        yF = amrex::max(0.0, yF / total_mass_fraction);
        yO = amrex::max(0.0, yO / total_mass_fraction);
        yP = amrex::max(0.0, yP / total_mass_fraction);

    }
}


// Calculate equivalence ratio (phi)
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real calculatePhi(Real yO, Real yF, Real yP, const ProbParm* prob_parm) {
    
    Real yO0 = amrex::max(1.e-20, yO + (prob_parm->OF_st / (1.0 + prob_parm->OF_st)) * yP);
    Real yF0 = amrex::max(1.e-20, yF + (1.0 / (1.0 + prob_parm->OF_st)) * yP);
    
    Real phi = prob_parm->OF_st * yF0 / yO0;
    
    return amrex::max(0.0, amrex::min(prob_parm->phi_max, phi));
}

// Calculate reaction progress variable
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real calculateYReact(Real phi, Real yO, Real yF, Real yP, const ProbParm* prob_parm) {

    Real yO0 = amrex::max(1.e-20, yO + (prob_parm->OF_st / (1.0 + prob_parm->OF_st)) * yP);
    Real yF0 = amrex::max(1.e-20, yF + (1.0 / (1.0 + prob_parm->OF_st)) * yP);

    if (phi < 1.0+ 1e-6) {
        return amrex::min(1.0, yF / yF0);
    } else {
        return amrex::min(1.0, yO / yO0);
    }
}

// Calculate species reaction rates
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void calculateSpeciesRates(Real phi, Real omegarho, Real s, Real& omega_F, Real& omega_Ox, Real& omega_Pr) {
    Real denom = s + phi;
    Real factor = (phi < 1.0) ? phi : 1.0;
    if (std::abs(phi - 1.0) < 1e-6) {
        omega_F = (1.0 / (s + 1.0)) * omegarho;
        omega_Ox = (s / (s + 1.0)) * omegarho;
        omega_Pr = -omegarho;
    } else {
        omega_F = (factor / denom) * omegarho;
        omega_Ox = (s * factor / denom) * omegarho;
        omega_Pr = -((s + 1.0) * factor / denom) * omegarho;
    }
}

// CDM Parameter Calculation
AMREX_GPU_DEVICE
void Calculate_CDM_Parameters(Real phi,
                              Real& pre_exp_out,
                              Real& Ea_nd_out,
                              Real& q_nd_out,
                              Real& kappa_0_out,
                              const Real phi_min,
                              const Real phi_max,
                              const Real penalty_slope);

#endif // FCT_ADVANCE_BOX_H