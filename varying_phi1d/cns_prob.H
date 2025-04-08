#ifndef CNS_PROB_H_
#define CNS_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

#include <AMReX_TagBox.H>
#include <AMReX_EBFArrayBox.H>

#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"
#include "FCT_advance_box.H"

#include <cmath>
#include <algorithm>

#define PI 3.141592653589793238
#define PREATM 101325

using namespace amrex;

AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx = geomdata.CellSize();

    Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
    Real ly = 0.1;  // 10 cm

    Real Pt, rhot, uxt, uyt, Y_f, Y_ox, Y_p;

    Real a0 = std::sqrt(parm.eos_gamma * parm.Rsp * prob_parm.T0);
    Real u0 = prob_parm.Mobj * a0;


    Real P_vn = 30.0 * prob_parm.p0;  // 30 atm -> Pascal
    Real T_vn = 2000.0;               // 2000 Kelvin
    Real epsilon = 0.02;              // %2 

    
    Real y1 = ly / 4;     // 0.025 m (2.5 cm)
    Real y2 = 2 * ly / 4; // 0.050 m (5 cm)
    Real y3 = 3 * ly / 4; // 0.075 m (7.5 cm)
    Real R = 0.01;        // (1 cm)


    bool in_cylinder1 = std::sqrt((x - 0.05) * (x - 0.05) + (y - y1) * (y - y1)) <= R;
    bool in_cylinder2 = std::sqrt((x - 0.05) * (x - 0.05) + (y - y2) * (y - y2)) <= R;
    bool in_cylinder3 = std::sqrt((x - 0.05) * (x - 0.05) + (y - y3) * (y - y3)) <= R;

    if (x < prob_parm.radhitemp) {  //  (x < 0.05)
       
        if (in_cylinder1 || in_cylinder2 || in_cylinder3) {
          
            Pt    = P_vn * (1 - epsilon);  
            rhot  = Pt / (parm.Rsp * T_vn * (1 - epsilon));  
            uxt   = prob_parm.upvn;  
            uyt   = Real(0.0);
        } else {
            // Von Neumann 
            Pt    = P_vn;
            rhot  = Pt / (parm.Rsp * T_vn);
            uxt   = prob_parm.upvn;
            uyt   = Real(0.0);
        }

        Y_f  = prob_parm.rich_Yf;
        Y_ox = prob_parm.rich_Yox;
        Y_p  = prob_parm.rich_Yp;

    } else {
        
        rhot  = prob_parm.rich_rhot;
        Pt    = prob_parm.p0;
        uxt   = u0;
        uyt   = Real(0.0);

        Y_f  = prob_parm.rich_Yf;
        Y_ox = prob_parm.rich_Yox;
        Y_p  = prob_parm.rich_Yp;
    }

   
    Real phi = calculatePhi(Y_ox, Y_f, Y_p, &prob_parm);
    Real Y_react = calculateYReact(phi, Y_ox, Y_f, Y_p, &prob_parm);

    Real pre_exp_tmp, Ea_nd_tmp, q_nd_tmp, kappa_0_tmp;
    Parm::Calculate_CDM_Parameters(phi, pre_exp_tmp, Ea_nd_tmp, q_nd_tmp, kappa_0_tmp,
                                   prob_parm.phi_min, prob_parm.phi_max, prob_parm.penalty_slope);

    pre_exp_tmp *= 1.e-3;
    kappa_0_tmp *= 1.e-1;

    Real q_dim  = q_nd_tmp  * parm.Rsp * prob_parm.T0;
    Real Ea_dim = Ea_nd_tmp * parm.Ru  * prob_parm.T0;

    Real cs = std::sqrt(parm.eos_gamma * Pt / rhot);
    
    
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot * uxt;
    state(i,j,k,UMY  ) = rhot * uyt;
#if (AMREX_SPACEDIM == 3)
    state(i,j,k,UMZ  ) = Real(0.0);
#endif
    state(i,j,k,URHOY_F) = rhot * Y_f;
    state(i,j,k,URHOY_O) = rhot * Y_ox;
    state(i,j,k,URHOY_P) = rhot * Y_p;
    
    Real et = Pt / (parm.eos_gamma - Real(1.0));
    state(i,j,k,UEINT) = et;
    state(i,j,k,UEDEN) = et + Real(0.5) * (rhot * (uxt * uxt + uyt * uyt));
    state(i,j,k,UTEMP) = T_vn;
    state(i,j,k,SFOIL) = Real(0.0);
}


AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
init_probparams (amrex::GeometryData const& geomdata, Parm const& parm, ProbParm& prob_parm, ProbParm& dprob_parm)
{
    using amrex::Real;

    prob_parm.rho0 = prob_parm.p0 / (parm.Rsp * prob_parm.T0);
    Real a0 = std::sqrt(parm.eos_gamma * parm.Rsp * prob_parm.T0);
    
    Real t1 = Real(0.5) * parm.q_dim * prob_parm.rho0 * (parm.eos_gamma*parm.eos_gamma-1.0) 
            / (prob_parm.p0 * parm.eos_gamma);
    prob_parm.Dcj = a0 * (std::sqrt(1.+t1) + std::sqrt(t1));
    prob_parm.Mcj = prob_parm.Dcj / a0;

    Real od = std::sqrt(prob_parm.od_factor);
    Real mach = od * prob_parm.Mcj;

    // Get the von-Neumann state variables (use normal shock relations)
    Real eta = 1.0 / (mach * mach);

    Real dvnd0 = (parm.eos_gamma+1.0) / (parm.eos_gamma - 1.0 + 2.0*eta);
    prob_parm.rhovn = prob_parm.rho0 * dvnd0;
    
    Real pvnp0 = 1. + (((2.0*parm.eos_gamma)/(parm.eos_gamma+1.0))*((1.0/eta) - 1.0));
    prob_parm.pvn = prob_parm.p0 * pvnp0;

    prob_parm.Tvn = prob_parm.pvn / (parm.Rsp * prob_parm.rhovn);

    Real u0 = prob_parm.Mobj * a0;
    prob_parm.upvn = (od * prob_parm.Dcj * (1. - (1.0/dvnd0))) + (u0 / dvnd0);
    prob_parm.uwvn = (od*prob_parm.Dcj) - prob_parm.upvn;

    // Get the lower and upper indices (global domain indices) for the ZND profile
    // Smaller index corresponds to CJ state and larger index corresponds to von-Neumann state
    // In the wave fixed frame, a detonation propagating from left to right has negative reactant and
    // product velocity
    // xsh is the location of the shock (von-Neumann state)
    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real d0d1 = (Real(1.0) + (parm.eos_gamma*mach*mach)) / ((parm.eos_gamma + Real(1.0))*(mach*mach));
    prob_parm.rhocj = prob_parm.rho0 / d0d1;

    Real p1p0 = (Real(1.0) + (parm.eos_gamma*mach*mach)) / (parm.eos_gamma + Real(1.0));
    prob_parm.pcj = prob_parm.p0 * p1p0;

    prob_parm.Tcj = prob_parm.pcj / (parm.Rsp * prob_parm.rhocj);
    prob_parm.upcj = od * prob_parm.Dcj * (1. - d0d1) + (d0d1 * u0);

    prob_parm.shloc = 0.0; prob_parm.flameloc = 0.0;
    dprob_parm.shloc = 0.0; dprob_parm.flameloc = 0.0;

    //Print() << "eb_wallloss = " << parm.eb_wallloss 
      //      << ", ksolid = " << parm.ksolid
        //    << ", Tsolidwall = " << parm.tempsolidwall << "\n"; 
       // Print() << "Dcj = " << prob_parm.Dcj << " Mcj = " << prob_parm.Mcj
        //<< ", D = " << od*prob_parm.Dcj << ", Ma = " << mach
        //<< ", Ea = " << parm.Ea_dim << ", q = " << parm.q_dim
        //<< ", Tvn = " << prob_parm.Tvn << ", pvn = " << prob_parm.pvn << "\n";
    
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_probspecific_bc(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NGROW][NUM_STATE],
  amrex::Real s_ext[NUM_STATE],
  const int idir,
  const int i,
  const int j,
#if AMREX_SPACEDIM==3
  const int k,
#endif
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  Parm const& parm,
  amrex::Array4<amrex::Real> const& dat)
{  
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cns_tag_probspecific(int i, int j, int k, amrex::Array4<char> const& tag, 
                    amrex::Array4<amrex::Real > const& sarr, 
                    amrex::GeometryData const& geomdata, 
                    char tagval, 
                    amrex::Array4<amrex::EBCellFlag const> const& flag,
                    Parm const& parm, ProbParm const& prob_parm, amrex::Real time,
                    int level)
{
    using amrex::Real;

    if(flag(i,j,k).isRegular() && flag(i+1,j,k).isRegular() && flag(i-1,j,k).isRegular() && level < prob_parm.refuptolev) {

        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx      = geomdata.CellSize();
        Real x = prob_lo[0] + (i+Real(0.5))*dx[0];

        Real axY_f = 0.0, axY_o = 0.0, axY_p = 0.0, axT = 0.0, axP = 0.0;

        for (int d = -1; d <= 1; d += 2) {
            if (flag(i+d, j, k).isConnected(d, 0, 0) && sarr(i+d, j, k, URHO) > 1.0e-12 && sarr(i, j, k, URHO) > 1.0e-12) {
                axY_f = amrex::max(axY_f, amrex::Math::abs((sarr(i+d, j, k, URHOY_F) / sarr(i+d, j, k, URHO)) 
                                                          - (sarr(i, j, k, URHOY_F) / sarr(i, j, k, URHO))));
                axY_o = amrex::max(axY_o, amrex::Math::abs((sarr(i+d, j, k, URHOY_O) / sarr(i+d, j, k, URHO)) 
                                                          - (sarr(i, j, k, URHOY_O) / sarr(i, j, k, URHO))));
                axY_p = amrex::max(axY_p, amrex::Math::abs((sarr(i+d, j, k, URHOY_P) / sarr(i+d, j, k, URHO)) 
                                                          - (sarr(i, j, k, URHOY_P) / sarr(i, j, k, URHO))));
                axT = amrex::max(axT, amrex::Math::abs(sarr(i+d, j, k, UTEMP) - sarr(i, j, k, UTEMP)));
                axP = amrex::max(axP, (parm.eos_gamma - 1.0) * amrex::Math::abs(sarr(i+d, j, k, UEINT) - sarr(i, j, k, UEINT)));
            }
        }

        Real gymag = std::sqrt(axY_f * axY_f + axY_o * axY_o + axY_p * axY_p);
        Real gTmag = std::sqrt(axT * axT);
        Real gPmag = std::sqrt(axP * axP);

        Real minloc = prob_parm.flameloc - prob_parm.xreflo;
        Real maxloc = prob_parm.flameloc + prob_parm.xrefhi;

        if (!flag(i, j, k).isCovered()) {
            if (gymag >= prob_parm.deltaY || gTmag >= prob_parm.deltaT || gPmag >= prob_parm.deltaP * PREATM
                || (x >= minloc && x <= maxloc)) {
                tag(i, j, k) = tagval;
            }
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real get_shock_location (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<Real const> const& state,
    Parm const& parm) noexcept      
{
    Real shloc = geomdata.ProbLo(0);
    Real pressure = (parm.eos_gamma - 1.0) * state(i, 0, 0, UEINT);

    // Shock location is the largest x where pressure is greater than threshold
    if (pressure > 1.5 * PREATM) {
        shloc = geomdata.ProbLo(0) + (i + 0.5) * geomdata.CellSize(0);
        
    }
    return shloc;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real get_flame_location (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<amrex::Real const> const& state,
    Parm const& parm,
    const ProbParm* prob_parm) noexcept      
{
    using amrex::Real; 
    Real flamelocate = geomdata.ProbLo(0);  

    // Retrieve mass fractions
    Real yF = state(i, 0, 0, URHOY_F) / state(i, 0, 0, URHO);
    Real yO = state(i, 0, 0, URHOY_O) / state(i, 0, 0, URHO);
    Real yP = state(i, 0, 0, URHOY_P) / state(i, 0, 0, URHO);

    normalizeMassFractions(yF, yO, yP);

    Real phi = calculatePhi(yO, yF, yP, prob_parm);
    Real Y_react = calculateYReact(phi, yO, yF, yP, prob_parm);

    if (Y_react < 0.5) {
        flamelocate = geomdata.ProbLo(0) + (i + 0.5) * geomdata.CellSize(0);
       
    }

    return flamelocate;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real global_energy_release_rate (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<amrex::Real const> const& state,
    Parm const& parm,
    const ProbParm* prob_parm) noexcept      
{
    using amrex::Real;

    // Retrieve local mass fractions
    Real yF = state(i, 0, 0, URHOY_F) / state(i, 0, 0, URHO);
    Real yO = state(i, 0, 0, URHOY_O) / state(i, 0, 0, URHO);
    Real yP = state(i, 0, 0, URHOY_P) / state(i, 0, 0, URHO);

    normalizeMassFractions(yF, yO, yP);

    Real phi = calculatePhi(yO, yF, yP, prob_parm);
    Real Y_react = calculateYReact(phi, yO, yF, yP, prob_parm);

    // Compute CDM parameters based on phi_unburnt
    Real pre_exp_tmp, Ea_nd_tmp, q_nd_tmp, kappa_0_tmp;
    Parm::Calculate_CDM_Parameters(phi,
                                   pre_exp_tmp,       
                                   Ea_nd_tmp,
                                   q_nd_tmp,
                                   kappa_0_tmp,
                                   prob_parm->phi_min,
                                   prob_parm->phi_max,
                                   prob_parm->penalty_slope
    );

    pre_exp_tmp *= 1.e-3;
    kappa_0_tmp *= 1.e-1;
    Real Ea_dim = Ea_nd_tmp * parm.Ru * prob_parm->Tref;
    Real q_dim  = q_nd_tmp  * parm.Rsp * prob_parm->Tref;

    // Compute energy release rate
    Real omegarhoq = state(i, 0, 0, URHO) * pre_exp_tmp * state(i, 0, 0, URHO) * Y_react 
                    * std::exp(-Ea_dim / (parm.Ru * state(i, 0, 0, UTEMP))) * q_dim;
    
    return amrex::max(0.0, omegarhoq);
}

// THIS FUNCTION IS CALLED BY THE CPU AT THE END OF EVERY TIMESTEP
// (IE) TIMESTEP AT LEVEL ZERO
// THIS FUNCTION CAN BE USED TO COMPUTE THE LOCAL SHOCK AND REACTION FRONT 
// LOCATION, REACTION FRONT SPEED ETC. THESE QUANTITIES MUST BE DEFINED 
// IN prob_parm

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
cns_probspecific_func (amrex::MultiFab& State, 
                       amrex::GeometryData const& geomdata, 
                       int write_data, Parm const& parm, 
                       ProbParm* hprob_parm,   
                       ProbParm* dprob_parm,   
                       amrex::Real time, amrex::Real dt, 
                       int level)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx      = geomdata.CellSize();

    Real shloc = 0.0, flameloc = 0.0, energy_grr = 0.0;

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(State.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    for (int ii = 0; ii < hprob_parm->nzones; ++ii) {

        int myproc = ParallelDescriptor::MyProc();

        // Reduce operations for flame and shock locations
        ReduceOps<ReduceOpMax> reduce_op_flame;
        ReduceData<Real> reduce_data_flame(reduce_op_flame);
        using ReduceTupleFlame = typename decltype(reduce_data_flame)::Type;

        ReduceOps<ReduceOpMax> reduce_op_shock;
        ReduceData<Real> reduce_data_shock(reduce_op_shock);
        using ReduceTupleShock = typename decltype(reduce_data_shock)::Type;

#ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(State, false); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            const auto& flag = flags[mfi];
            auto const& s_arr = State.array(mfi);

            if (flag.getType(bx) != FabType::covered) {
                
                reduce_op_flame.eval(bx, reduce_data_flame, [=]
                AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleFlame {
                    // Y ve Z yönünü sabit 0 alýyoruz
                    return get_flame_location(i, 0, 0, geomdata, s_arr, parm, hprob_parm);
                });

                reduce_op_shock.eval(bx, reduce_data_shock, [=]
                AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleShock {
                    // Y ve Z yönünü sabit 0 alýyoruz
                    return get_shock_location(i, 0, 0, geomdata, s_arr, parm);
                });
            }
        }

        // Gather the results from each processor
        ReduceTupleFlame host_tuple_flame = reduce_data_flame.value();
        Real zone_flameloc = amrex::get<0>(host_tuple_flame);

        ReduceTupleShock host_tuple_shock = reduce_data_shock.value();
        Real zone_shloc = amrex::get<0>(host_tuple_shock);

        ParallelDescriptor::ReduceRealMax(zone_flameloc);
        ParallelDescriptor::ReduceRealMax(zone_shloc);

        // Update flame and shock locations
        if (myproc == 0) {
            hprob_parm->flamel[ii] = amrex::max(zone_flameloc, hprob_parm->flamel[ii]); 
            dprob_parm->flamel[ii] = amrex::max(zone_flameloc, dprob_parm->flamel[ii]);
            hprob_parm->shl[ii] = amrex::max(zone_shloc, hprob_parm->shl[ii]); 
            dprob_parm->shl[ii] = amrex::max(zone_shloc, dprob_parm->shl[ii]);
        }

        flameloc = amrex::max(flameloc, zone_flameloc);
        shloc    = amrex::max(shloc, zone_shloc);
    }

    ParallelDescriptor::Barrier();

    // ENERGY RELEASE RATE 
    ReduceOps<ReduceOpSum> reduce_op_enrelease;
    ReduceData<Real> reduce_data_enrelease(reduce_op_enrelease);
    using ReduceTupleEnrelease = typename decltype(reduce_data_enrelease)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(State, false); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto& flag = flags[mfi];
        auto const& s_arr = State.array(mfi);

        if (flag.getType(bx) != FabType::covered) {
            reduce_op_enrelease.eval(bx, reduce_data_enrelease, [=]
            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleEnrelease {
                return global_energy_release_rate(i, 0, 0, geomdata, s_arr, parm, hprob_parm);
            });
        }
    }

    ReduceTupleEnrelease host_tuple_grr = reduce_data_enrelease.value();
    energy_grr = amrex::get<0>(host_tuple_grr);

    ParallelDescriptor::ReduceRealSum(energy_grr);

    int myproc = ParallelDescriptor::MyProc();

    if (myproc == 0) {
        hprob_parm->flameloc = flameloc;
        dprob_parm->flameloc = flameloc;
        hprob_parm->shloc    = shloc;
        dprob_parm->shloc    = shloc;

        if (hprob_parm->write_to_file == 1 && write_data == 1) {
            std::ofstream ofs(hprob_parm->data_file, std::ofstream::app);
            Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
                << std::left << std::setw(12) << flameloc*100 << "\t"  
                << std::left << std::setw(12) << shloc*100 << "\t"
                << std::left << std::setw(12) << energy_grr << "\n";
            ofs.close();
        }
    }

    ParallelDescriptor::Barrier();
}

AMREX_GPU_HOST
AMREX_FORCE_INLINE
void
init_eb_geometry (const amrex::Geometry& geom, const int max_coarsening_level)
{

}           
#endif
