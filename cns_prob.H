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
    const Real* dx      = geomdata.CellSize();

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real ly = prob_hi[1] - prob_lo[1];

    Real Pt, rhot, uxt, uyt, Yt;

    Real a0 = std::sqrt(parm.eos_gamma * parm.Rsp * prob_parm.T0);
    Real u0 = prob_parm.Mobj * a0;

    // Real rad = x*x + y*y - prob_parm.radhitemp*prob_parm.radhitemp;

    if(x > prob_parm.radhitemp){
    // if(rad > 0.0){
        uxt     = u0; 
        uyt     = Real(0.0);
        Pt      = prob_parm.p0;
        rhot    = 1.0;
    }else{
        // UNREACTED FUEL ENTERS AT VON-NEUMANN STATE
        Real pert = Real(0.0);
        Pt    = 30 * prob_parm.p0;  
        rhot  = Pt / (parm.Rsp * 2000);  
        uxt   = prob_parm.upvn;  
        uyt   = Real(0.0);
       

    }
    Yt      = Real(1.0);
    Real cs = std::sqrt(parm.eos_gamma * Pt / rhot);
    // TEST INFLOW CONDITIONS (THE ENTIRE DOMAIN IS AT MACH 3)
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot * uxt;
    state(i,j,k,UMY  ) = rhot * uyt;
#if (AMREX_SPACEDIM == 3)
    state(i,j,k,UMZ  ) = Real(0.0);
#endif
    state(i,j,k,URHOY) =  Yt;
    Real et = Pt/(parm.eos_gamma-Real(1.0));
    state(i,j,k,UEINT) = et;
    state(i,j,k,UEDEN) = et + Real(0.5)*(rhot * (uxt * uxt + uyt * uyt) );
    state(i,j,k,UTEMP) = Pt / (parm.Rsp * rhot);
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

    amrex::Print() << "Dcj = " << prob_parm.Dcj << " Mcj = " << prob_parm.Mcj
    << ", D = " << od*prob_parm.Dcj << ", Ma = " << mach
    << ", Ea = " << parm.Ea_dim << ", q = " << parm.q_dim
    << ", Tvn = " << prob_parm.Tvn << ", pvn = " << prob_parm.pvn << "\n";
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

    if(flag(i,j,k).isRegular() && flag(i+1,j,k).isRegular() && flag(i-1,j,k).isRegular() && level < prob_parm.refuptolev){

        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx      = geomdata.CellSize();
        Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
        Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
#if AMREX_SPACEDIM==3
        Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
#endif

        Real axY = 0.0, axT = 0.0, axP = 0.0;
        //Real ayY = 0.0, ayT = 0.0, ayP = 0.0;
        if(flag(i,j,k).isConnected(1,0,0)){
            axY = amrex::Math::abs( (sarr(i+1,j,k,URHOY)/sarr(i+1,j,k,URHO)) 
                - (sarr(i,j,k,URHOY)/sarr(i,j,k,URHO)) );    
            axT = amrex::Math::abs( sarr(i+1,j,k,UTEMP) - sarr(i,j,k,UTEMP) );
            axP = (parm.eos_gamma-1.)*amrex::Math::abs( sarr(i+1,j,k,UEINT) - sarr(i,j,k,UEINT) );
        }

    
        if(flag(i,j,k).isConnected(-1,0,0)){
            axY = amrex::max(axY, 
            amrex::Math::abs((sarr(i,j,k,URHOY)/sarr(i,j,k,URHO))-(sarr(i-1,j,k,URHOY)/sarr(i-1,j,k,URHO))));

            axT = amrex::max(axT,amrex::Math::abs(sarr(i,j,k,UTEMP)-sarr(i-1,j,k,UTEMP)));

            axP = amrex::max(axP,
                (parm.eos_gamma-1.)*amrex::Math::abs(sarr(i,j,k,UEINT)-sarr(i-1,j,k,UEINT)));
        }

        Real gymag = std::sqrt(axY*axY);
        Real gTmag = std::sqrt(axT*axT);
        Real gPmag = std::sqrt(axP*axP);

        Real minloc = prob_parm.flameloc - prob_parm.xreflo;
        Real maxloc = prob_parm.flameloc + prob_parm.xrefhi;
    
        if((!flag(i,j,k).isCovered())){
            if(gymag >= prob_parm.deltaY 
                || gTmag >= prob_parm.deltaT
                || gPmag >= prob_parm.deltaP*PREATM
                || (x >= minloc && x <= minloc
#if AMREX_SPACEDIM==3
                && z >= prob_parm.zreflo && z <= prob_parm.zrefhi
#endif
                ) 
                ){
                    tag(i,j,k) = tagval;
            } 
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real
get_flame_location (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<amrex::Real const> const& state,
    Parm const& parm) noexcept      
{
    using amrex::Real; 

    Real flamelocate = geomdata.ProbLo(0);    
    // FLAME LOCATION IS THE LARGEST VALUE OF x FOR
    // WHICH REACTANT MASS FRACTION DROPS BELOW 0.5
    if(state(i,0,0,URHOY) / state(i,0,0,URHO) < 0.5)
    	flamelocate = geomdata.ProbLo(0) + (i+0.5)*geomdata.CellSize(0);

    return flamelocate;
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
amrex::Real
global_energy_release_rate (int i, int j, int k, 
    amrex::GeometryData const& geomdata, 
    amrex::Array4<Real const> const& state,
    Parm const& parm) noexcept      
{
    Real omegarhoq = state(i,0,0,URHO) * parm.pre_exp * state(i,0,0,URHOY) 
                     * std::exp(-parm.Ea_dim / (parm.Ru * state(i,0,0,UTEMP)))
                     * parm.q_dim;
    return omegarhoq;
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
                       ProbParm& hprob_parm, 
                       ProbParm& dprob_parm, 
                       amrex::Real time, amrex::Real dt,
                       int level)
{
    using amrex::Real;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();

    Real shloc = 0.0, flameloc = 0.0, energy_grr = 0.0;

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(State.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    int myproc = ParallelDescriptor::MyProc();

    // Reduce operations for flame and shock locations
    ReduceOps<ReduceOpMax> reduce_op_flame;
    ReduceData<Real> reduce_data_flame(reduce_op_flame);
    using ReduceTupleFlame = typename decltype(reduce_data_flame)::Type;

    ReduceOps<ReduceOpMax> reduce_op_shock;
    ReduceData<Real> reduce_data_shock(reduce_op_shock);
    using ReduceTupleShock = typename decltype(reduce_data_shock)::Type;

    ReduceOps<ReduceOpSum> reduce_op_enrelease;
    ReduceData<Real> reduce_data_enrelease(reduce_op_enrelease);
    using ReduceTupleEnrelease = typename decltype(reduce_data_enrelease)::Type;

#ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(State, false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& flag = flags[mfi];
        auto const& s_arr = State.array(mfi);

        if (flag.getType(bx) != FabType::covered)
        {
            // FLAME LOCATION
            reduce_op_flame.eval(bx, reduce_data_flame, [=]
            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleFlame
            {
                return get_flame_location(i, 0, 0, geomdata, s_arr, parm);
            });

            // SHOCK LOCATION
            reduce_op_shock.eval(bx, reduce_data_shock, [=]
            AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleShock
            {
                return get_shock_location(i, 0, 0, geomdata, s_arr, parm);
            });

            // ENERGY RELEASE RATE (1D)
            if (level == 0 && write_data == 1)
            {
                reduce_op_enrelease.eval(bx, reduce_data_enrelease, [=]
                AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTupleEnrelease
                {
                    return global_energy_release_rate(i, 0, 0, geomdata, s_arr, parm);
                });
            }
        }
    } 

    ReduceTupleFlame host_tuple_flame = reduce_data_flame.value();
    flameloc = amrex::get<0>(host_tuple_flame);

    ReduceTupleShock host_tuple_shock = reduce_data_shock.value();
    shloc = amrex::get<0>(host_tuple_shock);

    ReduceTupleEnrelease host_tuple_grr = reduce_data_enrelease.value();
    energy_grr = amrex::get<0>(host_tuple_grr);

    ParallelDescriptor::ReduceRealMax(flameloc);
    ParallelDescriptor::ReduceRealMax(shloc);
    ParallelDescriptor::ReduceRealSum(energy_grr);

    if (myproc == 0)
    {
        hprob_parm.flameloc = amrex::max(flameloc, hprob_parm.flameloc); 
        dprob_parm.flameloc = amrex::max(flameloc, dprob_parm.flameloc);
        hprob_parm.shloc    = amrex::max(shloc, hprob_parm.shloc); 
        dprob_parm.shloc    = amrex::max(shloc, dprob_parm.shloc);

        if (hprob_parm.write_to_file == 1 && write_data == 1)
        {
            std::ofstream ofs;
            if (hprob_parm.append_file == 0)
            {
                hprob_parm.append_file = 1;
                ofs.open(hprob_parm.data_file, std::ofstream::out);
                if (!ofs) amrex::Abort("Failed to create file to write flame and shock location..exiting!");
                Print(myproc, ofs) << "# time flameloc (cm) shockloc (cm) global_energy_release_rate" << "\n";
            }
            else
            {
                ofs.open(hprob_parm.data_file, std::ofstream::app);
            }

            Print(myproc, ofs).SetPrecision(8) << std::left << std::setw(12) << time << "\t"
                << std::left << std::setw(12) << hprob_parm.flameloc * 100  << "\t"  
                << std::left << std::setw(12) << hprob_parm.shloc * 100 << "\t"
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
