#ifndef FCT_DIFFUSION_EB_K_H_
#define FCT_DIFFUSION_EB_K_H_

#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_CONSTANTS.H>
#include <cmath>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_diffcoef_eb (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& q,
                 amrex::Array4<amrex::EBCellFlag const> const& flag,
                 amrex::Array4<amrex::Real> const& coefs,
                 Parm const& parm) noexcept
{
    using amrex::Real;

    if (q(i,j,k,QTEMP) < 0.)
    {
       // std::cout << "Computing diff coeffs at " << i,j,k) <<
       //      " with  temp = " << q(i,j,k,UTEMP)  << std::endl;
        amrex::Abort("Negative temperature sent to cns_diffcoef_eb");
    }

    bool cov = flag(i,j,k).isCovered();
    coefs(i,j,k,0) = cov ? -1.e10 : parm.kappa_0 * parm.Pr * std::pow(q(i,j,k,QTEMP), Real(0.7));
    // coefs(i,j,k,CXI)  = cov ? -1.e10 : Real(0.0);
    // coefs(i,j,k,CLAM) = cov ? -1.e10 : coefs(i,j,k,CETA)*parm.cp/parm.Pr;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_constcoef_eb (int i, int j, int k,
                 amrex::Array4<amrex::EBCellFlag const> const& flag,
                 amrex::Array4<amrex::Real> const& coefs,
                 Parm const& parm) noexcept
{
    using amrex::Real;

     bool cov = flag(i,j,k).isCovered();
     coefs(i,j,k,0) = cov ? -1.e01 : parm.const_visc_mu;
     // coefs(i,j,k,CXI)  = cov ? -1.e01 : parm.const_visc_ki;
     // coefs(i,j,k,CLAM) = cov ? -1.e01 : parm.const_lambda;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_phys_diff_eb_x (int i, int j, int k,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& uo,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::Array4<amrex::EBCellFlag const> const& flag,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::GpuArray<amrex::Real,3> const& weights,
               amrex::Array4<amrex::Real> const& fx,
               int do_reac, Parm const& parm) noexcept
{
    using amrex::Real;

    Real  dTdx = (q(i,j,k,QTEMP)-q(i-1,j,k,QTEMP))*dxinv[0];
    AMREX_D_TERM(Real  dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];,
                 Real  dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];,
                 Real  dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];);

    int  jhip = j + (flag(i  ,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jhim = j - (flag(i  ,j,k).isConnected(0,-1,0) ? 1 : 0 );
    int  jlop = j + (flag(i-1,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jlom = j - (flag(i-1,j,k).isConnected(0,-1,0) ? 1 : 0 );

    Real  whi = weights[jhip-jhim];
    Real  wlo = weights[jlop-jlom];
    Real  dudy = (0.5*dxinv[1]) * ((q(i  ,jhip,k,QU)-q(i  ,jhim,k,QU))*whi
                                  +(q(i-1,jlop,k,QU)-q(i-1,jlom,k,QU))*wlo);
    Real  dvdy = (0.5*dxinv[1]) * ((q(i  ,jhip,k,QV)-q(i  ,jhim,k,QV))*whi
                                  +(q(i-1,jlop,k,QV)-q(i-1,jlom,k,QV))*wlo);

#if (AMREX_SPACEDIM == 3)
    int  khip = k + (flag(i  ,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  khim = k - (flag(i  ,j,k).isConnected(0,0,-1) ? 1 : 0 );
    int  klop = k + (flag(i-1,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  klom = k - (flag(i-1,j,k).isConnected(0,0,-1) ? 1 : 0 );

    whi = weights[khip-khim];
    wlo = weights[klop-klom];
    Real  dudz = (0.5*dxinv[2]) * ((q(i  ,j,khip,QU)-q(i  ,j,khim,QU))*whi
                                  +(q(i-1,j,klop,QU)-q(i-1,j,klom,QU))*wlo);
    Real  dwdz = (0.5*dxinv[2]) * ((q(i  ,j,khip,QW)-q(i  ,j,khim,QW))*whi
                                  +(q(i-1,j,klop,QW)-q(i-1,j,klom,QW))*wlo);
#endif

#if (AMREX_SPACEDIM == 2)
    Real  divu = dudx + dvdy;

    Real  muf = 0.5*(coeffs(i,j,k,0)+coeffs(i-1,j,k,0));
    // Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i-1,j,k,CETA));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i-1,j,k,CXI));
    Real  tauxx = muf*(2.0*dudx-(2.0/3.0)*divu);
    Real  tauxy = muf*(dudy+dvdx);

    fx(i,j,k,UMX)   = tauxx;
    fx(i,j,k,UMY)   = tauxy;
    fx(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx+
                            (q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy+
                            ((coeffs(i,j,k,0) +coeffs(i-1,j,k,0))*parm.cp/parm.Pr)*dTdx );
#else
    Real  divu = dudx + dvdy + dwdz;
    Real  muf  = 0.5*(coeffs(i,j,k,0)+coeffs(i-1,j,k,0));
    // Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i-1,j,k,CETA));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i-1,j,k,CXI));
    Real  tauxx = muf*(2.0*dudx-(2.0/3.0)*divu);
    Real  tauxy = muf*(dudy+dvdx);
    Real  tauxz = muf*(dudz+dwdx);

    fx(i,j,k,UMX)   = tauxx;
    fx(i,j,k,UMY)   = tauxy;
    fx(i,j,k,UMZ)   = tauxz;
    fx(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx+
                            (q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy+
                            (q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz+
                            (coeffs(i,j,k,0) +coeffs(i-1,j,k,0)*parm.cp/parm.Pr)*dTdx );
#endif

    if(do_reac == 1){

        Real dYdx_fuel = dxinv[0]*( (uo(i,j,k,URHOY_F)/uo(i,j,k,URHO)) - (uo(i-1,j,k,URHOY_F)/uo(i-1,j,k,URHO)) );

        Real dYdx_oxid = dxinv[0]*( (uo(i,j,k,URHOY_O)/uo(i,j,k,URHO)) - (uo(i-1,j,k,URHOY_O)/uo(i-1,j,k,URHO)) );
        
        Real rhoD = 0.5*(coeffs(i-1,j,k,0) + coeffs(i,j,k,0)) / ( parm.Pr * parm.Le );

        fx(i, j, k, URHOY_F) = rhoD * dYdx_fuel;
        fx(i, j, k, URHOY_O) = rhoD * dYdx_oxid;
    }

}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_phys_diff_eb_y (int i, int j, int k, 
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& uo,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::Array4<amrex::EBCellFlag const> const& flag,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::GpuArray<amrex::Real,3> const& weights,
               amrex::Array4<amrex::Real> const& fy,
               int do_reac, Parm const& parm) noexcept
{
    using amrex::Real;

    Real  dTdy = (q(i,j,k,QTEMP)-q(i,j-1,k,QTEMP))*dxinv[1];
    AMREX_D_TERM(Real  dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];,
                 Real  dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];,
                 Real  dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];);

    int  ihip = i + (flag(i,j  ,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j  ,k).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j-1,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j-1,k).isConnected(-1,0,0) ? 1 : 0 );

    Real  whi = weights[ihip-ihim];
    Real  wlo = weights[ilop-ilom];
    Real  dudx = (0.5*dxinv[0]) * ((q(ihip,j  ,k,QU)-q(ihim,j  ,k,QU))*whi
                                  +(q(ilop,j-1,k,QU)-q(ilom,j-1,k,QU))*wlo);
    Real  dvdx = (0.5*dxinv[0]) * ((q(ihip,j  ,k,QV)-q(ihim,j  ,k,QV))*whi
                                  +(q(ilop,j-1,k,QV)-q(ilom,j-1,k,QV))*wlo);

#if (AMREX_SPACEDIM == 3)
    int  khip = k + (flag(i,j  ,k).isConnected(0,0, 1) ? 1 : 0 );
    int  khim = k - (flag(i,j  ,k).isConnected(0,0,-1) ? 1 : 0 );
    int  klop = k + (flag(i,j-1,k).isConnected(0,0, 1) ? 1 : 0 );
    int  klom = k - (flag(i,j-1,k).isConnected(0,0,-1) ? 1 : 0 );

    whi = weights[khip-khim];
    wlo = weights[klop-klom];
    Real  dvdz = (0.5*dxinv[2]) * ((q(i,j  ,khip,QV)-q(i,j  ,khim,QV))*whi
                                  +(q(i,j-1,klop,QV)-q(i,j-1,klom,QV))*wlo);
    Real  dwdz = (0.5*dxinv[2]) * ((q(i,j  ,khip,QW)-q(i,j  ,khim,QW))*whi
                                  +(q(i,j-1,klop,QW)-q(i,j-1,klom,QW))*wlo) ;
#endif

#if (AMREX_SPACEDIM == 2)
    Real  divu = dudx + dvdy;
    Real  muf  = 0.5*(coeffs(i,j,k,0)+coeffs(i,j-1,k,0));
    // Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i,j-1,k,CETA));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j-1,k,CXI));
    Real  tauyy = muf*(2.0*dvdy-(2.0/3.0)*divu);
    Real  tauxy = muf*(dudy+dvdx);

    fy(i,j,k,UMX)   = tauxy;
    fy(i,j,k,UMY)   = tauyy;
    fy(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy
                             +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy
                             +((coeffs(i,j,k,0) +coeffs(i,j-1,k,0))*parm.cp/parm.Pr)*dTdy );
#else
    Real  divu = dudx + dvdy + dwdz;
    Real  muf  = 0.5*(coeffs(i,j,k,0)+coeffs(i,j-1,k,0));
    // Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i,j-1,k,CETA));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j-1,k,CXI));
    Real  tauyy = muf*(2.0*dvdy-(2.0/3.0)*divu);
    Real  tauxy = muf*(dudy+dvdx);
    Real  tauyz = muf*(dwdy+dvdz);

    fy(i,j,k,UMX)   = tauxy;
    fy(i,j,k,UMY)   = tauyy;
    fy(i,j,k,UMZ)   = tauyz;
    fy(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy
                             +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy
                             +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz
                             +(coeffs(i,j,k,0) +coeffs(i,j-1,k,0)*parm.cp/parm.Pr)*dTdy );
#endif

    if(do_reac == 1){

        Real dYdy_fuel = dxinv[1]*( (uo(i,j,k,URHOY_F)/uo(i,j,k,URHO)) - (uo(i,j-1,k,URHOY_F)/uo(i,j-1,k,URHO)) );

        Real dYdy_oxid = dxinv[1]*( (uo(i,j,k,URHOY_O)/uo(i,j,k,URHO)) - (uo(i,j-1,k,URHOY_O)/uo(i,j-1,k,URHO)) );

        Real rhoD = 0.5*( coeffs(i,j,k,0) + coeffs(i,j-1,k,0) ) / (parm.Pr * parm.Le);

        fy(i,j,k,URHOY_F) = rhoD * dYdy_fuel;
        fy(i,j,k,URHOY_O) = rhoD * dYdy_oxid;
    }

}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_phys_diff_eb_z (int i, int j, int k,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& uo,
               amrex::Array4<amrex::Real const> const& coeffs,
               amrex::Array4<amrex::EBCellFlag const> const& flag,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
               amrex::GpuArray<amrex::Real,3> const& weights,
               amrex::Array4<amrex::Real> const& fz,
               int do_reac, Parm const& parm) noexcept
{
    using amrex::Real;

    Real  dTdz = (q(i,j,k,QTEMP)-q(i,j,k-1,QTEMP))*dxinv[2];
    Real  dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real  dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real  dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];

    int  ihip = i + (flag(i,j,k  ).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j,k  ).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j,k-1).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j,k-1).isConnected(-1,0,0) ? 1 : 0 );

    Real   whi = weights[ihip-ihim];
    Real   wlo = weights[ilop-ilom];
    Real   dudx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QU)-q(ihim,j,k  ,QU))*whi
                                   +(q(ilop,j,k-1,QU)-q(ilom,j,k-1,QU))*wlo) ;
    Real  dwdx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QW)-q(ihim,j,k  ,QW))*whi
                                   +(q(ilop,j,k-1,QW)-q(ilom,j,k-1,QW))*wlo) ;


    int  jhip = j + (flag(i,j,k  ).isConnected(0, 1,0) ? 1 : 0 );
    int  jhim = j - (flag(i,j,k  ).isConnected(0,-1,0) ? 1 : 0 );
    int  jlop = j + (flag(i,j,k-1).isConnected(0, 1,0) ? 1 : 0 );
    int  jlom = j - (flag(i,j,k-1).isConnected(0,-1,0) ? 1 : 0 );

    whi = weights[jhip-jhim];
    wlo = weights[jlop-jlom];
    Real   dvdy = (0.5*dxinv[1]) * ((q(i,jhip,k  ,QV)-q(i,jhim,k  ,QV))*whi
                                   +(q(i,jlop,k-1,QV)-q(i,jlom,k-1,QV))*wlo);
    Real  dwdy = (0.5*dxinv[1]) *  ((q(i,jhip,k  ,QW)-q(i,jhim,k  ,QW))*whi
                                   +(q(i,jlop,k-1,QW)-q(i,jlom,k-1,QW))*wlo);

    Real  divu = dudx + dvdy + dwdz;
    Real  muf = 0.5*(coeffs(i,j,k,0)+coeffs(i,j,k-1,0));
    // Real  etaf = 0.5*(coeffs(i,j,k,CETA)+coeffs(i,j,k-1,CETA));
    // Real  xif  = 0.5*(coeffs(i,j,k,CXI)+coeffs(i,j,k-1,CXI));
    Real  tauxz = muf*(dudz+dwdx);
    Real  tauyz = muf*(dvdz+dwdy);
    Real  tauzz = muf*(2.0*dwdz-(2.0/3.0)*divu);

    fz(i,j,k,UMX)   = tauxz;
    fz(i,j,k,UMY)   = tauyz;
    fz(i,j,k,UMZ)   = tauzz;
    fz(i,j,k,UEDEN) = 0.5*( (q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz
                            +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz
                            +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz
                            +(coeffs(i,j,k,0) +coeffs(i,j,k-1,0)*parm.cp/parm.Pr)*dTdz );

    if(do_reac == 1){

        Real dYdz_fuel = dxinv[2]*( (uo(i,j,k,URHOY_F)/uo(i,j,k,URHO)) - (uo(i,j,k-1,URHOY_F)/uo(i,j,k-1,URHO)) );

        Real dYdz_oxid = dxinv[2]*( (uo(i,j,k,URHOY_O)/uo(i,j,k,URHO)) - (uo(i,j,k-1,URHOY_O)/uo(i,j,k-1,URHO)) );

        Real rhoD = 0.5*( coeffs(i,j,k,0) + coeffs(i,j,k-1,0) ) / (parm.Pr * parm.Le);

        fz(i,j,k,URHOY_F) = rhoD * dYdz_fuel;
        fz(i,j,k,URHOY_O) = rhoD * dYdz_oxid;
    }
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real
fct_compute_interp1d (amrex::Real cym,amrex::Real cy0,amrex::Real cyp, Array1D<Real, 1,3>& v)
{
    Real x = cym*v(1) + cy0*v(2) + cyp*v(3);
    return x;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real
fct_compute_interp2d(amrex::Real cym,amrex::Real cy0,amrex::Real cyp,
                 amrex::Real czm,amrex::Real cz0,amrex::Real czp, Array2D<Real, 1,3,1,3>& v)
{
    Real x = czm*(cym*v(1,1) + cy0*v(2,1) + cyp*v(3,1)) + cz0*(cym*v(1,2) + cy0*v(2,2) + cyp*v(3,2))
            +czp*(cym*v(1,3) + cy0*v(2,3) + cyp*v(3,3));
    return x;
}

#if (AMREX_SPACEDIM == 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_compute_diff_wallflux ( int i, int j, int k,
                        amrex::Array4<amrex::Real const> const& q,
                        amrex::Array4<amrex::Real const> const& coefs,
                        amrex::Array4<amrex::Real const> const& bcent,
                        const amrex::Real axm, const amrex::Real axp,
                        const amrex::Real aym, const amrex::Real ayp,
                        amrex::GpuArray<amrex::Real,NEQNS>& viscw,
                        Parm const& parm) noexcept
{
    //  This implementation assumes adiabatic walls

    using amrex::Real;

    for (int n = 0; n<NEQNS; n++) viscw[n]=0.;

    // dapx*ihat + dapy*jhat : vector pointing away from wall (outward normal to wall) 
    Real dapx = axp - axm;
    Real dapy = ayp - aym;


    Real apnorm = std::sqrt(dapx*dapx+dapy*dapy);

    if (apnorm == 0.0) {
        amrex::Abort("compute_diff_wallflux: we are in trouble.");
    }

    Real apnorminv = 1.0/apnorm;
    Real anrmx = -dapx * apnorminv; // unit vector pointing toward the wall
    Real anrmy = -dapy * apnorminv;

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bct;

    // The center of the wall
    bct[0] = bcent(i,j,k,0);
    bct[1] = bcent(i,j,k,1);

    Real u1,v1,u2,v2,d1,d2;

    if (std::abs(anrmx) >= std::abs(anrmy))
    {
       // y line: x = const
       // the equation for the line:  x = bct(0) - d*anrmx
       //                             y = bct(1) - d*anrmy
       Real s = -anrmx > 0 ? 1. : -1. ;
       int is = static_cast<int>(round(s));

       //
       // the line intersects the y line (x = s) at ...
       //
       d1 = (bct[0] - s) * (1.0/anrmx);  // this is also the distance from wall to intersection
       Real yit = bct[1] - d1*anrmy;
       int iyit = j + static_cast<int>(round(yit));
       yit = yit - static_cast<int>(round(yit));   // shift so that the center of the nine cells are (0.,0.)

       // coefficients for quadratic interpolation

       Real cym = 0.50*yit*(yit-1.0);
       Real cy0 = 1.0-yit*yit;
       Real cyp = 0.50*yit*(yit+1.0);

       // interpolation

       Array1D<Real,1,3> v;

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(i+is,iyit-2+ii,k,QU);
       }

       u1 = fct_compute_interp1d(cym,cy0,cyp, v);

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(i+is,iyit-2+ii,k,QV);
       }

       v1 = fct_compute_interp1d(cym,cy0,cyp, v);

       //
       // the line intersects the y-z plane (x = 2*s) at ...
       //

       d2 = (bct[0] - 2.0*s) * (1.0/anrmx);
       yit = bct[1] - d2*anrmy;
       iyit = j + static_cast<int>(round(yit)) ;
       yit = yit - static_cast<int>(round(yit));  // shift so that the center of the nine cells are (0.,0.)

       // coefficients for quadratic interpolation

       cym = 0.5*yit*(yit-1.);
       cy0 = 1.-yit*yit;
       cyp = 0.5*yit*(yit+1.);

       // interpolation

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(i+2*is,iyit-2+ii,k,QU);
       }


       u2 = fct_compute_interp1d(cym,cy0,cyp, v);

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(i+2*is,iyit-2+ii,k,QV);
       }

       v2 = fct_compute_interp1d(cym,cy0,cyp, v);

    } else {

       // x line
       Real s = -anrmy > 0. ? 1. : -1.;
       int is = static_cast<int>(round(s));

       d1 = (bct[1] - s) / anrmy;
       Real xit = bct[0] - d1*anrmx;

       int ixit = i + static_cast<int>(round(xit));
       xit = xit - static_cast<int>(round(xit));

       Real cxm = 0.5*xit*(xit-1.);
       Real cx0 = 1.-xit*xit;
       Real cxp = 0.5*xit*(xit+1.);

       Array1D<Real,1,3> v;

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(ixit-2+ii,j+is,k,QU);
       }

       u1 = fct_compute_interp1d(cxm,cx0,cxp, v);

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(ixit-2+ii,j+is,k,QV);
       }

       v1 = fct_compute_interp1d(cxm,cx0,cxp, v);

       d2 = (bct[1] - 2.0*s) * (1.0/anrmy);
       xit = bct[0] - d2*anrmx;
       ixit = i + static_cast<int>(round(xit));
       xit = xit - static_cast<int>(round(xit));

       cxm = 0.5*xit*(xit-1.);
       cx0 = 1.-xit*xit;
       cxp = 0.5*xit*(xit+1.);

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(ixit-2+ii,j+2*is,k,QU);
       }


       u2 = fct_compute_interp1d(cxm,cx0,cxp, v);

       for (int ii = 1; ii<4; ii++){
            v(ii) = q(ixit-2+ii,j+2*is,k,QV);
       }

       v2 = fct_compute_interp1d(cxm,cx0,cxp, v);


    }


    // compute derivatives on the wall given that velocity is zero on the wall.

    Real ddinv = 1./(d1*d2*(d2-d1));
    Real dudn = -ddinv*(d2*d2*u1-d1*d1*u2);  // note that the normal vector points toward the wall
    Real dvdn = -ddinv*(d2*d2*v1-d1*d1*v2);

    // transform them to d/dx, d/dy and d/dz given transverse derivatives are zero

    Real dudx = dudn * anrmx;
    Real dudy = dudn * anrmy;

    Real dvdx = dvdn * anrmx;
    Real dvdy = dvdn * anrmy;

    Real divu = dudx+dvdy;

    Real mu = coefs(i,j,k,0);
    if(parm.eb_wallloss == 1){
        Real Tl = q(i,j,k,QTEMP);
        Real kf = parm.kappa_0 * parm.cp * std::pow(Tl, 0.7);
        Real Tw = (Tl*kf + parm.tempsolidwall*parm.ksolid) / (kf + parm.ksolid); 
        mu = parm.kappa_0 * parm.Pr * std::pow(Tw, Real(0.7));
    }

    Real tautmp = -(2.0/3.0)*mu*divu;
    Real tauxx = mu*2.*dudx + tautmp;
    Real tauyy = mu*2.*dvdy + tautmp;
    Real tauxy = mu*(dudy+dvdx);

    // assumes  dx == dy == dz
    viscw[1] = (dapx*tauxx + dapy*tauxy);
    viscw[2] = (dapx*tauxy + dapy*tauyy);
}

#else // 3d version below

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
fct_compute_diff_wallflux ( int i, int j, int k,
                        amrex::Array4<amrex::Real const> const& q,
                        amrex::Array4<amrex::Real const> const& coefs,
                        amrex::Array4<amrex::Real const> const& bcent,
                        const amrex::Real axm, const amrex::Real axp,
                        const amrex::Real aym, const amrex::Real ayp,
                        const amrex::Real azm, const amrex::Real azp,
                        amrex::GpuArray<amrex::Real,NEQNS>& viscw,
                        Parm const& parm) noexcept
{
    //  This implementation assumes adiabatic walls

    using amrex::Real;

    for (int n = 0; n<NEQNS; n++) viscw[n]=0.;

    AMREX_D_TERM(Real dapx = axp - axm;,
                 Real dapy = ayp - aym;,
                 Real dapz = azp - azm;);

    Real apnorm = std::sqrt(dapx*dapx+dapy*dapy+dapz*dapz);

    if (apnorm == 0.0) {
        amrex::Abort("compute_diff_wallflux: we are in trouble.");
    }

    Real apnorminv = 1.0/apnorm;
    AMREX_D_TERM(Real anrmx = -dapx * apnorminv;, // unit vector pointing toward the wall
                 Real anrmy = -dapy * apnorminv;,
                 Real anrmz = -dapz * apnorminv;);

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bct;

    // The center of the wall
    AMREX_D_TERM(bct[0] = bcent(i,j,k,0);,
                 bct[1] = bcent(i,j,k,1);,
                 bct[2] = bcent(i,j,k,2););

    Real u1,v1,w1,u2,v2,w2,d1,d2;

    if ( (std::abs(anrmx) >= std::abs(anrmy) ) && ( std::abs(anrmx) >= std::abs(anrmz) ))
    {
       // y-z plane: x = const
       // the equation for the line:  x = bct(0) - d*anrmx
       //                             y = bct(1) - d*anrmy
       //                             z = bct(2) - d*anrmz
       Real s = -anrmx > 0 ? 1. : -1. ;
       int is = static_cast<int>(round(s));

       //
       // the line intersects the y-z plane (x = s) at ...
       //
       d1 = (bct[0] - s) * (1.0/anrmx);  // this is also the distance from wall to intersection
       Real yit = bct[1] - d1*anrmy;
       int iyit = j + static_cast<int>(round(yit));
       yit = yit - static_cast<int>(round(yit));   // shift so that the center of the nine cells are (0.,0.)

       Real zit = bct[2] - d1*anrmz;
       int izit = k + static_cast<int>(round(zit));
       zit = zit - static_cast<int>(round(zit));

       // coefficients for quadratic interpolation

       Real cym = 0.50*yit*(yit-1.0);
       Real cy0 = 1.0-yit*yit;
       Real cyp = 0.50*yit*(yit+1.0);

       Real czm = 0.50*zit*(zit-1.0);
       Real cz0 = 1.0-zit*zit;
       Real czp = 0.50*zit*(zit+1.0);

       // interpolation

       Array2D<Real,1,3,1,3> v;

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(i+is,iyit-2+ii,izit-2+jj,QU);
       }

       u1 = fct_compute_interp2d(cym,cy0,cyp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(i+is,iyit-2+ii,izit-2+jj,QV);
       }

       v1 = fct_compute_interp2d(cym,cy0,cyp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(i+is,iyit-2+ii,izit-2+jj,QW);
       }

       w1 = fct_compute_interp2d(cym,cy0,cyp,czm,cz0,czp, v);

       //
       // the line intersects the y-z plane (x = 2*s) at ...
       //

       d2 = (bct[0] - 2.0*s) * (1.0/anrmx);
       yit = bct[1] - d2*anrmy;
       zit = bct[2] - d2*anrmz;
       iyit = j + static_cast<int>(round(yit));
       izit = k + static_cast<int>(round(zit));
       yit = yit - static_cast<int>(round(yit));  // shift so that the center of the nine cells are (0.,0.)
       zit = zit - static_cast<int>(round(zit));

       // coefficients for quadratic interpolation

       cym = 0.5*yit*(yit-1.);
       cy0 = 1.-yit*yit;
       cyp = 0.5*yit*(yit+1.);
       czm = 0.5*zit*(zit-1.);
       cz0 = 1.-zit*zit;
       czp = 0.5*zit*(zit+1.);

       // interpolation

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(i+2*is,iyit-2+ii,izit-2+jj,QU);
       }


       u2 = fct_compute_interp2d(cym,cy0,cyp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(i+2*is,iyit-2+ii,izit-2+jj,QV);
       }

       v2 = fct_compute_interp2d(cym,cy0,cyp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(i+2*is,iyit-2+ii,izit-2+jj,QW);
       }

       w2 = fct_compute_interp2d(cym,cy0,cyp,czm,cz0,czp, v);

    } else if  (std::abs(anrmy) > std::abs(anrmx) && std::abs(anrmy) > std::abs(anrmz))
    {
       // z-x plane
       Real s = -anrmy > 0. ? 1. : -1.;
       int is = static_cast<int>(round(s));

       AMREX_D_TERM(d1 = (bct[1] - s) / anrmy;,
                    Real xit = bct[0] - d1*anrmx;,
                    Real zit = bct[2] - d1*anrmz;);
       int ixit = i + static_cast<int>(round(xit));
       xit = xit - static_cast<int>(round(xit));

       Real cxm = 0.5*xit*(xit-1.);
       Real cx0 = 1.-xit*xit;
       Real cxp = 0.5*xit*(xit+1.);

       int izit = k + static_cast<int>(round(zit));
       zit = zit - static_cast<int>(round(zit));
       Real czm = 0.5*zit*(zit-1.);
       Real cz0 = 1.-zit*zit;
       Real czp = 0.5*zit*(zit+1.);

       Array2D<Real,1,3,1,3> v;

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,j+is,izit-2+jj,QU);
       }

       u1 = fct_compute_interp2d(cxm,cx0,cxp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,j+is,izit-2+jj,QV);
       }

       v1 = fct_compute_interp2d(cxm,cx0,cxp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,j+is,izit-2+jj,QW);
       }

       w1 = fct_compute_interp2d(cxm,cx0,cxp,czm,cz0,czp, v);

       d2 = (bct[1] - 2.0*s) * (1.0/anrmy);
       xit = bct[0] - d2*anrmx;
       ixit = i + static_cast<int>(round(xit));
       xit = xit - static_cast<int>(round(xit));

       cxm = 0.5*xit*(xit-1.);
       cx0 = 1.-xit*xit;
       cxp = 0.5*xit*(xit+1.);

       zit = bct[2] - d2*anrmz;
       izit = k + static_cast<int>(round(zit));
       zit = zit - static_cast<int>(round(zit));
       czm = 0.5*zit*(zit-1.);
       cz0 = 1.-zit*zit;
       czp = 0.5*zit*(zit+1.);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,j+2*is,izit-2+jj,QU);
       }


       u2 = fct_compute_interp2d(cxm,cx0,cxp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,j+2*is,izit-2+jj,QV);
       }

       v2 = fct_compute_interp2d(cxm,cx0,cxp,czm,cz0,czp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,j+2*is,izit-2+jj,QW);
       }

       w2 = fct_compute_interp2d(cxm,cx0,cxp,czm,cz0,czp, v);

   } else {

       // x-y plane

       Real s = -anrmz > 0 ? 1. : -1.;
       int is = static_cast<int>(round(s));

       d1 = (bct[2] - s) * (1.0/anrmz);
       Real xit = bct[0] - d1*anrmx;
       Real yit = bct[1] - d1*anrmy;
       int ixit = i + static_cast<int>(round(xit));
       int iyit = j + static_cast<int>(round(yit));
       xit = xit - static_cast<int>(round(xit));
       yit = yit - static_cast<int>(round(yit));

       Real cxm = 0.5*xit*(xit-1.);
       Real cx0 = 1.-xit*xit;
       Real cxp = 0.5*xit*(xit+1.);
       Real cym = 0.5*yit*(yit-1.);
       Real cy0 = 1.-yit*yit;
       Real cyp = 0.5*yit*(yit+1.);

       Array2D<Real,1,3,1,3> v;

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,iyit-2+jj,k+is,QU);
       }

       u1 = fct_compute_interp2d(cxm,cx0,cxp,cym,cy0,cyp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,iyit-2+jj,k+is,QV);
       }

       v1 = fct_compute_interp2d(cxm,cx0,cxp,cym,cy0,cyp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,iyit-2+jj,k+is,QW);
       }

       w1 = fct_compute_interp2d(cxm,cx0,cxp,cym,cy0,cyp, v);

       d2 = (bct[2] - 2.*s) * (1.0/anrmz);
       xit = bct[0] - d2*anrmx;
       yit = bct[1] - d2*anrmy;
       ixit = i + static_cast<int>(round(xit));
       iyit = j + static_cast<int>(round(yit));
       xit = xit - static_cast<int>(round(xit));
       yit = yit - static_cast<int>(round(yit));

       cxm = 0.5*xit*(xit-1.);
       cx0 = 1.-xit*xit;
       cxp = 0.5*xit*(xit+1.);
       cym = 0.5*yit*(yit-1.);
       cy0 = 1.-yit*yit;
       cyp = 0.5*yit*(yit+1.);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,iyit-2+jj,k+2*is,QU);
       }


       u2 = fct_compute_interp2d(cxm,cx0,cxp,cym,cy0,cyp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,iyit-2+jj,k+2*is,QV);
       }

       v2 = fct_compute_interp2d(cxm,cx0,cxp,cym,cy0,cyp, v);

       for (int ii = 1; ii<4; ii++){
          for (int jj = 1; jj<4; jj++)
            v(ii,jj) = q(ixit-2+ii,iyit-2+jj,k+2*is,QW);
       }

       w2 =compute_interp2d(cxm,cx0,cxp,cym,cy0,cyp, v);
    }


    // compute derivatives on the wall given that velocity is zero on the wall.

    Real ddinv = 1./(d1*d2*(d2-d1));
    AMREX_D_TERM(Real dudn = -ddinv*(d2*d2*u1-d1*d1*u2);,  // note that the normal vector points toward the wall
                 Real dvdn = -ddinv*(d2*d2*v1-d1*d1*v2);,
                 Real dwdn = -ddinv*(d2*d2*w1-d1*d1*w2););

    // transform them to d/dx, d/dy and d/dz given transverse derivatives are zero

    Real dudx = dudn * anrmx;
    Real dvdx = dvdn * anrmx;
    Real dwdx = dwdn * anrmx;

    Real dudy = dudn * anrmy;
    Real dvdy = dvdn * anrmy;
    Real dwdy = dwdn * anrmy;

    Real dudz = dudn * anrmz;
    Real dvdz = dvdn * anrmz;
    Real dwdz = dwdn * anrmz;

    Real divu = dudx+dvdy+dwdz;
    
    Real mu = coefs(i,j,k,0);
    if(parm.eb_wallloss == 1){
        Real Tl = q(i,j,k,QTEMP);
        Real kf = parm.kappa_0 * parm.cp * std::pow(Tl, 0.7);
        Real Tw = (Tl*kf + parm.tempsolidwall*parm.ksolid) / (kf + parm.ksolid); 
        mu = parm.kappa_0 * parm.Pr * std::pow(Tw, Real(0.7));
    }

    Real tautmp = -(2.0/3.0)*mu*divu;
    Real tauxx = mu*2.*dudx + tautmp;
    Real tauyy = mu*2.*dvdy + tautmp;
    Real tauxy = mu*(dudy+dvdx);

    Real tauzz = mu*2.*dwdz + tautmp;
    Real tauxz = mu*(dudz+dwdx);
    Real tauyz = mu*(dwdy+dvdz);

    // assumes  dx == dy == dz
    viscw[1] = (dapx*tauxx + dapy*tauxy + dapz*tauxz);
    viscw[2] = (dapx*tauxy + dapy*tauyy + dapz*tauyz);
    viscw[3] = (dapx*tauxz + dapy*tauyz + dapz*tauzz);
}
#endif
#endif
