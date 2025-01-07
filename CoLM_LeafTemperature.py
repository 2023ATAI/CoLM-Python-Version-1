import numpy as np
import sys
from CoLM_Qsadv import qsadv
from CoLM_CanopyLayerProfile import cal_z0_displa, ueffect,frd
import CoLM_FrictionVelocity
import CoLM_TurbulenceLEddy
import CoLM_PlantHydraulic
import CoLM_AssimStomataConductance
import  CoLM_OrbCoszen
def dewfraction(sigf, lai, sai, dewmx, ldew, ldew_rain, ldew_snow):
    """
    Determine fraction of foliage covered by water and fraction of foliage that is dry and transpiring.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         : determine fraction of foliage covered by water and
                        ! fraction of foliage that is dry and transpiring
    """
    lsai = lai + sai
    dewmxi = 1.0 / dewmx
    vegt = lsai
    fwet = 0.0

    if ldew > 0.0:
        fwet = ((dewmxi / vegt) * ldew) ** 0.666666666666
        # Check for maximum limit of fwet
        fwet = min(fwet, 1.0)

    fdry = (1.0 - fwet) * lai / lsai

    return fwet, fdry

def LeafTemperature(co_lm, var_global, const_lc, const_physical, patchclass,
                    ipatch    ,ivt       ,deltim    ,csoilc    ,dewmx     ,htvp      ,
                    lai       ,sai       ,htop      ,hbot      ,sqrtdi    ,effcon    ,
                    vmax25    ,slti      ,hlti      ,shti      ,hhti      ,trda      ,
                    trdm      ,trop      ,g1        ,g0        ,gradm     ,binter    ,
                    extkn     ,extkb     ,extkd     ,hu        ,ht        ,hq        ,
                    us        ,vs        ,thm       ,th        ,thv       ,qm        ,
                    psrf      ,rhoair    ,parsun    ,parsha    ,sabv      ,frl       ,
                    fsun      ,thermk    ,rstfacsun ,rstfacsha ,gssun     ,gssha     ,
                    po2m      ,pco2m     ,z0h_g     ,obug      ,ustarg    ,zlnd      ,
                    zsno      ,fsno      ,sigf      ,etrc      ,tg        ,qg,rss    ,
                    t_soil    ,t_snow    ,q_soil    ,q_snow    ,dqgdT     ,emg       ,
                    tl        ,ldew      ,ldew_rain ,ldew_snow ,taux      ,tauy      ,
                    fseng     ,fseng_soil,fseng_snow,fevpg     ,fevpg_soil,fevpg_snow,
                    cgrnd     ,cgrndl    ,cgrnds    ,tref      ,qref      ,rst       ,
                    assim     ,respc     ,fsenl     ,fevpl     ,etr       ,dlrad     ,
                    ulrad     ,z0m       ,zol       ,rib       ,ustar     ,qstar     ,
                    tstar     ,fm        ,fh        ,fq        ,rootfr    ,
                    #!Plant Hydraulic variables
                    kmax_sun  ,kmax_sha  ,kmax_xyl  ,kmax_root ,psi50_sun ,psi50_sha ,
                    psi50_xyl ,psi50_root,ck        ,vegwp     ,gs0sun    ,gs0sha    ,
                    assimsun  ,etrsun    ,assimsha  ,etrsha    ,
                    #!Ozone stress variables
                    o3coefv_sun      ,o3coefv_sha      ,o3coefg_sun     ,o3coefg_sha ,
                    lai_old          ,o3uptakesun      ,o3uptakesha     ,forc_ozone  ,
                    #!End ozone stress variables
                    hpbl      ,
                    qintr_rain,qintr_snow,t_precip  ,hprl      ,smp       ,hk        ,
                    hksati    ,rootflux                                               ):

    # !=======================================================================
    # ! !DESCRIPTION:
    # ! Foliage energy conservation is given by foliage energy budget equation
    # !                      Rnet - Hf - LEf = 0
    # ! The equation is solved by Newton-Raphson iteration, in which this iteration
    # ! includes the calculation of the photosynthesis and stomatal resistance, and the
    # ! integration of turbulent flux profiles. The sensible and latent heat
    # ! transfer between foliage and atmosphere and ground is linked by the equations:
    # !                      Ha = Hf + Hg and Ea = Ef + Eg
    # !
    # ! Original author : Yongjiu Dai, August 15, 2001
    # !
    # ! REVISIONS:
    # ! Hua Yuan, 09/2014: imbalanced energy due to T/q adjustment is
    # !                    allocated to sensible heat flux.
    # !
    # ! Hua Yuan, 10/2017: added options for z0, displa, rb and rd calculation
    # !                    (Dai, Y., Yuan, H., Xin, Q., Wang, D., Shangguan, W.,
    # !                    Zhang, S., et al. (2019). Different representations of
    # !                    canopy structure—A large source of uncertainty in global
    # !                    land surface modeling. Agricultural and Forest Meteorology,
    # !                    269–270, 119–135. https://doi.org/10.1016/j.agrformet.2019.02.006
    # !
    # ! Hua Yuan, 10/2019: change only the leaf tempertature from two-leaf to one-leaf
    # !                    (due to large differences may exist btween sunlit/shaded
    # !                    leaf temperature.
    # !
    # ! Xingjie Lu and Nan Wei, 01/2021: added plant hydraulic process interface
    # !
    # ! Nan Wei,  01/2021: added interaction btw prec and canopy
    # !
    # ! Shaofeng Liu, 05/2023: add option to call moninobuk_leddy, the LargeEddy
    # !                        surface turbulence scheme (LZD2022);
    # !                        make a proper update of um.
    # !=======================================================================

    itmax  = 40   #maximum number of iteration
    itmin  = 6    #minimum number of iteration
    delmax = 3.0  #maximum change in leaf temperature [K]
    dtmin  = 0.01 #max limit for temperature convergence [K]
    dlemin = 0.1  #max limit for energy flux convergence [w/m2]
    
#-----------------------Local Variables---------------------------------
    dtl = np.zeros(itmax+1)

    displa = 0.0     # displacement height [m]
    zldis = 0.0      # reference height "minus" zero displacement heght [m]
    zii = 0.0        # convective boundary layer height [m]
    z0mv = 0.0       # roughness length = 0.0 momentum [m]
    z0hv = 0.0       # roughness length = 0.0 sensible heat [m]
    z0qv = 0.0       # roughness length = 0.0 latent heat [m]
    zeta = 0.0       # dimensionless height used in Monin-Obukhov theory
    beta = 0.0       # coefficient of conective velocity [-]
    wc = 0.0         # convective velocity [m/s]
    wc2 = 0.0        # wc**2
    dth = 0.0        # diff of virtual temp. between ref. height and surface
    dthv = 0.0       # diff of vir. poten. temp. between ref. height and surface
    dqh = 0.0        # diff of humidity between ref. height and surface
    obu = 0.0        # monin-obukhov length (m)
    um = 0.0         # wind speed including the stablity effect [m/s]
    ur = 0.0         # wind speed at reference height [m/s]
    uaf = 0.0        # velocity of air within foliage [m/s]
    fh2m = 0.0       # relation for temperature at 2m
    fq2m = 0.0       # relation for specific humidity at 2m
    fm10m = 0.0      # integral of profile function for momentum at 10m
    thvstar = 0.0    # virtual potential temperature scaling parameter
    taf = 0.0        # air temperature within canopy space [K]
    qaf = 0.0        # humidity of canopy air [kg/kg]
    eah = 0.0        # canopy air vapor pressure (pa)
    pco2g = 0.0      # co2 pressure (pa) at ground surface (pa)
    pco2a = 0.0      # canopy air co2 pressure (pa)

    fdry = 0.0       # fraction of foliage that is green and dry [-]
    fwet = 0.0       # fraction of foliage covered by water [-]
    cf = 0.0         # heat transfer coefficient from leaves [-]
    rb = 0.0         # leaf boundary layer resistance [s/m]
    rbsun = 0.0      # Sunlit leaf boundary layer resistance [s/m]
    rbsha = 0.0      # Shaded leaf boundary layer resistance [s/m]
    rd = 0.0         # aerodynamical resistance between ground and canopy air
    ram = 0.0        # aerodynamical resistance [s/m]
    rah = 0.0        # thermal resistance [s/m]
    raw = 0.0        # moisture resistance [s/m]
    clai = 0.0       # canopy heat capacity [Jm-2K-1]
    cah = 0.0        # heat conductance for air [m/s]
    cgh = 0.0        # heat conductance for ground [m/s]
    cfh = 0.0        # heat conductance for leaf [m/s]
    caw = 0.0        # latent heat conductance for air [m/s]
    cgw = 0.0        # latent heat conductance for ground [m/s]
    cfw = 0.0        # latent heat conductance for leaf [m/s]
    wtshi = 0.0      # sensible heat resistance for air = 0.0 grd and leaf [-]
    wtsqi = 0.0      # latent heat resistance for air = 0.0 grd and leaf [-]
    wta0 = 0.0       # normalized heat conductance for air [-]
    wtg0 = 0.0       # normalized heat conductance for ground [-]
    wtl0 = 0.0       # normalized heat conductance for air and leaf [-]
    wtaq0 = 0.0      # normalized latent heat conductance for air [-]
    wtgq0 = 0.0      # normalized heat conductance for ground [-]
    wtlq0 = 0.0      # normalized latent heat cond. for air and leaf [-]

    ei = 0.0         # vapor pressure on leaf surface [pa]
    deidT = 0.0      # derivative of "ei" on "tl" [pa/K]
    qsatl = 0.0      # leaf specific humidity [kg/kg]
    qsatldT = 0.0    # derivative of "qsatl" on "tlef"

    del1 = 0.0        # absolute change in leaf temp in current iteration [K]
    del2 = 0.0       # change in leaf temperature in previous iteration [K]
    dele = 0.0       # change in heat fluxes from leaf [W/m2]
    dele2 = 0.0      # change in heat fluxes from leaf in previous iteration [W/m2]
    det = 0.0        # maximum leaf temp. change in two consecutive iter [K]
    dee = 0.0        # maximum leaf heat fluxes change in two consecutive iter [W/m2]

    obuold = 0.0     # monin-obukhov length from previous iteration
    tlbef = 0.0      # leaf temperature from previous iteration [K]
    ecidif = 0.0     # excess energies [W/m2]
    err = 0.0        # balance error

    rssun = 0.0      # sunlit leaf stomatal resistance [s/m]
    rssha = 0.0      # shaded leaf stomatal resistance [s/m]
    fsha = 0.0       # shaded fraction of canopy
    laisun = 0.0     # sunlit leaf area index = 0.0 one-sided
    laisha = 0.0     # shaded leaf area index = 0.0 one-sided
    respcsun = 0.0   # sunlit leaf respiration rate [umol co2 /m**2/ s] [+]
    respcsha = 0.0   # shaded leaf respiration rate [umol co2 /m**2/ s] [+]
    rsoil = 0.0      # soil respiration
    gah2o = 0.0      # conductance between canopy and atmosphere
    gdh2o = 0.0      # conductance between canopy and ground
    tprcor = 0.0        # tf*psur*100./1.013e5


    # Define iteration parameters
    itmax   = 40  # maximum number of iterations
    itmin   = 6  # minimum number of iterations
    delmax  = 3.0  # maximum change in leaf temperature [K]
    dtmin   = 0.01  # max limit for temperature convergence [K]
    dlemin  = 0.1  # max limit for energy flux convergence [W/m2]

    k_soil_root = np.zeros(var_global.nl_soil, dtype=np.float64)
    k_ax_root = np.zeros(var_global.nl_soil, dtype=np.float64)

    zd_opt = 3
    rb_opt = 3
    rd_opt = 3

    # !======================End Variable List==================================
    # initialization of errors and iteration parameters
    it = 1  # counter for leaf temperature iteration
    del_ = 0.0  # change in leaf temperature from previous iteration
    dele = 0.0  # latent heat flux from leaf for previous iteration

    dtl = np.zeros(itmax + 2)  # difference of tl between two iterative steps
    fevpl_bef = 0.0  # previous value of evaporation+transpiration from leaves [mm/s]

    fht = 0.0  # integral of profile function for heat
    fqt = 0.0  # integral of profile function for moisture

    # scaling-up coefficients from leaf to canopy
    fsha = 1.0 - fsun
    laisun = lai * fsun
    laisha = lai * fsha

    # scaling-up coefficients from leaf to canopy
    cintsun = [(1.0 - np.exp(-(0.110 + extkb) * lai)) / (0.110 + extkb),
               (1.0 - np.exp(-(extkb + extkd) * lai)) / (extkb + extkd),
               (1.0 - np.exp(-extkb * lai)) / extkb]

    cintsha = [(1.0 - np.exp(-0.110 * lai)) / 0.110 - cintsun[0],
               (1.0 - np.exp(-extkd * lai)) / extkd - cintsun[1],
               lai - cintsun[2]]

    # !=======================================================================
    # get fraction of wet and dry canopy surface (fwet & fdry)
    # initial saturated vapor pressure and humidity and their derivation
    # !=======================================================================
    clai = 0.0

    fwet, fdry = dewfraction(sigf, lai, sai, dewmx, ldew, ldew_rain, ldew_snow)

    ei, deiDT, qsatl, qsatlDT = qsadv(tl, psrf)
    # !=======================================================================
    # initial for fluxes profile
    # !=======================================================================
    nmozsgn = 0  # number of times moz changes sign
    obuold = 0.0  # monin-obukhov length from previous iteration
    zii = 1000.0  # m (pbl height)
    beta = 1.0  # - (in computing W_*)
    z0mg = (1.0 - fsno) * zlnd + fsno * zsno
    z0hg = z0mg
    z0qg = z0mg

    z0m = htop * const_lc.z0mr[patchclass]
    displa = htop * const_lc.displar[patchclass]

    z0mv = z0m
    z0hv = z0m
    z0qv = z0m

    # Modify aerodynamic parameters for sparse/dense canopy (X. Zeng)
    lt = min(lai + sai, 2.0)
    egvf = (1.0 - np.exp(-lt)) / (1.0 - np.exp(-2.0))
    displa *= egvf
    z0mv = np.exp(egvf * np.log(z0mv) + (1.0 - egvf) * np.log(z0mg))
    z0hv = z0mv
    z0qv = z0mv

    # 10/17/2017, yuan: 3D z0m and displa
    if zd_opt == 3:
        z0mv, displa = cal_z0_displa(lai + sai, htop, 1.0,const_physical.vonkar)
        # NOTE: adjusted for samll displa
        displasink = max(htop / 2.0, displa)
        hsink = z0mv + displasink
        z0hv = z0mv
        z0qv = z0mv

    fai = 1.0 - np.exp(-0.5 * (lai + sai))
    sqrtdragc = min((0.003 + 0.3 * fai) ** 0.5, 0.3)
    a_k71 = htop / (htop - displa) / (const_physical.vonkar / sqrtdragc)
    taf = 0.5 * (tg + thm)
    qaf = 0.5 * (qm + qg)
    pco2a = pco2m
    tprcor = 44.6 * 273.16 * psrf / 1.013e5
    rsoil = 0.0  # respiration (mol m-2 s-1)
    rsoil = 0.22 * 1.e-6

    ur = max(0.1, np.sqrt(us * us + vs * vs))  # limit set to 0.1
    dth = thm - taf  # θ*:dth;  q*:dqh
    dqh = qm - qaf
    dthv = dth * (1.0 + 0.61 * qm) + 0.61 * th * dqh
    zldis = hu - displa

    if zldis <= 0.0:
        print('the obs height of u less than the zero displacement height')
        sys.exit()

    um, obu = CoLM_FrictionVelocity.moninobukini(ur, th, thm, thv, dth, dqh, dthv, zldis, z0mv,const_physical.grav)

    # !=======================================================================
    # BEGIN stability iteration
    # !=======================================================================
    while it <= itmax:
        tlbef = tl
        del2 =  del_
        dele2 = dele

    # !=======================================================================
    # Aerodynamical resistances
    # !=======================================================================
    # Evaluate stability-dependent variables using moz from prior iteration
        if rd_opt == 3:
            if co_lm['DEF_USE_CBL_HEIGHT']:
                ustar,fh2m,fq2m,htop,fmtop,fm,fh,fq,fht,fqt,phih =\
                    CoLM_TurbulenceLEddy.moninobukm_leddy(hu, ht, hq, displa, z0mv, z0hv, z0qv, obu, um, displasink, z0mv, hpbl)
            else:
                ustar, fh2m, fq2m, fmtop, fm, fh, fq, fht, fqt, phih=CoLM_FrictionVelocity.moninobukm(const_physical, hu, ht, hq, displa, z0mv, z0hv, z0qv, obu, um, displasink, z0mv, htop)

            # Aerodynamic resistance
            ram = 1.0 / (ustar * ustar / um)
            rah = 1.0 / (const_physical.vonkar / (fh - fht) * ustar)
            raw = 1.0 / (const_physical.vonkar / (fq - fqt) * ustar)
        else:
            if co_lm['DEF_USE_CBL_HEIGHT']:
                ustar,fh2m,fq2m,fm10m,fm,fh,fq = (
                    CoLM_TurbulenceLEddy.moninobuk_leddy(hu, ht, hq, displa, z0mv, z0hv, z0qv, obu, um, hpbl,const_physical.vonkar))
            else:
                ustar, fh2m, fq2m, fm10m, fm, fh, fq = (
                    CoLM_FrictionVelocity.moninobuk(hu, ht, hq, displa, z0mv, z0hv, z0qv, obu, um))

            # Aerodynamic resistance
            ram = 1.0 / (ustar * ustar / um)
            rah = 1.0 / (const_physical.vonkar / fh * ustar)
            raw = 1.0 / (const_physical.vonkar / fq * ustar)

        z0hg = z0mg / np.exp(0.13 * (ustar * z0mg / 1.5e-5) ** 0.45)
        z0qg = z0hg

        # Bulk boundary layer resistance of leaves
        uaf = ustar
        cf = 0.01 * sqrtdi / np.sqrt(uaf)
        rb = 1 / (cf * uaf)

        # 3D rb calculation
        if rb_opt == 3:
            utop = ustar / const_physical.vonkar * fmtop
            ueff = ueffect(utop, htop, z0mg, z0mg, a_k71, 1.0, 1.0)
            cf = 0.01 * sqrtdi * np.sqrt(ueff)
            rb = 1.0 / cf

        # Calculate rd using modified formula
        w = np.exp(-(lai + sai))
        csoilcn = (const_physical.vonkar / (0.13 * (z0mg * uaf / 1.5e-5) ** 0.45)) * w + csoilc * (1.0 - w)
        rd = 1.0 / (csoilcn * uaf)

        # 3D rd calculation
        if rd_opt == 3:
            ktop = const_physical.vonkar * (htop - displa) * ustar / phih
            rd = frd(const_physical, ktop, htop, z0qg, hsink, z0qg, displa / htop, z0qg, obug, ustar, z0mg, a_k71, 1.0, 1.0)
        # !=======================================================================
        # stomatal resistances
        # !=======================================================================
        if lai > 0.001:
            eah = qaf * psrf / (0.622 + 0.378 * qaf)  # pa

            # Check if DEF_USE_PLANTHYDRAULICS is enabled
            if co_lm['DEF_USE_PLANTHYDRAULICS']:
                rstfacsun = 1.
                rstfacsha = 1.

            # Calculate sunlit and shaded leaf boundary layer resistance
            rbsun = rb / laisun
            rbsha = rb / laisha

            # print (rssun, '----rssun1-----')
            # Sunlit leaves stomata calculation
            assimsun, respcsun, rssun= CoLM_AssimStomataConductance.stomata(co_lm, vmax25, effcon, slti, hlti, shti, hhti, trda, trdm, trop, g1, g0, gradm, binter, thm, psrf, po2m, pco2m,
                    pco2a,
                    eah, ei, tl, parsun, o3coefv_sun, o3coefg_sun, rbsun, raw, rstfacsun, cintsun, assimsun, respcsun,
                    rssun)
            # print(rssun, '----rssun2-----')
            # Shaded leaves stomata calculation
            assimsha, respcsha, rssha= CoLM_AssimStomataConductance.stomata(co_lm, vmax25, effcon, slti, hlti, shti, hhti, trda, trdm, trop, g1, g0, gradm, binter, thm, psrf, po2m, pco2m,
                    pco2a,
                    eah, ei, tl, parsha, o3coefv_sha, o3coefg_sha, rbsha, raw, rstfacsha, cintsha, assimsha, respcsha,
                    rssha)



            # If DEF_USE_PLANTHYDRAULICS is enabled, adjust conductance
            if co_lm['DEF_USE_PLANTHYDRAULICS']:
                pass
                # Calculate conductance for sunlit and shaded leaves
                # gssun = min(1.e6, 1. / (rssun * tl / tprcor)) / laisun * 1.e6
                # gssha = min(1.e6, 1. / (rssha * tl / tprcor)) / laisha * 1.e6
                # sai = max(sai, 0.1)
                # # Call PlantHydraulicStress_twoleaf subroutine
                # CoLM_PlantHydraulic.PlantHydraulicStress_twoleaf(var_global.nl_soil, var_global.nvegwcs, var_global.z_soi, var_global.dz_soi, rootfr, psrf, qsatl, qaf, tl, rb,
                #                                   rss,
                #                                   raw, rd, rstfacsun, rstfacsha, cintsun, cintsha, laisun, laisha,
                #                                   rhoair,
                #                                   fwet, sai, kmax_sun, kmax_sha, kmax_xyl, kmax_root, psi50_sun,
                #                                   psi50_sha,
                #                                   psi50_xyl, psi50_root, htop, ck, smp, hk, hksati, vegwp, etrsun,
                #                                   etrsha,
                #                                   rootflux, qg, qm, gs0sun, gs0sha, k_soil_root, k_ax_root)
                # etr = etrsun + etrsha
                # gssun = gssun * laisun
                # gssha = gssha * laisha
                #
                #
                # gs0sun = gssun / max(rstfacsun, 1.e-2)
                # gs0sha = gssha / max(rstfacsha, 1.e-2)
                #
                # # Calculate boundary layer conductance
                # gb_mol = 1. / rb * tprcor / tl * 1.e6
                # CoLM_PlantHydraulic.getvegwp_twoleaf(vegwp, var_global.nvegwcs, var_global.nl_soil, var_global.z_soi, gb_mol, gssun, gssha, qsatl, qaf, qg, qm, rhoair, psrf,
                #                  fwet,
                #                  laisun, laisha, htop, sai, tl, rss, raw, rd, smp, k_soil_root, k_ax_root, kmax_xyl,
                #                  kmax_root, rstfacsun, rstfacsha, psi50_sun, psi50_sha, psi50_xyl, psi50_root, ck,
                #                  rootflux, etrsun, etrsha)

                # Calculate stomatal resistance for sunlit and shaded leaves
                # rssun = tprcor / tl * 1.e6 / gssun
                # rssha = tprcor / tl * 1.e6 / gssha

        # If LAI is less than or equal to 0.001
        else:
            # Assign default values
            rssun = 2.e4
            assimsun = 0.
            respcsun = 0.
            rssha = 2.e4
            assimsha = 0.
            respcsha = 0.
            gssun = 0.0
            gssha = 0.0

            # If DEF_USE_PLANTHYDRAULICS is enabled, adjust values
            if co_lm['DEF_USE_PLANTHYDRAULICS']:
                etr = 0.0
                etrsun = 0.0
                etrsha = 0.0
                rootflux = 0.0
        rssun = rssun * laisun
        # print(rssun, '----rssun3-----')
        rssha = rssha * laisha
        # !=======================================================================
        # dimensional and non-dimensional sensible and latent heat conductances
        # ! for canopy and soil flux calculations.
        # !=======================================================================

        delta = 0.0
        if qsatl - qaf > 0:
            delta = 1.0

        cah = 1.0 / rah
        cgh = 1.0 / rd
        cfh = (lai + sai) / rb

        caw = 1.0 / raw
        if qg < qaf:
            cgw = 1.0 / rd  # dew case, no soil resistance
        else:
            if co_lm['DEF_RSS_SCHEME'] == 4:
                cgw = rss / rd
            else:
                cgw = 1.0 / (rd + rss)

        cfw = (1.0 - delta * (1.0 - fwet)) * (lai + sai) / rb + (1.0 - fwet) * delta * \
              (laisun / (rb + rssun) + laisha / (rb + rssha))

        wtshi = 1.0 / (cah + cgh + cfh)
        wtsqi = 1.0 / (caw + cgw + cfw)

        wta0 = cah * wtshi
        wtg0 = cgh * wtshi
        wtl0 = cfh * wtshi

        wtaq0 = caw * wtsqi
        wtgq0 = cgw * wtsqi
        wtlq0 = cfw * wtsqi

        # !=======================================================================
        # IR radiation, sensible and latent heat fluxes and their derivatives
        # !=======================================================================
        #the partial derivatives of areodynamical resistance are ignored
        # ! which cannot be determined analtically
        fac = 1. - thermk
        #  longwave absorption and their derivatives
        #           ! 10/16/2017, yuan: added reflected longwave by the ground
        if not co_lm['DEF_SPLIT_SOILSNOW']:
            irab = (frl - 2.0 * const_physical.stefnc * tl ** 4 + emg * const_physical.stefnc * tg ** 4) * fac + \
                   (1 - emg) * thermk * fac * frl + (1 - emg) * (1 - thermk) * fac * const_physical.stefnc * tl ** 4
        else:
            irab = (frl - 2.0 * const_physical.stefnc * tl ** 4 + (
                        1.0 - fsno) * emg * const_physical.stefnc * t_soil ** 4 + fsno * emg * const_physical.stefnc * t_snow ** 4) * fac + \
                   (1 - emg) * thermk * fac * frl + (1 - emg) * (1 - thermk) * fac * const_physical.stefnc * tl ** 4

        dirab_dtl = -8.0 * const_physical.stefnc * tl ** 3 * fac + 4.0 * (1 - emg) * (1 - thermk) * fac * const_physical.stefnc * tl ** 3

        fsenl = rhoair * const_physical.cpair * cfh * ((wta0 + wtg0) * tl - wta0 * thm - wtg0 * tg)
        fsenl_dtl = rhoair * const_physical.cpair * cfh * (wta0 + wtg0)
        # print (etr, '---etr1-----')
        etr = rhoair * (1 - fwet) * delta * \
              (laisun / (rb + rssun) + laisha / (rb + rssha)) * ((wtaq0 + wtgq0) * qsatl - wtaq0 * qm - wtgq0 * qg)
        # print(rhoair, fwet, delta, laisun, rb, rssun, laisha, rssha, wtaq0, wtgq0, qsatl, qm , qg, '-----------------')
        # print(etr, '---etr2-----')
        etrsun = rhoair * (1 - fwet) * delta * (laisun / (rb + rssun)) * ((
                    wtaq0 + wtgq0) * qsatl - wtaq0 * qm - wtgq0 * qg)

        etrsha = rhoair * (1 - fwet) * delta * (laisha / (rb + rssha)) * ((
                    wtaq0 + wtgq0) * qsatl - wtaq0 * qm - wtgq0 * qg)

        etr_dtl = rhoair * (1 - fwet) * delta * (
                    laisun / (rb + rssun) + laisha / (rb + rssha)) * (wtaq0 + wtgq0) * qsatlDT

        if not co_lm['DEF_USE_PLANTHYDRAULICS']:
            if etr >= etrc:
                etr = etrc
                # print(etr, '---etr3-----')
                etr_dtl = 0.0
        else:
            if rstfacsun <= 1.0e-2 or etrsun <= 1.0e-7:
                etrsun = 0.0
            if rstfacsha <= 1.0e-2 or etrsha <= 1.0e-7:
                etrsha = 0.0
            etr = etrsun + etrsha
            # print(etr, '---etr4-----')

        evplwet = rhoair * (1 - delta * (1 - fwet)) * (lai + sai) / rb * (
                    (wtaq0 + wtgq0) * qsatl - wtaq0 * qm - wtgq0 * qg)
        evplwet_dtl = rhoair * (1 - delta * (1 - fwet)) * (lai + sai) / rb * (wtaq0 + wtgq0) * qsatlDT

        if evplwet >= ldew / deltim:
            evplwet = ldew / deltim
            evplwet_dtl = 0.0

        fevpl = etr + evplwet
        fevpl_dtl = etr_dtl + evplwet_dtl

        erre = 0.0
        fevpl_noadj = fevpl
        if fevpl * fevpl_bef < 0.0:
            erre = -0.9 * fevpl
            fevpl = 0.1 * fevpl

        # !========================================================================================================
        # difference of temperatures by quasi-newton-raphson method for the non-linear system equations
        # !========================================================================================================

        dtl[it] = (sabv + irab - fsenl - const_physical.hvap * fevpl + const_physical.cpliq * qintr_rain * (t_precip - tl) + const_physical.cpice * qintr_snow * (
                    t_precip - tl)) / \
                  ((
                               lai + sai) * clai / deltim - dirab_dtl + fsenl_dtl + const_physical.hvap * fevpl_dtl + const_physical.cpliq * qintr_rain + const_physical.cpice * qintr_snow)
        dtl_noadj = dtl[it]

        # check magnitude of change in leaf temperature limit to maximum allowed value
        if it <= itmax:
            # Put brakes on large temperature excursions
            if abs(dtl[it]) > delmax:
                dtl[it] = delmax * dtl[it] / abs(dtl[it])

            # NOTE: could be a bug IF dtl*dtl==0, changed from lt->le
            if it >= 2 and dtl[it - 1] * dtl[it] <= 0:
                dtl[it] = 0.5 * (dtl[it - 1] + dtl[it])

        tl = tlbef + dtl[it]

        # !========================================================================================================
        # square roots differences of temperatures and fluxes for use as the condition of convergences
        # !========================================================================================================

        dela = np.sqrt(dtl[it] ** 2)
        dele = dtl[it] * dtl[it] * (dirab_dtl ** 2 + fsenl_dtl ** 2 + const_physical.hvap * fevpl_dtl ** 2)
        dele = np.sqrt(dele)

        # !========================================================================================================
        # saturated vapor pressures and canopy air temperature, canopy air humidity
        # !========================================================================================================
        #Recalculate leaf saturated vapor pressure (ei_)for updated leaf temperature
        # ! and adjust specific humidity (qsatl_) proportionately
        ei,deiDT,qsatl,qsatlDT = qsadv(tl, psrf)

        #update vegetation/ground surface temperature, canopy air temperature,canopy air humidity
        taf = wta0 * thm + wtg0 * tg + wtl0 * tl
        qaf = wtaq0 * qm + wtgq0 * qg + wtlq0 * qsatl

        # update co2 partial pressure within canopy air
        gah2o = 1.0 / raw * tprcor / thm  # mol m-2 s-1
        if co_lm['DEF_RSS_SCHEME'] == 4:
            gdh2o = rss / rd * tprcor / thm  # mol m-2 s-1
        else:
            gdh2o = 1.0 / (rd + rss) * tprcor / thm  # mol m-2 s-1
        pco2a = pco2m - 1.37 * psrf / max(0.446, gah2o) * (assimsun + assimsha - respcsun - respcsha - rsoil)

        # !========================================================================================================
        # Update monin-obukhov length and wind speed including the stability effect
        # !========================================================================================================
        dth = thm - taf
        dqh = qm - qaf

        tstar = const_physical.vonkar / (fh - fht) * dth
        qstar = const_physical.vonkar / (fq - fqt) * dqh

        thvstar = tstar * (1.0 + 0.61 * qm) + 0.61 * th * qstar
        zeta = zldis * const_physical.vonkar * const_physical.grav * thvstar / (ustar ** 2 * thv)
        if zeta >= 0:  # stable
            zeta = min(2.0, max(zeta, 1e-6))
        else:  # unstable
            zeta = max(-100.0, min(zeta, -1e-6))
        obu = zldis / zeta

        if zeta >= 0:
            um = max(ur, 0.1)
        else:
            if co_lm['DEF_USE_CBL_HEIGHT']:
                zii = max(5.0 * hu, hpbl)
            wc = (-const_physical.grav * ustar * thvstar * zii / thv) ** (1.0 / 3.0)
            wc2 = beta * beta * (wc * wc)
            um = np.sqrt(ur * ur + wc2)

        if obuold * obu < 0:
            nmozsgn = nmozsgn + 1
        if nmozsgn >= 4:
            obu = zldis / (-0.01)
        obuold = obu

        # !========================================================================================================
        # Test for convergence
        # !========================================================================================================
        it = it + 1

        if it > itmin:
            fevpl_bef = fevpl
            det = max(dela, del2)
            # 10/03/2017, yuan: possible bugs here, solution:
            # define dee, change del => dee
            dee = max(dele, dele2)
            if det < dtmin and dee < dlemin:
                break

    if co_lm['DEF_USE_OZONESTRESS']:
        # Call function to calculate ozone stress for sunlit leaves
        o3coefv_sun, o3coefg_sun = CoLM_OrbCoszen.CalcOzoneStress( forc_ozone, psrf, th, ram, rssun, rb, lai, lai_old, ivt, o3uptakesun,
                        deltim)

        # Call function to calculate ozone stress for shaded leaves
        o3coefv_sha, o3coefg_sha = CoLM_OrbCoszen.CalcOzoneStress( forc_ozone, psrf, th, ram, rssha, rb, lai, lai_old, ivt, o3uptakesha,
                        deltim)

        # Update the old leaf area index
        lai_old = lai
        assimsun *= o3coefv_sun
        assimsha *= o3coefv_sha
        rssun /= o3coefg_sun
        rssha /= o3coefg_sha

    # !========================================================================================================
    # END stability iteration
    # !========================================================================================================
    z0m = z0mv
    zol = zeta
    rib = min(5., zol * ustar ** 2 / (const_physical.vonkar ** 2 / fh * um ** 2))

    # canopy fluxes and total assimilation and respiration
    if lai > 0.001:
        rst = 1. / (laisun / rssun + laisha / rssha)
    else:
        rssun = 2.0e4
        rasha = 2.0e4
        assimsun = 0.
        assimsha = 0.
        respcsun = 0.
        respcsha = 0.
        rst = 2.0e4

    assim = assimsun + assimsha
    respc = respcsun + respcsha  # + rsoil

    # Canopy fluxes and total assimilation and respiration
    fsenl += fsenl_dtl * dtl[it - 1] + (dtl_noadj - dtl[it - 1]) * (((
            lai + sai) * clai / deltim) - dirab_dtl + fsenl_dtl + const_physical.hvap * fevpl_dtl + const_physical.cpliq * qintr_rain + const_physical.cpice * qintr_snow) + const_physical.hvap * erre
    etr0 = etr
    etr += etr_dtl * dtl[it - 1]
    # print(etr, '---etr5-----')

    if co_lm['DEF_USE_PLANTHYDRAULICS']:
        # TODO@yuan: rootflux may not be consistent with etr,
        # water imbalance could happen.
        if abs(etr0) >= 1.e-15:
            rootflux *= etr / etr0
        else:
            rootflux += var_global.dz_soi / sum(var_global.dz_soi) * etr_dtl * dtl[it - 1]

    evplwet += evplwet_dtl * dtl[it - 1]
    fevpl = fevpl_noadj
    fevpl += fevpl_dtl * dtl[it - 1]

    elwmax = ldew / deltim
    elwdif = max(0.0, evplwet - elwmax)
    evplwet = min(evplwet, elwmax)

    fevpl -= elwdif
    fsenl += const_physical.hvap * elwdif

    taux = -rhoair * us / ram
    tauy = -rhoair * vs / ram

    # !========================================================================================================
    # fluxes from ground to canopy space
    # !========================================================================================================
    fseng = const_physical.cpair * rhoair * cgh * (tg - taf)
    # 03/07/2020, yuan: calculate fseng_soil/snow
    # taf = wta0*thm + wtg0*tg + wtl0*tl
    fseng_soil = const_physical.cpair * rhoair * cgh * ((1.0 - wtg0) * t_soil - wta0 * thm - wtl0 * tl)
    fseng_snow = const_physical.cpair * rhoair * cgh * ((1.0 - wtg0) * t_snow - wta0 * thm - wtl0 * tl)

    # 03/07/2020, yuan: calculate fevpg_soil/snow
    # qaf = wtaq0*qm + wtgq0*qg + wtlq0*qsatl

    fevpg = rhoair * cgw * (qg - qaf)
    fevpg_soil = rhoair * cgw * ((1.0 - wtgq0) * q_soil - wtaq0 * qm - wtlq0 * qsatl)
    fevpg_snow = rhoair * cgw * ((1.0 - wtgq0) * q_snow - wtaq0 * qm - wtlq0 * qsatl)

    # !========================================================================================================
    # downward (upward) longwave radiation below (above) the canopy and prec. sensible heat
    # !========================================================================================================
    dlrad = thermk * frl + const_physical.stefnc * fac * tlbef ** 3 * (tlbef + 4.0 * dtl[it - 1])

    if not co_lm['DEF_SPLIT_SOILSNOW']:
        ulrad = const_physical.stefnc * (fac * tlbef ** 3 * (tlbef + 4.0 * dtl[it - 1]) +
                          thermk * emg * tg ** 4) + \
                (1.0 - emg) * thermk * thermk * frl + \
                (1.0 - emg) * thermk * fac * const_physical.stefnc * tlbef ** 4 + \
                4.0 * (1.0 - emg) * thermk * fac * const_physical.stefnc * tlbef ** 3 * dtl[it - 1]
    else:
        ulrad = const_physical.stefnc * (fac * tlbef ** 3 * (tlbef + 4.0 * dtl[it - 1]) +
                          (1.0 - fsno) * thermk * emg * t_soil ** 4 +
                          fsno * thermk * emg * t_snow ** 4) + \
                (1.0 - emg) * thermk * thermk * frl + \
                (1.0 - emg) * thermk * fac * const_physical.stefnc * tlbef ** 4 + \
                4.0 * (1.0 - emg) * thermk * fac * const_physical.stefnc * tlbef ** 3 * dtl[it - 1]

    hprl = const_physical.cpliq * qintr_rain * (t_precip - tl) + const_physical.cpice * qintr_snow * (t_precip - tl)

    # !========================================================================================================
    #  Derivative of soil energy flux with respect to soil temperature (cgrnd)
    # !========================================================================================================

    cgrnds = const_physical.cpair * rhoair * cgh * (1.0 - wtg0)
    cgrndl = rhoair * cgw * (1.0 - wtgq0) * dqgdT
    cgrnd = cgrnds + cgrndl * htvp

    # !========================================================================================================
    #  balance check
    # ! (the computational error was created by the assumed 'dtl' in line 406-408)
    # !========================================================================================================
    err = sabv + irab + dirab_dtl * dtl[it - 1] - fsenl - const_physical.hvap * fevpl + hprl

    # if defined CoLMDEBUG:
    if abs(err) > 0.2:
        print('energy imbalance in LeafTemperature.F90', it - 1, err, sabv, irab, fsenl, const_physical.hvap * fevpl, hprl)

    # !========================================================================================================
    #  Update dew accumulation (kg/m2)
    # !========================================================================================================
    if co_lm['DEF_Interception_scheme'] == 1:
        ldew = max(0., ldew - evplwet * deltim)
    elif co_lm['DEF_Interception_scheme'] == 2:
        ldew = max(0., ldew - evplwet * deltim)
    elif co_lm['DEF_Interception_scheme'] == 3:
        if ldew_rain > evplwet * deltim:
            ldew_rain -= evplwet * deltim
            ldew_snow = ldew_snow
            ldew = ldew_rain + ldew_snow
        else:
            ldew_rain = 0.0
            ldew_snow = max(0., ldew - evplwet * deltim)
            ldew = ldew_snow
    elif co_lm['DEF_Interception_scheme'] == 4:
        if ldew_rain > evplwet * deltim:
            ldew_rain -= evplwet * deltim
            ldew_snow = ldew_snow
            ldew = ldew_rain + ldew_snow
        else:
            ldew_rain = 0.0
            ldew_snow = max(0., ldew - evplwet * deltim)
            ldew = ldew_snow
    elif co_lm['DEF_Interception_scheme'] == 5:
        if ldew_rain > evplwet * deltim:
            ldew_rain -= evplwet * deltim
            ldew_snow = ldew_snow
            ldew = ldew_rain + ldew_snow
        else:
            ldew_rain = 0.0
            ldew_snow = max(0., ldew - evplwet * deltim)
            ldew = ldew_snow
    elif co_lm['DEF_Interception_scheme'] == 6:
        if ldew_rain > evplwet * deltim:
            ldew_rain -= evplwet * deltim
            ldew_snow = ldew_snow
            ldew = ldew_rain + ldew_snow
        else:
            ldew_rain = 0.0
            ldew_snow = max(0., ldew - evplwet * deltim)
            ldew = ldew_snow
    elif co_lm['DEF_Interception_scheme'] == 7:
        if ldew_rain > evplwet * deltim:
            ldew_rain -= evplwet * deltim
            ldew_snow = ldew_snow
            ldew = ldew_rain + ldew_snow
        else:
            ldew_rain = 0.0
            ldew_snow = max(0., ldew - evplwet * deltim)
            ldew = ldew_snow
    elif co_lm['DEF_Interception_scheme'] == 8:
        if ldew_rain > evplwet * deltim:
            ldew_rain -= evplwet * deltim
            ldew_snow = ldew_snow
            ldew = ldew_rain + ldew_snow
        else:
            ldew_rain = 0.0
            ldew_snow = max(0., ldew - evplwet * deltim)
            ldew = ldew_snow
    else:
        raise ValueError("Invalid value for DEF_Interception_scheme")

    # !========================================================================================================
    #  Update dew accumulation (kg/m2)
    # !========================================================================================================
    tref = thm + const_physical.vonkar / (fh - fht) * dth * (fh2m / const_physical.vonkar - fh / const_physical.vonkar)
    qref = qm + const_physical.vonkar / (fq - fqt) * dqh * (fq2m / const_physical.vonkar - fq / const_physical.vonkar)

    return sai, vegwp, gs0sun, gs0sha, tl, ldew, ldew_rain, ldew_snow, lai_old, o3uptakesun, o3uptakesha, forc_ozone, taux, tauy, fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, cgrnd, cgrndl, cgrnds, tref, qref, rstfacsun, rstfacsha, gssun, gssha, rootflux, assimsun, etrsun, assimsha, etrsha,  rst, assim, respc, fsenl, fevpl, etr, dlrad, ulrad, hprl, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq


