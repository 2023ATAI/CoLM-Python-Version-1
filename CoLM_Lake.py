import copy

import CoLM_Utils
import CoLM_SoilThermalParameters
import CoLM_FrictionVelocity
import CoLM_TurbulenceLEddy
import CoLM_SnowLayersCombineDivide
import CoLM_SoilSnowHydrology
import CoLM_Qsadv
import math
import numpy as np

#!-----------------------------------------------------------------------
#! DESCRIPTION:
#! Simulating energy balance processes of land water body
#!
#! REFERENCE:
#! Dai et al, 2018, The lake scheme of the common land model and its performance evaluation.
#! Chinese Science Bulletin, 63(28-29), 3002–3021, https://doi.org/10.1360/N972018-00609
#!
#! Original author: Yongjiu Dai 04/2014/
#!
#! Revisions:
#! Nan Wei,  01/2018: interaction btw prec and lake surface including phase change of prec and water body
#! Nan Wei,  06/2018: update heat conductivity of water body and soil below and snow hydrology
#! Hua Yuan, 01/2023: added snow layer absorption in melting calculation
#!-----------------------------------------------------------------------
def hConductivity_lake(const_physical, nl_lake, snl, t_grnd, z_lake, t_lake, lake_icefrac, rhow, dlat, ustar, z0mg,
                        lakedepth, depthcrit):
    mixfact = 5.0  # Mixing enhancement factor
    p0 = 1.0       # Neutral value of turbulent Prandtl number
    
    cwat = const_physical.cpliq * const_physical.denh2o
    tkice_eff = const_physical.tkice * const_physical.denice / const_physical.denh2o  # Effective conductivity since layer depth is constant
    km = const_physical.tkwat / cwat                    # Molecular diffusivity (constant)
    u2m = max(0.1, ustar / const_physical.vonkar * np.log(2.0 / z0mg))
    ws = 1.2e-03 * u2m
    ks = 6.6 * np.sqrt(abs(np.sin(dlat))) * (u2m ** (-1.84))
    
    tk_lake = np.zeros(nl_lake)
    kme = np.zeros(nl_lake)
    
    for j in range(nl_lake - 1):
        drhodz = (rhow[j + 1] - rhow[j]) / (z_lake[j + 1] - z_lake[j])
        n2 = max(7.5e-5, const_physical.grav / rhow[j] * drhodz)
        num = 40.0 * n2 * (const_physical.vonkar * z_lake[j]) ** 2
        tmp = -2.0 * ks * z_lake[j]
        if tmp < -40.0:
            tmp = -40.0
        den = max((ws ** 2) * np.exp(tmp), 1.0e-10)
        ri = (-1.0 + np.sqrt(max(1.0 + num / den, 0.0))) / 20.0
        
        if t_grnd > const_physical.tfrz and t_lake[0] > const_physical.tfrz and snl == 0:
            tmp = -ks * z_lake[j]
            if tmp < -40.0:
                tmp = -40.0
            ke = const_physical.vonkar * ws * z_lake[j] / p0 * np.exp(tmp) / (1.0 + 37.0 * ri ** 2)
            kme[j] = km + ke

            fangkm = 1.039e-8 * max(n2, 7.5e-5) ** (-0.43)
            kme[j] += fangkm

            if lakedepth >= depthcrit:
                kme[j] *= mixfact  # Mixing enhancement factor for lakes deeper than depthcrit
            tk_lake[j] = kme[j] * cwat
        else:
            kme[j] = km
            fangkm = 1.039e-8 * max(n2, 7.5e-5) ** (-0.43)
            kme[j] += fangkm
            if lakedepth >= depthcrit:
                kme[j] *= mixfact
            tk_lake[j] = kme[j] * cwat * tkice_eff / ((1.0 - lake_icefrac[j]) * tkice_eff + kme[j] * cwat * lake_icefrac[j])
    
    kme[nl_lake-1] = kme[nl_lake-2]
    savedtke1 = kme[0] * cwat
    
    if t_grnd > const_physical.tfrz and t_lake[0] > const_physical.tfrz and snl == 0:
        tk_lake[nl_lake-1] = tk_lake[nl_lake-2]
    else:
        tk_lake[nl_lake-1] = kme[nl_lake-1] * cwat * tkice_eff / ((1.0 - lake_icefrac[nl_lake-1]) * tkice_eff + kme[nl_lake-1] * cwat * lake_icefrac[nl_lake-1])
    
    return tk_lake, savedtke1

def roughness_lake(const_physical, snl, t_grnd, t_lake, lake_icefrac, forc_psrf, cur, ustar):
    # Constants
    # tfrz = 273.15     # Freezing point of water (K)
    # vonkar = 0.4      # von Karman constant
    # grav = 9.81       # Acceleration due to gravity (m/s^2)
    
    cus = 0.1         # Empirical constant for roughness under smooth flow
    kva0 = 1.51e-5    # Kinematic viscosity of air (m^2/s) at 20C and 1.013e5 Pa
    prn = 0.713       # Prandtl number for air at neutral stability
    sch = 0.66        # Schmidt number for water in air at neutral stability

    # Initialize output variables
    z0mg = 0.0
    z0hg = 0.0
    z0qg = 0.0

    if t_grnd > const_physical.tfrz and t_lake > const_physical.tfrz and snl == 0:
        kva = kva0 * (t_grnd / 293.15) ** 1.5 * 1.013e5 / forc_psrf  # Kinematic viscosity of air
        z0mg = max(cus * kva / max(ustar, 1.e-4), cur * ustar ** 2 / const_physical.grav)  # Momentum roughness length
        z0mg = max(z0mg, 1.0e-5)  # This limit is redundant with current values
        sqre0 = (max(z0mg * ustar / kva, 0.1)) ** 0.5  # Square root of roughness Reynolds number
        z0hg = z0mg * np.exp(-const_physical.vonkar / prn * (4.0 * sqre0 - 3.2))  # SH roughness length
        z0qg = z0mg * np.exp(-const_physical.vonkar / sch * (4.0 * sqre0 - 4.2))  # LH roughness length
        z0qg = max(z0qg, 1.0e-5)  # Minimum allowed roughness length for unfrozen lakes
        z0hg = max(z0hg, 1.0e-5)  # Set low to avoid floating point exceptions
    elif snl == 0:  # Frozen lake with ice, and no snow cover
        z0mg = 0.001  # z0mg won't have changed
        z0hg = z0mg / np.exp(0.13 * (ustar * z0mg / 1.5e-5) ** 0.45)
        z0qg = z0hg
    else:  # Use roughness over snow
        z0mg = 0.0024  # z0mg won't have changed
        z0hg = z0mg / np.exp(0.13 * (ustar * z0mg / 1.5e-5) ** 0.45)
        z0qg = z0hg

    return z0mg, z0hg, z0qg

def newsnow_lake(const_physical,
    maxsnl, nl_lake, deltim, dz_lake, pg_rain, pg_snow, t_precip, bifall,
    t_lake, zi_soisno, z_soisno, dz_soisno, t_soisno, wliq_soisno, wice_soisno,
    fiold, snl, sag, scv, snowdp, lake_icefrac):
    """
    DESCRIPTION:
    Add new snow nodes and interaction between precipitation and lake surface 
    including phase change of precipitation and water body.

    Original author : Yongjiu Dai, 04/2014
    Revisions:
    Nan Wei,  01/2018: update interaction between precipitation and lake surface
    """

    # Local variables
    newnode = 0
    dz_snowf = pg_snow / bifall
    snowdp += dz_snowf * deltim
    scv += pg_snow * deltim  # snow water equivalent (mm)
    wice_lake = np.zeros(nl_lake)
    wliq_lake = np.zeros(nl_lake)

    zi_soisno[0] = 0.0

    if snl == 0 and snowdp < 0.01:  # no snow layer, energy exchange between precipitation and lake surface
        a = const_physical.cpliq * pg_rain * deltim * (t_precip - const_physical.tfrz)  # cool down rainfall to const_physical.tfrz
        b = pg_rain * deltim * const_physical.hfus  # all rainfall frozen
        c = const_physical.cpice * const_physical.denh2o * dz_lake[0] * lake_icefrac[0] * (const_physical.tfrz - t_lake[0])  # warm up lake surface ice to const_physical.tfrz
        d = const_physical.denh2o * dz_lake[0] * lake_icefrac[0] * const_physical.hfus  # all lake surface ice melt
        e = const_physical.cpice * pg_snow * deltim * (const_physical.tfrz - t_precip)  # warm up snowfall to const_physical.tfrz
        f = pg_snow * deltim * const_physical.hfus  # all snowfall melt
        g = const_physical.cpliq * const_physical.denh2o * dz_lake[0] * (1 - lake_icefrac[0]) * (t_lake[0] - const_physical.tfrz)  # cool down lake surface water to const_physical.tfrz
        h = const_physical.denh2o * dz_lake[0] * (1 - lake_icefrac[0]) * const_physical.hfus  # all lake surface water frozen
        sag = 0.0

        if lake_icefrac[0] > 0.999:
            if a + b <= c:
                tw = min(const_physical.tfrz, t_precip)
                t_lake[0] = (a + b + const_physical.cpice * (pg_rain + pg_snow) * deltim * tw + const_physical.cpice * const_physical.denh2o * dz_lake[0] * t_lake[0] * lake_icefrac[0]) / \
                            (const_physical.cpice * const_physical.denh2o * dz_lake[0] * lake_icefrac[0] + const_physical.cpice * (pg_rain + pg_snow) * deltim)
                scv += pg_rain * deltim
                snowdp += pg_rain * deltim / bifall
                pg_snow += pg_rain
                pg_rain = 0.0
            elif a <= c:
                t_lake[0] = const_physical.tfrz
                scv += (c - a) / const_physical.hfus
                snowdp += (c - a) / (const_physical.hfus * bifall)
                pg_snow += min(pg_rain, (c - a) / (const_physical.hfus * deltim))
                pg_rain = max(0.0, pg_rain - (c - a) / (const_physical.hfus * deltim))
            elif a <= c + d:
                t_lake[0] = const_physical.tfrz
                wice_lake = const_physical.denh2o * dz_lake[0] - (a - c) / const_physical.hfus
                wliq_lake = (a - c) / const_physical.hfus
                lake_icefrac[0] = wice_lake / (wice_lake + wliq_lake)
            else:
                t_lake[0] = (const_physical.cpliq * pg_rain * deltim * t_precip + const_physical.cpliq * const_physical.denh2o * dz_lake[0] * const_physical.tfrz - c - d) / \
                            (const_physical.cpliq * const_physical.denh2o * dz_lake[0] + const_physical.cpliq * pg_rain * deltim)
                lake_icefrac[0] = 0.0

            if snowdp >= 0.01:  # frozen rain may make new snow layer
                snl = -1
                newnode = 1
                dz_soisno[0] = snowdp  # meter
                z_soisno[0] = -0.5 * dz_soisno[0]
                zi_soisno[-1] = -dz_soisno[0]
                sag = 0.0  # snow age

                t_soisno[0] = t_lake[0]  # K
                wice_soisno[0] = scv  # kg/m2
                wliq_soisno[0] = 0.0  # kg/m2
                fiold[0] = 1.0

        elif lake_icefrac[0] >= 0.001:
            if pg_rain > 0.0 and pg_snow > 0.0:
                t_lake[0] = const_physical.tfrz
            elif pg_rain > 0.0:
                if a >= d:
                    t_lake[0] = (const_physical.cpliq * pg_rain * deltim * t_precip + const_physical.cpliq * const_physical.denh2o * dz_lake[0] * const_physical.tfrz - d) / \
                                (const_physical.cpliq * const_physical.denh2o * dz_lake[0] + const_physical.cpliq * pg_rain * deltim)
                    lake_icefrac[0] = 0.0
                else:
                    t_lake[0] = const_physical.tfrz
                    wice_lake[0] = const_physical.denh2o * dz_lake[0] * lake_icefrac[0] - a / const_physical.hfus
                    wliq_lake[0] = const_physical.denh2o * dz_lake[0] * (1 - lake_icefrac[0]) + a / const_physical.hfus
                    lake_icefrac[0] = wice_lake[0] / (wice_lake[0] + wliq_lake[0])
            elif pg_snow > 0.0:
                if e >= h:
                    t_lake[0] = (h + const_physical.cpice * const_physical.denh2o * dz_lake[0] * const_physical.tfrz + const_physical.cpice * pg_snow * deltim * t_precip) / \
                                (const_physical.cpice * pg_snow * deltim + const_physical.cpice * const_physical.denh2o * dz_lake[0])
                    lake_icefrac[0] = 1.0
                else:
                    t_lake[0] = const_physical.tfrz
                    wice_lake[0] = const_physical.denh2o * dz_lake[0] * lake_icefrac[0] + e / const_physical.hfus
                    wliq_lake[0] = const_physical.denh2o * dz_lake[0] * (1 - lake_icefrac[0]) - e / const_physical.hfus
                    lake_icefrac[0] = wice_lake[0] / (wice_lake[0] + wliq_lake[0])

        else:
            if e + f <= g:
                tw = max(const_physical.tfrz, t_precip)
                t_lake[0] = (const_physical.cpliq * const_physical.denh2o * dz_lake[0] * t_lake[0] * (1 - lake_icefrac[0]) + const_physical.cpliq * (pg_rain + pg_snow) * deltim * tw - e - f) / \
                            (const_physical.cpliq * (pg_rain + pg_snow) * deltim + const_physical.cpliq * const_physical.denh2o * dz_lake[0] * (1 - lake_icefrac[0]))
                scv -= pg_snow * deltim
                snowdp -= dz_snowf * deltim
                pg_rain += pg_snow
                pg_snow = 0.0
            elif e <= g:
                t_lake[0] = const_physical.tfrz
                scv -= (g - e) / const_physical.hfus
                snowdp -= (g - e) / (const_physical.hfus * bifall)
                pg_rain += min(pg_snow, (g - e) / (const_physical.hfus * deltim))
                pg_snow = max(0.0, pg_snow - (g - e) / (const_physical.hfus * deltim))
            elif e <= g + h:
                t_lake[0] = const_physical.tfrz
                wice_lake[0] = (e - g) / const_physical.hfus
                wliq_lake[0] = const_physical.denh2o * dz_lake[0] - (e - g) / const_physical.hfus
                lake_icefrac[0] = wice_lake[0] / (wice_lake[0] + wliq_lake[0])
            else:
                t_lake[0] = (const_physical.cpice * pg_snow * deltim * t_precip + const_physical.cpliq * const_physical.denh2o * dz_lake[0] * const_physical.tfrz + g + h) / \
                            (const_physical.cpliq * const_physical.denh2o * dz_lake[0] + const_physical.cpice * pg_snow * deltim)
                lake_icefrac[0] = 1.0
          
    elif snl==0 and snowdp >= 0.01:
        dz_soisno[0] = snowdp  # meter
        z_soisno[0] = -0.5 * dz_soisno[0]
        zi_soisno[-1] = -dz_soisno[0]
        sag = 0.0  # snow age

        t_soisno[0] = min(const_physical.tfrz, t_precip)  # K
        wice_soisno[0] = scv  # kg/m2
        wliq_soisno[0] = 0.0  # kg/m2
        fiold[0] = 1.0

    else:  # there are snow layers, assume the new snow has the same temperature with the surface lake water
        lb = snl + 1
        t_soisno[lb] = (
            (wice_soisno[lb] * const_physical.cpice + wliq_soisno[lb] * const_physical.cpliq) * t_soisno[lb]
            + (pg_rain * const_physical.cpliq + pg_snow * const_physical.cpice) * deltim * t_precip
        ) / (
            wice_soisno[lb] * const_physical.cpice + wliq_soisno[lb] * const_physical.cpliq
            + pg_rain * deltim * const_physical.cpliq + pg_snow * deltim * const_physical.cpice
        )

        t_soisno[lb] = min(const_physical.tfrz, t_soisno[lb])
        wice_soisno[lb] = wice_soisno[lb] + deltim * pg_snow
        dz_soisno[lb] = dz_soisno[lb] + dz_snowf * deltim
        z_soisno[lb] = zi_soisno[lb] - 0.5 * dz_soisno[lb]
        zi_soisno[lb - 1] = zi_soisno[lb] - dz_soisno[lb]

    return pg_rain, pg_snow, zi_soisno, z_soisno, dz_soisno, t_soisno, wliq_soisno, wice_soisno, fiold, snl, sag, scv, snowdp, lake_icefrac, t_lake
    
def laketem(nl_colm, const_physical, patchtype, maxsnl, nl_soil, nl_lake, dlat, deltim, forc_hgt_u, forc_hgt_t, forc_hgt_q,
        forc_us, forc_vs, forc_t, forc_q, forc_rhoair, forc_psrf, forc_sols, forc_soll, forc_solsd,
        forc_solld, sabg, forc_frl, dz_soisno, z_soisno, zi_soisno, dz_lake, lakedepth, vf_quartz,
        vf_gravels, vf_om, vf_sand, wf_gravels, wf_sand, porsl, csol, k_solids, dksatu, dksatf,
        dkdry, BA_alpha, BA_beta, hpbl, t_grnd, scv, snowdp, t_soisno, wliq_soisno, wice_soisno,
        imelt_soisno, t_lake, lake_icefrac, savedtke1, snofrz, sabg_snow_lyr, taux , tauy , fsena ,
           fevpa        , lfevpa      , fseng        , fevpg     ,
           qseva        , qsubl       , qsdew        , qfros     ,
           olrg         , fgrnd       , tref         , qref      ,
           trad         , emis        , z0m          , zol       ,
           rib          , ustar       , qstar        , tstar     ,
           fm           , fh          , fq           , sm, urban_call=None):

    # Lakes have variable depth, possible snow layers above, freezing & thawing of lake water,
    # and soil layers with active temperature and gas diffusion below.
    #
    # Calculates temperatures in the 25-30 layer column of (possible) snow,
    # lake water, soil, and bedrock beneath lake.
    # Snow and soil temperatures are determined as in SoilTemperature, except
    # for appropriate boundary conditions at the top of the snow (the flux is fixed
    # to be the ground heat flux), the bottom of the snow (adjacent to top lake layer),
    # and the top of the soil (adjacent to the bottom lake layer).
    # Also, the soil is kept fully saturated.
    # The whole column is solved simultaneously as one tridiagonal matrix.
    itmax  = 40   # maximum number of iteration
    itmin  = 6    # minimum number of iteration
    delmax = 3.0  # maximum change in lake temperature [K]
    dtmin  = 0.01 # max limit for temperature convergence [K]
    dlemin = 0.1  # max limit for energy flux convergence [w/m2]
    # Initialize local variables
    # Constants
    cur0 = 0.01       # min. Charnock parameter
    curm = 0.1        # maximum Charnock parameter
    fcrit = 22.0      # critical dimensionless fetch for Charnock parameter (Vickers & Mahrt 1997)
                      # but converted to USE u instead of u* (Subin et al. 2011)
    mixfact = 5.0     # Mixing enhancement factor.
    depthcrit = 25.0   # (m) Depth beneath which to enhance mixing
    fangmult = 5.0     # Multiplier for unfrozen diffusivity
    minmultdepth = 20.0  # (m) Minimum depth for imposing fangmult
    cnfac = 0.5        # Crank Nicholson factor between 0 and 1

    # SNICAR model variables
    snofrz = np.zeros(-maxsnl)  # snow freezing rate (col,lyr) [kg m-2 s-1]
    sabg_snow_lyr = np.zeros(-maxsnl+1)  # solar radiation absorbed by ground [W/m2]

    #---------------- local variables in surface temp and fluxes calculation -----------------
    idlak = 0     # index of lake, 1 = deep lake, 2 = shallow lake
    z_lake = np.zeros (nl_lake)  # lake node depth (middle point of layer) (m)

    ax = 0.0      # used in iteration loop for calculating t_grnd (numerator of NR solution)
    bx   = 0.0     # used in iteration loop for calculating t_grnd (denomin. of NR solution)
    beta1  = 0.0   # coefficient of conective velocity [-]
    degdT  = 0.0   # d(eg)/dT
    displax  = 0.0 # zero- displacement height [m]
    dqh    = 0.0   # diff of humidity between ref. height and surface
    dth    = 0.0   # diff of virtual temp. between ref. height and surface
    dthv    = 0.0  # diff of vir. poten. temp. between ref. height and surface
    dzsur  = 0.0   # 1/2 the top layer thickness (m)
    tsur   = 0.0   # top layer temperature
    rhosnow  = 0.0 # partitial density of water (ice + liquid)
    eg   = 0.0     # water vapor pressure at temperature T [pa]
    emg   = 0.0    # ground emissivity (0.97 for snow,
    errore = 0.0   # lake temperature energy conservation error (w/m**2)
    hm    = 0.0    # energy residual [W/m2]
    htvp  = 0.0    # latent heat of vapor of water (or sublimation) [j/kg]
    obu   = 0.0    # monin-obukhov length (m)
    obuold = 0.0   # monin-obukhov length of previous iteration
    qsatg  = 0.0   # saturated humidity [kg/kg]
    qsatgdT = 0.0  # d(qsatg)/dT

    ram  = 0.0     # aerodynamical resistance [s/m]
    rah   = 0.0    # thermal resistance [s/m]
    raw   = 0.0    # moisture resistance [s/m]
    stftg3  = 0.0  # emg*sb*t_grnd*t_grnd*t_grnd
    fh2m   = 0.0   # relation for temperature at 2m
    fq2m   = 0.0   # relation for specific humidity at 2m
    fm10m   = 0.0  # integral of profile function for momentum at 10m
    t_grnd_bef0 = 0.0   # initial ground temperature
    t_grnd_bef  = 0.0   # initial ground temperature
    thm   = 0.0    # intermediate variable (forc_t+0.0098*forc_hgt_t)
    th   = 0.0     # potential temperature (kelvin)
    thv  = 0.0     # virtual potential temperature (kelvin)
    thvstar = 0.0  # virtual potential temperature scaling parameter
    tksur = 0.0    # thermal conductivity of snow/soil (w/m/kelvin)
    um   = 0.0     # wind speed including the stablity effect [m/s]
    ur   = 0.0     # wind speed at reference height [m/s]
    visa   = 0.0   # kinematic viscosity of dry air [m2/s]
    wc    = 0.0    # convective velocity [m/s]
    wc2    = 0.0   # wc*wc
    zeta   = 0.0   # dimensionless height used in Monin-Obukhov theory
    zii    = 0.0   # convective boundary height [m]
    zldis  = 0.0   # reference height "minus" zero displacement heght [m]
    z0mg   = 0.0   # roughness length over ground, momentum [m]
    z0hg   = 0.0   # roughness length over ground, sensible heat [m]
    z0qg   = 0.0   # roughness length over ground, latent heat [m]
    zsum = 0.0

    #output----------------------

    wliq_lake = np.zeros (nl_lake)  # lake liquid water (kg/m2)
    wice_lake = np.zeros (nl_lake)  # lake ice lens (kg/m2)
    vf_water = np.zeros (nl_soil) # volumetric fraction liquid water within underlying soil
    vf_ice = np.zeros (nl_soil)   # volumetric fraction ice len within underlying soil

    fgrnd1 = 0.0  # ground heat flux into the first snow/lake layer [W/m2]

    #--------------------
    rhow = np.zeros(nl_lake) # density of water (kg/m**3)
    fin      = 0.0      # heat flux into lake - flux out of lake (w/m**2)
    phi = np.zeros(nl_lake)  # solar radiation absorbed by layer (w/m**2)
    phi_soil  = 0.0     # solar radiation into top soil layer (W/m^2)
    phidum   = 0.0      # temporary value of phi

    imelt_lake = np.zeros(nl_lake,dtype=int)       # lake flag for melting or freezing snow and soil layer [-]
    cv_lake = np.zeros(nl_lake)          # heat capacity [J/(m2 K)]
    tk_lake = np.zeros(nl_lake)          # thermal conductivity at layer node [W/(m K)]
    cv_soisno = np.zeros(nl_soil - maxsnl) # heat capacity of soil/snow [J/(m2 K)]
    tk_soisno = np.zeros(nl_soil - maxsnl) # thermal conductivity of soil/snow [W/(m K)] (at interface below, except for j=0)
    hcap = np.zeros(nl_soil)           # J/(m3 K)
    thk = np.zeros(nl_soil - maxsnl)     # W/(m K)
    tktopsoil   = 0.0                 # thermal conductivity of the top soil layer [W/(m K)]

    t_soisno_bef = np.zeros(nl_soil - maxsnl ) # beginning soil/snow temp for E cons. check [K]
    t_lake_bef = np.zeros(nl_lake)          # beginning lake temp for energy conservation check [K]
    wice_soisno_bef = np.zeros(-maxsnl-1)    # ice lens [kg/m2]

    cvx   = np.zeros  (nl_lake+nl_soil -  maxsnl ) # heat capacity for whole column [J/(m2 K)]
    tkix   = np.zeros (nl_lake+nl_soil -  maxsnl ) # thermal conductivity at layer interfaces for whole column [W/(m K)]
    phix  = np.zeros  (nl_lake+nl_soil -  maxsnl ) # solar source term for whole column [W/m**2]
    zx    = np.zeros  (nl_lake+nl_soil -  maxsnl ) # interface depth (+ below surface) for whole column [m]
    tx    = np.zeros  (nl_lake+nl_soil -  maxsnl ) # temperature of whole column [K]
    tx_bef = np.zeros (nl_lake+nl_soil -  maxsnl ) # beginning lake/snow/soil temp for energy conservation check [K]
    factx  = np.zeros (nl_lake+nl_soil -  maxsnl ) # coefficient used in computing tridiagonal matrix
    fnx   = np.zeros  (nl_lake+nl_soil -  maxsnl ) # heat diffusion through the layer interface below [W/m2]
    a     = np.zeros  (nl_lake+nl_soil -  maxsnl ) # "a" vector for tridiagonal matrix
    b    = np.zeros   (nl_lake+nl_soil -  maxsnl ) # "b" vector for tridiagonal matrix
    c    = np.zeros   (nl_lake+nl_soil -  maxsnl ) # "c" vector for tridiagonal matrix
    r    = np.zeros   (nl_lake+nl_soil -  maxsnl ) # "r" vector for tridiagonal solution
    fn1  = np.zeros   (nl_lake+nl_soil -  maxsnl ) # heat diffusion through the layer interface below [W/m2]
    brr   = np.zeros  (nl_lake+nl_soil -  maxsnl ) #
    
    # ======================================================================
    # [1] constants and model parameters
    # ======================================================================

    # Constants for lake temperature model
    za = [0.5, 0.6]
    cwat = const_physical.cpliq * const_physical.denh2o     # water heat capacity per unit volume
    cice_eff = const_physical.cpice * const_physical.denh2o # use water density because layer depth is not adjusted for freezing
    cfus = const_physical.hfus * const_physical.denh2o      # latent heat per unit volume
    tkice_eff = const_physical.tkice * const_physical.denice / const_physical.denh2o # effective conductivity since layer depth is constant
    emg = 0.97                 # surface emissivity

    # Define snow layer on ice lake
    snl = 0
    for j in range(5):
        if wliq_soisno[j] + wice_soisno[j] > 0.:
            snl -= 1
    lb = snl + 1

    # Latent heat
    if t_grnd > const_physical.tfrz:
        htvp = const_physical.hvap
    else:
        htvp = const_physical.hsub

    # Define levels
    # z_lake = [0.0] * nl_lake
    z_lake[0] = dz_lake[0] / 2.0
    for j in range(1, nl_lake):
        z_lake[j] = z_lake[j-1] + (dz_lake[j-1] + dz_lake[j]) / 2.0

    # Base on lake depth, assuming that small lakes are likely to be shallower
    # Estimate crudely based on lake depth
    if z_lake[nl_lake - 1] < 4.0:
        idlak = 0
        fetch = 100.0  # shallow lake
    else:
        idlak = 1
        fetch = 25.0 * z_lake[nl_lake - 1]  # deep lake
        
    # ======================================================================
    # [2] pre-processing for the calculation of the surface temperature and fluxes
    # ======================================================================

    if not nl_colm['DEF_USE_SNICAR'] or urban_call is not None:
        if snl == 0:
            # Calculate the nir fraction of absorbed solar.
            betaprime = (forc_soll + forc_solld) / max(1e-5, forc_sols + forc_soll + forc_solsd + forc_solld)
            betavis = 0.0  # The fraction of the visible (e.g., vis not nir from atm) sunlight
                           # absorbed in ~1 m of water (the surface layer za_lake).
                           # This is roughly the fraction over 700 nm but may depend on the details
                           # of atmospheric radiative transfer.
                           # As long as NIR = 700 nm and up, this can be zero.
            betaprime = betaprime + (1.0 - betaprime) * betavis
        else:
            # or frozen but no snow layers or
            # currently ignore the transmission of solar in snow and ice layers
            # to be updated in the future version
            betaprime = 1.0
    else:
        # Calculate the nir fraction of absorbed solar.
        betaprime = (forc_soll + forc_solld) / max(1e-5, forc_sols + forc_soll + forc_solsd + forc_solld)
        betavis = 0.0  # The fraction of the visible (e.g., vis not nir from atm) sunlight
                       # absorbed in ~1 m of water (the surface layer za_lake).
                       # This is roughly the fraction over 700 nm but may depend on the details
                       # of atmospheric radiative transfer.
                       # As long as NIR = 700 nm and up, this can be zero.
        betaprime = betaprime + (1.0 - betaprime) * betavis

    # Call to qsadv function
    eg, degdT, qsatg, qsatgdT = CoLM_Qsadv.qsadv(t_grnd, forc_psrf)

    # Potential temperature at the reference height
    beta1 = 1.0  # (in computing W_*)
    zii = 1000.0  # m (pbl height)
    thm = forc_t + 0.0098 * forc_hgt_t  # intermediate variable equivalent to
                                        # forc_t * (pgcm / forc_psrf) ** (const_physical.rgas / const_physical.cpair)
    th = forc_t * (100000.0 / forc_psrf) ** (const_physical.rgas / const_physical.cpair)  # potential T
    thv = th * (1.0 + 0.61 * forc_q)  # virtual potential T
    ur = max(0.1, math.sqrt(forc_us * forc_us + forc_vs * forc_vs))  # limit set to 0.1

    # Initialization variables
    nmozsgn = 0
    obuold = 0.0
    dth = thm - t_grnd
    dqh = forc_q - qsatg
    dthv = dth * (1.0 + 0.61 * forc_q) + 0.61 * th * dqh
    zldis = forc_hgt_u - 0.0

    # Roughness lengths, allow all roughness lengths to be prognostic
    ustar = 0.06
    wc = 0.5

    # Kinematic viscosity of dry air (m2/s) - Andreas (1989) CRREL Rep. 89-11
    visa = 1.326e-5 * (1.0 + 6.542e-3 * (forc_t - const_physical.tfrz)
           + 8.301e-6 * (forc_t - const_physical.tfrz) ** 2 - 4.84e-9 * (forc_t - const_physical.tfrz) ** 3)

    cur = cur0 + curm * math.exp(max(-(fetch * const_physical.grav / ur / ur) ** (1.0 / 3.0) / fcrit,
                                 -(z_lake[nl_lake - 1] * const_physical.grav) ** 0.5 / ur))  # depth-limited

    if dthv >= 0.0:
        um = max(ur, 0.1)
    else:
        um = math.sqrt(ur * ur + wc * wc)

    for i in range(1, 5):
        z0mg = 0.013 * ustar * ustar / const_physical.grav + 0.11 * visa / ustar
        ustar = const_physical.vonkar * um / math.log(zldis / z0mg)

    # Call to roughness_lake function
    z0mg, z0hg, z0qg = roughness_lake(const_physical, snl, t_grnd, t_lake[0], lake_icefrac[0], forc_psrf, cur, ustar)

    # Call to moninobukini function
    um, obu = CoLM_FrictionVelocity.moninobukini(ur, th, thm, thv, dth, dqh, dthv, zldis, z0mg, const_physical.grav)

    if snl == 0:
        dzsur = dz_lake[0] / 2.0
    else:
        dzsur = z_soisno[maxsnl- lb+1] - zi_soisno[maxsnl - lb]

    iter = 1
    del_T_grnd = 1.0  # t_grnd diff
    convernum = 0  # number of times when del_T_grnd <= 0.01
    
    # ======================================================================
    # [3] Begin stability iteration and temperature and fluxes calculation
    # ======================================================================

    # Iteration loop
    while iter <= itmax:
        t_grnd_bef = t_grnd

        if t_grnd_bef > const_physical.tfrz and t_lake[0] > const_physical.tfrz and snl == 0:
            tksur = savedtke1  # water molecular conductivity
            tsur = t_lake[0]
            htvp = const_physical.hvap
        elif snl == 0:  # frozen but no snow layers
            tksur = const_physical.tkice  # This is an approximation because the whole layer may not be frozen, and it is not
                           # accounting for the physical (but not nominal) expansion of the frozen layer.
            tsur = t_lake[0]
            htvp = const_physical.hsub
        else:
            # need to calculate thermal conductivity of the top snow layer
            rhosnow = (wice_soisno[maxsnl- lb+1] + wliq_soisno[maxsnl- lb+1]) / dz_soisno[maxsnl- lb+1]
            tksur = const_physical.tkair + (7.75e-5 * rhosnow + 1.105e-6 * rhosnow * rhosnow) * (const_physical.tkice - const_physical.tkair)
            tsur = t_soisno[maxsnl- lb+1]
            htvp = const_physical.hsub

        # Evaluate stability-dependent variables using moz from prior iteration
        displax = 0.0
        if nl_colm['DEF_USE_CBL_HEIGHT']:
            ustar, fh2m, fq2m, fm10m, fm, fh, fq = CoLM_TurbulenceLEddy.moninobuk_leddy(forc_hgt_u, forc_hgt_t, forc_hgt_q, displax, z0mg, z0hg, z0qg, obu, um, hpbl, const_physical.vonkar)
        else:
            ustar,fh2m,fq2m,fm10m,fm,fh,fq = CoLM_FrictionVelocity.moninobuk(const_physical, forc_hgt_u, forc_hgt_t, forc_hgt_q, displax, z0mg, z0hg, z0qg, obu, um)

        # Get derivative of fluxes with respect to ground temperature
        ram = 1.0 / (ustar * ustar / um)
        rah = 1.0 / (const_physical.vonkar / fh * ustar)
        raw = 1.0 / (const_physical.vonkar / fq * ustar)
        stftg3 = emg * const_physical.stefnc * t_grnd_bef * t_grnd_bef * t_grnd_bef

        ax = betaprime * sabg + emg * forc_frl + 3.0 * stftg3 * t_grnd_bef \
            + forc_rhoair * const_physical.cpair / rah * thm \
            - htvp * forc_rhoair / raw * (qsatg - qsatgdT * t_grnd_bef - forc_q) \
            + tksur * tsur / dzsur

        bx = 4.0 * stftg3 + forc_rhoair * const_physical.cpair / rah \
            + htvp * forc_rhoair / raw * qsatgdT + tksur / dzsur

        t_grnd = ax / bx

        # Surface fluxes of momentum, sensible and latent using ground temperatures from previous time step
        fseng = forc_rhoair * const_physical.cpair * (t_grnd - thm) / rah
        fevpg = forc_rhoair * (qsatg + qsatgdT * (t_grnd - t_grnd_bef) - forc_q) / raw

        eg, degdT, qsatg, qsatgdT = CoLM_Qsadv.qsadv(t_grnd, forc_psrf)
        dth = thm - t_grnd
        dqh = forc_q - qsatg
        tstar = const_physical.vonkar / fh * dth
        qstar = const_physical.vonkar / fq * dqh
        thvstar = tstar * (1.0 + 0.61 * forc_q) + 0.61 * th * qstar
        zeta = zldis * const_physical.vonkar * const_physical.grav * thvstar / (ustar ** 2 * thv)

        if zeta >= 0.0:  # stable
            zeta = min(2.0, max(zeta, 1e-6))
        else:  # unstable
            zeta = max(-100.0, min(zeta, -1e-6))

        obu = zldis / zeta

        if zeta >= 0.0:
            um = max(ur, 0.1)
        else:
            if nl_colm['DEF_USE_CBL_HEIGHT']:
                zii = max(5.0 * forc_hgt_u, hpbl)

            wc = (-const_physical.grav * ustar * thvstar * zii / thv) ** (1.0 / 3.0)
            wc2 = beta1 * beta1 * (wc * wc)
            um = math.sqrt(ur * ur + wc2)

        z0mg, z0hg, z0qg = roughness_lake(const_physical, snl, t_grnd, t_lake[0], lake_icefrac[0], forc_psrf,
                       cur, ustar)

        iter += 1
        del_T_grnd = abs(t_grnd - t_grnd_bef)

        if iter > itmin:
            if del_T_grnd <= dtmin:
                convernum += 1

            if convernum >= 4:
                break
                
    # ----------------------------------------------------------------------
    # Zack Subin, 3/27/09
    # Since they are now a function of whatever t_grnd was before cooling
    # to freezing temperature, then this value should be used in the derivative correction term.
    # Allow convection if ground temp is colder than lake but warmer than 4C, or warmer than
    # lake which is warmer than freezing but less than 4C.

    tdmax = const_physical.tfrz + 4.0

    if (snl < 0 or t_lake[0] <= const_physical.tfrz) and t_grnd > const_physical.tfrz:
        t_grnd_bef = t_grnd
        t_grnd = const_physical.tfrz
        fseng = forc_rhoair * const_physical.cpair * (t_grnd - thm) / rah
        fevpg = forc_rhoair * (qsatg + qsatgdT * (t_grnd - t_grnd_bef) - forc_q) / raw
    elif (t_lake[0] > t_grnd and t_grnd > tdmax) or (t_lake[0] < t_grnd and t_lake[0] > const_physical.tfrz and t_grnd < tdmax):
        # Convective mixing will occur at surface
        t_grnd_bef = t_grnd
        t_grnd = t_lake[0]
        fseng = forc_rhoair * const_physical.cpair * (t_grnd - thm) / rah
        fevpg = forc_rhoair * (qsatg + qsatgdT * (t_grnd - t_grnd_bef) - forc_q) / raw
    # ----------------------------------------------------------------------

    # net longwave from ground to atmosphere
    stftg3 = emg * const_physical.stefnc * t_grnd_bef * t_grnd_bef * t_grnd_bef
    olrg = (1.0 - emg) * forc_frl + emg * const_physical.stefnc * t_grnd_bef ** 4 + 4.0 * stftg3 * (t_grnd - t_grnd_bef)

    if t_grnd > const_physical.tfrz:
        htvp = const_physical.hvap
    else:
        htvp = const_physical.hsub

    # The actual heat flux from the ground interface into the lake, not including the light that penetrates the surface.
    fgrnd1 = betaprime * sabg + forc_frl - olrg - fseng - htvp * fevpg

    # January 12, 2023 by Yongjiu Dai
    if nl_colm['DEF_USE_SNICAR'] and not urban_call:
        hs = sabg_snow_lyr[maxsnl- lb+1] + forc_frl - olrg - fseng - htvp * fevpg
        dhsdT = 0.0

    # ------------------------------------------------------------
    # Set up vector r and vectors a, b, c that define tridiagonal matrix
    # snow and lake and soil layer temperature
    # ------------------------------------------------------------
    
    # ------------------------------------------------------------
    # Lake density
    # ------------------------------------------------------------

    for j in range(nl_lake):
        rhow[j] = ((1.0 - lake_icefrac[j]) * const_physical.denh2o * (1.0 - 1.9549e-05 * (abs(t_lake[j] - 277.0))**1.68) +
                   lake_icefrac[j] * const_physical.denice)
        # Allow for ice fraction; assume constant ice density.
        # This is not the correct average-weighting but that's OK because the density will only
        # be used for convection for lakes with ice, and the ice fraction will dominate the
        # density differences between layers.
        # Using this average will make sure that surface ice is treated properly during
        # convective mixing.

    # ------------------------------------------------------------
    # Diffusivity and implied thermal "conductivity" = diffusivity * cwat
    # ------------------------------------------------------------

    for j in range(nl_lake):
        cv_lake[j] = dz_lake[j] * (cwat * (1.0 - lake_icefrac[j]) + cice_eff * lake_icefrac[j])

    tk_lake, savedtke1 = hConductivity_lake(const_physical, nl_lake, snl, t_grnd, z_lake, t_lake, lake_icefrac, rhow,
                       dlat, ustar, z0mg, lakedepth, depthcrit)

    # ------------------------------------------------------------
    # Set the thermal properties of the snow above frozen lake and underlying soil
    # and check initial energy content.
    # ------------------------------------------------------------

    lb = snl + 1
    for i in range(nl_soil):
        vf_water[i] = wliq_soisno[i-maxsnl] / (dz_soisno[i-maxsnl] * const_physical.denh2o)
        vf_ice[i] = wice_soisno[i-maxsnl] / (dz_soisno[i-maxsnl] * const_physical.denice)
        hcap[i], thk[i-maxsnl] = CoLM_SoilThermalParameters.soil_hcap_cond(nl_colm, const_physical, vf_gravels[i], vf_om[i], vf_sand[i], porsl[i],
                       wf_gravels[i], wf_sand[i], k_solids[i],
                       csol[i], dkdry[i], dksatu[i], dksatf[i],
                       BA_alpha[i], BA_beta[i],
                       t_soisno[i-maxsnl], vf_water[i], vf_ice[i], hcap[i], thk[i-maxsnl])
        cv_soisno[i-maxsnl] = hcap[i] * dz_soisno[i-maxsnl]

    # Snow heat capacity and conductivity
    if lb <= 0:
        for j in range(lb-maxsnl-1, -maxsnl):
            cv_soisno[j] = const_physical.cpliq * wliq_soisno[j] + const_physical.cpice * wice_soisno[j]
            rhosnow = (wice_soisno[j] + wliq_soisno[j]) / dz_soisno[j]
            thk[j] = const_physical.tkair + (7.75e-5 * rhosnow + 1.105e-6 * rhosnow * rhosnow) * (const_physical.tkice - const_physical.tkair)

    # Thermal conductivity at the layer interface
    for i in range(-maxsnl-lb+1, nl_soil -maxsnl-1):
        # Avoid the snow conductivity to be dominant in the thermal conductivity of the interface.
        # Modified by Nan Wei, 08/25/2014
        if i != 4:
            tk_soisno[i] = (thk[i] * thk[i + 1] * (z_soisno[i + 1] - z_soisno[i]) /
                            (thk[i] * (z_soisno[i + 1] - zi_soisno[i]) + thk[i + 1] * (zi_soisno[i] - z_soisno[i])))
        else:
            tk_soisno[i] = thk[i]

    tk_soisno[nl_soil-maxsnl-1] = 0.0
    tktopsoil = thk[5]

    # Sum cv_lake * t_lake for energy check
    # Include latent heat term, and use const_physical.tfrz as reference temperature
    # to prevent abrupt change in heat content due to changing heat capacity with phase change.

    # This will need to be over all soil / lake / snow layers. Lake is below.
    ocvts = 0.0
    for j in range(nl_lake):
        ocvts += cv_lake[j] * (t_lake[j] - const_physical.tfrz) + cfus * dz_lake[j] * (1.0 - lake_icefrac[j])

    # Now DO for soil / snow layers
    for j in range(-maxsnl-lb+1, nl_soil-maxsnl):
        ocvts += cv_soisno[j] * (t_soisno[j] - const_physical.tfrz) + const_physical.hfus * wliq_soisno[j]
        if j == 5 and scv > 0.0 and j == lb:
            ocvts -= scv * const_physical.hfus

    # Set up solar source terms (phix)
    # Modified January 12, 2023 by Yongjiu Dai
    if not nl_colm['DEF_USE_SNICAR'] or urban_call is not None:
        if t_grnd > const_physical.tfrz and t_lake[0] > const_physical.tfrz and snl == 0:  # No snow cover, unfrozen layer lakes
            for j in range(nl_lake):
                # Extinction coefficient from surface data (1/m), if no eta from surface data,
                # set eta, the extinction coefficient, according to L Hakanson, Aquatic Sciences, 1995
                # (regression of secchi depth with lake depth for small glacial basin lakes), and the
                # Poole & Atkins expression for extinction coefficient of 1.7 / secchi Depth (m).

                eta = 1.1925 * max(lakedepth, 1.0)**(-0.424)
                zin = z_lake[j] - 0.5 * dz_lake[j]
                zout = z_lake[j] + 0.5 * dz_lake[j]
                rsfin = math.exp(-eta * max(zin - za[idlak], 0.0))  # The radiation within surface layer (z < za)
                rsfout = math.exp(-eta * max(zout - za[idlak], 0.0))  # Fixed at (1-beta) * sabg
                # Let rsfout for bottom layer go into soil.
                # Robust even for pathological cases,
                # like lakes thinner than za[idlak].

                phi[j] = (rsfin - rsfout) * sabg * (1.0 - betaprime)
                if j == nl_lake - 1:
                    phi_soil = rsfout * sabg * (1.0 - betaprime)
        elif snl == 0:  # No snow-covered layers, but partially frozen
            phi[0] = sabg * (1.0 - betaprime)
            phi[1:nl_lake] = 0.0
            phi_soil = 0.0
        else:  # Snow covered, needs improvement; Mironov 2002 suggests that SW can penetrate thin ice and may cause spring convection.
            phi[:] = 0.0
            phi_soil = 0.0
    else:
        for j in range(nl_lake):
            # Extinction coefficient from surface data (1/m), if no eta from surface data,
            # set eta, the extinction coefficient, according to L Hakanson, Aquatic Sciences, 1995
            # (regression of secchi depth with lake depth for small glacial basin lakes), and the
            # Poole & Atkins expression for extinction coefficient of 1.7 / secchi Depth (m).

            eta = 1.1925 * max(lakedepth, 1.0)**(-0.424)
            zin = z_lake[j] - 0.5 * dz_lake[j]
            zout = z_lake[j] + 0.5 * dz_lake[j]
            rsfin =math. exp(-eta * max(zin - za[idlak], 0.0))  # The radiation within surface layer (z < za)
            rsfout = math.exp(-eta * max(zout - za[idlak], 0.0))  # Fixed at (1-beta) * sabg
            # Let rsfout for bottom layer go into soil.
            # Robust even for pathological cases,
            # like lakes thinner than za[idlak].

            phi[j] = (rsfin - rsfout) * sabg_snow_lyr[5] * (1.0 - betaprime)
            if j == nl_lake - 1:
                phi_soil = rsfout * sabg_snow_lyr[5] * (1.0 - betaprime)

    phix[:] = 0
    phix[5:nl_lake+5] = phi[:nl_lake]  # Lake layer
    phix[nl_lake+5] = phi_soil  # Top soil layer

    # Set up interface depths(zx), and temperatures (tx).
    for j in range(lb-maxsnl-1, nl_lake + nl_soil-maxsnl):
        jprime = j - nl_lake
        if j < 5:  # Snow layer
            zx[j] = z_soisno[j]
            tx[j] = t_soisno[j]
        elif j <= nl_lake-maxsnl-1:  # Lake layer
            zx[j] = z_lake[j+maxsnl]
            tx[j] = t_lake[j+maxsnl]
        else:
            zx[j] = z_lake[nl_lake-1] + dz_lake[nl_lake-1]/2. + z_soisno[jprime]
            tx[j] = t_soisno[jprime]

    tx_bef = copy.copy(tx)

    # Heat capacity and resistance of snow without snow layers (<1cm) is ignored during diffusion,
    # but its capacity to absorb latent heat may be used during phase change.

    for j in range(lb-maxsnl-1, nl_lake + nl_soil-maxsnl):
        jprime = j - nl_lake

        # heat capacity [J/(m2 K)]
        if j < 5:  # snow layer
            cvx[j] = cv_soisno[j]
        elif j <= nl_lake-maxsnl-1:  # lake layer
            cvx[j] = cv_lake[j+maxsnl]
        else:  # soil layer
            cvx[j] = cv_soisno[jprime]

        # Determine interface thermal conductivities at layer interfaces [W/(m K)]
        if j < 4:  # non-bottom snow layer
            tkix[j] = tk_soisno[j]
        elif j == 4:  # bottom snow layer
            dzp = zx[j+1] - zx[j]
            tkix[j] = tk_lake[0] * tk_soisno[j] * dzp / (tk_soisno[j] * z_lake[0] + tk_lake[0] * (-zx[j]))
            # tk_soisno[0] is the conductivity at the middle of that layer
        elif j < nl_lake-maxsnl-1:  # non-bottom lake layer
            tkix[j] = (tk_lake[j+maxsnl] * tk_lake[j+maxsnl+1] * (dz_lake[j+maxsnl+1] + dz_lake[j+maxsnl])) / (tk_lake[j+maxsnl] * dz_lake[j+maxsnl+1] + tk_lake[j+maxsnl+1] * dz_lake[j+maxsnl])
        elif j == nl_lake-maxsnl-1:  # bottom lake layer
            dzp = zx[j+1] - zx[j]
            tkix[j] = tktopsoil * tk_lake[j+maxsnl] * dzp / (tktopsoil * dz_lake[j+maxsnl] / 2. + tk_lake[j+maxsnl] * z_soisno[5])
        else:  # soil layer
            tkix[j] = tk_soisno[jprime]

    # Determine heat diffusion through the layer interface
    for j in range(lb-maxsnl-1, nl_lake + nl_soil-maxsnl):
        factx[j] = deltim / cvx[j]
        if j < nl_lake + nl_soil-maxsnl-1:  # top or interior layer
            fnx[j] = tkix[j] * (tx[j + 1] - tx[j]) / (zx[j + 1] - zx[j])
        else:  # bottom soil layer
            fnx[j] = 0.0  # not used

    if nl_colm['DEF_USE_SNICAR'] and urban_call is None:
        if lb <= 0:  # snow covered
            for j in range(-maxsnl+lb-1, 6):
                if j == -maxsnl+lb-1:  # top snow layer
                    dzp = zx[j + 1] - zx[j]
                    a[j] = 0.0
                    b[j] = 1.0 + (1.0 - cnfac) * factx[j] * tkix[j] / dzp
                    c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
                    r[j] = tx_bef[j] + factx[j] * (hs - dhsdT * tx_bef[j] + cnfac * fnx[j])
                elif j <= 4:  # non-top snow layers
                    dzm = zx[j] - zx[j - 1]
                    dzp = zx[j + 1] - zx[j]
                    a[j] = -(1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                    b[j] = 1.0 + (1.0 - cnfac) * factx[j] * (tkix[j] / dzp + tkix[j - 1] / dzm)
                    c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
                    r[j] = tx_bef[j] + cnfac * factx[j] * (fnx[j] - fnx[j - 1]) + factx[j] * sabg_snow_lyr[j]
                else:  # snow covered top lake layer
                    dzm = zx[j] - zx[j - 1]
                    dzp = zx[j + 1] - zx[j]
                    a[j] = -(1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                    b[j] = 1.0 + (1.0 - cnfac) * factx[j] * (tkix[j] / dzp + tkix[j - 1] / dzm)
                    c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
                    r[j] = tx_bef[j] + cnfac * factx[j] * (fnx[j] - fnx[j - 1]) + factx[j] * (phix[j] + betaprime * sabg_snow_lyr[j])
        else:
            j = 5  # no snow covered top lake layer
            dzp = zx[j + 1] - zx[j]
            a[j] = 0.0
            b[j] = 1.0 + (1.0 - cnfac) * factx[j] * tkix[j] / dzp
            c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
            r[j] = tx_bef[j] + factx[j] * (cnfac * fnx[j] + phix[j] + fgrnd1)

        for j in range(6, nl_lake + nl_soil-maxsnl):
            if j < nl_lake + nl_soil-maxsnl-1:  # middle lake and soil layers
                dzm = zx[j] - zx[j - 1]
                dzp = zx[j + 1] - zx[j]
                a[j] = -(1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                b[j] = 1.0 + (1.0 - cnfac) * factx[j] * (tkix[j] / dzp + tkix[j - 1] / dzm)
                c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
                r[j] = tx_bef[j] + cnfac * factx[j] * (fnx[j] - fnx[j - 1]) + factx[j] * phix[j]
            else:  # bottom soil layer
                dzm = zx[j] - zx[j - 1]
                a[j] = -(1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                b[j] = 1.0 + (1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                c[j] = 0.0
                r[j] = tx_bef[j] - cnfac * factx[j] * fnx[j - 1]

    else:
        for j in range(-maxsnl+lb-1, nl_lake + nl_soil-maxsnl):
            if j == -maxsnl+lb-1:  # top layer
                dzp = zx[j + 1] - zx[j]
                a[j] = 0.0
                b[j] = 1.0 + (1.0 - cnfac) * factx[j] * tkix[j] / dzp
                c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
                r[j] = tx_bef[j] + factx[j] * (cnfac * fnx[j] + phix[j] + fgrnd1)
            elif j < nl_lake + nl_soil-maxsnl-1:  # middle layer
                dzm = zx[j] - zx[j - 1]
                dzp = zx[j + 1] - zx[j]
                a[j] = -(1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                b[j] = 1.0 + (1.0 - cnfac) * factx[j] * (tkix[j] / dzp + tkix[j - 1] / dzm)
                c[j] = -(1.0 - cnfac) * factx[j] * tkix[j] / dzp
                r[j] = tx_bef[j] + cnfac * factx[j] * (fnx[j] - fnx[j - 1]) + factx[j] * phix[j]
            else:  # bottom soil layer
                dzm = zx[j] - zx[j - 1]
                a[j] = -(1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                b[j] = 1.0 + (1.0 - cnfac) * factx[j] * tkix[j - 1] / dzm
                c[j] = 0.0
                r[j] = tx_bef[j] - cnfac * factx[j] * fnx[j - 1]
    
    #------------------------------------------------------------
    # Solve for tdsolution
    #------------------------------------------------------------

    nl_sls = abs(snl) + nl_lake + nl_soil

    # Assuming tridia is a function that solves a tridiagonal matrix system
    tx[-maxsnl+lb-1:] = CoLM_Utils.tridia(nl_sls, a[-maxsnl+lb-1:], b[-maxsnl+lb-1:], c[-maxsnl+lb-1:], r[-maxsnl+lb-1:],tx[-maxsnl+lb-1:])

    for j in range(-maxsnl+lb-1, nl_lake + nl_soil-maxsnl):
        jprime = j - nl_lake
        if j < 5:  # snow layer
            t_soisno[j] = tx[j]
        elif j <= nl_lake-maxsnl-1:  # lake layer
            t_lake[j+maxsnl] = tx[j]
        else:  # soil layer
            t_soisno[jprime] = tx[j]

    # Additional variables for CWRF or other atmospheric models
    emis = emg
    z0m = z0mg
    zol = zeta
    rib = min(5.0, zol * ustar**2 / (const_physical.vonkar * const_physical.vonkar / fh * um**2))

    # Radiative temperature
    trad = (olrg / const_physical.stefnc)**0.25

    # Solar absorption below the surface
    fgrnd = sabg + forc_frl - olrg - fseng - htvp * fevpg
    taux = -forc_rhoair * forc_us / ram
    tauy = -forc_rhoair * forc_vs / ram
    fsena = fseng
    fevpa = fevpg
    lfevpa = htvp * fevpg

    # 2 m height air temperature and specific humidity
    tref = thm + const_physical.vonkar / fh * dth * (fh2m / const_physical.vonkar - fh / const_physical.vonkar)
    qref = forc_q + const_physical.vonkar / fq * dqh * (fq2m / const_physical.vonkar - fq / const_physical.vonkar)

    # Calculate sublimation, frosting, dewing
    qseva = 0.0
    qsubl = 0.0
    qsdew = 0.0
    qfros = 0.0

    if fevpg >= 0.0:
        if lb < 0:
            qseva = min(wliq_soisno[-maxsnl+lb-1] / deltim, fevpg)
            qsubl = fevpg - qseva
        else:
            qseva = min((1.0 - lake_icefrac[0]) * 1000.0 * dz_lake[0] / deltim, fevpg)
            qsubl = fevpg - qseva
    else:
        if t_grnd < const_physical.tfrz:
            qfros = abs(fevpg)
        else:
            qsdew = abs(fevpg)

    # Energy conservation check
    esum1 = 0.0
    esum2 = 0.0
    for j in range(-maxsnl-lb+1, nl_lake + nl_soil-maxsnl):
        # print (tx[j], tx_bef[j], tx[j] - tx_bef[j],'------esum1-----')
        esum1 += (tx[j] - tx_bef[j]) * cvx[j]
        esum2 += (tx[j] - const_physical.tfrz) * cvx[j]

    errsoi = esum1 / deltim - fgrnd
    if abs(errsoi) > 0.1:
        print('energy conservation error in LAND WATER COLUMN during tridiagonal solution,',
              'error (W/m^2):', errsoi, fgrnd)
    #------------------------------------------------------------
    #*[4] Phase change
    #------------------------------------------------------------

    sm = 0.0
    xmf = 0.0
    imelt_soisno[:] = 0
    imelt_lake[:] = 0

    if nl_colm['DEF_USE_SNICAR'] and urban_call is None:
        wice_soisno_bef[lb-maxsnl-1:4] = wice_soisno[lb-maxsnl-1:4]

    # Check for case of snow without snow layers and top lake layer temp above freezing.
    if snl == 0 and scv > 0.0 and t_lake[0] > const_physical.tfrz:
        heatavail = (t_lake[0] - const_physical.tfrz) * cv_lake[0]
        melt = min(scv, heatavail / const_physical.hfus)
        heatrem = max(heatavail - melt * const_physical.hfus, 0.0)  # catch small negative value to keep t at const_physical.tfrz
        t_lake[0] = const_physical.tfrz + heatrem / cv_lake[0]

        snowdp = max(0.0, snowdp * (1.0 - melt / scv))
        scv -= melt

        if scv < 1.e-12:
            scv = 0.0  # prevent tiny residuals
        if snowdp < 1.e-12:
            snowdp = 0.0  # prevent tiny residuals
        sm += melt / deltim
        xmf += melt * const_physical.hfus

    # Lake phase change
    for j in range(nl_lake):
        if t_lake[j] > const_physical.tfrz and lake_icefrac[j] > 0.0:  # melting
            imelt_lake[j] = 1
            heatavail = (t_lake[j] - const_physical.tfrz) * cv_lake[j]
            melt = min(lake_icefrac[j] * const_physical.denh2o * dz_lake[j], heatavail / const_physical.hfus)
            heatrem = max(heatavail - melt * const_physical.hfus, 0.0)  # catch small negative value to keep t at const_physical.tfrz
        elif t_lake[j] < const_physical.tfrz and lake_icefrac[j] < 1.0:  # freezing
            imelt_lake[j] = 2
            heatavail = (t_lake[j] - const_physical.tfrz) * cv_lake[j]
            melt = max(-(1.0 - lake_icefrac[j]) * const_physical.denh2o * dz_lake[j], heatavail / const_physical.hfus)
            heatrem = min(heatavail - melt * const_physical.hfus, 0.0)  # catch small positive value to keep t at const_physical.tfrz

        # Update temperature and ice fraction.
        if imelt_lake[j] > 0:
            lake_icefrac[j] -= melt / (const_physical.denh2o * dz_lake[j])
            if lake_icefrac[j] > 1.0 - 1.e-12:
                lake_icefrac[j] = 1.0  # prevent tiny residuals
            if lake_icefrac[j] < 1.e-12:
                lake_icefrac[j] = 0.0  # prevent tiny residuals
            cv_lake[j] += melt * (const_physical.cpliq - const_physical.cpice)  # update heat capacity
            t_lake[j] = const_physical.tfrz + heatrem / cv_lake[j]
            xmf += melt * const_physical.hfus

    # Snow & soil phase change. Currently does not DO freezing point depression.
    for j in range(-maxsnl-snl, nl_soil-maxsnl):
        if t_soisno[j] > const_physical.tfrz and wice_soisno[j] > 0.0:  # melting
            imelt_soisno[j] = 1
            heatavail = (t_soisno[j] - const_physical.tfrz) * cv_soisno[j]
            melt = min(wice_soisno[j], heatavail / const_physical.hfus)
            heatrem = max(heatavail - melt * const_physical.hfus, 0.0)  # catch small negative value to keep t at const_physical.tfrz
            if j <= 4:
                sm += melt / deltim
        elif t_soisno[j] < const_physical.tfrz and wliq_soisno[j] > 0.0:  # freezing
            imelt_soisno[j] = 2
            heatavail = (t_soisno[j] - const_physical.tfrz) * cv_soisno[j]
            melt = max(-wliq_soisno[j], heatavail / const_physical.hfus)
            heatrem = min(heatavail - melt * const_physical.hfus, 0.0)  # catch small positive value to keep t at const_physical.tfrz

        # Update temperature and soil components.
        if imelt_soisno[j] > 0:
            wice_soisno[j] -= melt
            wliq_soisno[j] += melt
            if wice_soisno[j] < 1.e-12:
                wice_soisno[j] = 0.0  # prevent tiny residuals
            if wliq_soisno[j] < 1.e-12:
                wliq_soisno[j] = 0.0  # prevent tiny residuals
            cv_soisno[j] += melt * (const_physical.cpliq - const_physical.cpice)  # update heat capacity
            t_soisno[j] = const_physical.tfrz + heatrem / cv_soisno[j]
            xmf += melt * const_physical.hfus

    #------------------------------------------------------------

    if nl_colm['DEF_USE_SNICAR'] and urban_call is None:
        # for SNICAR: layer freezing mass flux (positive):
        for j in range(-maxsnl+lb-1, 4):
            if imelt_soisno[j] == 2 and j < 1:
                snofrz[j] = max(0.0, (wice_soisno[j] - wice_soisno_bef[j])) / deltim

    #if defined CoLMDEBUG
    # second energy check and water check. now check energy balance before and after phase
    # change, considering the possibility of changed heat capacity during phase change, by
    # using initial heat capacity in the first step, final heat capacity in the second step,
    # and differences from const_physical.tfrz only to avoid enthalpy correction for (const_physical.cpliq-const_physical.cpice)*melt*const_physical.tfrz.
    # also check soil water sum.
    if nl_colm['CoLMDEBUG']:
        for j in range(nl_lake):
            esum2 -= (t_lake[j] - const_physical.tfrz) * cv_lake[j]

        for j in range(-maxsnl+lb-1, nl_soil-maxsnl):
            esum2 -= (t_soisno[j] - const_physical.tfrz) * cv_soisno[j]

        esum2 -= xmf
        errsoi = esum2 / deltim

        if abs(errsoi) > 0.1:
            print('energy conservation error in LAND WATER COLUMN during phase change, error (W/m^2):', errsoi)
            
    #------------------------------------------------------------
#*[5] Convective mixing: make sure fracice*dz is conserved, heat content c*dz*T is conserved, and
# all ice ends up at the top. Done over all lakes even IF frozen.
# Either an unstable density profile or ice in a layer below an incompletely frozen layer will trigger.
#------------------------------------------------------------

    # recalculate density
    for j in range(nl_lake):

        rhow[j] = (1.0 - lake_icefrac[j]) * 1000.0 * (1.0 - 1.9549e-05 * (abs(t_lake[j] - 277.0))**1.68) + lake_icefrac[j] * const_physical.denice
        # print(lake_icefrac[j], t_lake[j], rhow[j],'-----==------')
    for j in range(nl_lake-1):
        qav = 0.0
        nav = 0.0
        iceav = 0.0

        # print(j,rhow[j] > rhow[j + 1],(lake_icefrac[j] < 1.0 and lake_icefrac[j + 1] > 0.0),'-----rhow-if----')

        if rhow[j] > rhow[j + 1] or (lake_icefrac[j] < 1.0 and lake_icefrac[j + 1] > 0.0):
            for i in range(j + 2):
                qav += dz_lake[i] * (t_lake[i] - const_physical.tfrz) * ((1.0 - lake_icefrac[i]) * cwat + lake_icefrac[i] * cice_eff)
                iceav += lake_icefrac[i] * dz_lake[i]
                nav += dz_lake[i]

            qav /= nav
            iceav /= nav



            # If the average temperature is above freezing, put the extra energy into the water.
            # If it is below freezing, take it away from the ice.

            if qav > 0.0:
                tav_froz = 0.0  # Celsius
                tav_unfr = qav / ((1.0 - iceav) * cwat)
                # print(j,tav_unfr, qav,'----if ---')
            elif qav < 0.0:
                tav_froz = qav / (iceav * cice_eff)
                tav_unfr = 0.0  # Celsius
                # print(j,tav_unfr, qav,'----elif ---')
            else:
                tav_froz = 0.0
                tav_unfr = 0.0
                # print(j,tav_unfr, qav,'----else ---')

        if nav > 0.0:
            for i in range(j + 2):
                # put all the ice at the top.
                # if the average temperature is above freezing, put the extra energy into the water.
                # if it is below freezing, take it away from the ice.
                # for the layer with both ice & water, be careful to use the average temperature
                # that preserves the correct total heat content given what the heat capacity of that
                # layer will actually be.
                if i == 0:
                    zsum = 0.0

                if (zsum + dz_lake[i]) / nav <= iceav:
                    lake_icefrac[i] = 1.0
                    t_lake[i] = tav_froz + const_physical.tfrz
                elif zsum / nav < iceav:
                    lake_icefrac[i] = (iceav * nav - zsum) / dz_lake[i]
                    # Find average value that preserves correct heat content.
                    t_lake[i] = (lake_icefrac[i] * tav_froz * cice_eff +
                                (1.0 - lake_icefrac[i]) * tav_unfr * cwat) / \
                                (lake_icefrac[i] * cice_eff + (1 - lake_icefrac[i]) * cwat) + const_physical.tfrz
                else:
                    lake_icefrac[i] = 0.0
                    t_lake[i] = tav_unfr + const_physical.tfrz

                zsum += dz_lake[i]
                rhow[i] = (1.0 - lake_icefrac[i]) * 1000.0 * (1.0 - 1.9549e-05 * (abs(t_lake[i] - 277.0))**1.68) + lake_icefrac[i] * const_physical.denice
                # print (rhow[i],lake_icefrac[i],t_lake[i] ,const_physical.denice,(1.0 - lake_icefrac[i]) * 1000.0 ,(abs(t_lake[i] - 277.0))**1.68,'-=-=-=-=-=-=-=')
    #------------------------------------------------------------
    #*[6] Re-evaluate thermal properties and sum energy content.
    #------------------------------------------------------------

    # for lake
    for j in range(nl_lake):
        cv_lake[j] = dz_lake[j] * (cwat * (1.0 - lake_icefrac[j]) + cice_eff * lake_icefrac[j])

    # do as above to sum energy content
    ncvts = 0.0
    for j in range(nl_lake):
        ncvts += cv_lake[j] * (t_lake[j] - const_physical.tfrz) + cfus * dz_lake[j] * (1.0 - lake_icefrac[j])

    for j in range(-maxsnl+lb-1, nl_soil-maxsnl):
        ncvts += cv_soisno[j] * (t_soisno[j] - const_physical.tfrz) + const_physical.hfus * wliq_soisno[j]
        if j == 5 and scv > 0 and j == -maxsnl+lb-1:
            ncvts -= scv * const_physical.hfus

    # check energy conservation.
    errsoi = (ncvts - ocvts) / deltim - fgrnd
    if abs(errsoi) < 0.10:
        fseng -= errsoi
        fsena = fseng
        fgrnd += errsoi
        errsoi = 0.0
    else:
        print("energy conservation error in LAND WATER COLUMN during convective mixing", errsoi, fgrnd, ncvts, ocvts)

    return t_grnd, scv, snowdp, t_soisno, wliq_soisno, wice_soisno, imelt_soisno, t_lake, lake_icefrac, savedtke1, snofrz, taux,tauy,fsena,fevpa,lfevpa,fseng,fevpg,qseva,qsubl,qsdew,qfros,olrg,fgrnd,tref,qref,trad,emis,z0m, zol,rib,ustar,qstar,tstar,fm,fh,fq,sm
        
def snowwater_lake(nl_colm, const_physical, maxsnl, nl_soil, nl_lake, deltim, ssi, wimp, porsl, pg_rain, pg_snow, dz_lake, 
                   imelt, fiold, qseva, qsubl, qsdew, qfros, z_soisno, dz_soisno, zi_soisno, t_soisno, 
                   wice_soisno, wliq_soisno, t_lake, lake_icefrac, qout_snowb, fseng, fgrnd, snl, scv, 
                   snowdp, sm, forc_us, forc_vs, forc_aer, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, 
                   mss_dst1, mss_dst2, mss_dst3, mss_dst4, urban_call=None):
    """
    Calculation of Lake Hydrology. Lake water mass is kept constant. The soil is simply maintained at
    volumetric saturation if ice melting frees up pore space.
    
    Called:
        -> snowwater:                  change of snow mass and snow water onto soil
        -> snowcompaction:             compaction of snow layers
        -> combinesnowlayers:          combine snow layers that are thinner than minimum
        -> dividesnowlayers:           subdivide snow layers that are thicker than maximum
    
    Initial: Yongjiu Dai, December, 2012
                              April, 2014
    REVISIONS:
    Nan Wei, 06/2018: update snow hydrology above lake
    Yongjiu Dai, 01/2023: added for SNICAR model effects for snowwater,
    combinesnowlayers, dividesnowlayers processes by calling snowwater_snicar(),
    SnowLayersCombine_snicar, SnowLayersDivide_snicar()
    """
    # for runoff calculation (assumed no mass change in the land water bodies)
    lb = snl + 1  # lower bound of array
    qout_snowb = 0.0

    # ----------------------------------------------------------
    # *[1] snow layer on frozen lake
    # ----------------------------------------------------------
    if snl < 0:
        lb = snl + 1
        if nl_colm['DEF_USE_SNICAR'] and urban_call is None:
            qout_snowb, mss_bcphi, mss_bcpho, mss_ocphi, mss_ocpho, mss_dst1, mss_dst2, 
            mss_dst3, mss_dst4, wice_soisno, wliq_soisno = CoLM_SoilSnowHydrology.snow_water_snicar(lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros,
                             dz_soisno[-maxsnl+lb-1:0], wice_soisno[-maxsnl+lb-1:0], wliq_soisno[-maxsnl+lb-1:0], qout_snowb,
                             forc_aer,
                             mss_bcpho[lb:0], mss_bcphi[lb:0], mss_ocpho[lb:0], mss_ocphi[lb:0],
                             mss_dst1[lb:0], mss_dst2[lb:0], mss_dst3[lb:0], mss_dst4[lb:0])
        else:
            qout_snowb = CoLM_SoilSnowHydrology.snowwater(lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros,
                      dz_soisno[-maxsnl+lb-1:3], wice_soisno[-maxsnl+lb-1:3], wliq_soisno[-maxsnl+lb-1:3], qout_snowb)

        # Natural compaction and metamorphosis.
        lb = snl + 1
        dz_soisno[-maxsnl+lb-1:3] = CoLM_SnowLayersCombineDivide.snowcompaction(lb, deltim, imelt[-maxsnl+lb-1:3], fiold[-maxsnl+lb-1:3], t_soisno[-maxsnl+lb-1:3],
                       wliq_soisno[-maxsnl+lb-1:3], wice_soisno[-maxsnl+lb-1:3], forc_us, forc_vs, dz_soisno[-maxsnl+lb-1:3])

        # Combine thin snow elements
        lb = maxsnl + 1
        if nl_colm['DEF_USE_SNICAR'] and urban_call is None :
            wice_soisno[-maxsnl+lb-1:6], wliq_soisno[-maxsnl+lb-1:6], t_soisno[-maxsnl+lb-1:6], dz_soisno[-maxsnl+lb-1:6], z_soisno[-maxsnl+lb-1:6], zi_soisno[-maxsnl+lb-1:6], snowdp, scv, snl = CoLM_SnowLayersCombineDivide.snowlayerscombine_snicar(lb, snl, z_soisno[-maxsnl+lb-1:6], dz_soisno[-maxsnl+lb-1:6], zi_soisno[-maxsnl+lb-1:6],
                                     wliq_soisno[-maxsnl+lb-1:6], wice_soisno[-maxsnl+lb-1:6], t_soisno[-maxsnl+lb-1:6], scv, snowdp,
                                     mss_bcpho[-maxsnl+lb-1:3], mss_bcphi[-maxsnl+lb-1:3], mss_ocpho[-maxsnl+lb-1:3], mss_ocphi[-maxsnl+lb-1:3],
                                     mss_dst1[-maxsnl+lb-1:3], mss_dst2[-maxsnl+lb-1:3], mss_dst3[-maxsnl+lb-1:3], mss_dst4[-maxsnl+lb-1:3])
        else:
            wliq_soisno[-maxsnl+lb-1:6], wice_soisno[-maxsnl+lb-1:6], t_soisno[-maxsnl+lb-1:6], dz_soisno[-maxsnl+lb-1:6], zi_soisno[-maxsnl+lb-1:6], snowdp, scv, snl =CoLM_SnowLayersCombineDivide.snowlayerscombine(lb, snl, z_soisno[-maxsnl+lb-1:6], dz_soisno[-maxsnl+lb-1:6], zi_soisno[-maxsnl+lb-1:6],
                              wliq_soisno[-maxsnl+lb-1:6], wice_soisno[-maxsnl+lb-1:6], t_soisno[-maxsnl+lb-1:6], scv, snowdp)

        # Divide thick snow elements
        if snl < 0:
            if nl_colm['DEF_USE_SNICAR'] and urban_call is None:
                wice_soisno[-maxsnl+lb-1:3], wliq_soisno[-maxsnl+lb-1:3], t_soisno[-maxsnl+lb-1:3], dz_soisno[-maxsnl+lb-1:3], z_soisno[-maxsnl+lb-1:3], zi_soisno[-maxsnl+lb-1:3], snl = CoLM_SnowLayersCombineDivide.snowlayersdivide_snicar(lb, snl, z_soisno[-maxsnl+lb-1:3], dz_soisno[-maxsnl+lb-1:3], zi_soisno[-maxsnl+lb-1:3],
                                        wliq_soisno[-maxsnl+lb-1:3], wice_soisno[-maxsnl+lb-1:3], t_soisno[-maxsnl+lb-1:3],
                                        mss_bcpho[-maxsnl+lb-1:3], mss_bcphi[-maxsnl+lb-1:3], mss_ocpho[-maxsnl+lb-1:3], mss_ocphi[-maxsnl+lb-1:3],
                                        mss_dst1[-maxsnl+lb-1:3], mss_dst2[-maxsnl+lb-1:3], mss_dst3[-maxsnl+lb-1:3], mss_dst4[-maxsnl+lb-1:3])
            else:
                snl,  wliq_soisno[-maxsnl+lb-1:3], wice_soisno[-maxsnl+lb-1:3], t_soisno[-maxsnl+lb-1:3], dz_soisno[-maxsnl+lb-1:3], z_soisno[-maxsnl+lb-1:3], zi_soisno[-maxsnl+lb-1:3] =CoLM_SnowLayersCombineDivide.snowlayersdivide(lb, snl, z_soisno[-maxsnl+lb-1:3], dz_soisno[-maxsnl+lb-1:3], zi_soisno[-maxsnl+lb-1:3],
                                 wliq_soisno[-maxsnl+lb-1:3], wice_soisno[-maxsnl+lb-1:3], t_soisno[-maxsnl+lb-1:3])
    # ----------------------------------------------------------
    # *[2] check for single completely unfrozen snow layer over lake.
    #     Modeling this ponding is unnecessary and can cause instability after the timestep
    #     when melt is completed, as the temperature after melt can be excessive
    #     because the fluxes were calculated with a fixed ground temperature of freezing, but the
    #     phase change was unable to restore the temperature to freezing.  (Zack Subnin 05/2010)
    # ----------------------------------------------------------

    if snl == -1 and wice_soisno[4] == 0.0:
        # Remove layer
        # Take extra heat of layer and release to sensible heat in order to maintain energy conservation.
        heatrem = const_physical.cpliq * wliq_soisno[4] * (t_soisno[4] - const_physical.tfrz)
        fseng += heatrem / deltim
        fgrnd -= heatrem / deltim

        snl = 0
        scv = 0.0
        snowdp = 0.0

    # ----------------------------------------------------------
    # *[3] check for snow layers above lake with unfrozen top layer. Mechanically,
    #     the snow will fall into the lake and melt or turn to ice. IF the top layer has
    #     sufficient heat to melt the snow without freezing, THEN that will be done.
    #     Otherwise, the top layer will undergo freezing, but only IF the top layer will
    #     not freeze completely. Otherwise, let the snow layers persist and melt by diffusion.
    # ----------------------------------------------------------

    if t_lake[0] > const_physical.tfrz and snl < 0 and lake_icefrac[0] < 0.001:  # for unfrozen lake
        unfrozen = True
    else:
        unfrozen = False

    sumsnowice = 0.0
    sumsnowliq = 0.0
    heatsum = 0.0

    for j in range(snl-maxsnl, 3):
        if unfrozen:
            sumsnowice += wice_soisno[j]
            sumsnowliq += wliq_soisno[j]
            heatsum += wice_soisno[j] * const_physical.cpice * (const_physical.tfrz - t_soisno[j]) + wliq_soisno[j] * const_physical.cpliq * (const_physical.tfrz - t_soisno[j])

    if unfrozen:
        # changed by weinan as the subroutine newsnow_lake
        # Remove snow and subtract the latent heat from the top layer.

        t_ave = const_physical.tfrz - heatsum / (sumsnowice * const_physical.cpice + sumsnowliq * const_physical.cpliq)

        a = heatsum
        b = sumsnowice * const_physical.hfus
        c = (t_lake[0] - const_physical.tfrz) * const_physical.cpliq * const_physical.denh2o * dz_lake[0]
        d = const_physical.denh2o * dz_lake[0] * const_physical.hfus

        # all snow melt
        if c >= a + b:
            t_lake[0] = (const_physical.cpliq * (const_physical.denh2o * dz_lake[0] * t_lake[0] + (sumsnowice + sumsnowliq) * const_physical.tfrz) - a - b) / \
                        (const_physical.cpliq * (const_physical.denh2o * dz_lake[0] + sumsnowice + sumsnowliq))
            sm += scv / deltim
            scv = 0.0
            snowdp = 0.0
            snl = 0
        # lake partially freezing to melt all snow
        elif c + d >= a + b:
            t_lake[0] = const_physical.tfrz
            sm += scv / deltim
            scv = 0.0
            snowdp = 0.0
            snl = 0
            lake_icefrac[0] = (a + b - c) / d

        # snow do not melt while all lake freezing
        # elif c + d < a:
        #     t_lake[1] = (c + d + const_physical.cpice * (sumsnowice * t_ave + const_physical.denh2o * dz_lake[1] * const_physical.tfrz) + const_physical.cpliq * sumsnowliq * t_ave) / \
        #                 (const_physical.cpice * (sumsnowice + const_physical.denh2o * dz_lake[1]) + const_physical.cpliq * sumsnowliq)
        #     lake_icefrac[1] = 1.0
        
    # ----------------------------------------------------------
    # *[4] Soil water and ending water balance
    # ----------------------------------------------------------
    # Here this consists only of making sure that soil is saturated even as it melts and
    # pore space opens up. Conversely, if excess ice is melting and the liquid water exceeds the
    # saturation value, then remove water.

    for j in range(5, nl_soil-maxsnl):
        a = wliq_soisno[j] / (dz_soisno[j] * const_physical.denh2o) + wice_soisno[j] / (dz_soisno[j] * const_physical.denice)

        if a < porsl[j+maxsnl]:
            wliq_soisno[j] = max(0.0, (porsl[j+maxsnl] * dz_soisno[j] - wice_soisno[j] / const_physical.denice) * const_physical.denh2o)
            wice_soisno[j] = max(0.0, (porsl[j+maxsnl] * dz_soisno[j] - wliq_soisno[j] / const_physical.denh2o) * const_physical.denice)
        else:
            wliq_soisno[j] = max(0.0, wliq_soisno[j] - (a - porsl[j+maxsnl]) * const_physical.denh2o * dz_soisno[j])
            wice_soisno[j] = max(0.0, (porsl[j+maxsnl] * dz_soisno[j] - wliq_soisno[j] / const_physical.denh2o) * const_physical.denice)

        if wliq_soisno[j] > porsl[j+maxsnl] * const_physical.denh2o * dz_soisno[j]:
            wliq_soisno[j] = porsl[j+maxsnl] * const_physical.denh2o * dz_soisno[j]
            wice_soisno[j] = 0.0
    return  z_soisno, dz_soisno, zi_soisno, t_soisno, wliq_soisno, wice_soisno, t_lake, lake_icefrac, qout_snowb,  fseng, fgrnd, snl, scv, snowdp,sm ,mss_bcpho,mss_bcphi,mss_ocpho , mss_ocphi, mss_dst1 , mss_dst2 , mss_dst3  , mss_dst4