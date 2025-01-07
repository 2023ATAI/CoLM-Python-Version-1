import CoLM_Qsadv
import CoLM_Hydro_SoilFunction
import CoLM_GroundFluxes
import CoLM_LeafTemperature
import CoLM_Eroot
import CoLM_SoilSurfaceResistance
import CoLM_GroundTemperature
import math
import numpy as np

def THERMAL(nl_colm,const_physical, var_global, const_lc,patchclass,
            ipatch      ,patchtype   ,lb          ,deltim     ,
            trsmx0      ,zlnd        ,zsno        ,csoilc     ,
            dewmx       ,capr        ,cnfac       ,vf_quartz  ,
            vf_gravels  ,vf_om       ,vf_sand     ,wf_gravels ,
            wf_sand     ,csol        ,porsl       ,psi0       ,bsw,
            theta_r     ,alpha_vgm   ,n_vgm       ,L_vgm      ,
            sc_vgm      ,fc_vgm      ,                         
            k_solids    ,dksatu      ,dksatf      ,dkdry      ,
            BA_alpha    ,BA_beta                              ,
            lai         ,laisun      ,laisha                  ,
            sai         ,htop        ,hbot        ,sqrtdi     ,
            rootfr      ,rstfacsun_out   ,rstfacsha_out       , rss,
            gssun_out   ,gssha_out   ,
assimsun_out      ,etrsun_out        ,assimsha_out      ,etrsha_out        ,
            effcon      ,vmax25      ,hksati      ,smp        ,hk,
            kmax_sun    ,kmax_sha    ,kmax_xyl    ,kmax_root  ,
            psi50_sun   ,psi50_sha   ,psi50_xyl   ,psi50_root ,
            ck          ,vegwp       ,gs0sun      ,gs0sha     ,
            lai_old     ,o3uptakesun ,o3uptakesha ,forc_ozone, 
            slti        ,hlti        ,shti        ,hhti       ,
            trda        ,trdm        ,trop        ,g1         ,
            g0          ,gradm       ,binter      ,extkn      ,
            forc_hgt_u  ,forc_hgt_t  ,forc_hgt_q  ,forc_us    ,
            forc_vs     ,forc_t      ,forc_q      ,forc_rhoair,
            forc_psrf   ,forc_pco2m  ,forc_hpbl   ,forc_po2m  ,
            coszen      ,parsun      ,parsha      ,sabvsun    ,
            sabvsha     ,sabg,sabg_soil,sabg_snow ,frl        ,
            extkb       ,extkd       ,thermk      ,fsno       ,
            sigf        ,dz_soisno   ,z_soisno    ,zi_soisno  ,
            tleaf       ,t_soisno    ,wice_soisno ,wliq_soisno,
            ldew,ldew_rain,ldew_snow ,scv,snowdp  ,imelt      ,
            taux        ,tauy        ,fsena       ,fevpa      ,
            lfevpa      ,fsenl       ,fevpl       ,etr        ,
            fseng       ,fevpg       ,olrg        ,fgrnd      ,
            rootr       ,rootflux    ,
            qseva       ,qsdew       ,qsubl       ,qfros      ,
            qseva_soil  ,qsdew_soil  ,qsubl_soil  ,qfros_soil ,
            qseva_snow  ,qsdew_snow  ,qsubl_snow  ,qfros_snow ,
            sm          ,tref        ,qref        ,
            trad        ,rst         ,assim       ,respc      ,
            errore      ,emis        ,z0m         ,zol        ,
            rib         ,ustar       ,qstar       ,tstar      ,
            fm          ,fh          ,fq          ,pg_rain    ,
            pg_snow     ,t_precip    ,qintr_rain  ,qintr_snow ,
            snofrz      ,sabg_snow_lyr):
    # This is the main subroutine to execute the calculation of thermal processes and surface fluxes.
    # Original author: Yongjiu Dai, 09/15/1999; 08/30/2002
    #
    # FLOW DIAGRAM FOR THERMAL.F90
    #
    # THERMAL ===> es, esdT, qs, qsdT = CoLM_Qsadv.qsadv
    #              groundfluxes
    #              eroot                      |dewfraction
    #              LeafTemp   |               |es, esdT, qs, qsdT = CoLM_Qsadv.qsadv
    #              LeafTempPC |  ---------->  |moninobukini
    #                                         |moninobuk
    #                                         |MOD_AssimStomataConductance
    #
    #              groundTem     ---------->   meltf
    #
    #
    # REVISIONS:
    # Hua Yuan, 12/2019: added initial codes for PFT and Plant Community (PC)
    #                    vegetation classification processes
    # Nan Wei,  01/2021: added variables passing of plant hydraulics and precipitation sensible heat
    #                    with canopy and ground for PFT and Plant Community (PC)

    """
    Determine fraction of foliage covered by water and fraction of foliage that is dry and transpiring.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         : xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    Args:
        ipatch (INTEGER):  patch index [-]
        lb (INTEGER): lower bound of array [-]
        patchtype (INTEGER):! land water TYPE (0=soil, 1=urban or built-up, 2=wetland, 3=glacier/ice sheet, 4=land water bodies)


        deltim (float): model time step [second]
        trsmx0 (float): max transpiration for moist soil+100% veg.  [mm/s]
        zlnd (float)  : roughness length for soil [m]
        zsno (float)  : roughness length for snow [m]
        deltim (float): drag coefficient for soil under canopy [-]
        trsmx0 (float): maximum dew
        zlnd (float)  : tuning factor to turn first layer T into surface T
        zsno (float)  : Crank Nicholson factor between 0 and 1

        ##! soil physical parameters
        vf_quartz [var_global.nl_soil](array):  volumetric fraction of quartz within mineral soil
        vf_gravels[var_global.nl_soil](array):  volumetric fraction of gravels
        vf_om     [var_global.nl_soil](array):  volumetric fraction of organic matter
        vf_sand   [var_global.nl_soil](array):  volumetric fraction of sand
        wf_gravels[var_global.nl_soil](array):  gravimetric fraction of gravels
        wf_sand   [var_global.nl_soil](array):  gravimetric fraction of sand
        csol      [var_global.nl_soil](array):  heat capacity of soil solids [J/(m3 K)]
        porsl     [var_global.nl_soil](array):  soil porosity [-]
        psi0      [var_global.nl_soil](array):  soil water suction, negative potential [mm]

        theta_r  [var_global.nl_soil](array):  VG parameters for soil water retention curve
        alpha_vgm[var_global.nl_soil](array):  VG parameters for soil water retention curve
        n_vgm    [var_global.nl_soil](array):  VG parameters for soil water retention curve
        L_vgm    [var_global.nl_soil](array):  VG parameters for soil water retention curve
        sc_vgm   [var_global.nl_soil](array):  VG parameters for soil water retention curve
        fc_vgm   [var_global.nl_soil](array):  VG parameters for soil water retention curve

        k_solids  [var_global.nl_soil](array):  thermal conductivity of minerals soil [W/m-K]
        dkdry     [var_global.nl_soil](array):  thermal conductivity of dry soil [W/m-K]
        dksatu    [var_global.nl_soil](array):  thermal conductivity of saturated unfrozen soil [W/m-K]
        dksatf    [var_global.nl_soil](array):  thermal conductivity of saturated frozen soil [W/m-K]
        hksati    [var_global.nl_soil](array):  hydraulic conductivity at saturation [mm h2o/s]
        BA_alpha  [var_global.nl_soil](array):  alpha in Balland and Arp(2005) thermal conductivity scheme
        BA_beta   [var_global.nl_soil](array):  beta in Balland and Arp(2005) thermal conductivity scheme

        
        ##!  vegetation parameters
        lai (float):  adjusted leaf area index for seasonal variation [-]
        sai (float):  stem area index  [-]
        htop (float):  canopy crown top height [m]
        hbot (float):  canopy crown bottom height [m]
        sqrtdi (float):  inverse sqrt of leaf dimension [m**-0.5]
        rootfr[var_global.nl_soil](array):root fraction

        effcon (float):  quantum efficiency of RuBP regeneration (mol CO2/mol quanta)
        vmax25 (float):  maximum carboxylation rate at 25 C at canopy top
        kmax_sun (float): 
        kmax_sha (float): 
        kmax_xyl (float): 
        kmax_root (float): 
        psi50_sun (float): water potential at 50% loss of sunlit leaf tissue conductance (mmH2O)
        psi50_sha (float): water potential at 50% loss of shaded leaf tissue conductance (mmH2O)
        psi50_xyl (float): water potential at 50% loss of xylem tissue conductance (mmH2O)
        psi50_root (float): ! water potential at 50% loss of root tissue conductance (mmH2O)
        ck (float):  shape-fitting parameter for vulnerability curve (-)
        slti (float):  slope of low temperature inhibition function      [s3]
        hlti (float):  1/2 point of low temperature inhibition function  [s4]
        shti (float):  slope of high temperature inhibition function     [s1]
        hhti (float):  1/2 point of high temperature inhibition function [s2]
        trda (float):  temperature coefficient in gs-a model             [s5]
        trdm (float):  temperature coefficient in gs-a model             [s6]
        trop (float):  temperature coefficient in gs-a model
        gradm (float):  conductance-photosynthesis slope parameter
        binter (float):  conductance-photosynthesis intercept
        extkn (float):  coefficient of leaf nitrogen allocation


        ##! atmospherical variables and observational height
        forc_hgt_u (float):  observational height of wind [m]
        forc_hgt_t (float):  observational height of temperature [m]
        forc_hgt_q (float):  observational height of humidity [m]
        forc_u (float):  wind component in eastward direction [m/s]
        forc_v (float):  wind component in northward direction [m/s]
        forc_t (float):  temperature at agcm reference height [kelvin]
        forc_q (float):  specific humidity at agcm reference height [kg/kg]
        forc_rhoair (float):  density air [kg/m3]
        forc_psrf (float): atmosphere pressure at the surface [pa]
        forc_pco2m (float):  CO2 concentration in atmos. (pascals)
        forc_po2m (float): O2 concentration in atmos. (pascals)
        forc_hpbl (float): atmospheric boundary layer height [m]
        pg_rain (float):  rainfall onto ground including canopy runoff [kg/(m2 s)]
        pg_snow (float):  snowfall onto ground including canopy runoff [kg/(m2 s)]
        t_precip (float):  snowfall/rainfall temperature [kelvin]
        qintr_rain (float):  rainfall interception [(mm h2o/s)]
        qintr_snow (float):  snowfall interception [(mm h2o/s)]

        ##!  radiative fluxes
        coszen (float):  cosine of the solar zenith angle
        parsun (float):  photosynthetic active radiation by sunlit leaves (W m-2)
        parsha (float):  photosynthetic active radiation by shaded leaves (W m-2)
        sabvsun (float):  solar radiation absorbed by vegetation [W/m2]
        sabvsha (float):  solar radiation absorbed by vegetation [W/m2]
        sabg (float):  solar radiation absorbed by ground [W/m2]
        frl (float):  atmospheric infrared (longwave) radiation [W/m2]
        extkb (float):  (k, g(mu)/mu) direct solar extinction coefficient
        extkd (float):  diffuse and scattered diffuse PAR extinction coefficient
        thermk (float):  canopy gap fraction for tir radiation

        ##! state variable (1)
        fsno (float):  fraction of ground covered by snow
        sigf (float):  fraction of veg cover, excluding snow-covered veg [-]
        dz_soisno[lb:var_global.nl_soil] (array):  layer thickiness [m]
        z_soisno [lb:var_global.nl_soil] (array):  node depth [m]
        zi_soisno[lb-1:var_global.nl_soil]  ! interface depth [m]


    Returns:
        fwet (float): fraction of foliage covered by water [-]
        fdry (float): fraction of foliage that is green and dry [-]
    """
# ----------------------------------------------------------------------------------------------------------------------------
# [1] Initial set and propositional variables
# ----------------------------------------------------------------------------------------------------------------------------
    # emissivity
    emg = 0.96
    if scv > 0 or patchtype == 3:
        emg = 0.97

    taux = 0.
    tauy = 0.
    fsena = 0.
    fevpa = 0.
    lfevpa = 0.
    fsenl = 0.
    fevpl = 0.
    etr = 0.
    fseng = 0.
    fevpg = 0.

    cgrnd = 0.0  # deriv. of soil energy flux wrt to soil temp [w/m2/k]
    cgrndl = 0.0  # deriv=0.0  of soil sensible heat flux wrt soil temp [w/m2/k]
    cgrnds = 0.0  # deriv of soil latent heat flux wrt soil temp [w/m**2/k]
    tref = 0.0  # 2 m height air temperature [kelvin]
    qref = 0.0  # 2 m height air specific humidity
    rst = 2.0e4  # stomatal resistance (s m-1)
    assim = 0.0  # assimilation
    respc = 0.0  # respiration
    hprl = 0

    emis = 0.
    z0m = 0.
    zol = 0.
    rib = 0.
    ustar = 0.
    qstar = 0.
    tstar = 0.
    rootr = np.zeros(var_global.nl_soil)
    rootflux = np.zeros(var_global.nl_soil)

    dlrad = 0.0  # downward longwave radiation blow the canopy [W/m2]
    t_soil = 0.0  # ground soil temperature
    t_snow = 0.0  # ground snow temperature
#---------------------Local Variables-----------------------------------
    fseng_soil=0.0    # sensible heat flux from soil fraction
    fseng_snow=0.0    # sensible heat flux from snow fraction
    fevpg_soil=0.0    # latent heat flux from soil fraction
    fevpg_snow=0.0    # latent heat flux from snow fraction


    degdT=0.0         # d(eg)/dT
    dqgdT=0.0         # d(qg)/dT

    eg=0.0            # water vapor pressure at temperature T [pa]
    egsmax=0.0        # max. evaporation which soil can provide at one time step
    egidif=0.0        # the excess of evaporation over "egsmax"
                    #glaciers and water surface; 0.96 for soil and wetland)
    errore=0.0        # energy balnce error [w/m2]
    etrc=0.0          # maximum possible transpiration rate [mm/s]
    fac=0.0           # soil wetness of surface layer
    fact = np.zeros(var_global.nl_soil + 1 - lb)  # used in computing tridiagonal matrix
    fsun=0.0          # fraction of sunlit canopy
    hr=0.0            # relative humidity
    htvp=0.0          # latent heat of vapor of water (or sublimation) [j/kg]
    olru=0.0          # olrg excluding dwonwelling reflection [W/m2]
    olrb=0.0          # olrg assuming blackbody emission [W/m2]
    psit=0.0          # negative potential of soil
    qg=0.0            # ground specific humidity [kg/kg]

    fm = 0.0
    fh = 0.0
    fq = 0.0

    q_soil= 0.0       # ground soil specific humudity [kg/kg]
    q_snow= 0.0       # ground snow specific humudity [kg/kg]
    qsatg= 0.0        # saturated humidity [kg/kg]
    qsatgdT= 0.0      # d(qsatg)/dT
    qred= 0.0         # soil surface relative humidity
    sabv= 0.0         # solar absorbed by canopy [W/m2]
    thm= 0.0          # intermediate variable (forc_t+0.0098*forc_hgt_t)
    th= 0.0           # potential temperature (kelvin)
    thv= 0.0          # virtual potential temperature (kelvin)
    rstfac= 0.0       # factor of soil water stress
    t_grnd= 0.0       # ground surface temperature [K]
    t_grnd_bef= 0.0   # ground surface temperature [K]

    t_soisno_bef=np.zeros(var_global.nl_soil + 1 - lb) # soil/snow temperature before update
    tinc= 0.0         # temperature difference of two time step
    ur= 0.0           # wind speed at reference height [m/s]
    ulrad= 0.0        # upward longwave radiation above the canopy [W/m2]
    wice0=np.zeros(var_global.nl_soil+ 1- lb)# ice mass from previous time-step
    wliq0=np.zeros(var_global.nl_soil+ 1- lb)# liquid mass from previous time-step
    wx= 0.0           # patitial volume of ice and water of surface layer
    xmf= 0.0          # total latent heat of phase change of ground water
    hprl=0.0           # precipitation sensible heat from canopy
    z0m_g=0.0   
    z0h_g=0.0  
    zol_g=0.0  
    obu_g=0.0  
    rib_g=0.0  
    ustar_g=0.0  
    qstar_g=0.0  
    tstar_g=0.0
    fm10m=0.0  
    fm_g=0.0  
    fh_g=0.0   
    fq_g=0.0  
    fh2m=0.0  
    fq2m=0.0  
    um=0.0   
    obu=0.0
    #Ozone stress variables
    o3coefv_sun=0.0   
    o3coefv_sha=0.0   
    o3coefg_sun=0.0   
    o3coefg_sha=0.0

    dlrad = frl

    t_soil = t_soisno[0]
    t_snow = t_soisno[lb-1]

    if not nl_colm['DEF_SPLIT_SOILSNOW']:
        t_grnd = t_soisno[lb-1]
        ulrad = frl * (1. - emg) + emg * const_physical.stefnc * t_grnd ** 4
    else:
        t_grnd = fsno * t_snow + (1. - fsno) * t_soil
        ulrad = frl * (1. - emg) + fsno * emg * const_physical.stefnc * t_snow ** 4 + (1. - fsno) * emg * const_physical.stefnc * t_soil ** 4

    # temperature and water mass from previous time step
    t_soisno_bef[lb-1:] = t_soisno[lb-1:]
    t_grnd_bef = t_grnd
    wice0[lb-1:] = wice_soisno[lb-1:]
    wliq0[lb-1:] = wliq_soisno[lb-1:]

    # latent heat, assumed that the sublimation occured only as wliq_soisno=0
    htvp = const_physical.hvap
    if wliq_soisno[lb-1] <= 0 and wice_soisno[lb-1] > 0:
        htvp = const_physical.hsub

    # potential temperature at the reference height
    thm = forc_t + 0.0098 * forc_hgt_t
    th = forc_t * (100000. / forc_psrf) ** (const_physical.rgas / const_physical.cpair)
    thv = th * (1. + 0.61 * forc_q)
    ur = max(0.1, np.sqrt(forc_us ** 2 + forc_vs ** 2))
# ----------------------------------------------------------------------------------------------------------------------------
# [2] specific humidity and its derivative at ground surface
# ----------------------------------------------------------------------------------------------------------------------------
    qred = 1.
    hr = 1.

    if patchtype <= 1:  # soil ground
        wx = (wliq_soisno[0] / const_physical.denh2o + wice_soisno[0] / const_physical.denice) / dz_soisno[0]
        if porsl[0] < 1.e-6:  # bed rock
            fac = 0.001
        else:
            fac = min(1., wx / porsl[0])
            fac = max(fac, 0.001)

        # Calculate soil water potential based on different soil models
        psit = 0
        if nl_colm["Campbell_SOIL_MODEL"]:
            psit = psi0[0] * fac ** (-bsw[0])  # psit = max(smpmin, psit)
        if nl_colm["vanGenuchten_Mualem_SOIL_MODEL"]:
            psit = CoLM_Hydro_SoilFunction.soil_psi_from_vliq(nl_colm, fac * (porsl[0] - theta_r[0]) + theta_r[0], porsl[0], theta_r[0], psi0[0],
                                      5, [alpha_vgm[0], n_vgm[0], L_vgm[0], sc_vgm[0], fc_vgm[0]])
        psit = max(-1.e8, psit)
        hr = math.exp(psit / const_physical.roverg / t_grnd)
        qred = (1. - fsno) * hr + fsno

    if not nl_colm['DEF_SPLIT_SOILSNOW']:
        eg, degdT, qsatg, qsatgdT = CoLM_Qsadv.qsadv(t_grnd, forc_psrf)

        qg = qred * qsatg
        dqgdT = qred * qsatgdT

        if qsatg > forc_q > qred * qsatg:
            qsatg = forc_q
            qsatgdT = 0.

        q_soil = qg
        q_snow = qg

    else:
        eg, degdT, qsatg, qsatgdT = CoLM_Qsadv.qsadv(t_soil, forc_psrf)

        q_soil = hr * qsatg
        dqgdT = (1. - fsno) * hr * qsatgdT

        if qsatg > forc_q > hr * qsatg:
            qsatg = forc_q
            qsatgdT = 0.
        eg, degdT, qsatg, qsatgdT = CoLM_Qsadv.qsadv(t_snow, forc_psrf)

        q_snow = qsatg
        dqgdT = dqgdT + fsno * qsatgdT

        # weighted average qg
        qg = (1. - fsno) * q_soil + fsno * q_snow

    # Calculate soil surface resistance (rss)
    # Do NOT calculate rss for the first timestep
    if nl_colm['DEF_RSS_SCHEME'] > 0 and rss != var_global. spval:
        rss = CoLM_SoilSurfaceResistance.SoilSurfaceResistance(nl_colm, const_physical, var_global.nl_soil, forc_rhoair, hksati, porsl, psi0,
                                     bsw ,
                                     theta_r,
                                     alpha_vgm,
                                     n_vgm,
                                     L_vgm,
                                     sc_vgm,
                                     fc_vgm,
                                     dz_soisno, t_soisno, wliq_soisno, wice_soisno, fsno, qg)
    else:
        rss = 0.
# ----------------------------------------------------------------------------------------------------------------------------
#  [3] Compute sensible and latent fluxes and their derivatives with respect
#      to ground temperature using ground temperatures from previous time step.
#
# ----------------------------------------------------------------------------------------------------------------------------
# Always call GroundFluxes for bare ground CASE
    taux, tauy, fseng, fseng_soil, fseng_snow,fevpg, fevpg_soil, fevpg_snow, cgrnd, cgrndl, cgrnds, tref, qref, z0m_g, z0h_g, zol_g, rib_g, ustar_g, tstar_g, qstar_g, fm_g, fh_g, fq_g = CoLM_GroundFluxes.GroundFluxes(const_physical, zlnd, zsno, forc_hgt_u, forc_hgt_t, forc_hgt_q, forc_hpbl,
                 forc_us, forc_vs, forc_t, forc_q, forc_rhoair, forc_psrf,
                 ur, thm, th, thv, t_grnd, qg, rss, dqgdT, htvp,
                 fsno, cgrnd, cgrndl, cgrnds,
                 t_soil, t_snow, q_soil, q_snow,
                 taux, tauy, fseng, fseng_soil, fseng_snow,
                 fevpg, fevpg_soil, fevpg_snow, tref, qref,
                 z0m_g, z0h_g, zol_g, rib_g, ustar_g, qstar_g, tstar_g, fm_g, fh_g, fq_g,nl_colm,
                                        const_physical.cpair, const_physical.vonkar, const_physical.grav)
    obu_g = forc_hgt_u / zol_g #divide by zero
# ----------------------------------------------------------------------------------------------------------------------------
#  [4] Canopy temperature, fluxes from the canopy
# ----------------------------------------------------------------------------------------------------------------------------
    if patchtype == 0 and nl_colm['DEF_USE_LCT'] or patchtype > 0:

        sabv = sabvsun + sabvsha

        if lai + sai > 1e-6:
            # soil water stress factor on stomatal resistance

            rootr, etrc, rstfac = CoLM_Eroot.eroot(nl_colm,var_global.nl_soil, trsmx0, porsl,
                                                   psi0, rootfr, dz_soisno, t_soisno, wliq_soisno,
                                                   const_physical.tfrz,
                                                   bsw, theta_r, alpha_vgm, n_vgm, L_vgm, sc_vgm, fc_vgm)
            # print(rstfac, '-----rstfac2-----')
            # fraction of sunlit and shaded leaves of canopy
            fsun = (1. - math.exp(-min(extkb * lai, 40.))) / max(min(extkb * lai, 40.), 1.e-6)

            if coszen <= 0.0 or sabv < 1.:
                fsun = 0.5

            laisun = lai * fsun
            laisha = lai * (1 - fsun)
            rstfacsun_out = rstfac
            rstfacsha_out = rstfac
            # print (etr,'---etr0----')

            sai, vegwp, gs0sun, gs0sha, tleaf, ldew, ldew_rain, ldew_snow, lai_old, o3uptakesun, o3uptakesha, forc_ozone, taux, tauy, fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, cgrnd, cgrndl, cgrnds, tref, qref, rstfacsun_out, rstfacsha_out, gssun_out, gssha_out, rootflux, assimsun_out, etrsun_out, assimsha_out, etrsha_out, rst, assim, respc, fsenl, fevpl, etr, dlrad, ulrad, hprl, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq = CoLM_LeafTemperature.LeafTemperature(
                nl_colm,
                var_global,
                const_lc,
                const_physical,
                patchclass,
                ipatch, 1, deltim, csoilc, dewmx, htvp,
                lai, sai, htop, hbot, sqrtdi,
                effcon, vmax25, slti, hlti, shti,
                hhti, trda, trdm, trop, g1, g0, gradm,
                binter, extkn, extkb, extkd, forc_hgt_u,
                forc_hgt_t, forc_hgt_q, forc_us, forc_vs, thm,
                th, thv, forc_q, forc_psrf, forc_rhoair,
                parsun, parsha, sabv, frl, fsun,
                thermk, rstfacsun_out, rstfacsha_out,
                gssun_out, gssha_out, forc_po2m, forc_pco2m, z0h_g,
                obu_g, ustar_g, zlnd, zsno, fsno, sigf, etrc, t_grnd,
                qg, rss, t_soil, t_snow, q_soil, q_snow, dqgdT,
                emg, tleaf, ldew, ldew_rain, ldew_snow,
                taux, tauy, fseng, fseng_soil, fseng_snow,
                fevpg, fevpg_soil, fevpg_snow,
                cgrnd, cgrndl, cgrnds,
                tref, qref, rst, assim, respc,
                fsenl, fevpl, etr, dlrad, ulrad,
                z0m, zol, rib, ustar, qstar,
                tstar, fm, fh, fq, rootfr,
                kmax_sun, kmax_sha, kmax_xyl, kmax_root, psi50_sun,
                psi50_sha, psi50_xyl, psi50_root, ck, vegwp,
                gs0sun, gs0sha, assimsun_out, etrsun_out,
                assimsha_out, etrsha_out,
                # Ozone stress variables
                o3coefv_sun, o3coefv_sha, o3coefg_sun, o3coefg_sha,
                lai_old, o3uptakesun, o3uptakesha, forc_ozone,
                # end ozone stress variables
                forc_hpbl,
                qintr_rain, qintr_snow, t_precip, hprl, smp,
                hk, hksati, rootflux)

        else:
            tleaf = forc_t
            laisun = 0.
            laisha = 0.
            ldew_rain = 0.
            ldew_snow = 0.
            ldew = 0.
            rstfacsun_out = 0.
            rstfacsha_out = 0.
            if nl_colm['DEF_USE_PLANTHYDRAULICS']:
                vegwp = -2.5e4
    if (nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']):
        pass
        # if patchtype == 0:
        #     ps = landpft. patch_pft_s[ipatch]
        #     pe = landpft. patch_pft_e[ipatch]

        #     rootr_p = np.zeros((var_global.nl_soil, pe - ps))
        #     rootflux_p = np.zeros((var_global.nl_soil, pe - ps))
        #     etrc_p = np.zeros(pe - ps)
        #     rstfac_p = np.zeros(pe - ps)
        #     rstfacsun_p = np.zeros(pe - ps)
        #     rstfacsha_p = np.zeros(pe - ps)
        #     gssun_p = np.zeros(pe - ps)
        #     gssha_p = np.zeros(pe - ps)
        #     fsun_p = np.zeros(pe - ps)
        #     sabv_p = np.zeros(pe - ps)
            
        #     # IF (DEF_USE_PFT .or. time_invariants.patchclass(ipatch)==CROPLAND) THEN
        #     if nl_colm['DEF_USE_PFT'] or time_invariants.patchclass[ipatch] == nl_colm['CROPLAND']:
        #         fseng_soil_p = np.zeros(pe - ps)
        #         fseng_snow_p = np.zeros(pe - ps)
        #         fevpg_soil_p = np.zeros(pe - ps)
        #         fevpg_snow_p = np.zeros(pe - ps)
        #         cgrnd_p = np.zeros(pe - ps)
        #         cgrnds_p = np.zeros(pe - ps)
        #         cgrndl_p = np.zeros(pe - ps)
        #         dlrad_p = np.zeros(pe - ps)
        #         ulrad_p = np.zeros(pe - ps)
        #         zol_p = np.zeros(pe - ps)
        #         rib_p = np.zeros(pe - ps)
        #         ustar_p = np.zeros(pe - ps)
        #         qstar_p = np.zeros(pe - ps)
        #         tstar_p = np.zeros(pe - ps)
        #         fm_p = np.zeros(pe - ps)
        #         fh_p = np.zeros(pe - ps)
        #         fq_p = np.zeros(pe - ps)
            
        #     hprl_p = np.zeros(pe - ps)
        #     assimsun_p = np.zeros(pe - ps)
        #     etrsun_p = np.zeros(pe - ps)
        #     assimsha_p = np.zeros(pe - ps)
        #     etrsha_p = np.zeros(pe - ps)

        #     sabv_p[ps:pe+1] = sabvsun_p[ps:pe+1] + sabvsha_p[ps:pe+1]
        #     sabv = sabvsun + sabvsha

        #     for i in range(ps, pe + 1):
        #         p = time_invariants.pftclass[i]

        #         if lai_p[i] + sai_p[i] > 1e-6:
        #             rootr, etrc, rstfac = CoLM_Eroot.eroot(var_global.nl_soil, trsmx0, porsl, psi0, constpft.rootfr_p[:, p], dz_soisno, t_soisno, wliq_soisno, rootr_p[:, i], etrc_p[i], rstfac_p[i])

        #             # fraction of sunlit and shaded leaves of canopy
        #             fsun_p[i] = (1. - np.exp(-min(extkb_p[i] * lai_p[i], 40.))) / max(min(extkb_p[i] * lai_p[i], 40.), 1.e-6)

        #             if coszen <= 0.0 or sabv_p[i] < 1.:
        #                 fsun_p[i] = 0.5

        #             laisun_p[i] = lai_p[i] * fsun_p[i]
        #             laisha_p[i] = lai_p[i] * (1 - fsun_p[i])
        #             rstfacsun_p[i] = rstfac_p[i]
        #             rstfacsha_p[i] = rstfac_p[i]
        #         else:
        #             laisun_p[i] = 0.
        #             laisha_p[i] = 0.
        #             ldew_rain_p[i] = 0.
        #             ldew_snow_p[i] = 0.
        #             ldew_p[i] = 0.
        #             rootr_p[:, i] = 0.
        #             rootflux_p[:, i] = 0.
        #             rstfacsun_p[i] = 0.
        #             rstfacsha_p[i] = 0.

        #     if nl_colm['DEF_USE_PFT'] or time_invariants.patchclass[ipatch] == nl_colm['CROPLAND']:
        #         for i in range(ps, pe + 1):
        #             p = time_invariants.pftclass[i]
        #             if lai_p[i] + sai_p[i] > 1e-6:
        #                 sai, vegwp, gs0sun, gs0sha, tleaf, ldew, ldew_rain, ldew_snow, lai_old, o3uptakesun, o3uptakesha, 
        #                 forc_ozone, taux, tauy, fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, cgrnd,
        #                 cgrndl, cgrnds, tref, qref, rstfacsun_out, rstfacsha_out, gssun_out, gssha_out, rootflux, 
        #                 assimsun_out, etrsun_out, assimsha_out, etrsha_out, rst, assim, respc, fsenl, fevpl, etr, 
        #                 dlrad, ulrad, hprl, z0m, zol, rib, 
        #                 ustar, qstar, tstar, fm, fh, fq = CoLM_LeafTemperature.LeafTemperature(ipatch, p, deltim, csoilc, dewmx, htvp, lai_p[i], sai_p[i], 
        #                                 htop_p[i], hbot_p[i], sqrtdi_p[p], effcon_p[p], vmax25_p[p], 
        #                                 slti_p[p], hlti_p[p], shti_p[p], hhti_p[p], trda_p[p], 
        #                                 trdm_p[p], trop_p[p], g1_p[p], g0_p[p], gradm_p[p], 
        #                                 binter_p[p], extkn_p[p], extkb_p[i], extkd_p[i], forc_hgt_u, 
        #                                 forc_hgt_t, forc_hgt_q, forc_us, forc_vs, thm, th, thv, 
        #                                 forc_q, forc_psrf, forc_rhoair, parsun_p[i], parsha_p[i], 
        #                                 sabv_p[i], frl, fsun_p[i], thermk_p[i], rstfacsun_p[i], 
        #                                 rstfacsha_p[i], gssun_p[i], gssha_p[i], forc_po2m, forc_pco2m, 
        #                                 z0h_g, obu_g, ustar_g, zlnd, zsno, fsno, sigf_p[i], etrc_p[i], 
        #                                 t_grnd, qg, rss, t_soil, t_snow, q_soil, q_snow, dqgdT, 
        #                                 emg, tleaf_p[i], ldew_p[i], ldew_rain_p[i], ldew_snow_p[i], 
        #                                 taux_p[i], tauy_p[i], fseng_p[i], fseng_soil_p[i], fseng_snow_p[i], 
        #                                 fevpg_p[i], fevpg_soil_p[i], fevpg_snow_p[i], cgrnd_p[i], 
        #                                 cgrndl_p[i], cgrnds_p[i], tref_p[i], qref_p[i], rst_p[i], 
        #                                 assim_p[i], respc_p[i], fsenl_p[i], fevpl_p[i], etr_p[i], 
        #                                 dlrad_p[i], ulrad_p[i], z0m_p[i], zol_p[i], rib_p[i], ustar_p[i], 
        #                                 qstar_p[i], tstar_p[i], fm_p[i], fh_p[i], fq_p[i], constpft.rootfr_p[:, p], 
        #                                 kmax_sun_p[p], kmax_sha_p[p], kmax_xyl_p[p], kmax_root_p[p], 
        #                                 psi50_sun_p[p], psi50_sha_p[p], psi50_xyl_p[p], psi50_root_p[p], 
        #                                 ck_p[p], vegwp_p[:, i], gs0sun_p[i], gs0sha_p[i], assimsun_p[i], 
        #                                 etrsun_p[i], assimsha_p[i], etrsha_p[i], o3coefv_sun_p[i], 
        #                                 o3coefv_sha_p[i], o3coefg_sun_p[i], o3coefg_sha_p[i], 
        #                                 lai_old_p[i], o3uptakesun_p[i], o3uptakesha_p[i], forc_ozone, 
        #                                 forc_hpbl, qintr_rain_p[i], qintr_snow_p[i], t_precip, hprl_p[i], smp, 
        #                                 hk, hksati, rootflux_p[1:, i])
        #             else:
        #                 taux, tauy, fseng, fseng_soil, fseng_snow,fevpg, fevpg_soil, fevpg_snow, cgrnd, 
        #                 cgrndl, cgrnds, tref, qref, z0m, z0h_g, zol, rib, ustar, tstar, qstar, fm, fh, 
        #                 fq = CoLM_GroundFluxes.GroundFluxes(zlnd, zsno, forc_hgt_u, forc_hgt_t, forc_hgt_q, forc_hpbl, forc_us, 
        #                             forc_vs, forc_t, forc_q, forc_rhoair, forc_psrf, ur, thm, th, thv, 
        #                             t_grnd, qg, rss, dqgdT, htvp, fsno, cgrnd_p[i], cgrndl_p[i], 
        #                             cgrnds_p[i], t_soil, t_snow, q_soil, q_snow, taux_p[i], tauy_p[i], 
        #                             fseng_p[i], fseng_soil_p[i], fseng_snow_p[i], fevpg_p[i], 
        #                             fevpg_soil_p[i], fevpg_snow_p[i], tref_p[i], qref_p[i], z0m_p[i], 
        #                             z0h_g, zol_p[i], rib_p[i], ustar_p[i], qstar_p[i], tstar_p[i], 
        #                             fm_p[i], fh_p[i], fq_p[i])
        #                 tleaf_p[i] = forc_t
        #                 gssun_p[i] = 0
        #                 gssha_p[i] = 0
        #                 assimsun_p[i] = 0
        #                 etrsun_p[i] = 0
        #                 assimsha_p[i] = 0
        #                 etrsha_p[i] = 0
        #                 rst_p[i] = 2.0e4
        #                 assim_p[i] = 0
        #                 respc_p[i] = 0
        #                 fsenl_p[i] = 0
        #                 fevpl_p[i] = 0
        #                 etr_p[i] = 0
        #                 dlrad_p[i] = frl
        #                 if not nl_colm['DEF_SPLIT_SOILSNOW']:
        #                     ulrad_p[i] = frl * (1 - emg) + emg * const_physical.stefnc * t_grnd ** 4
        #                 else:
        #                     ulrad_p[i] = frl * (1 - emg) + fsno * emg * const_physical.stefnc * t_snow ** 4 + (1 - fsno) * emg * const_physical.stefnc * t_soil ** 4
        #                 hprl_p[i] = 0
        #                 if nl_colm['DEF_USE_PLANTHYDRAULICS']:
        #                     vegwp_p[:, i] = -2.5e4

        #     if nl_colm['DEF_USE_PC'] and time_invariants.patchclass[ipatch] != nl_colm['CROPLAND']:
        #         rst_p[ps:pe] = 2.0e4
        #         assim_p[ps:pe] = 0.0
        #         respc_p[ps:pe] = 0.0
        #         fsenl_p[ps:pe] = 0.0
        #         fevpl_p[ps:pe] = 0.0
        #         etr_p[ps:pe] = 0.0
        #         hprl_p[ps:pe] = 0.0
        #         z0m_p[ps:pe] = (1.0 - fsno) * zlnd + fsno * zsno

        #         if nl_colm['DEF_USE_PLANTHYDRAULICS']:
        #             vegwp_p[:, ps:pe] = -2.5e4

        #         LeafTemperaturePC(ipatch, ps, pe, deltim, csoilc, dewmx, htvp, time_invariants.pftclass[ps:pe], pftfrac[ps:pe], 
        #                         htop_p[ps:pe], hbot_p[ps:pe], lai_p[ps:pe], sai_p[ps:pe], extkb_p[ps:pe], extkd_p[ps:pe],
        #                         forc_hgt_u, forc_hgt_t, forc_hgt_q, forc_us, forc_vs, forc_t, thm, th, thv, forc_q, 
        #                         forc_psrf, forc_rhoair, parsun_p[ps:pe], parsha_p[ps:pe], fsun_p, sabv_p, frl, thermk_p[ps:pe],
        #                         fshade_p[ps:pe], rstfacsun_p, rstfacsha_p, gssun_p, gssha_p, forc_po2m, forc_pco2m, z0h_g, 
        #                         obu_g, ustar_g, zlnd, zsno, fsno, sigf_p[ps:pe], etrc_p, t_grnd, qg, rss, dqgdT, emg, t_soil, 
        #                         t_snow, q_soil, q_snow, z0m_p[ps:pe], tleaf_p[ps:pe], ldew_p[ps:pe], ldew_rain_p[ps:pe], 
        #                         ldew_snow_p[ps:pe], taux, tauy, fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, 
        #                         cgrnd, cgrndl, cgrnds, tref, qref, rst_p[ps:pe], assim_p[ps:pe], respc_p[ps:pe], fsenl_p[ps:pe], 
        #                         fevpl_p[ps:pe], etr_p[ps:pe], dlrad, ulrad, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq, 
        #                         vegwp_p[:, ps:pe], gs0sun_p[ps:pe], gs0sha_p[ps:pe], assimsun_p, etrsun_p, assimsha_p, etrsha_p,
        #                         o3coefv_sun_p[ps:pe], o3coefv_sha_p[ps:pe], o3coefg_sun_p[ps:pe], o3coefg_sha_p[ps:pe], 
        #                         lai_old_p[ps:pe], o3uptakesun_p[ps:pe], o3uptakesha_p[ps:pe], forc_ozone, forc_hpbl, 
        #                         qintr_rain_p[ps:pe], qintr_snow_p[ps:pe], t_precip, hprl_p, smp, hk, hksati, rootflux_p)

        #         laisun = np.sum(laisun_p[ps:pe] * pftfrac[ps:pe])
        #         laisha = np.sum(laisha_p[ps:pe] * pftfrac[ps:pe])
        #         tleaf = np.sum(tleaf_p[ps:pe] * pftfrac[ps:pe])
        #         ldew_rain = np.sum(ldew_rain_p[ps:pe] * pftfrac[ps:pe])
        #         ldew_snow = np.sum(ldew_snow_p[ps:pe] * pftfrac[ps:pe])
        #         ldew = np.sum(ldew_p[ps:pe] * pftfrac[ps:pe])
        #         rst = np.sum(rst_p[ps:pe] * pftfrac[ps:pe])
        #         assim = np.sum(assim_p[ps:pe] * pftfrac[ps:pe])
        #         respc = np.sum(respc_p[ps:pe] * pftfrac[ps:pe])
        #         fsenl = np.sum(fsenl_p[ps:pe] * pftfrac[ps:pe])
        #         fevpl = np.sum(fevpl_p[ps:pe] * pftfrac[ps:pe])
        #         etr = np.sum(etr_p[ps:pe] * pftfrac[ps:pe])

        #         if nl_colm['DEF_USE_PFT'] or time_invariants.patchclass[ipatch] == nl_colm['CROPLAND']:
        #             dlrad = np.sum(dlrad_p[ps:pe] * pftfrac[ps:pe])
        #             ulrad = np.sum(ulrad_p[ps:pe] * pftfrac[ps:pe])
        #             tref = np.sum(tref_p[ps:pe] * pftfrac[ps:pe])
        #             qref = np.sum(qref_p[ps:pe] * pftfrac[ps:pe])
        #             taux = np.sum(taux_p[ps:pe] * pftfrac[ps:pe])
        #             tauy = np.sum(tauy_p[ps:pe] * pftfrac[ps:pe])
        #             fseng = np.sum(fseng_p[ps:pe] * pftfrac[ps:pe])
        #             fseng_soil = np.sum(fseng_soil_p[ps:pe] * pftfrac[ps:pe])
        #             fseng_snow = np.sum(fseng_snow_p[ps:pe] * pftfrac[ps:pe])
        #             fevpg = np.sum(fevpg_p[ps:pe] * pftfrac[ps:pe])
        #             fevpg_soil = np.sum(fevpg_soil_p[ps:pe] * pftfrac[ps:pe])
        #             fevpg_snow = np.sum(fevpg_snow_p[ps:pe] * pftfrac[ps:pe])
        #             cgrnd = np.sum(cgrnd_p[ps:pe] * pftfrac[ps:pe])
        #             cgrndl = np.sum(cgrndl_p[ps:pe] * pftfrac[ps:pe])
        #             cgrnds = np.sum(cgrnds_p[ps:pe] * pftfrac[ps:pe])
        #             z0m = np.sum(z0m_p[ps:pe] * pftfrac[ps:pe])
        #             zol = np.sum(zol_p[ps:pe] * pftfrac[ps:pe])
        #             rib = np.sum(rib_p[ps:pe] * pftfrac[ps:pe])
        #             ustar = np.sum(ustar_p[ps:pe] * pftfrac[ps:pe])
        #             qstar = np.sum(qstar_p[ps:pe] * pftfrac[ps:pe])
        #             tstar = np.sum(tstar_p[ps:pe] * pftfrac[ps:pe])
        #             fm = np.sum(fm_p[ps:pe] * pftfrac[ps:pe])
        #             fh = np.sum(fh_p[ps:pe] * pftfrac[ps:pe])
        #             fq = np.sum(fq_p[ps:pe] * pftfrac[ps:pe])

        #         rstfacsun_out = np.sum(rstfacsun_p[ps:pe] * pftfrac[ps:pe])
        #         rstfacsha_out = np.sum(rstfacsha_p[ps:pe] * pftfrac[ps:pe])
        #         gssun_out = np.sum(gssun_p[ps:pe] * pftfrac[ps:pe])
        #         gssha_out = np.sum(gssha_p[ps:pe] * pftfrac[ps:pe])
        #         assimsun_out = np.sum(assimsun_p[ps:pe] * pftfrac[ps:pe])
        #         etrsun_out = np.sum(etrsun_p[ps:pe] * pftfrac[ps:pe])
        #         assimsha_out = np.sum(assimsha_p[ps:pe] * pftfrac[ps:pe])
        #         etrsha_out = np.sum(etrsha_p[ps:pe] * pftfrac[ps:pe])
        #         hprl = np.sum(hprl_p[ps:pe] * pftfrac[ps:pe])

        #         if nl_colm['DEF_USE_PLANTHYDRAULICS']:
        #             for j in range(nvegwcs):
        #                 vegwp[j] = np.sum(vegw_p[j, ps:pe] * pftfrac[ps:pe])

        #             if abs(etr) > 0.0:
        #                 for j in range(nl_soil):
        #                     rootflux[j] = np.sum(rootflux_p[j, ps:pe] * pftfrac[ps:pe])
        #         else:
        #             if abs(etr) > 0.0:
        #                 for j in range(nl_soil):
        #                     rootr[j] = np.sum(rootr_p[j, ps:pe] * etr_p[ps:pe] * pftfrac[ps:pe]) / etr

        #         del rootflux_p
        #         del etrc_p
        #         del rstfac_p
        #         del rstfacsun_p
        #         del rstfacsha_p
        #         del gssun_p
        #         del gssha_p
        #         del fsun_p
        #         del sabv_p

        #         if nl_colm['DEF_USE_PFT'] or time_invariants.patchclass[ipatch] == nl_colm['CROPLAND']:
        #             del fseng_soil_p
        #             del fseng_snow_p
        #             del fevpg_soil_p
        #             del fevpg_snow_p
        #             del cgrnd_p
        #             del cgrnds_p
        #             del cgrndl_p
        #             del dlrad_p
        #             del ulrad_p
        #             del zol_p
        #             del rib_p
        #             del ustar_p
        #             del qstar_p
        #             del tstar_p
        #             del fm_p
        #             del fh_p
        #             del fq_p

        #         del hprl_p
        #         del assimsun_p
        #         del etrsun_p
        #         del assimsha_p
        #         del etrsha_p

    # =======================================================================
    # [5] Ground temperature
    # =======================================================================

    t_soisno, wice_soisno, wliq_soisno, scv, snowdp, sm, xmf, fact, imelt, snofrz = CoLM_GroundTemperature.GroundTemperature(
        nl_colm, const_physical, patchtype, lb, var_global.nl_soil, deltim,
        capr, cnfac, vf_quartz, vf_gravels, vf_om, vf_sand, wf_gravels, wf_sand,
        porsl, psi0,
        #ifdef Campbell_SOIL_MODEL
        bsw,
        #endif
        #ifdef vanGenuchten_Mualem_SOIL_MODEL
        theta_r, alpha_vgm, n_vgm, L_vgm,
        sc_vgm, fc_vgm,
        #endif
        csol, k_solids, dksatu, dksatf, dkdry,
        BA_alpha, BA_beta,
        sigf, dz_soisno, z_soisno, zi_soisno,
        t_soisno, t_grnd, t_soil, t_snow, wice_soisno, wliq_soisno, scv, snowdp, fsno,
        frl, dlrad, sabg, sabg_soil, sabg_snow, sabg_snow_lyr,
        fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, cgrnd, htvp, emg,
        imelt, snofrz, sm, xmf, fact, pg_rain, pg_snow, t_precip
    )
    
    # =======================================================================
    # [6] Correct fluxes to present soil temperature
    # =======================================================================

    if not nl_colm['DEF_SPLIT_SOILSNOW']:
        t_grnd = t_soisno[lb-1]
        tinc = t_soisno[lb-1] - t_soisno_bef[lb-1]
    else:
        t_grnd = fsno * t_soisno[lb-1] + (1.0 - fsno) * t_soisno[0]
        tinc = t_grnd - t_grnd_bef

    fseng += tinc * cgrnds
    fseng_soil += tinc * cgrnds
    fseng_snow += tinc * cgrnds
    fevpg += tinc * cgrndl
    fevpg_soil += tinc * cgrndl
    fevpg_snow += tinc * cgrndl

    # calculation of evaporative potential; flux in kg m-2 s-1.
    # egidif holds the excess energy IF all water is evaporated
    # during the timestep.  this energy is later added to the sensible heat flux.

    qseva = 0.
    qsubl = 0.
    qfros = 0.
    qsdew = 0.
    qseva_soil = 0.
    qsubl_soil = 0.
    qfros_soil = 0.
    qsdew_soil = 0.
    qseva_snow = 0.
    qsubl_snow = 0.
    qfros_snow = 0.
    qsdew_snow = 0.

    if not nl_colm['DEF_SPLIT_SOILSNOW']:
        egsmax = (wice_soisno[lb-1] + wliq_soisno[lb-1]) / deltim
        egidif = max(0., fevpg - egsmax)
        fevpg = min(fevpg, egsmax)
        fseng += htvp * egidif

        if fevpg >= 0.:
            # not allow for sublimation in melting (melting ==> evap. ==> sublimation)
            qseva = min(wliq_soisno[lb-1] / deltim, fevpg)
            qsubl = fevpg - qseva
        else:
            if t_grnd < const_physical.tfrz:
                qfros = abs(fevpg)
            else:
                qsdew = abs(fevpg)
    else:
        if lb < 1:  # snow layer exists
            egsmax = (wice_soisno[lb-1] + wliq_soisno[lb-1]) / deltim
            egidif = max(0., fevpg_snow - egsmax)
            fevpg_snow = min(fevpg_snow, egsmax)
            fseng_snow += htvp * egidif
        else:  # no snow layer, attribute to soil
            fevpg_soil = fevpg_soil * (1. - fsno) + fevpg_snow * fsno

        egsmax = (wice_soisno[0] + wliq_soisno[0]) / deltim
        egidif = max(0., fevpg_soil - egsmax)
        fevpg_soil = min(fevpg_soil, egsmax)
        fseng_soil += htvp * egidif

        if lb < 1:  # snow layer exists
            fseng = fseng_soil * (1. - fsno) + fseng_snow * fsno
            fevpg = fevpg_soil * (1. - fsno) + fevpg_snow * fsno
        else:  # no snow layer, attribute to soil
            fseng = fseng_soil
            fseng_snow = 0.
            fevpg = fevpg_soil
            fevpg_snow = 0.

        if fevpg_snow >= 0.:
            # not allow for sublimation in melting (melting ==> evap. ==> sublimation)
            qseva_snow = min(wliq_soisno[lb-1] / deltim, fevpg_snow)
            qsubl_snow = fevpg_snow - qseva_snow
            qseva_snow *= fsno
            qsubl_snow *= fsno
        else:
            # snow temperature < const_physical.tfrz
            if t_soisno[lb-1] < const_physical.tfrz:
                qfros_snow = abs(fevpg_snow * fsno)
            else:
                qsdew_snow = abs(fevpg_snow * fsno)

        if fevpg_soil >= 0.:
            # not allow for sublimation in melting (melting ==> evap. ==> sublimation)
            qseva_soil = min(wliq_soisno[0] / deltim, fevpg_soil)
            qsubl_soil = fevpg_soil - qseva_soil
        else:
            # soil temperature < const_physical.tfrz
            if t_soisno[0] < const_physical.tfrz:
                qfros_soil = abs(fevpg_soil)
            else:
                qsdew_soil = abs(fevpg_soil)

        if lb < 1:  # snow layer exists
            qseva_soil *= (1. - fsno)
            qsubl_soil *= (1. - fsno)
            qfros_soil *= (1. - fsno)
            qsdew_soil *= (1. - fsno)

    # total fluxes to atmosphere
    fsena = fsenl + fseng
    fevpa = fevpl + fevpg
    lfevpa = const_physical.hvap * fevpl + htvp * fevpg  # W/m^2 (accounting for sublimation)

    # ground heat flux
    if not nl_colm['DEF_SPLIT_SOILSNOW']:
        fgrnd = (sabg + dlrad * emg -
                 emg * const_physical.stefnc * t_grnd_bef**4 -
                 emg * const_physical.stefnc * t_grnd_bef**3 * (4. * tinc) -
                 (fseng + fevpg * htvp) +
                 const_physical.cpliq * pg_rain * (t_precip - t_grnd) +
                 const_physical.cpice * pg_snow * (t_precip - t_grnd))
    else:
        fgrnd = (sabg + dlrad * emg -
                 fsno * emg * const_physical.stefnc * t_snow**4 -
                 (1. - fsno) * emg * const_physical.stefnc * t_soil**4 -
                 emg * const_physical.stefnc * t_grnd_bef**3 * (4. * tinc) -
                 (fseng + fevpg * htvp) +
                 const_physical.cpliq * pg_rain * (t_precip - t_grnd) +
                 const_physical.cpice * pg_snow * (t_precip - t_grnd))

    # outgoing long-wave radiation from canopy + ground
    olrg = ulrad + 4. * emg * const_physical.stefnc * t_grnd_bef**3 * tinc

    # averaged bulk surface emissivity
    olrb = const_physical.stefnc * t_grnd_bef**3 * (4. * tinc)
    olru = ulrad + emg * olrb
    olrb = ulrad + olrb
    emis = olru / olrb

    # radiative temperature
    if olrg < 0:
        print("MOD_Thermal.F90: Error! Negative outgoing longwave radiation flux:")
        print(ipatch, olrg, tinc, ulrad)
        print(ipatch, errore, sabv, sabg, frl, olrg, fsenl, fseng, const_physical.hvap * fevpl, htvp * fevpg, xmf, fgrnd)

    trad = (olrg / const_physical.stefnc)**0.25

    # additional variables required by WRF and RSM model
    if lai + sai <= 1e-6:
        ustar = ustar_g
        tstar = tstar_g
        qstar = qstar_g
        rib = rib_g
        zol = zol_g
        z0m = z0m_g
        fm = fm_g
        fh = fh_g
        fq = fq_g
    
    # =======================================================================
    # [7] energy balance error
    # =======================================================================

    # one way to check energy
    # errore = (sabv + sabg + frl - olrg - fsena - lfevpa - fgrnd + hprl +
    #           const_physical.cpliq * pg_rain * (t_precip - t_grnd) + const_physical.cpice * pg_snow * (t_precip - t_grnd))

    # another way to check energy
    errore = (sabv + sabg + frl - olrg - fsena - lfevpa - xmf + hprl +
              const_physical.cpliq * pg_rain * (t_precip - t_grnd) + const_physical.cpice * pg_snow * (t_precip - t_grnd))

    for j in range(lb-1, var_global.nl_soil):
        errore -= (t_soisno[j] - t_soisno_bef[j]) / fact[j]

    if nl_colm['CoLMDEBUG']:
        if abs(errore) > 0.5:
            print('MOD_Thermal.F90: energy balance violation')
            print(ipatch, errore, sabv, sabg, frl, olrg, fsenl, fseng, const_physical.hvap * fevpl, htvp * fevpg, xmf, hprl)
            print(const_physical.cpliq * pg_rain * (t_precip - t_grnd), const_physical.cpice * pg_snow * (t_precip - t_grnd))
            # CoLM_stop()
    # 100 format(10(f15.3))
    return sai, vegwp, gs0sun, gs0sun, lai_old, o3uptakesun, o3uptakesha, forc_ozone, tleaf, smp, hk, ldew, ldew_rain, ldew_snow, scv, snowdp, snofrz, imelt, laisun, laisha, gssun_out, gssha_out, rstfacsun_out, rstfacsha_out, assimsun_out, etrsun_out, assimsha_out, etrsha_out, taux, tauy, fsena, fevpa, fsenl, fevpl, etr, fseng, fevpg, olrg, fgrnd, rootr, rootflux, qseva, qsdew, qsubl, qfros, qseva_soil, qsdew_soil, qsubl_soil, qseva_snow, qsdew_snow, qsubl_snow, qfros_snow, sm, tref, qref, trad, rss, rst, assim, respc, emis, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq