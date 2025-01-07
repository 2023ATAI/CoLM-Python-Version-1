"""
Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
software         : Initialization of Land Characteristic Parameters and Initial State Variables
Reference        : [1] Dai et al., 2003: The Common Land Model (CoLM). Bull. of Amer. Meter. Soc., 84: 1013-1023
                   [2] Dai et al., 2004: A two-big-leaf model for canopy temperature, photosynthesis
                   and stomatal conductance. Journal of Climate
                   [3] Dai et al., 2014: The Terrestrial Modeling System (TMS).
Args:
    patchtype
    lc_year
Returns:
    patchtype
"""
import numpy as np
import CoLM_NetSolar
import CoLM_RainSnowTemp
import CoLM_NewSnow
from CoLM_LeafInterception import LeafInterception
import CoLM_SnowLayersCombineDivide
import CoLM_SoilSnowHydrology
import CoLM_Glacier
import CoLM_Utils
import CoLM_OrbCoszen
import CoLM_Albland
import CoLM_LAIEmpirical
import CoLM_SnowFraction
import CoLM_Albedo
import CoLM_Thermal
import CoLM_Lake
import CoLM_TimeManager


def colm_main(nl_colm, var_global, const_physical, const_lc, gblock, mpi, VTV, VTI, isgreenwich, landpft,
           ipatch,       idate,        coszen,       deltim,        
           patchlonr,    patchlatr,    patchclass,   patchtype,     
           doalb,        dolai,        dosst,        oro,           
           soil_s_v_alb, soil_d_v_alb, soil_s_n_alb, soil_d_n_alb,
           vf_quartz,    vf_gravels,   vf_om,        vf_sand,       
           wf_gravels,   wf_sand,      porsl,        psi0,          
           bsw,          theta_r,      
           alpha_vgm,    n_vgm,        L_vgm,         
           sc_vgm,       fc_vgm,       

           hksati,       csol,         k_solids,     dksatu,        
           dksatf,       dkdry,        BA_alpha,     BA_beta,       
           rootfr,       lakedepth,    dz_lake,      topostd, BVIC, 

           # flddepth,     fldfrc,       fevpg_fld,    qinfl_fld,

           htop,         hbot,         sqrtdi,       
           effcon,       vmax25,                                    
           kmax_sun,     kmax_sha,     kmax_xyl,     kmax_root,     
           psi50_sun,    psi50_sha,    psi50_xyl,    psi50_root,    
           ck,           slti,         hlti,         shti,          
           hhti,         trda,         trdm,         trop,          
           g1,           g0,           gradm,        binter,        
           extkn,        chil,         rho,          tau,           

           forc_pco2m,   forc_po2m,    forc_us,      forc_vs,       
           forc_t,       forc_q,       forc_prc,     forc_prl,      
           forc_rain,    forc_snow,    forc_psrf,    forc_pbot,     
           forc_sols,    forc_soll,    forc_solsd,   forc_solld,    
           forc_frl,     forc_hgt_u,   forc_hgt_t,   forc_hgt_q,    
           forc_rhoair,  

           forc_hpbl,    

           forc_aerdep,  

           z_sno,        dz_sno,       t_soisno,     wliq_soisno,   
           wice_soisno,  smp,          hk,           t_grnd,        
           tleaf,        ldew,         ldew_rain,    ldew_snow,     
           sag,          scv,          snowdp,       fveg,          
           fsno,         sigf,         green,        lai,           
           sai,          alb,          ssun,         ssha,          
           ssoi,         ssno,         thermk,       extkb,         
           extkd,        vegwp,        gs0sun,       gs0sha,        

           lai_old,      o3uptakesun,  o3uptakesha,  forc_ozone,    

           zwt,          wdsrf,        wa,           wetwat,        
           t_lake,       lake_icefrac, savedtke1,    

           snw_rds,      ssno_lyr,     
           mss_bcpho,    mss_bcphi,    mss_ocpho,     mss_ocphi,    
           mss_dst1,     mss_dst2,     mss_dst3,      mss_dst4,     

           laisun,       laisha,       rootr,rootflux,rss,          
           rstfacsun_out,rstfacsha_out,gssun_out,    gssha_out,     
           assimsun_out, etrsun_out,   assimsha_out, etrsha_out,    
           h2osoi,       wat,          

           taux,         tauy,         fsena,        fevpa,         
           lfevpa,       fsenl,        fevpl,        etr,           
           fseng,        fevpg,        olrg,         fgrnd,         
           trad,         tref,         qref,                        
           rsur,         rsur_se,      rsur_ie,      rnof,          
           qintr,        qinfl,        qdrip,                       
           rst,          assim,        respc,        sabvsun,       
           sabvsha,      sabg,         sr,           solvd,         
           solvi,        solnd,        solni,        srvd,          
           srvi,         srnd,         srni,         solvdln,       
           solviln,      solndln,      solniln,      srvdln,        
           srviln,       srndln,       srniln,       qcharge,       
           xerr,         zerr,         

           zlnd,         zsno,         csoilc,       dewmx,         
           wtfact,       capr,         cnfac,        ssi,           
           wimp,         pondmx,       smpmax,       smpmin,        
           trsmx0,       tcrit,        

           emis,         z0m,          zol,          rib,           
           ustar,        qstar,        tstar,        fm,            
           fh,           fq, vic_b_infilt, vic_Dsmax, vic_Ds, vic_Ws, vic_c
):
    #  ----------------------- Local  Variables -----------------------------
    maxsnl = var_global.maxsnl
    calday = 0.0  # Julian cal day (1.xx to 365.xx)
    endwb = 0.0  # water mass at the end of time step
    errore = 0.0  # energy balnce errore (Wm-2)
    errorw = 0.0  # water balnce errore (mm)
    fiold = np.zeros(var_global.nl_soil - maxsnl)  # fraction of ice relative to the total water
    w_old = 0.0  # liquid water mass of the column at the previous time step (mm)

    sabg_soil = 0.0  # solar absorbed by soil fraction
    sabg_snow = 0.0  # solar absorbed by snow fraction
    parsun = 0.0  # PAR by sunlit leaves [W/m2]
    parsha = 0.0  # PAR by shaded leaves [W/m2]
    qseva = 0.0  # ground surface evaporation rate (mm h2o/s)
    qsdew = 0.0  # ground surface dew formation (mm h2o /s) [+]
    qsubl = 0.0  # sublimation rate from snow pack (mm h2o /s) [+]
    qfros = 0.0  # surface dew added to snow pack (mm h2o /s) [+]
    qseva_soil = 0.0  # ground soil surface evaporation rate (mm h2o/s)
    qsdew_soil = 0.0  # ground soil surface dew formation (mm h2o /s) [+]
    qsubl_soil = 0.0  # sublimation rate from soil ice pack (mm h2o /s) [+]
    qfros_soil = 0.0  # surface dew added to soil ice pack (mm h2o /s) [+]
    qseva_snow = 0.0  # ground snow surface evaporation rate (mm h2o/s)
    qsdew_snow = 0.0  # ground snow surface dew formation (mm h2o /s) [+]
    qsubl_snow = 0.0  # sublimation rate from snow pack (mm h2o /s) [+]
    qfros_snow = 0.0  # surface dew added to snow pack (mm h2o /s) [+]
    scvold = 0.0  # snow cover for previous time step [mm]
    sm = 0.0  # rate of snowmelt [kg/(m2 s)]
    ssw = 0.0  # water volumetric content of soil surface layer [m3/m3]
    tssub = np.zeros(7)  # surface/sub-surface temperatures [K]
    tssea = 0.0  # sea surface temperature [K]
    totwb = 0.0  # water mass at the begining of time step
    wt = 0.0  # fraction of vegetation buried (covered) by snow [-]
    z_soisno = np.zeros(var_global.nl_soil - maxsnl)  # layer depth (m)
    dz_soisno = np.zeros(var_global.nl_soil - maxsnl)  # layer thickness (m)
    zi_soisno = np.zeros(var_global.nl_soil - maxsnl+1)  # interface level below a "z" level (m)

    prc_rain = 0.0  # convective rainfall [kg/(m2 s)]
    prc_snow = 0.0  # convective snowfall [kg/(m2 s)]
    prl_rain = 0.0  # large scale rainfall [kg/(m2 s)]
    prl_snow = 0.0  # large scale snowfall [kg/(m2 s)]
    t_precip = 0.0  # snowfall/rainfall temperature [kelvin]
    bifall = 0.0  # bulk density of newly fallen dry snow [kg/m3]
    pg_rain = 0.0  # rainfall onto ground including canopy runoff [kg/(m2 s)]
    pg_snow = 0.0  # snowfall onto ground including canopy runoff [kg/(m2 s)]
    qintr_rain = 0.0  # rainfall interception (mm h2o/s)
    qintr_snow = 0.0  # snowfall interception (mm h2o/s)
    errw_rsub = 0.0  # the possible subsurface runoff deficit after PHS is included

    # number of snow layers
    imelt = np.zeros(var_global.nl_soil - maxsnl, dtype=int)  # flag for: melting=1= freezing=2= Nothing happended=0
    lb = 0
    lbsn = 0  # lower bound of arrays
    j = 0  # do looping index

    # For SNICAR snow model
    #    ----------------------------------------------------------------------
    a = 0.0
    aa = 0.0
    gwat = 0.0
    wextra = 0.0
    t_rain = 0.0
    t_snow = 0.0
    ps = 0
    pe = 0
    pc = 0

    # ======================================================================
    if nl_colm['CaMa_Flood']:
        #    add variables for flood evaporation [mm/s] and re-infiltration [mm/s] calculation.
        kk = 0.0  #
        taux_fld = 0.0  # wind stress: E-W [kg/m/s**2]
        tauy_fld = 0.0  # wind stress: N-S [kg/m/s**2]
        fsena_fld = 0.0  # sensible heat from agcm reference height to atmosphere [W/m2]
        fevpa_fld = 0.0  # evaporation from agcm reference height to atmosphere [mm/s]
        fseng_fld = 0.0  # sensible heat flux from ground [W/m2]
        tref_fld = 0.0  # 2 m height air temperature [kelvin]
        qref_fld = 0.0  # 2 m height air humidity
        z0m_fld = 0.0  # effective roughness [m]
        zol_fld = 0.0  # dimensionless height (z/L) used in Monin-Obukhov theory
        rib_fld = 0.0  # bulk Richardson number in surface layer
        ustar_fld = 0.0  # friction velocity [m/s]
        tstar_fld = 0.0  # temperature scaling parameter
        qstar_fld = 0.0  # moisture scaling parameter
        fm_fld = 0.0  # integral of profile function for momentum
        fh_fld = 0.0  # integral of profile function for heat
        fq_fld = 0.0  # integral of profile function for moisture

    # Variables
    snl_bef = 0  # number of snow layers (integer)
    forc_aer = np.zeros(14, dtype=np.float64)  # aerosol deposition from atmosphere model [kg m-1 s-1]
    snofrz = np.zeros(-maxsnl, dtype=np.float64)  # snow freezing rate [kg m-2 s-1]
    t_soisno_ = np.zeros((-maxsnl+1), dtype=np.float64)  # soil + snow layer temperature [K]
    dz_soisno_ = np.zeros((-maxsnl+1), dtype=np.float64)  # layer thickness (m)
    sabg_snow_lyr = np.zeros((-maxsnl+1), dtype=np.float64)  # snow layer absorption [W/m-2]

    # 初始化 z_soisno 和 dz_soisno
    z_soisno[:-maxsnl] = z_sno[:-maxsnl]
    z_soisno[-maxsnl:var_global.nl_soil-maxsnl] = var_global.z_soi[:var_global.nl_soil]
    dz_soisno[:-maxsnl] = dz_sno[:-maxsnl]
    dz_soisno[-maxsnl:var_global.nl_soil-maxsnl] = var_global.dz_soi[:var_global.nl_soil]

    # SNICAR initialization
    # ---------------------

    # snow freezing rate (col,lyr) [kg m-2 s-1]
    snofrz[:] = 0.0

    # aerosol deposition value
    if nl_colm['DEF_Aerosol_Readin']:
        forc_aer[:] = forc_aerdep  # read from outside forcing file
    else:
        forc_aer[:] = 0.0  # manual setting
        # forc_aer[:] = 4.2E-7  # 手动设置示例

        # ======================================================================
    #  [1] Solar absorbed by vegetation and ground
    #      and precipitation information (rain/snow fall and precip temperature
    # ======================================================================
    ssun, ssha, ssoi, ssno, ssno_lyr, parsun, parsha, sabvsun, sabvsha, sabg,  sabg_soil, sabg_snow, fsno, sr, solvd, solvi, solnd, solni, srvd, srvi, srnd, srni, solvdln, solviln, solndln, solniln, srvdln, srviln, srndln, srniln, sabg_snow_lyr = CoLM_NetSolar.netsolar(nl_colm, var_global, ipatch, idate, deltim,
                                           patchlonr, patchtype,
                                           forc_sols, forc_soll,
                                           forc_solsd, forc_solld,
                                           alb, ssun,
                                           ssha,
                                           lai, sai,
                                           rho, tau,
                                           ssoi, ssno,
                                           ssno_lyr,
                                           parsun, parsha,
                                           sabvsun, sabvsha,
                                           sabg,
                                           sabg_soil, sabg_snow,
                                           fsno, sabg_snow_lyr, sr,
                                           solvd, solvi,
                                           solnd,
                                           solni, srvd,
                                           srvi, srnd, srni,
                                           solvdln, solviln,
                                           solndln,
                                           solniln, srvdln,
                                           srviln,
                                           srndln, srniln)

    # 调用 rain_snow_temp 函数
    prc_rain, prc_snow, prl_rain, prl_snow, t_precip, bifall = CoLM_RainSnowTemp.rain_snow_temp(nl_colm, const_physical,
                                                                                                var_global,
                                                                                                patchtype,
                                                                                                forc_t,
                                                                                                forc_q,
                                                                                                forc_psrf,
                                                                                                forc_prc,
                                                                                                forc_prl,
                                                                                                forc_us,
                                                                                                forc_vs,
                                                                                                tcrit, prc_rain,
                                                                                                prc_snow, prl_rain,
                                                                                                prl_snow, t_precip,
                                                                                                bifall)

    forc_rain = prc_rain + prl_rain
    forc_snow = prc_snow + prl_snow

    # ======================================================================
    if patchtype <= 2:  # <=== is - URBAN and BUILT-UP   (patchtype = 1)

        scvold = scv  # snow mass at previous time step

        snl = 0
        for j in range(-maxsnl):
            if wliq_soisno[j] + wice_soisno[j] > 0.0:
                snl -= 1

        zi_soisno[5] = 0.0
        if snl < 0:
            for j in range(-maxsnl-1, snl-maxsnl-1, -1):
                zi_soisno[j] = zi_soisno[j + 1] - dz_soisno[j]

        for j in range(6, var_global.nl_soil + 6):
            zi_soisno[j] = zi_soisno[j - 1] + dz_soisno[j-1]

        totwb = ldew + scv + np.sum(wice_soisno[5:] + wliq_soisno[5:]) + wa

        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            totwb += wdsrf
        if patchtype == 2:
            totwb += wetwat

        fiold[:] = 0.0
        if snl < 0:
            fiold[-maxsnl + snl:5] = wice_soisno[-maxsnl + snl:5] / (wliq_soisno[-maxsnl + snl:5] + wice_soisno[-maxsnl + snl:5])

        # ----------------------------------------------------------------------
        # [2] Canopy interception and precipitation onto ground surface
        # ----------------------------------------------------------------------
        # CoLM_SoilSnowHydrology.py文件里的
        qflx_irrig_sprinkler = 0.0
        leaf = LeafInterception(nl_colm,const_physical)
        if patchtype == 0:
            # 如果定义了 LULC_USGS 或 LULC_IGBP
            if nl_colm['LULC_USGS'] or nl_colm['LULC_IGBP']:

                tleaf, ldew,ldew_rain, ldew_snow, pg_rain, pg_snow, qintr,qintr_rain, qintr_snow = leaf.LEAF_interception_wrap(const_physical.tfrz, deltim,dewmx, forc_us,
                                                                     forc_vs,chil,
                                                                     sigf,lai,sai,forc_t,
                                                                     tleaf, prc_rain,prc_snow,prl_rain,prl_snow, ldew,ldew_rain,
                                                                     ldew_snow, z0m,forc_hgt_u,
                                                                     pg_rain, pg_snow, qintr,
                                                                     qintr_rain, qintr_snow)

            # 如果定义了 LULC_IGBP_PFT 或 LULC_IGBP_PC
            if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
                pass
            # tleaf, ldew,ldew_rain,ldew_snow,pg_rain, pg_snow,qintr,qintr_rain,qintr_snow = leaf.LEAF_interception_pftwrap(ipatch, deltim, dewmx, forc_us, forc_vs, forc_t,
            #                         prc_rain, prc_snow, prl_rain, prl_snow,
            #                         ldew, ldew_rain, ldew_snow, z0m, forc_hgt_u, pg_rain, pg_snow, qintr, qintr_rain, qintr_snow)
        else:

            tleaf, ldew,ldew_rain,ldew_snow,pg_rain, pg_snow,qintr,qintr_rain,qintr_snow= leaf.LEAF_interception_wrap(
                const_physical.tfrz, deltim, dewmx, forc_us,
                forc_vs,
                chil,
                sigf,
                lai,
                sai,
                forc_t,
                tleaf,
                prc_rain, prc_snow, prl_rain, prl_snow,
                ldew,
                ldew_rain,
                ldew_snow,
                z0m,
                forc_hgt_u,
                pg_rain,
                pg_snow,
                qintr,
                qintr_rain,
                qintr_snow)

        qdrip = pg_rain + pg_snow

        # ----------------------------------------------------------------------
        # [3] Initialize new snow nodes for snowfall / sleet
        # ----------------------------------------------------------------------
        snl_bef = snl
        zi_soisno[:6], z_soisno[:5], dz_soisno[:5], t_soisno[:5], wliq_soisno[:5], wice_soisno[:5], fiold[:5], snl, sag, scv, snowdp, fsno, wetwat = CoLM_NewSnow.newsnow(nl_colm, const_physical, patchtype,
                                                       maxsnl, deltim, t_grnd, pg_rain, pg_snow, bifall,
                                                       t_precip, zi_soisno[:6], z_soisno[:5], dz_soisno[:5],
                                                       t_soisno[:5], wliq_soisno[:5], wice_soisno[:5],
                                                       fiold[:5],
                                                       snl, sag, scv,
                                                       snowdp, fsno, wetwat)

        # ----------------------------------------------------------------------
        # [4] Energy and Water balance
        # ----------------------------------------------------------------------
        lb = snl + 1  # lower bound of array
        lbsn = min(lb, 0)
        # print (etr, 'etr-------1')
        sai, vegwp, gs0sun, gs0sun, lai_old, o3uptakesun, o3uptakesha, forc_ozone, tleaf, smp, hk, ldew, ldew_rain, ldew_snow, scv, snowdp, snofrz, imelt, laisun, laisha, gssun_out, gssha_out, rstfacsun_out, rstfacsha_out,    assimsun_out, etrsun_out, assimsha_out, etrsha_out, taux, tauy, fsena, fevpa, fsenl, fevpl, etr, fseng, fevpg, olrg, fgrnd, rootr, rootflux, qseva, qsdew, qsubl, qfros, qseva_soil, qsdew_soil, qsubl_soil, qseva_snow, qsdew_snow, qsubl_snow, qfros_snow, sm, tref,    qref, trad, rss, rst, assim, respc, emis, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq = CoLM_Thermal.THERMAL(nl_colm, const_physical, var_global, const_lc,patchclass,ipatch ,patchtype ,lb ,deltim,
                  trsmx0            ,zlnd              ,zsno              ,csoilc            ,
                  dewmx             ,capr              ,cnfac             ,vf_quartz         ,
                  vf_gravels        ,vf_om             ,vf_sand           ,wf_gravels        ,
                  wf_sand           ,csol              ,porsl             ,psi0              ,
    #ifdef Campbell_SOIL_MODEL
                  bsw               ,
    #endif
    #ifdef vanGenuchten_Mualem_SOIL_MODEL
                  theta_r           ,alpha_vgm         ,n_vgm             ,L_vgm             ,
                  sc_vgm            ,fc_vgm            ,
    #endif
                  k_solids          ,dksatu            ,dksatf            ,dkdry             ,
                  BA_alpha          ,BA_beta           ,
                  lai               ,laisun            ,laisha            ,
                  sai               ,htop              ,hbot              ,sqrtdi            ,
                  rootfr            ,rstfacsun_out     ,rstfacsha_out     ,rss               ,
                  gssun_out         ,gssha_out         ,
                  assimsun_out      ,etrsun_out        ,assimsha_out      ,etrsha_out        ,

                  effcon            ,
                  vmax25            ,hksati            ,smp               ,hk                ,
                  kmax_sun          ,kmax_sha          ,kmax_xyl          ,kmax_root         ,
                  psi50_sun         ,psi50_sha         ,psi50_xyl         ,psi50_root        ,
                  ck                ,vegwp             ,gs0sun            ,gs0sha            ,
                  #Ozone stress variables
                  lai_old           ,o3uptakesun       ,o3uptakesha       ,forc_ozone        ,
                  #End ozone stress variables
                  slti              ,hlti              ,shti              ,hhti              ,
                  trda              ,trdm              ,trop              ,g1                ,
                  g0                ,gradm             ,binter            ,extkn             ,
                  forc_hgt_u        ,forc_hgt_t        ,forc_hgt_q        ,forc_us           ,
                  forc_vs           ,forc_t            ,forc_q            ,forc_rhoair       ,
                  forc_psrf         ,forc_pco2m        ,forc_hpbl         ,
                  forc_po2m         ,coszen            ,parsun            ,parsha            ,
                  sabvsun           ,sabvsha           ,sabg,sabg_soil,sabg_snow,forc_frl    ,
                  extkb             ,extkd             ,thermk            ,fsno              ,
                  sigf              ,dz_soisno[-maxsnl-lb+1:]    ,z_soisno[-maxsnl-lb+1:]     ,zi_soisno[-maxsnl-lb+1:]  ,
                  tleaf             ,t_soisno[-maxsnl-lb+1:]     ,wice_soisno[-maxsnl-lb+1:]  ,wliq_soisno[-maxsnl-lb+1:]  ,
                  ldew,ldew_rain,ldew_snow,scv         ,snowdp            ,imelt[-maxsnl-lb+1:]        ,
                  taux              ,tauy              ,fsena             ,fevpa             ,
                  lfevpa            ,fsenl             ,fevpl             ,etr               ,
                  fseng             ,fevpg             ,olrg              ,fgrnd             ,
                  rootr             ,rootflux          ,
                  qseva             ,qsdew             ,qsubl             ,qfros             ,
                  qseva_soil        ,qsdew_soil        ,qsubl_soil        ,qfros_soil        ,
                  qseva_snow        ,qsdew_snow        ,qsubl_snow        ,qfros_snow        ,
                  sm                ,tref              ,qref              ,
                  trad              ,rst               ,assim             ,respc             ,

                  errore            ,emis              ,z0m               ,zol               ,
                  rib               ,ustar             ,qstar             ,tstar             ,
                  fm                ,fh                ,fq                ,pg_rain           ,
                  pg_snow           ,t_precip          ,qintr_rain        ,qintr_snow        ,
                  snofrz[-maxsnl+lbsn:6]    ,sabg_snow_lyr[-maxsnl+lb:7] )
        # print(etr, '-----et-r---')
        if not nl_colm['DEF_USE_VariablySaturatedFlow']:
            pass
        else:
            wice_soisno[-maxsnl-lb+1:], wliq_soisno[-maxsnl-lb+1:],smp,hk,zwt,wdsrf,wa, wetwat,rsur, rsur_se, rsur_ie, rnof, qinfl, mss_bcpho[lbsn: 5] ,mss_bcphi[lbsn: 5] ,mss_ocpho [lbsn: 5],mss_ocphi [lbsn: 5],mss_dst1 [lbsn: 5],mss_dst2 [lbsn: 5],mss_dst3 [lbsn: 5],mss_dst4[lbsn: 5] = CoLM_SoilSnowHydrology.water_vsf(nl_colm,const_physical,landpft,VTI.wetwatmax,ipatch, patchtype,lb, var_global.nl_soil,
                                                          deltim, z_soisno[-maxsnl-lb+1:], dz_soisno[-maxsnl-lb+1:],zi_soisno[-maxsnl-lb+1:],
                                                          bsw, theta_r, topostd, BVIC, alpha_vgm, n_vgm, L_vgm, sc_vgm, fc_vgm,
                                                         porsl, psi0, hksati,
                                                            rootr, rootflux, t_soisno[-maxsnl-lb+1:], wliq_soisno[-maxsnl-lb+1:],
                                                            wice_soisno[-maxsnl-lb+1:], smp, hk, pg_rain,
                                                            sm, etr, qseva, qsdew,  qsubl, qfros, qseva_soil, qsdew_soil,
                                                            qsubl_soil, qfros_soil, qseva_snow, qsdew_snow,
                                                            qsubl_snow, qfros_snow, fsno, rsur, rsur_se,
                                                         rsur_ie, rnof, qinfl, wtfact,  ssi, pondmx, wimp, zwt, wdsrf, wa, wetwat,
                                                          # flddepth          ,fldfrc            ,qinfl_fld,
                                                          forc_aer,vic_b_infilt, vic_Dsmax, vic_Ds, vic_Ws, vic_c, fevpg,
                                                        mss_bcpho[lbsn:5], mss_bcphi[lbsn: 5], mss_ocpho[lbsn: 5], mss_ocphi[lbsn: 5],
                                                        mss_dst1[lbsn: 5], mss_dst2[lbsn: 5], mss_dst3[lbsn: 5], mss_dst4[lbsn: 5])


        # 执行条件检查并调用相应函数
        if snl < 0:
            # Compaction rate for snow
            lb = snl + 1  # lower bound of array
            dz_soisno[-maxsnl+lb-1:5] = CoLM_SnowLayersCombineDivide.snowcompaction(const_physical, lb, deltim, imelt[-maxsnl+lb-1:5], fiold[-maxsnl+lb-1:5], t_soisno[-maxsnl+lb-1:5], wliq_soisno[-maxsnl+lb-1:5],wice_soisno[-maxsnl+lb-1:5],forc_us,forc_vs,dz_soisno[-maxsnl+lb-1:5])

            # Combine thin snow elements
            lb = maxsnl + 1

            if nl_colm['DEF_USE_SNICAR']:
                wliq_soisno[-maxsnl+lb-1: 6], wice_soisno[-maxsnl+lb-1: 6], t_soisno[-maxsnl+lb-1: 6], dz_soisno[-maxsnl+lb-1: 6], z_soisno[-maxsnl+lb-1: 6], zi_soisno[-maxsnl+lb-1: 7], snowdp, scv, snl,mss_bcpho[-maxsnl+lb-1: 5], mss_bcphi[-maxsnl+lb-1: 5], mss_ocpho[-maxsnl+lb-1: 5], mss_ocphi[-maxsnl+lb-1: 5], mss_dst1[-maxsnl+lb-1: 5], mss_dst2[-maxsnl+lb-1: 5], mss_dst3[-maxsnl+lb-1: 5], mss_dst4[-maxsnl+lb-1: 5] = CoLM_SnowLayersCombineDivide.snowlayerscombine_snicar(lb, snl, z_soisno[-maxsnl+lb-1:6],
                                                                            dz_soisno[-maxsnl+lb-1:6], zi_soisno[-maxsnl+lb-1:7], wliq_soisno[-maxsnl+lb-1:6], wice_soisno[-maxsnl+lb-1: 6], t_soisno[-maxsnl+lb-1: 6], scv, snowdp, mss_bcpho[-maxsnl+lb-1: 5], mss_bcphi[-maxsnl+lb-1: 5], mss_ocpho[-maxsnl+lb-1: 5], mss_ocphi[-maxsnl+lb-1: 5], mss_dst1[-maxsnl+lb-1: 5], mss_dst2[-maxsnl+lb-1: 5], mss_dst3[-maxsnl+lb-1: 5], mss_dst4[-maxsnl+lb-1: 5])
            else:
                wliq_soisno[-maxsnl+lb-1: 6], wice_soisno[-maxsnl+lb-1: 6], t_soisno[-maxsnl+lb-1: 6], dz_soisno[-maxsnl+lb-1: 7], zi_soisno[:6], snowdp, scv, snl= CoLM_SnowLayersCombineDivide.snowlayerscombine(var_global, lb, snl,
                                                                                           z_soisno[-maxsnl+lb-1: 6],
                                                                                           dz_soisno[-maxsnl+lb-1: 6],
                                                                                           zi_soisno[-maxsnl+lb-1: 7],
                                                                                           wliq_soisno[-maxsnl+lb-1: 6],
                                                                                           wice_soisno[-maxsnl+lb-1: 6],
                                                                                           t_soisno[-maxsnl+lb-1: 6], scv,
                                                                                           snowdp)

            # Divide thick snow elements
            if snl < 0:
                if nl_colm['DEF_USE_SNICAR']:
                    wice_soisno[-maxsnl+lb-1:5], wliq_soisno[-maxsnl+lb-1:5], t_soisno[-maxsnl+lb-1:5], dz_soisno[-maxsnl+lb-1:5], z_soisno[-maxsnl+lb-1:5], zi_soisno[-maxsnl+lb-1:6], snl = CoLM_SnowLayersCombineDivide.snowlayersdivide_snicar(var_global, lb, snl, z_soisno[-maxsnl+lb-1:5], dz_soisno[-maxsnl+lb-1:5], zi_soisno[-maxsnl+lb-1:6], wliq_soisno[-maxsnl+lb-1:5], wice_soisno[-maxsnl+lb-1:5], t_soisno[-maxsnl+lb-1:5], mss_bcpho[-maxsnl+lb-1: 5], mss_bcphi[-maxsnl+lb-1: 5], mss_ocpho[-maxsnl+lb-1: 5], mss_ocphi[-maxsnl+lb-1: 5], mss_dst1[-maxsnl+lb-1: 5], mss_dst2[-maxsnl+lb-1: 5], mss_dst3[-maxsnl+lb-1: 5], mss_dst4[-maxsnl+lb-1: 5])
            else:
                snl, wice_soisno[-maxsnl+lb-1:5], wliq_soisno[-maxsnl+lb-1:5], t_soisno[-maxsnl+lb-1:5], dz_soisno[-maxsnl+lb-1:5], z_soisno[-maxsnl+lb-1:5], zi_soisno[-maxsnl+lb-1:6] = CoLM_SnowLayersCombineDivide.snowlayersdivide(var_global, lb, snl, z_soisno[-maxsnl+lb-1:5], dz_soisno[-maxsnl+lb-1:5], zi_soisno[-maxsnl+lb-1:6], wliq_soisno[-maxsnl+lb-1:5], wice_soisno[-maxsnl+lb-1:5], t_soisno[-maxsnl+lb-1:5])

        # 设置空节点为零
        if snl > maxsnl:
            wice_soisno[:snl -maxsnl] = 0.0
            wliq_soisno[:snl -maxsnl] = 0.0
            t_soisno[:snl -maxsnl] = 0.0
            z_soisno[:snl -maxsnl] = 0.0
            dz_soisno[:snl -maxsnl] = 0.0

        lb = snl + 1
        t_grnd = t_soisno[-maxsnl-lb+1]

        # ----------------------------------------
        # energy balance
        # ----------------------------------------
        zerr = errore

        if nl_colm['CoLMDEBUG']:
            if abs(errore) > 0.5:
                print(f'Warning: energy balance violation {errore}, {patchclass}')

        # ----------------------------------------
        # water balance
        # ----------------------------------------
        endwb = sum(wice_soisno[5:]) + sum(wliq_soisno[5:]) + ldew + scv + wa

        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            endwb += wdsrf
            if patchtype == 2:
                endwb += wetwat

        if nl_colm['CaMa_Flood']:
            pass
        # if nl_colm['LWINFILT']:
        #     if patchtype == 0:
        #         endwb -= qinfl_fld * deltim

        if not nl_colm['CatchLateralFlow']:
            errorw = (endwb - totwb) - (
                    forc_prc + forc_prl - fevpa - rnof - errw_rsub) * deltim
        else:
            errorw=(endwb-totwb)-(forc_prc+forc_prl-fevpa-errw_rsub)*deltim
        # errorw=(endwb-totwb)-(forc_prc+forc_prl-fevpa-errw_rsub)*deltim

        # if nl_colm['CROP']:
        #     if  nl_colm['DEF_USE_IRRIGATION']:
        #         errorw -= irrig_rate[ipatch] * deltim

        if not nl_colm['DEF_USE_VariablySaturatedFlow']:
            if patchtype == 2:
                errorw = 0.0  # wetland

        xerr = errorw / deltim

        if nl_colm['CoLMDEBUG']:
            if abs(errorw) > 1e-3:
                if patchtype <= 1:
                    print('Warning: water balance violation in CoLMMAIN (soil)', errorw)
                elif patchtype == 2:
                    print('Warning: water balance violation in CoLMMAIN (wetland)', errorw)
        # CoLM_stop()
            if abs(errw_rsub * deltim) > 1e-3:
                print('Subsurface runoff deficit due to PHS', errw_rsub * deltim)

    # !======================================================================
    elif patchtype == 3:  # ! <=== is LAND ICE (glacier/ice sheet) (patchtype = 3)
    # !======================================================================
        # initial set
        scvold = scv  # snow mass at previous time step

        snl = 0
        for j in range(-maxsnl):
            if wliq_soisno[j] + wice_soisno[j] > 0.0:
                snl -= 1

        zi_soisno[5] = 0.0
        if snl < 0:
            for j in range(4, 4+snl, -1):
                zi_soisno[j] = zi_soisno[j + 1] - dz_soisno[j]

        for j in range(6, var_global.nl_soil-maxsnl + 1):
            zi_soisno[j] = zi_soisno[j - 1] + dz_soisno[j-1]

        totwb = scv + sum(wice_soisno[5:] + wliq_soisno[5:])
        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            totwb += wdsrf

        fiold[:] = 0.0
        if snl < 0:
            fiold[-maxsnl-snl:5] = wice_soisno[-maxsnl-snl:5] / (wliq_soisno[-maxsnl-snl:5] + wice_soisno[-maxsnl-snl:5])

        pg_rain = prc_rain + prl_rain
        pg_snow = prc_snow + prl_snow

        t_rain = t_precip
        if wliq_soisno[5] > dz_soisno[5] * const_physical.denh2o:
            wextra = (wliq_soisno[5] - dz_soisno[5] * const_physical.denh2o) / deltim
        t_rain = (pg_rain * t_precip + wextra * t_soisno[5]) / (pg_rain + wextra)
        pg_rain += wextra
        wliq_soisno[5] = dz_soisno[5] * const_physical.denh2o
        totwb -= wextra * deltim

        t_snow = t_precip
        if wice_soisno[5] > dz_soisno[5] * const_physical.denice:
            wextra = (wice_soisno[5] - dz_soisno[5] * const_physical.denice) / deltim
        t_snow = (pg_snow * t_precip + wextra * t_soisno[5]) / (pg_snow + wextra)
        pg_snow += wextra
        wice_soisno[5] = dz_soisno[5] * const_physical.denice
        totwb -= wextra * deltim

        if pg_rain + pg_snow > 0:
            t_precip = (pg_rain * const_physical.cpliq * t_rain + pg_snow * const_physical.cpice * t_snow) / (
                    pg_rain * const_physical.cpliq + pg_snow * const_physical.cpice)

        # ----------------------------------------------------------------
        # Initilize new snow nodes for snowfall / sleet
        # ----------------------------------------------------------------
        # 初始化新的雪节点

        snl_bef = snl

    # 调用newsnow函数
        zi_soisno[:5], z_soisno[:5], dz_soisno[:5], t_soisno[:5], wliq_soisno[:5], wice_soisno[:5], fiold[:5], snl, sag, scv, snowdp, fsno, wetwat = CoLM_NewSnow.newsnow(nl_colm, const_physical, patchtype,
                                                       maxsnl, deltim, t_grnd, pg_rain, pg_snow, bifall,
                                                       t_precip, zi_soisno[:6], z_soisno[:5], dz_soisno[:5],
                                                       t_soisno[:5], wliq_soisno[:5], wice_soisno[:5],
                                                       fiold[:5],
                                                       snl, sag, scv,
                                                       snowdp, fsno, wetwat)

    # ----------------------------------------------------------------
        # Energy and Water balance
        # ----------------------------------------------------------------
        lb = snl + 1  # lower bound of array
        lbsn = min(lb, 0)

        snofrz, imelt[-maxsnl-lb+1:], qseva, qsdew, qsubl, qfros, sm = CoLM_Glacier.GLACIER_TEMP(nl_colm, const_physical, patchtype,
                                                                                  lb, var_global.nl_soil, deltim,
                                                                                  zlnd, zsno, capr, cnfac,
                                                        forc_hgt_u, forc_hgt_t, forc_hgt_q, forc_us,
                                                        forc_vs, forc_t, forc_q, forc_hpbl,
                                                        forc_rhoair, forc_psrf, coszen, sabg,
                                                        forc_frl, fsno,
                                                                                  dz_soisno[-maxsnl-lb+1:],
                                                                                  z_soisno[-maxsnl-lb+1:], zi_soisno[-maxsnl-lb+1:],
                                                                                  t_soisno[-maxsnl-lb+1:], wice_soisno[-maxsnl-lb+1:], wliq_soisno[-maxsnl-lb+1:],
                                                                                scv, snowdp, imelt[-maxsnl-lb+1:], taux,
                                                                                tauy, fsena, fevpa, lfevpa,
                                                                                fseng, fevpg, olrg, fgrnd,
                                                                                qseva, qsdew, qsubl, qfros,
                                                                                sm, tref, qref, trad,
                                                                                errore, emis, z0m, zol,
                                                                                rib, ustar, qstar, tstar,
                                                                                fm, fh, fq, pg_rain, pg_snow, t_precip,
                                                                                  snofrz[-maxsnl-lbsn-1:5], sabg_snow_lyr[-maxsnl-lb+1:6])

        # 调用GLACIER_WATER或GLACIER_WATER_snicar函数
        if nl_colm['DEF_USE_SNICAR']:
            snl,  z_soisno, dz_soisno, zi_soisno, t_soisno, wliq_soisno, wice_soisno, scv, snowdp, gwat, mss_bcpho   ,mss_bcphi   ,mss_ocpho,mss_ocphi, mss_dst1    ,mss_dst2    ,mss_dst3  ,mss_dst4 = CoLM_Glacier.GLACIER_WATER_snicar(var_global.nl_soil, maxsnl, deltim, z_soisno,
                                                     dz_soisno, zi_soisno, t_soisno, wliq_soisno, wice_soisno,
                                                     pg_rain, pg_snow, sm, scv, snowdp, imelt,
                                                     fiold, snl, qseva, qsdew,
                                                     qsubl, qfros, gwat,
                                                     ssi, wimp, forc_us, forc_vs,
                                                     forc_aer,
                                                     mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi,
                                                     mss_dst1, mss_dst2, mss_dst3, mss_dst4)

        else:

            snl,  z_soisno, dz_soisno, zi_soisno, t_soisno, wice_soisno, wliq_soisno, scv, snowdp, gwat = CoLM_Glacier.GLACIER_WATER(var_global.nl_soil, maxsnl, deltim, z_soisno,
                                              dz_soisno, zi_soisno, t_soisno, wliq_soisno, wice_soisno,
                                              pg_rain, pg_snow, sm, scv, snowdp, imelt,
                                              fiold, snl, qseva, qsdew,
                                              qsubl, qfros, gwat,
                                              ssi, wimp, forc_us, forc_vs)

        # 更新水流变量
        if not nl_colm['DEF_USE_VariablySaturatedFlow']:
            rsur = max(0.0, gwat)
            rnof = rsur
        else:
            a = wdsrf + wliq_soisno[5] + gwat * deltim
        if a > dz_soisno[5] * const_physical.denh2o:
            wliq_soisno[5] = dz_soisno[5] * const_physical.denh2o
            wdsrf = a - wliq_soisno[5]
        else:
            wdsrf = 0.0
            wliq_soisno[5] = max(a, 1.0e-8)

        if nl_colm['CatchLateralFlow']:
            if wdsrf > pondmx:
                rsur = (wdsrf - pondmx) / deltim
                wdsrf = pondmx
            else:
                rsur = 0.0
            rnof = rsur
            rsur_se = rsur
            rsur_ie = 0.

        lb = snl + 1
        t_grnd = t_soisno[-maxsnl - lb +1]

        # ----------------------------------------
        # energy and water balance check
        # ----------------------------------------
        zerr = errore

        endwb = scv + sum(wice_soisno[5:] + wliq_soisno[5:])
        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            endwb = wdsrf + endwb

        # Define CatchLateralFlow if not defined
        if not nl_colm['CatchLateralFlow']:
            errorw = (endwb - totwb) - (pg_rain + pg_snow - fevpa - rnof) * deltim
        else:
            errorw = (endwb - totwb) - (pg_rain + pg_snow - fevpa) * deltim

        # Define CoLMDEBUG if not defined
        if nl_colm['CoLMDEBUG']:
            if nl_colm['DEF_USE_VariablySaturatedFlow']:
                if abs(errorw) > 1.0e-3:
                    print('Warning: water balance violation in CoLMMAIN (land ice)', errorw)
        # CALL CoLM_stop() equivalent in Python

        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            xerr = errorw / deltim
        else:
            xerr = 0.0
    # !=================================================================================================
    elif patchtype == 4:  # ! <=== is LAND WATER BODIES (lake, reservior and river) (patchtype = 4)
    # !=================================================================================================
        # Initialize total water balance
        totwb = scv + sum(wice_soisno[5:] + wliq_soisno[5:]) + wa
        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            totwb += wdsrf

        # Initialize snow layer
        snl = 0
        for j in range(-maxsnl):
            if wliq_soisno[j] + wice_soisno[j] > 0:
                snl -= 1

        zi_soisno[6] = 0
        if snl < 0:
            for j in range(4, 4+snl -1, -1):
                zi_soisno[j] = zi_soisno[j + 1] - dz_soisno[j-1]

        for j in range(6, var_global.nl_soil+6):
            zi_soisno[j] = zi_soisno[j - 1] + dz_soisno[j-1]

        scvold = scv  # snow mass at previous time step
        fiold[:] = 0.0

        if snl < 0:
            fiold[-maxsnl+snl:5] = wice_soisno[-maxsnl+snl:5] / (wliq_soisno[-maxsnl+snl:5] + wice_soisno[-maxsnl+snl:5])

        w_old = sum(wliq_soisno[5:]) + sum(wice_soisno[5:])

        pg_rain = prc_rain + prl_rain
        pg_snow = prc_snow + prl_snow

        pg_rain, pg_snow, zi_soisno[:6], z_soisno[:5], dz_soisno[:5], t_soisno[:5], wliq_soisno[:5], wice_soisno[:5], fiold[:5], snl, sag, scv, snowdp, lake_icefrac, t_lake = CoLM_Lake.newsnow_lake(const_physical,maxsnl, var_global.nl_lake, deltim, dz_lake, pg_rain,
                                                           pg_snow, t_precip, bifall, t_lake, zi_soisno[:6], z_soisno[:5],
                                                           dz_soisno[:5], t_soisno[:5], wliq_soisno[:5], wice_soisno[:5],
                                                           fiold[:5], snl, sag, scv,  snowdp, lake_icefrac)
        t_grnd, scv, snowdp, t_soisno, wliq_soisno, wice_soisno, imelt_soisno, t_lake, lake_icefrac, savedtke1, snofrz, taux,tauy,fsena,fevpa,lfevpa,fseng,fevpg,qseva,qsubl,qsdew,qfros,olrg,fgrnd,tref,qref,trad,emis,z0m, zol,rib,ustar,qstar,tstar,fm,fh,fq,sm = CoLM_Lake.laketem(nl_colm, const_physical, patchtype,maxsnl,var_global.nl_soil,var_global.nl_lake,
                                                                                                 patchlatr,
                                                                                                 deltim, forc_hgt_u, forc_hgt_t,
                                                                                                forc_hgt_q, forc_us, forc_vs, forc_t,
                                                                                                forc_q, forc_rhoair, forc_psrf, forc_sols,
                                                                                                forc_soll, forc_solsd, forc_solld, sabg,
                                                                                                forc_frl, dz_soisno, z_soisno, zi_soisno,
                                                                                                dz_lake, lakedepth, vf_quartz, vf_gravels,
                                                                                                vf_om, vf_sand, wf_gravels, wf_sand,
                                                                                                porsl, csol, k_solids,
                                                                                                dksatu, dksatf, dkdry,
                                                                                                BA_alpha, BA_beta, forc_hpbl, t_grnd, scv, snowdp, t_soisno,
                                                                                                wliq_soisno, wice_soisno, imelt, t_lake,
                                                                                                lake_icefrac, savedtke1, snofrz, sabg_snow_lyr, taux, tauy, fsena,
                                                                                                fevpa, lfevpa, fseng, fevpg,
                                                                                                qseva, qsubl, qsdew, qfros,
                                                                                                olrg, fgrnd, tref, qref,
                                                                                                trad, emis, z0m, zol,
                                                                                                rib, ustar, qstar, tstar,
                                                                                                fm, fh, fq, sm )

        z_soisno, dz_soisno, zi_soisno, t_soisno, wliq_soisno, wice_soisno, t_lake, lake_icefrac, qout_snowb,  fseng, fgrnd, snl, scv, snowdp,sm ,mss_bcpho,mss_bcphi,mss_ocpho , mss_ocphi, mss_dst1 , mss_dst2 , mss_dst3  , mss_dst4 = CoLM_Lake.snowwater_lake(nl_colm, const_physical, maxsnl, var_global.nl_soil,
                                       var_global.nl_lake, deltim,  ssi, wimp, porsl,  pg_rain, pg_snow, dz_lake, imelt[:5], fiold[: 5], qseva, qsubl, qsdew, qfros, z_soisno, dz_soisno, zi_soisno, t_soisno, wice_soisno, wliq_soisno, t_lake, lake_icefrac,
    gwat,  fseng, fgrnd, snl, scv,  snowdp, sm, forc_us, forc_vs, forc_aer,  mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4 )

        a = (sum(wliq_soisno[5:]) + sum(wice_soisno[5:]) + scv - w_old - scvold) / deltim
        aa = qseva + qsubl - qsdew - qfros

        if not nl_colm['DEF_USE_VariablySaturatedFlow']:
            rsur = max(0.0, pg_rain + pg_snow - aa - a)
            rnof = rsur
        else:
            wdsrf += (pg_rain + pg_snow - aa - a) * deltim

            if wdsrf + wa < 0:
                wa += wdsrf
                wdsrf = 0
            else:
                wdsrf += wa
                wa = 0

            if not nl_colm['CatchLateralFlow']:
                if    wdsrf > pondmx:
                    rsur = (wdsrf - pondmx) / deltim
                    wdsrf = pondmx
                else:
                    rsur = 0.0
                rnof = rsur
                rsur_se = rsur
                rsur_ie = 0.

        endwb = scv + sum(wice_soisno[5:] + wliq_soisno[5:]) + wa
        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            endwb += wdsrf

        errorw = (endwb - totwb) - (forc_prc + forc_prl - fevpa) * deltim
        if not nl_colm['CatchLateralFlow']:
            errorw += rnof * deltim

        if nl_colm['CoLMDEBUG']:
            if  nl_colm['DEF_USE_VariablySaturatedFlow']:
                if abs(errorw) > 1.0e-3:
                    print('Warning: water balance violation in CoLMMAIN (lake)', errorw)
        # CoLM_stop()

        if nl_colm['DEF_USE_VariablySaturatedFlow']:
            xerr = errorw / deltim
        else:
            xerr = 0.0

        if snl > maxsnl:
            wice_soisno[:-maxsnl+snl] = 0.0
            wliq_soisno[:-maxsnl+snl] = 0.0
            t_soisno[:-maxsnl+snl] = 0.0
            z_soisno[:-maxsnl+snl] = 0.0
            dz_soisno[:-maxsnl+snl] = 0.0

    # !=================================================================================================
    else:  # ! <=== is OCEAN (patchtype >= 99)
    # !=================================================================================================
        print('OCEAN need to add!!!!!!!!')

    # !=================================================================================================
    if nl_colm['CaMa_Flood']:
        pass

    # ======================================================================
    # Preparation for the next time step
    # 1) time-varying parameters for vegetation
    # 2) fraction of snow cover
    # 3) solar zenith angle and
    # 4) albedos
    # ======================================================================

    # cosine of solar zenith angle
    calday = CoLM_TimeManager.calendarday_date(idate, isgreenwich)
    coszen = CoLM_OrbCoszen.orb_coszen(calday, patchlonr, patchlatr)

    if patchtype <= 5:  # LAND
        if nl_colm['DYN_PHENOLOGY']:
    # need to update lai and sai, fveg, green, they are done once in a day only
            if dolai:
                lai,sai,fveg,green = CoLM_LAIEmpirical.LAI_empirical(nl_colm, patchclass, var_global.nl_soil, rootfr, t_soisno[5:])

        # only for soil patches
        # NOTE: lai from remote sensing has already considered snow coverage
        if patchtype == 0:
            if nl_colm['LULC_USGS'] or nl_colm['LULC_IGBP']:
                wt, sigf, fsno = CoLM_SnowFraction.snowfraction(VTV.tlai[ipatch], VTV.tsai[ipatch], z0m, zlnd,
                                            scv, snowdp)
                lai = VTV.tlai[ipatch]
                sai = VTV.tsai[ipatch] * sigf
            if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
                pass

        else:

            wt, sigf, fsno = CoLM_SnowFraction.snowfraction(VTV.tlai[ipatch], VTV.tsai[ipatch], z0m, zlnd, scv,
                                                snowdp)
            lai = VTV.tlai[ipatch]
            sai = VTV.tsai[ipatch] * sigf

        # water volumetric content of soil surface layer [m3/m3]
        ssw = min(1.0, 1.e-3 * wliq_soisno[5] / dz_soisno[5])
        if patchtype >= 3:
            ssw = 1.0

        # ============================================================================
        # Snow aging routine based on Flanner and Zender (2006), Linking snowpack
        # microphysics and albedo evolution, JGR, and Brun (1989), Investigation of
        # wet-snow metamorphism in respect of liquid-water content, Ann. Glaciol.

        dz_soisno_[:6] = dz_soisno[:6]
        t_soisno_[:6] = t_soisno[:6]

        if patchtype == 4:
            dz_soisno_[5] = dz_lake[0]
            t_soisno_[5] = t_lake[0]

        # ============================================================================
        # albedos
        # we supposed CALL it every time-step, because
        # other vegetation related parameters are needed to create
        if doalb:
            snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, sag, alb, ssun, ssha, thermk, extkb, extkd, ssoi, ssno, ssno_lyr = CoLM_Albedo.albland(nl_colm,var_global,gblock, const_physical.tfrz, ipatch, patchtype,deltim,
                     soil_s_v_alb,soil_d_v_alb,soil_s_n_alb,soil_d_n_alb,
                     chil,rho,tau,fveg,green,lai,sai,coszen,
                     wt,fsno,scv,scvold,sag,ssw,pg_snow,forc_t,t_grnd,t_soisno_,dz_soisno_,
                     snl,wliq_soisno,wice_soisno,snw_rds,snofrz,
                     mss_bcpho,mss_bcphi,mss_ocpho,mss_ocphi,
                     mss_dst1,mss_dst2,mss_dst3,mss_dst4,
                     alb,ssun,ssha,ssoi,ssno,ssno_lyr,thermk,extkb,extkd,mpi)

    else:  # OCEAN
        sag = 0.0
        if doalb:
            alb = CoLM_Albedo.albocean(oro, scv, coszen)

    # zero-filling set for glacier/ice-sheet/land water bodies/ocean components
    if patchtype > 2:
        lai = 0.0
        sai = 0.0
        laisun = 0.0
        laisha = 0.0
        green = 0.0
        fveg = 0.0
        sigf = 0.0

        ssun[:, :] = 0.0
        ssha[:, :] = 0.0
        thermk = 0.0
        extkb = 0.0
        extkd = 0.0

        tleaf = forc_t
        ldew_rain = 0.0
        ldew_snow = 0.0
        ldew = 0.0
        fsenl = 0.0
        fevpl = 0.0
        etr = 0.0
        assim = 0.0
        respc = 0.0

        zerr = 0.

        qinfl = 0.
        qdrip = forc_rain + forc_snow
        qintr = 0.
        h2osoi = 0.
        rstfacsun_out = 0.
        rstfacsha_out = 0.
        gssun_out = 0.
        gssha_out = 0.
        assimsun_out = 0.
        etrsun_out = 0.
        assimsha_out = 0.
        etrsha_out = 0.
        rootr = 0.
        rootflux = 0.
        zwt = 0.

    if not nl_colm['DEF_USE_VariablySaturatedFlow']:
        wa = 4800.

    qcharge = 0.
    if nl_colm['DEF_USE_PLANTHYDRAULICS']:
        vegwp = -2.5e4

    h2osoi = wliq_soisno[5:] / (dz_soisno[5:] * const_physical.denh2o) + wice_soisno[5:] / (
            dz_soisno[5:] * const_physical.denice)

    if nl_colm['DEF_USE_VariablySaturatedFlow']:
        wat = sum(wice_soisno[5:] + wliq_soisno[5:]) + ldew + scv + wetwat
    else:
        wat = sum(wice_soisno[5:] + wliq_soisno[5:]) + ldew + scv + wa

    z_sno[:] = z_soisno[:-maxsnl]
    dz_sno[:] = dz_soisno[:-maxsnl]
    # return oro, olrg
    return oro, z_sno, dz_sno, t_soisno, wliq_soisno, wice_soisno, hk, smp, t_lake, lake_icefrac, savedtke1, vegwp, gs0sun, gs0sha, lai_old, o3uptakesun, o3uptakesha, forc_ozone, t_grnd, tleaf, ldew, ldew_rain, ldew_snow, sag, scv, snowdp, zwt, wdsrf, wa, wetwat, snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, ssno_lyr, \
        fveg, fsno, sigf, green, lai, sai, coszen, alb[:], ssun[:], ssha[:], ssoi[:], ssno[:],\
        thermk, extkb,extkd, laisun, laisha, rstfacsun_out, rstfacsun_out, gssun_out, gssha_out,\
        wat, rss, rootr, rootflux, h2osoi, assimsun_out, etrsun_out, assimsha_out, \
        etrsha_out, taux, tauy, fsena, fevpa, lfevpa, fsenl, fevpl, etr, fseng, fevpg, olrg, fgrnd,\
        xerr, zerr, tref, qref, trad, rsur, rsur_se, rsur_ie, rnof, qintr, qinfl, qdrip, qcharge, \
        rst, assim, respc, sabvsun, sabvsha, sabg, sr, solvd, solvi, solnd, solni, srvd, srvi, srnd, \
        srni, solvdln, solviln, solndln, solniln, srvdln, srviln, srndln, srniln, forc_rain, forc_snow, \
        emis, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq
