import numpy as np
import CoLM_Utils
import CoLM_SoilThermalParameters
import CoLM_PhaseChange
import decimal

def GroundTemperature (nl_colm, const_physical, patchtype, lb, nl_soil, deltim, capr, cnfac, vf_quartz, vf_gravels, vf_om, vf_sand, wf_gravels, wf_sand, 
                      porsl, psi0, 
                      bsw, 
                      theta_r, alpha_vgm, n_vgm, L_vgm, sc_vgm, fc_vgm, 
                      csol, k_solids, dksatu, dksatf, dkdry, BA_alpha, BA_beta, 
                      sigf, dz_soisno, z_soisno, zi_soisno, 
                      t_soisno, t_grnd, t_soil, t_snow, wice_soisno, wliq_soisno, scv, snowdp, fsno, 
                      frl, dlrad, sabg, sabg_soil, sabg_snow, sabg_snow_lyr, 
                      fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, cgrnd, htvp, emg, 
                      imelt, snofrz, sm, xmf, fact, pg_rain, pg_snow, t_precip):
    """
    determine net radiation.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         :     Calculate snow and soil temperatures.
    The volumetric heat capacity is calculated as a linear combination
    in terms of the volumetric fraction of the constituent phases.
    The thermal conductivity of soil is computed from
    the algorithm of Johansen (as reported by Farouki 1981), and of snow is from
    the formulation used in SNTHERM (Jordan 1991).
    Boundary conditions:
        F = Rnet - Hg - LEg + Hpr(top),  F= 0 (base of the soil column).
    Soil / snow temperature is predicted from heat conduction
    in 10 soil layers and up to 5 snow layers.
    The thermal conductivities at the interfaces between two neighbor layers
    (j, j+1) are derived from an assumption that the flux across the interface
    is equal to that from the node j to the interface and the flux from the
    interface to the node j+1. The equation is solved using the Crank-Nicholson
    method and resulted in a tridiagonal system equation.
    Phase change (see meltf.F90)
    Original author : Yongjiu Dai, 09/15/1999; 08/30/2002; 05/2018
    REVISIONS:
    Nan Wei,  07/2017: interaction btw prec and land surface
    Nan Wei,  01/2019: USE the new version of soil thermal parameters to calculate soil temperature
    Hua Yuan, 01/2023: modified ground heat flux, temperature and meltf
                       calculation for SNICAR model
                       

    Args:
        patchtype    (int): land water type (3=glaciers)
        forc_t       (float): temperature at agcm reference height [kelvin]
        forc_q       (float): specific humidity at agcm reference height [kg/kg]
        forc_psrf    (float): atmosphere pressure at the surface [pa]
        forc_prc     (float): convective precipitation [mm/s]
        forc_prl     (float): large scale precipitation [mm/s]

        tcrit        (float): critical temp. to determine rain or snow

    Returns:
        prc_rain     (float): convective rainfall [kg/(m2 s)]
        prc_snow     (float): convective snowfall [kg/(m2 s)]
        prl_rain     (float): large scale rainfall [kg/(m2 s)]
        prl_snow     (float): large scale snowfall [kg/(m2 s)]
        t_precip     (float): snowfall/rainfall temperature [kelvin]
        bifall       (float): bulk density of newly fallen dry snow [kg/m3]
    """
        # Local variables
    cv = np.zeros(nl_soil)
    tk = np.zeros(nl_soil)
    hcap = np.zeros(nl_soil)
    thk = np.zeros(nl_soil)
    
    at = np.zeros(nl_soil)
    bt = np.zeros(nl_soil)
    ct = np.zeros(nl_soil)
    rt = np.zeros(nl_soil)
    
    fn = np.zeros(nl_soil)
    fn1 = np.zeros(nl_soil)
    dzm = 0.0
    dzp = 0.0
    
    t_soisno_bef = np.copy(t_soisno)
    wice_soisno_bef = np.copy(wice_soisno)
    hs = 0.0
    hs_soil = 0.0
    hs_snow = 0.0
    dhsdT = 0.0
    brr = np.zeros(nl_soil)
    vf_water = np.zeros(nl_soil)
    vf_ice = np.zeros(nl_soil)
    rhosnow = 0.0
    # =======================================================================
    # Soil ground and wetland heat capacity
    for i in range(nl_soil):
        vf_water[i] = wliq_soisno[i] / (dz_soisno[i] * const_physical.denh2o)
        vf_ice[i] = wice_soisno[i] / (dz_soisno[i] * const_physical.denice)
        hcap[i], thk[i] = CoLM_SoilThermalParameters.soil_hcap_cond(nl_colm, const_physical, vf_gravels[i], vf_om[i], vf_sand[i], porsl[i], wf_gravels[i], wf_sand[i], k_solids[i],
                       csol[i], dkdry[i], dksatu[i], dksatf[i], BA_alpha[i], BA_beta[i],
                       t_soisno[i], vf_water[i], vf_ice[i],hcap[i], thk[i])
        cv[i] = hcap[i] * dz_soisno[i]
    if lb == 1 and scv > 0.0:
        cv[0] += const_physical.cpice * scv

    # Snow heat capacity
    if lb <= 0:
        cv[:0] = const_physical.cpliq * wliq_soisno[:0] + const_physical.cpice * wice_soisno[:0]

    # Snow thermal conductivity
    if lb <= 0:
        for i in range(1-lb):
            rhosnow = (wice_soisno[i] + wliq_soisno[i]) / dz_soisno[i]
            # presently option [1] is the default option
            # [1] Jordan (1991) pp. 18
            thk[i] = const_physical.tkair + (7.75e-5 * rhosnow + 1.105e-6 * rhosnow**2) * (const_physical.tkice - const_physical.tkair)
            # ! [2] Sturm et al (1997)
            # ! thk(i) = 0.0138 + 1.01e-3*rhosnow + 3.233e-6*rhosnow**2
            # ! [3] Ostin and Andersson presented in Sturm et al., (1997)
            # ! thk(i) = -0.871e-2 + 0.439e-3*rhosnow + 1.05e-6*rhosnow**2
            # ! [4] Jansson(1901) presented in Sturm et al. (1997)
            # ! thk(i) = 0.0293 + 0.7953e-3*rhosnow + 1.512e-12*rhosnow**2
            # ! [5] Douville et al., (1995)
            # ! thk(i) = 2.2*(rhosnow/const_physical.denice)**1.88
            # ! [6] van Dusen (1992) presented in Sturm et al. (1997)
            # ! thk(i) = 0.021 + 0.42e-3*rhosnow + 0.22e-6*rhosnow**2
    
    # Thermal conductivity at the layer interface
    for i in range(lb-1, nl_soil-1):
           # ! the following consideration is try to avoid the snow conductivity
           # ! to be dominant in the thermal conductivity of the interface.
           # ! Because when the distance of bottom snow node to the interfacee
           # ! is larger than that of interface to top soil node,
           # ! the snow thermal conductivity will be dominant, and the result is that
           # ! lees heat tranfer between snow and soil
    
        if i == -1 and (z_soisno[i+1] - zi_soisno[i] < zi_soisno[i] - z_soisno[i]):
            tk[i] = 2.0 * thk[i] * thk[i+1] / (thk[i] + thk[i+1])
            tk[i] = max(0.5 * thk[i+1], tk[i])
        else:
            tk[i] = thk[i] * thk[i+1] * (z_soisno[i+1] - z_soisno[i]) / (thk[i] * (z_soisno[i+1] - zi_soisno[i+1]) + thk[i+1] * (zi_soisno[i+1] - z_soisno[i]))
    tk[nl_soil-1] = 0.0
    
    # Net ground heat flux into the surface and its temperature derivative
    #! 08/19/2021, yuan: NOTE! removed sigf, LAI->100% coverni
    if nl_colm['DEF_USE_SNICAR'] and lb < 1:
        hs = sabg_snow_lyr[lb] + sabg_soil + dlrad * emg - (fseng + fevpg * htvp) + const_physical.cpliq * pg_rain * (t_precip - t_grnd) + const_physical.cpice * pg_snow * (t_precip - t_grnd)
    else:
        hs = sabg + dlrad * emg - (fseng + fevpg * htvp) + const_physical.cpliq * pg_rain * (t_precip - t_grnd) + const_physical.cpice * pg_snow * (t_precip - t_grnd)
    
    if not nl_colm['DEF_SPLIT_SOILSNOW']:
        hs = hs - emg * const_physical.stefnc * t_grnd**4
    else:
        #! 03/08/2020, yuan: Separate soil and snow
        hs = hs - fsno * emg * const_physical.stefnc * t_snow**4 - (1.0 - fsno) * emg * const_physical.stefnc * t_soil**4
        #! 03/08/2020, yuan: calculate hs_soil, hs_snow for
        # Calculate hs_soil, hs_snow for soil/snow fractional cover separately.
        hs_soil = dlrad * emg - emg * const_physical.stefnc * t_soil**4 - (fseng_soil + fevpg_soil * htvp) + \
                  const_physical.cpliq * pg_rain * (t_precip - t_soil) + const_physical.cpice * pg_snow * (t_precip - t_soil)
        hs_soil = hs_soil * (1.0 - fsno) + sabg_soil
        
        hs_snow = dlrad * emg - emg * const_physical.stefnc * t_snow**4 - (fseng_snow + fevpg_snow * htvp) + \
                  const_physical.cpliq * pg_rain * (t_precip - t_snow) + const_physical.cpice * pg_snow * (t_precip - t_snow)
        
        if nl_colm['DEF_USE_SNICAR'] and lb < 1:
            hs_snow = hs_snow * fsno + sabg_snow_lyr[lb]
        else:
            hs_snow = hs_snow * fsno + sabg_snow
        
        dhsdT = -cgrnd - 4.0 * emg * const_physical.stefnc * t_grnd**3 - const_physical.cpliq * pg_rain - const_physical.cpice * pg_snow
        
        if abs(sabg_soil + sabg_snow - sabg) > 1.e-6 or abs(hs_soil + hs_snow - hs) > 1.e-6:
            print("MOD_GroundTemperature.F90: Error in splitting soil and snow surface!")
            print("sabg:", sabg, "sabg_soil:", sabg_soil, "sabg_snow:", sabg_snow)
            print("hs:", hs, "hs_soil:", hs_soil, "hs_snow:", hs_snow)
            # CoLM_stop()
    
    dhsdT = -cgrnd - 4.0 * emg * const_physical.stefnc * t_grnd**3 - const_physical.cpliq * pg_rain - const_physical.cpice * pg_snow
    
    # if abs(sabg_soil + sabg_snow - sabg) > 1.e-6 or abs(hs_soil + hs_snow - hs) > 1.e-6:
    #     raise ValueError("Energy balance error")

    t_soisno_bef[lb:] = t_soisno[lb:]

    j = lb -1
    fact[j] = deltim / cv[j] * dz_soisno[j] / (0.5 * (z_soisno[j] - zi_soisno[j] + capr * (z_soisno[j+1] - zi_soisno[j])))

    for j in range(lb, nl_soil):
        fact[j] = deltim / cv[j]

    for j in range(lb-1, nl_soil - 1):
        fn[j] = tk[j] * (t_soisno[j+1] - t_soisno[j]) / (z_soisno[j+1] - z_soisno[j])
    fn[nl_soil - 1] = 0.0

    # set up vector r and vectors a, b, c that define tridiagonal matrix
    j = lb - 1
    dzp = z_soisno[j+1] - z_soisno[j]
    at[j] = 0.0
    ct[j] = -(1.0 - cnfac) * fact[j] * tk[j] / dzp

    # the first layer
    if j < 0 and nl_colm['DEF_SPLIT_SOILSNOW']:
        bt[j] = 1 + (1.0 - cnfac) * fact[j] * tk[j] / dzp - fact[j] * fsno * dhsdT
        rt[j] = t_soisno[j] + fact[j] * (hs_snow - fsno * dhsdT * t_soisno[j] + cnfac * fn[j])
    else:
        bt[j] = 1 + (1.0 - cnfac) * fact[j] * tk[j] / dzp - fact[j] * dhsdT
        rt[j] = t_soisno[j] + fact[j] * (hs - dhsdT * t_soisno[j] + cnfac * fn[j])

    for j in range(lb, nl_soil - 1):
        dzm = z_soisno[j] - z_soisno[j-1]
        dzp = z_soisno[j+1] - z_soisno[j]

        if j < 0:  # snow layer
            at[j] = - (1.0 - cnfac) * fact[j] * tk[j-1] / dzm
            bt[j] = 1.0 + (1.0 - cnfac) * fact[j] * (tk[j] / dzp + tk[j-1] / dzm)
            ct[j] = - (1.0 - cnfac) * fact[j] * tk[j] / dzp
            if nl_colm['DEF_USE_SNICAR']:
                rt[j] = t_soisno[j] + fact[j] * sabg_snow_lyr[j] + cnfac * fact[j] * (fn[j] - fn[j-1])
            else:
                rt[j] = t_soisno[j] + cnfac * fact[j] * (fn[j] - fn[j-1])

        if j == 0:  # the first soil layer
            at[j] = - (1.0 - cnfac) * fact[j] * tk[j-1] / dzm
            ct[j] = - (1.0 - cnfac) * fact[j] * tk[j] / dzp
            if not nl_colm['DEF_SPLIT_SOILSNOW']:
                bt[j] = 1.0 + (1.0 - cnfac) * fact[j] * (tk[j] / dzp + tk[j-1] / dzm)
                rt[j] = t_soisno[j] + cnfac * fact[j] * (fn[j] - fn[j-1])
            else:
                bt[j] = 1.0 + (1.0 - cnfac) * fact[j] * (tk[j] / dzp + tk[j-1] / dzm) - (1.0 - fsno) * dhsdT * fact[j]
                rt[j] = t_soisno[j] + cnfac * fact[j] * (fn[j] - fn[j-1]) + fact[j] * (hs_soil - (1.0 - fsno) * dhsdT * t_soisno[j])

        if j > 0:  # inner soil layer
            at[j] = - (1.0 - cnfac) * fact[j] * tk[j-1] / dzm
            bt[j] = 1.0 + (1.0 - cnfac) * fact[j] * (tk[j] / dzp + tk[j-1] / dzm)
            ct[j] = - (1.0 - cnfac) * fact[j] * tk[j] / dzp
            rt[j] = t_soisno[j] + cnfac * fact[j] * (fn[j] - fn[j-1])

    j = nl_soil - 1
    dzm = z_soisno[j] - z_soisno[j-1]
    at[j] = - (1.0 - cnfac) * fact[j] * tk[j-1] / dzm
    bt[j] = 1.0 + (1.0 - cnfac) * fact[j] * tk[j-1] / dzm
    ct[j] = 0.0
    rt[j] = t_soisno[j] - cnfac * fact[j] * fn[j-1]

    i = len(at)

    t_soisno = CoLM_Utils.tridia(len(at), at, bt, ct, rt, t_soisno)
    #===========================================================================
    # melting or freezing
    #===========================================================================

    for j in range(lb, nl_soil - 1):
        fn1[j] = tk[j] * (t_soisno[j+1] - t_soisno[j]) / (z_soisno[j+1] - z_soisno[j])
    fn1[nl_soil - 1] = 0.

    j = lb
    brr[j] = cnfac * fn[j] + (1. - cnfac) * fn1[j]

    for j in range(lb + 1, nl_soil):
        brr[j] = cnfac * (fn[j] - fn[j-1]) + (1. - cnfac) * (fn1[j] - fn1[j-1])

    if nl_colm['DEF_USE_SNICAR']:
        wice_soisno_bef[lb:0] = wice_soisno[lb:0]

        t_soisno, wliq_soisno, wice_soisno, imelt, sm, xmf, snowdp, scv = CoLM_PhaseChange.meltf_snicar(patchtype, lb, nl_soil, deltim,
                     fact[lb:], brr[lb:], hs, hs_soil, hs_snow, fsno, sabg_snow_lyr[lb:], dhsdT,
                     t_soisno_bef[lb:], t_soisno[lb:], wliq_soisno[lb:], wice_soisno[lb:], imelt[lb:],
                     scv, snowdp, sm, xmf, porsl, psi0,
                     bsw ,
                     theta_r, alpha_vgm, n_vgm, L_vgm,
                     sc_vgm, fc_vgm ,
                     dz_soisno[1:nl_soil])

        # Layer freezing mass flux (positive)
        for j in range(lb, 0):
            if imelt[j] == 2 and j < 1:
                snofrz[j] = max(0., (wice_soisno[j] - wice_soisno_bef[j])) / deltim

    else:
        t_soisno, wice_soisno, wliq_soisno, scv, snowdp, sm, xmf, imelt = CoLM_PhaseChange.meltf(nl_colm,const_physical, patchtype, lb, nl_soil, deltim,
              fact[lb-1:], brr[lb-1:], hs, hs_soil, hs_snow, fsno, dhsdT,
              t_soisno_bef[lb-1:], t_soisno[lb-1:], wliq_soisno[lb-1:], wice_soisno[lb-1:], imelt[lb-1:],
              scv, snowdp, sm, xmf, porsl, psi0,
              bsw ,
              theta_r, alpha_vgm, n_vgm, L_vgm,
              sc_vgm, fc_vgm ,
              dz_soisno[:nl_soil])

    return t_soisno, wice_soisno, wliq_soisno, scv, snowdp, sm, xmf, fact, imelt, snofrz