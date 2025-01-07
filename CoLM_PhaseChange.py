import numpy as np
import CoLM_Hydro_SoilFunction

def meltf(co_lm, const_physical, patchtype,lb,nl_soil,deltim, 
                     fact,brr,hs,hs_soil,hs_snow,fsno,dhsdT, 
                     t_soisno_bef,t_soisno,wliq_soisno,wice_soisno,imelt, 
                     scv,snowdp,sm,xmf,porsl,psi0, bsw, theta_r,alpha_vgm,n_vgm,L_vgm,
                     sc_vgm,fc_vgm, dz):
    """
    Original author  : Qinghliang Li, 17/02/2024; Jinlong Zhu,   17/02/2024;
    software         : calculation of the phase change within snow and soil layers:
                    ! (1) check the conditions which the phase change may take place,
                    !     i.e., the layer temperature is great than the freezing point
                    !     and the ice mass is not equal to zero (i.e., melting),
                    !     or layer temperature is less than the freezing point
                    !     and the liquid water mass is not equal to zero (i.e., freezing);
                    ! (2) assess the rate of phase change from the energy excess (or deficit)
                    !     after setting the layer temperature to freezing point;
                    ! (3) re-adjust the ice and liquid mass, and the layer temperature

    Args:
            Code     (type)           Standard name                              Units

        |-- calday   (float)       : Julian cal day                           [1.xx to 365.xx]
        |-- dlat     (float)       : Centered latitude                        [radians]
        |-- dlon     (float)       : Centered longitude                       [radians]

    Returns:
            cosz     (float)       :the cosine of the solar zenith angle      [-]

    """

    # Declare local variables
    hm = np.zeros(nl_soil + 1)  # energy residual [W/m2]
    xm = np.zeros(nl_soil + 1)  # metling or freezing within a time step [kg/m2]
    wmass0 = np.zeros(nl_soil + 1)  # initial water mass [kg/m2]
    wice0 = np.zeros(nl_soil + 1)  # initial ice lens [kg/m2]
    wliq0 = np.zeros(nl_soil + 1)  # initial liquid water [kg/m2]
    supercool = np.zeros(nl_soil + 1)  # the maximum liquid water when the soil temperature is below the freezing point [mm3/mm3]

    # -----------------------------------------------------------------------
    # Initialize variables
    sm = 0.
    xmf = 0.
    for j in range(lb-1, nl_soil):
        imelt[j] = 0
        hm[j] = 0.
        xm[j] = 0.
        wice0[j] = wice_soisno[j]
        wliq0[j] = wliq_soisno[j]
        wmass0[j] = wice_soisno[j] + wliq_soisno[j]

    scvold = scv
    we = 0.
    if lb <= 0:
        we = np.sum(wice_soisno[lb:0] + wliq_soisno[lb:0])

    # supercooling water
    if co_lm['DEF_USE_SUPERCOOL_WATER']:
        for j in range(0, nl_soil):
            supercool[j] = 0.0
            if t_soisno[j] < const_physical.tfrz and patchtype <= 2:
                smp = const_physical.hfus * (t_soisno[j] - const_physical.tfrz) / (const_physical.grav * t_soisno[j]) * 1000.  # mm
                if porsl[j] > 0.:
                    if co_lm['Campbell_SOIL_MODEL']:
                        supercool[j] = porsl[j] * (smp / psi0[j]) ** (-1.0 / bsw[j])
                    else:
                        supercool[j] = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(smp, porsl[j], theta_r[j], -10.0, 5,
                                                          [alpha_vgm[j], n_vgm[j], L_vgm[j], sc_vgm[j], fc_vgm[j]])
                else:
                    supercool[j] = 0.
                supercool[j] = supercool[j] * dz[j] * 1000.  # mm

    for j in range(lb-1, nl_soil):
        # Melting identification
        # If ice exists above the melting point, melt some to liquid.
        if wice_soisno[j] > 0. and t_soisno[j] > const_physical.tfrz:
            imelt[j] = 1
            t_soisno[j] = const_physical.tfrz

        # Freezing identification
        # If liquid exists below the freezing point, freeze some to ice.
        if j <= 0:
            if wliq_soisno[j] > 0. and t_soisno[j] < const_physical.tfrz:
                imelt[j] = 2
                t_soisno[j] = const_physical.tfrz
        else:
            if co_lm['DEF_USE_SUPERCOOL_WATER']:
                if wliq_soisno[j] > supercool[j] and t_soisno[j] < const_physical.tfrz:
                    imelt[j] = 2
                    t_soisno[j] = const_physical.tfrz
            else:
                if wliq_soisno[j] > 0. and t_soisno[j] < const_physical.tfrz:
                    imelt[j] = 2
                    t_soisno[j] = const_physical.tfrz

    # If snow exists, but its thickness is less than the critical value (0.01 m)
    if lb == 1 and scv > 0.:
        if t_soisno[1] > const_physical.tfrz:
            imelt[1] = 1
            t_soisno[1] = const_physical.tfrz

    # Calculate the energy surplus and loss for melting and freezing
    for j in range(lb-1, nl_soil):
        if imelt[j] > 0:
            tinc = t_soisno[j] - t_soisno_bef[j]

            if j > lb:  # Not the top layer
                if j == 1 and co_lm['DEF_SPLIT_SOILSNOW'] and patchtype < 3:  # Interface soil layer
                    # Separate soil/snow heat flux, exclude glacier (3)
                    hm[j] = hs_soil + (1. - fsno) * dhsdT * tinc + brr[j] - tinc / fact[j]
                else:  # Internal layers other than the interface soil layer
                    hm[j] = brr[j] - tinc / fact[j]
            else:  # Top layer
                if j == 1 or (not co_lm['DEF_SPLIT_SOILSNOW']) or patchtype == 3:  # Soil layer
                    hm[j] = hs + dhsdT * tinc + brr[j] - tinc / fact[j]
                else:  # Snow cover
                    # Separate soil/snow heat flux, exclude glacier (3)
                    hm[j] = hs_snow + fsno * dhsdT * tinc + brr[j] - tinc / fact[j]

    for j in range(lb-1, nl_soil):
        if imelt[j] == 1 and hm[j] < 0.:
            hm[j] = 0.
            imelt[j] = 0

        # This error was checked carefully, it results from the computed error
        # of "Tridiagonal-Matrix" in SUBROUTINE "thermal".
        if imelt[j] == 2 and hm[j] > 0.:
            hm[j] = 0.
            imelt[j] = 0

    #! The rate of melting and freezing
    for j in range(lb-1, nl_soil):
        if imelt[j] > 0 and abs(hm[j]) > 0.:
            xm[j] = hm[j] * deltim / const_physical.hfus  # kg/m2

            # IF snow exists, but its thickness less than the critical value (1 cm)
            # Note: more work is need on how to tune the snow depth at this case
            if j == 0 and lb == 1 and scv > 0. and xm[j] > 0.:
                temp1 = scv  # kg/m2
                scv = max(0., temp1 - xm[j])
                propor = scv / temp1
                snowdp = propor * snowdp
                heatr = hm[j] - const_physical.hfus * (temp1 - scv) / deltim  # W/m2
                if heatr > 0.:
                    xm[j] = heatr * deltim / const_physical.hfus  # kg/m2
                    hm[j] = heatr  # W/m2
                else:
                    xm[j] = 0.
                    hm[j] = 0.
                sm = max(0., (temp1 - scv)) / deltim  # kg/(m2 s)
                xmf = const_physical.hfus * sm

            heatr = 0.
            if xm[j] > 0.:
                wice_soisno[j] = max(0., wice0[j] - xm[j])
                heatr = hm[j] - const_physical.hfus * (wice0[j] - wice_soisno[j]) / deltim
            else:
                if j <= 0:
                    wice_soisno[j] = min(wmass0[j], wice0[j] - xm[j])
                else:
                    if co_lm['DEF_USE_SUPERCOOL_WATER']:
                        if wmass0[j] < supercool[j]:
                            wice_soisno[j] = 0.
                        else:
                            wice_soisno[j] = min(wmass0[j] - supercool[j], wice0[j] - xm[j])
                    else:
                        wice_soisno[j] = min(wmass0[j], wice0[j] - xm[j])
                heatr = hm[j] - const_physical.hfus * (wice0[j] - wice_soisno[j]) / deltim

            wliq_soisno[j] = max(0., wmass0[j] - wice_soisno[j])

            if abs(heatr) > 0.:
                if j > lb:  # => not the top layer
                    if j == 1 and co_lm['DEF_SPLIT_SOILSNOW'] and patchtype < 3:
                        # -> interface soil layer
                        t_soisno[j] += fact[j] * heatr / (1. - fact[j] * (1. - fsno) * dhsdT)
                    else:
                        # -> internal layers other than the interface soil layer
                        t_soisno[j] += fact[j] * heatr
                else:  # => top layer
                    if j == 1 or (not co_lm['DEF_SPLIT_SOILSNOW']) or patchtype == 3:
                        # -> soil layer
                        t_soisno[j] += fact[j] * heatr / (1. - fact[j] * dhsdT)
                    else:
                        # -> snow cover
                        t_soisno[j] += fact[j] * heatr / (1. - fact[j] * fsno * dhsdT)

                if co_lm['DEF_USE_SUPERCOOL_WATER']:
                    if j <= 0 or patchtype == 3:  # snow
                        if wliq_soisno[j] * wice_soisno[j] > 0.:
                            t_soisno[j] = const_physical.tfrz
                else:
                    if wliq_soisno[j] * wice_soisno[j] > 0.:
                        t_soisno[j] = const_physical.tfrz

            xmf += const_physical.hfus * (wice0[j] - wice_soisno[j]) / deltim

            if imelt[j] == 1 and j < 1:
                sm += max(0., (wice0[j] - wice_soisno[j])) / deltim

    scvold = scv

    if lb <= 0:
        we = sum(wice_soisno[lb:0] + wliq_soisno[lb:0]) - we
        if abs(we) > 1.e-6:
            print('meltf err :', we)
            # CoLM_stop()
    return t_soisno, wice_soisno, wliq_soisno, scv, snowdp, sm, xmf, imelt

def meltf_snicar(co_lm, const_physical, patchtype, lb, nl_soil, deltim, fact, brr, hs, hs_soil, hs_snow, fsno, 
                 sabg_snow_lyr, dhsdT, t_soisno_bef, t_soisno, wliq_soisno, wice_soisno, 
                 imelt, scv, snowdp, sm, xmf, porsl, psi0, dz, 
                 bsw=None, theta_r=None, alpha_vgm=None, n_vgm=None, L_vgm=None, 
                 sc_vgm=None, fc_vgm=None):

    # Constants (assuming these are defined somewhere)
    # Local
    hm = np.zeros(nl_soil-lb)                  # energy residual [W/m2]
    xm = np.zeros(nl_soil-lb)                 # metling or freezing within a time step [kg/m2]
    heatr = 0.0                           # energy residual or loss after melting or freezing
    temp1 = 0.0                           # temporary variables [kg/m2]
    temp2 = 0.0                           # temporary variables [kg/m2]
    smp = 0.0
    supercool = np.zeros(nl_soil)            # the maximum liquid water when the soil temperature is below the   freezing point [mm3/mm3]
    wmass0 = np.zeros(nl_soil-lb)
    wice0 = np.zeros(nl_soil-lb)
    wliq0 = np.zeros(nl_soil-lb)
    propor = 0.0
    tinc = 0.0
    we = 0.0
    scvold = 0.0
    sm =0
    xmf = 0
    
    for j in range(lb,nl_soil+1):
        imelt[j] = 0
        hm[j] = 0.
        xm[j] = 0.
        wice0[j] = wice_soisno[j]
        wliq0[j] = wliq_soisno[j]
        wmass0[j] = wice_soisno[j] + wliq_soisno[j]

    scvold = scv
    we = 0.
    if lb <= 0:
        we = np.sum(wice_soisno[lb:0] + wliq_soisno[lb:0])

    # Supercooling water
    if co_lm['DEF_USE_SUPERCOOL_WATER']:
        for j in range(nl_soil):
            supercool[j] = 0.0
            if t_soisno[j] < const_physical.tfrz and patchtype <= 2:
                smp = const_physical.hfus * (t_soisno[j] - const_physical.tfrz) / (const_physical.grav * t_soisno[j]) * 1000.
                if porsl[j] > 0.:
                    if co_lm['Campbell_SOIL_MODEL']:
                        supercool[j] = porsl[j] * (smp / psi0[j]) ** (-1.0 / bsw[j])
                    else:
                        supercool[j] = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(smp, porsl[j], theta_r[j], -10.0, 5, 
                                                          [alpha_vgm[j], n_vgm[j], L_vgm[j], sc_vgm[j], fc_vgm[j]])
                else:
                    supercool[j] = 0.0
                supercool[j] = supercool[j] * dz[j] * 1000.
    
    for j in range(lb, nl_soil):
        if wice_soisno[j] > 0. and t_soisno[j] > const_physical.tfrz:
            imelt[j] = 1
            t_soisno[j] = const_physical.tfrz
        
        if j <= 0:
            if wliq_soisno[j] > 0. and t_soisno[j] < const_physical.tfrz:
                imelt[j] = 2
                t_soisno[j] = const_physical.tfrz
        else:
            if co_lm['DEF_USE_SUPERCOOL_WATER']:
                if wliq_soisno[j] > supercool[j] and t_soisno[j] < const_physical.tfrz:
                    imelt[j] = 2
                    t_soisno[j] = const_physical.tfrz
            else:
                if wliq_soisno[j] > 0. and t_soisno[j] < const_physical.tfrz:
                    imelt[j] = 2
                    t_soisno[j] = const_physical.tfrz

    if lb == 1 and scv > 0.:
        if t_soisno[1] > const_physical.tfrz:
            imelt[1] = 1
            t_soisno[1] = const_physical.tfrz

    for j in range(lb, nl_soil):
        if imelt[j] > 0:
            tinc = t_soisno[j] - t_soisno_bef[j]
            if j > lb:
                if j == 1 and co_lm['DEF_SPLIT_SOILSNOW'] and patchtype < 3:
                    hm[j] = hs_soil + (1. - fsno) * dhsdT * tinc + brr[j] - tinc / fact[j]
                else:
                    if j < 1 or (j == 1 and patchtype == 3):
                        hm[j] = brr[j] - tinc / fact[j] + sabg_snow_lyr[j]
                    else:
                        hm[j] = brr[j] - tinc / fact[j]
            else:
                if j == 1 or (not co_lm['DEF_SPLIT_SOILSNOW']) or patchtype == 3:
                    hm[j] = hs + dhsdT * tinc + brr[j] - tinc / fact[j]
                else:
                    hm[j] = hs_snow + fsno * dhsdT * tinc + brr[j] - tinc / fact[j]

    for j in range(lb, nl_soil):
        if imelt[j] == 1 and hm[j] < 0.:
            hm[j] = 0.
            imelt[j] = 0
        if imelt[j] == 2 and hm[j] > 0.:
            hm[j] = 0.
            imelt[j] = 0

    for j in range(lb, nl_soil):
        if imelt[j] > 0 and abs(hm[j]) > 0.:
            xm[j] = hm[j] * deltim / const_physical.hfus
            if j == 1 and lb == 1 and scv > 0. and xm[j] > 0.:
                temp1 = scv
                scv = max(0., temp1 - xm[j])
                propor = scv / temp1
                snowdp = propor * snowdp
                heatr = hm[j] - const_physical.hfus * (temp1 - scv) / deltim
                if heatr > 0.:
                    xm[j] = heatr * deltim / const_physical.hfus
                    hm[j] = heatr
                else:
                    xm[j] = 0.
                    hm[j] = 0.
                sm = max(0., (temp1 - scv)) / deltim
                xmf = const_physical.hfus * sm

            heatr = 0.
            if xm[j] > 0.:
                wice_soisno[j] = max(0., wice0[j] - xm[j])
                heatr = hm[j] - const_physical.hfus * (wice0[j] - wice_soisno[j]) / deltim
            else:
                if j <= 0:
                    wice_soisno[j] = min(wmass0[j], wice0[j] - xm[j])
                else:
                    if co_lm['DEF_USE_SUPERCOOL_WATER']:
                        if wmass0[j] < supercool[j]:
                            wice_soisno[j] = 0.
                        else:
                            wice_soisno[j] = min(wmass0[j] - supercool[j], wice0[j] - xm[j])
                    else:
                        wice_soisno[j] = min(wmass0[j], wice0[j] - xm[j])
                heatr = hm[j] - const_physical.hfus * (wice0[j] - wice_soisno[j]) / deltim

            wliq_soisno[j] = max(0., wmass0[j] - wice_soisno[j])

            if abs(heatr) > 0.:
                if j > lb:
                    if j == 1 and co_lm['DEF_SPLIT_SOILSNOW'] and patchtype < 3:
                        t_soisno[j] = t_soisno[j] + fact[j] * heatr / (1. - fact[j] * (1. - fsno) * dhsdT)
                    else:
                        t_soisno[j] = t_soisno[j] + fact[j] * heatr
                else:
                    if j == 1 or (not co_lm['DEF_SPLIT_SOILSNOW']) or patchtype == 3:
                        t_soisno[j] = t_soisno[j] + fact[j] * heatr / (1. - fact[j] * dhsdT)
                    else:
                        t_soisno[j] = t_soisno[j] + fact[j] * heatr / (1. - fact[j] * fsno * dhsdT)

                if co_lm['DEF_USE_SUPERCOOL_WATER']:
                    if j <= 0 or patchtype == 3:
                        if wliq_soisno[j] * wice_soisno[j] > 0.:
                            t_soisno[j] = const_physical.tfrz
                else:
                    if wliq_soisno[j] * wice_soisno[j] > 0.:
                        t_soisno[j] = const_physical.tfrz

            xmf += const_physical.hfus * (wice0[j] - wice_soisno[j]) / deltim

            if imelt[j] == 1:
                sm += wice0[j] - wice_soisno[j]
    if lb <= 0:
        we = np.sum(wice_soisno[lb:0+1] + wliq_soisno[lb:0+1]) - we
        if abs(we) > 1.e-6:
            print('meltf err : ', we)
            # CoLM_stop()

    return t_soisno, wliq_soisno, wice_soisno, imelt, sm, xmf, snowdp, scv
