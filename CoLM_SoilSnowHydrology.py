import numpy as np
import math
import CoLM_Utils
from CoLM_Hydro_SoilWater import Hydro_SoilWater
import CoLM_Runoff
import CoLM_Hydro_VIC_Variables
import CoLM_Hydro_VIC

#=======================================================================
# snowwater function
# Original author : Yongjiu Dai, /09/1999; /04/2014
#
# Water flow within snow is computed by an explicit and non-physical based scheme,
# which permits a part of liquid water over the holding capacity (a tentative value
# is used, i.e., equal to 0.033*porosity) to percolate into the underlying layer,
# except the case of that the porosity of one of the two neighboring layers is
# less than 0.05, the zero flow is assumed. The water flow out of the bottom
# snow pack will participate as the input of the soil water and runoff.
#=======================================================================

def snowwater(lb, deltim, ssi, wimp, 
              pg_rain, qseva, qsdew, qsubl, qfros, 
              dz_soisno, wice_soisno, wliq_soisno):

    denice = 917.0  # ice density in kg/m^3
    denh2o = 1000.0  # water density in kg/m^3

    # Local variables
    qin = 0.0
    qout = 0.0
    zwice = 0.0
    wgdif = 0.0

    vol_liq = np.zeros(len(dz_soisno))
    vol_ice = np.zeros(len(dz_soisno))
    eff_porosity = np.zeros(len(dz_soisno))

    #=======================================================================
    # renew the mass of ice lens (wice_soisno) and liquid (wliq_soisno) in the surface snow layer,
    # resulted by sublimation (frost) / evaporation (condense)

    wgdif = wice_soisno[lb] + (qfros - qsubl) * deltim
    wice_soisno[lb] = wgdif
    if wgdif < 0.0:
        wice_soisno[lb] = 0.0
        wliq_soisno[lb] += wgdif
    wliq_soisno[lb] += (pg_rain + qsdew - qseva) * deltim
    wliq_soisno[lb] = max(0.0, wliq_soisno[lb])

    # Porosity and partial volume
    for j in range(lb, -1, -1):
        vol_ice[j] = min(1.0, wice_soisno[j] / (dz_soisno[j] * denice))
        eff_porosity[j] = max(0.01, 1.0 - vol_ice[j])
        vol_liq[j] = min(eff_porosity[j], wliq_soisno[j] / (dz_soisno[j] * denh2o))

    # Capillary force within snow could be two or more orders of magnitude
    # less than those of gravity, this term may be ignored.
    # Here we could keep the gravity term only. The general expression
    # for water flow is "K * ss**3", however, no effective parameterization
    # for "K". Thus, a very simple treatment (not physical based) is introduced:
    # when the liquid water of layer exceeds the layer's holding
    # capacity, the excess meltwater adds to the underlying neighbor layer.

    for j in range(lb, -1, -1):
        wliq_soisno[j] += qin

        if j <= -1:
            # no runoff over snow surface, just ponding on surface
            if eff_porosity[j] < wimp or eff_porosity[j + 1] < wimp:
                qout = 0.0
            else:
                qout = max(0.0, (vol_liq[j] - ssi * eff_porosity[j]) * dz_soisno[j])
                qout = min(qout, (1.0 - vol_ice[j + 1] - vol_liq[j + 1]) * dz_soisno[j + 1])
        else:
            qout = max(0.0, (vol_liq[j] - ssi * eff_porosity[j]) * dz_soisno[j])

        qout = qout * 1000.0
        wliq_soisno[j] -= qout
        qin = qout

    qout_snowb = qout / deltim

    return wice_soisno, wliq_soisno, qout_snowb

def surfacerunoff(nl_soil, wtfact, wimp, porsl, psi0, hksati, z_soisno, dz_soisno, zi_soisno, eff_porosity, icefrac, gwat, zwt):
    """
    Calculate surface runoff.

    Parameters:
    nl_soil (int): Number of soil layers
    wtfact (float): Fraction of model area with high water table
    wimp (float): Water impermeable if porosity less than wimp
    porsl (array): Saturated volumetric soil water content (porosity)
    psi0 (array): Saturated soil suction (mm) (NEGATIVE)
    hksati (array): Hydraulic conductivity at saturation (mm H2O/s)
    z_soisno (array): Layer depth (m)
    dz_soisno (array): Layer thickness (m)
    zi_soisno (array): Interface level below a "z" level (m)
    eff_porosity (array): Effective porosity = porosity - vol_ice
    icefrac (array): Ice fraction (-)
    gwat (float): Net water input from top
    zwt (float): The depth from ground (soil) surface to water table (m)

    Returns:
    float: Surface runoff (mm H2O/s)
    """
    
    # Local variables
    fff = 0.5  # Runoff decay factor (m-1)
    
    # Fraction of saturated area
    fsat = wtfact * min(1.0, np.exp(-0.5 * fff * zwt))
    
    # Maximum infiltration capacity
    qinmax = np.min(10.0 ** (-6.0 * icefrac[:min(3, nl_soil)]) * hksati[:min(3, nl_soil)])
    if eff_porosity[0] < wimp:
        qinmax = 0.0
    
    # Surface runoff
    rsur = fsat * max(0.0, gwat) + (1.0 - fsat) * max(0.0, gwat - qinmax)
    
    return rsur

import numpy as np

def soilwater(nl_colm, patchtype, nl_soil, deltim, wimp, smpmin, qinfl, etr,
              z_soisno, dz_soisno, zi_soisno, t_soisno, vol_liq, vol_ice,
              smp, hk, icefrac, eff_porosity, porsl, hksati, bsw, psi0,
              rootr, rootflux, zwt):
    
    # Constants
    grav = 9.80616
    tfrz = 273.15
    smpmin = -1e8
    e_ice = 6.0

    # Initialize arrays
    amx = np.zeros(nl_soil)
    bmx = np.zeros(nl_soil)
    cmx = np.zeros(nl_soil)
    rmx = np.zeros(nl_soil)
    zmm = np.zeros(nl_soil)
    dzmm = np.zeros(nl_soil)
    zimm = np.zeros(nl_soil + 1)
    den = np.zeros(nl_soil)
    alpha = np.zeros(nl_soil)
    qin = np.zeros(nl_soil)
    qout = np.zeros(nl_soil)
    dqidw0 = np.zeros(nl_soil)
    dqidw1 = np.zeros(nl_soil)
    dqodw1 = np.zeros(nl_soil)
    dqodw2 = np.zeros(nl_soil)
    dsmpdw = np.zeros(nl_soil)
    dhkdw1 = np.zeros(nl_soil)
    dhkdw2 = np.zeros(nl_soil)
    imped = np.zeros(nl_soil)
    dwat = np.zeros(nl_soil)

    # Compute jwt index
    jwt = nl_soil
    for j in range(nl_soil):
        if zwt <= zi_soisno[j]:
            jwt = j
            break

    # Convert depths to mm
    for j in range(nl_soil):
        zmm[j] = z_soisno[j] * 1000
        dzmm[j] = dz_soisno[j] * 1000
        zimm[j] = zi_soisno[j] * 1000
    zimm[0] = 0.0

    # Compute matric potential and derivative based on liquid water content only
    for j in range(nl_soil):
        if nl_colm['DEF_USE_PLANTHYDRAULICS'] and patchtype!=1 or not nl_colm['DEF_URBAN_RUN']:
            if t_soisno[j] >= tfrz:
                if porsl[j] < 1e-6:
                    s_node = 0.001
                    smp[j] = psi0[j]
                    dsmpdw[j] = 0.0
                else:
                    s_node = max(vol_liq[j] / porsl[j], 0.01)
                    s_node = min(1.0, s_node)
                    smp[j] = psi0[j] * s_node**(-bsw[j])
                    smp[j] = max(smpmin, smp[j])
                    dsmpdw[j] = -bsw[j] * smp[j] / (s_node * porsl[j])
            else:
                smp[j] = 1e3 * 0.3336e6 / 9.80616 * (t_soisno[j] - tfrz) / t_soisno[j]
                smp[j] = max(smpmin, smp[j])
                dsmpdw[j] = 0.0
        else:
        # Compute hydraulic conductivity and soil matric potential and their derivatives
            if t_soisno[j] > tfrz:
                if porsl[j] < 1e-6:  # bed rock
                    s_node = 0.001
                    smp[j] = psi0[j]
                    dsmpdw[j] = 0.0
                else:
                    s_node = max(vol_liq[j] / porsl[j], 0.01)
                    s_node = min(1.0, s_node)
                    smp[j] = psi0[j] * s_node**(-bsw[j])
                    smp[j] = max(smpmin, smp[j])
                    dsmpdw[j] = -bsw[j] * smp[j] / (s_node * porsl[j])
            else:
                # when ice is present, the matric potential is only related to temperature
                # by (Fuchs et al., 1978: Soil Sci. Soc. Amer. J. 42(3):379-385)
                # Unit 1 Joule = 1 (kg m^2/s^2), J/kg /(m/s^2) ==> m ==> 1e3 mm
                smp[j] = 1e3 * 0.3336e6 / 9.80616 * (t_soisno[j] - tfrz) / t_soisno[j]
                smp[j] = max(smpmin, smp[j])  # Limit soil suction
                dsmpdw[j] = 0.0
    
    # Hydraulic conductivity and soil matric potential and their derivatives
    for j in range(nl_soil):
        if j < nl_soil:
            den[j] = zmm[j + 1] - zmm[j]
            alpha[j] = (smp[j + 1] - smp[j]) / den[j] - 1
        else:
            den[j] = 0.0  # not used
            alpha[j] = 0.0  # not used

        if (eff_porosity[j] < wimp or eff_porosity[min(nl_soil, j + 1)] < wimp or vol_liq[j] <= 1e-3):
            imped[j] = 0.0
            hk[j] = 0.0
            dhkdw1[j] = 0.0
            dhkdw2[j] = 0.0
        else:
            # The average conductivity between two heterogeneous medium layers (j and j + 1)
            if j < nl_soil:
                # Method I: UPSTREAM MEAN
                if alpha[j] <= 0:
                    hk[j] = hksati[j] * (vol_liq[j] / porsl[j]) ** (2 * bsw[j] + 3)
                    dhkdw1[j] = hksati[j] * (2 * bsw[j] + 3) * (vol_liq[j] / porsl[j]) ** (2 * bsw[j] + 2) / porsl[j]
                    dhkdw2[j] = 0.0
                else:
                    hk[j] = hksati[j + 1] * (vol_liq[j + 1] / porsl[j + 1]) ** (2 * bsw[j + 1] + 3)
                    dhkdw1[j] = 0.0
                    dhkdw2[j] = hksati[j + 1] * (2 * bsw[j + 1] + 3) * (vol_liq[j + 1] / porsl[j + 1]) ** (2 * bsw[j + 1] + 2) / porsl[j + 1]
            else:
                hk[j] = hksati[j] * (vol_liq[j] / porsl[j]) ** (2 * bsw[j] + 3)
                dhkdw1[j] = hksati[j] * (2 * bsw[j] + 3) * (vol_liq[j] / porsl[j]) ** (2 * bsw[j] + 2) / porsl[j]
                dhkdw2[j] = 0.0

            # replace fracice with impedance factor
            imped[j] = 10 ** (-e_ice * (0.5 * (icefrac[j] + icefrac[min(nl_soil - 1, j + 1)])))
            hk[j] = imped[j] * hk[j]
            dhkdw1[j] = imped[j] * dhkdw1[j]
            dhkdw2[j] = imped[j] * dhkdw2[j]

    # Set up r, a, b, and c vectors for tridiagonal solution

    # Node j=1 (top)
    j = 0
    qin[j] = qinfl
    qout[j] = -hk[j] * alpha[j]
    dqodw1[j] = -(alpha[j] * dhkdw1[j] - hk[j] * dsmpdw[j] / den[j])
    dqodw2[j] = -(alpha[j] * dhkdw2[j] + hk[j] * dsmpdw[j + 1] / den[j])

    amx[j] = 0.0
    bmx[j] = dzmm[j] / deltim + dqodw1[j]
    cmx[j] = dqodw2[j]

    if nl_colm['DEF_USE_PLANTHYDRAULICS'] and (patchtype!=1 or not nl_colm['DEF_URBAN_RUN']):
         rmx[j] =  qin[j]- qout[j] - rootflux[j]
    else:
         rmx[j] =  qin[j] - qout[j] - etr*rootr[j]

    # Nodes j=2 to j=nl_soil-1
    for j in range(1, nl_soil - 1):
        qin[j] = -hk[j - 1] * alpha[j - 1]
        dqidw0[j] = -(alpha[j - 1] * dhkdw1[j - 1] - hk[j - 1] * dsmpdw[j - 1] / den[j - 1])
        dqidw1[j] = -(alpha[j - 1] * dhkdw2[j - 1] + hk[j - 1] * dsmpdw[j] / den[j - 1])

        qout[j] = -hk[j] * alpha[j]
        dqodw1[j] = -(alpha[j] * dhkdw1[j] - hk[j] * dsmpdw[j] / den[j])
        dqodw2[j] = -(alpha[j] * dhkdw2[j] + hk[j] * dsmpdw[j + 1] / den[j])

        amx[j] = -dqidw0[j]
        bmx[j] = dzmm[j] / deltim - dqidw1[j] + dqodw1[j]
        cmx[j] = dqodw2[j]
        rmx[j] = qin[j] - qout[j] - rootflux[j]

    # Node j=nl_soil (bottom)
    j = nl_soil - 1
    qin[j] = -hk[j - 1] * alpha[j - 1]
    dqidw0[j] = -(alpha[j - 1] * dhkdw1[j - 1] - hk[j - 1] * dsmpdw[j - 1] / den[j - 1])
    dqidw1[j] = -(alpha[j - 1] * dhkdw2[j - 1] + hk[j - 1] * dsmpdw[j] / den[j - 1])

    qout[j] = hk[j]
    dqodw1[j] = dhkdw1[j]
    dqodw2[j] = 0.0

    amx[j] = -dqidw0[j]
    bmx[j] = dzmm[j] / deltim - dqidw1[j]+ dqodw1[j]
    cmx[j] = dqodw2[j]

    if nl_colm['DEF_USE_PLANTHYDRAULICS'] and (patchtype != 1 or not nl_colm['DEF_URBAN_RUN']):
        rmx[j] = qin[j] - qout[j] - rootflux[j]
    else:
        rmx[j] = qin[j] - qout[j] - etr * rootr[j]

    # Solve tridiagonal system for change in soil water
    dwat = CoLM_Utils.tridia(nl_soil, amx, bmx, cmx, rmx,dwat)


    if nl_colm['CoLMDEBUG']:
        errorw = -deltim * (qin[0] - qout[nl_soil - 1] - dqodw1[nl_soil - 1] * dwat[nl_soil - 1])

        for j in range(nl_soil):
            if nl_colm['DEF_USE_PLANTHYDRAULICS'] and (patchtype != 1 or not nl_colm['DEF_URBAN_RUN']):
                errorw += dwat[j] * dzmm[j] + rootflux[j] * deltim
            else:
                errorw += dwat[j] * dzmm[j] + etr * rootr[j] * deltim

        if abs(errorw) > 1.e-3:
            print('mass balance error in time step =', errorw)
    qcharge = qout[nl_soil] + dqodw1[nl_soil]*dwat[nl_soil]

    return dwat


def subsurfacerunoff(nl_soil, deltim, pondmx, eff_porosity, icefrac, dz_soisno, zi_soisno, wice_soisno, wliq_soisno,
                    porsl, psi0, bsw, zwt, wa, qcharge, rsubst, errw_rsub):
    watmin = 0.01  # Limit irreducible wrapping liquid water
    rsbmx = 5.0    # baseflow coefficient [mm/s]
    timean = 10.5  # global mean topographic index

    dzmm = np.zeros(nl_soil)
    for j in range(nl_soil):
        dzmm[j] = dz_soisno[j] * 1000.

    jwt = nl_soil
    for j in range(nl_soil):
        if zwt <= zi_soisno[j]:
            jwt = j - 1
            break

    #!============================== QCHARGE =========================================
    rous = porsl[nl_soil - 1] * (1. - (1. - 1.e3 * zwt / psi0[nl_soil - 1]) ** (-1. / bsw[nl_soil - 1]))
    rous = max(rous, 0.02)
    wa = wa + qcharge * deltim

    if jwt == nl_soil:
        zwt = max(0., zwt - (qcharge * deltim) / 1000. / rous)
    else:
        qcharge_tot = qcharge * deltim
        if qcharge_tot > 0.:
            for j in range(jwt, 0, -1):
                s_y = porsl[j - 1] * (1. - (1. - 1.e3 * zwt / psi0[j - 1]) ** (-1. / bsw[j - 1]))
                s_y = max(s_y, 0.02)
                qcharge_layer = min(qcharge_tot, (s_y * (zwt - zi_soisno[j - 1]) * 1.e3))
                qcharge_layer = max(qcharge_layer, 0.)
                zwt = max(0., zwt - qcharge_layer / s_y / 1000.)
                qcharge_tot = qcharge_tot - qcharge_layer
                if qcharge_tot <= 0.:
                    break
        else:
            for j in range(jwt + 1, nl_soil + 1):
                s_y = porsl[j - 1] * (1. - (1. - 1.e3 * zwt / psi0[j - 1]) ** (-1. / bsw[j - 1]))
                s_y = max(s_y, 0.02)
                qcharge_layer = max(qcharge_tot, -(s_y * (zi_soisno[j] - zwt) * 1.e3))
                qcharge_layer = min(qcharge_layer, 0.)
                qcharge_tot = qcharge_tot - qcharge_layer
                if qcharge_tot >= 0.:
                    zwt = max(0., zwt - qcharge_layer / s_y / 1000.)
                    break
                else:
                    zwt = zi_soisno[j - 1]
            if qcharge_tot > 0.:
                zwt = max(0., zwt - qcharge_tot / 1000. / rous)

#!-- Topographic runoff  ----------------------------------------------------------
    dzsum = 0
    icefracsum = 0.
    for j in range( max(jwt,1), nl_soil):
         dzsum = dzsum + dzmm[j]
         icefracsum = icefracsum + icefrac[j] * dzmm[j]

    fracice_rsub = max(0., np.exp(-3. * (1. - (icefracsum / dzsum))) - np.exp(-3.)) / (1.0 - np.exp(-3.))
    imped = max(0., 1. - fracice_rsub)
    drainage = imped * 5.5e-3 * np.exp(-2.5 * zwt)

    if jwt == nl_soil:
        wa = wa - drainage * deltim
        zwt = max(0., zwt + (drainage * deltim) / 1000. / rous)
        wliq_soisno[nl_soil - 1] = wliq_soisno[nl_soil - 1] + max(0., (wa - 5000.))
        wa = min(wa, 5000.)
    else:
        drainage_tot = -drainage * deltim
        for j in range(jwt + 1, nl_soil + 1):
            s_y = porsl[j - 1] * (1. - (1. - 1.e3 * zwt / psi0[j - 1]) ** (-1. / bsw[j - 1]))
            s_y = max(s_y, 0.02)
            drainage_layer = max(drainage_tot, -(s_y * (zi_soisno[j] - zwt) * 1.e3))
            drainage_layer = min(drainage_layer, 0.)
            wliq_soisno[j - 1] = wliq_soisno[j - 1] + drainage_layer
            drainage_tot = drainage_tot - drainage_layer
            if drainage_tot >= 0.:
                zwt = max(0., zwt - drainage_layer / s_y / 1000.)
                break
            else:
                zwt = zi_soisno[j - 1]

        zwt = max(0., zwt - drainage_tot / 1000. / rous)
        wa = wa + drainage_tot
        jwt = nl_soil
        for j in range(nl_soil):
            if zwt <= zi_soisno[j]:
                jwt = j - 1
                break

    zwt = max(0.0, zwt)
    zwt = min(80., zwt)
    rsubst = drainage

    for j in range(nl_soil - 1, 0, -1):
        xsi = max(wliq_soisno[j] - eff_porosity[j] * dzmm[j], 0.)
        wliq_soisno[j] = min(eff_porosity[j] * dzmm[j], wliq_soisno[j])
        wliq_soisno[j - 1] = wliq_soisno[j - 1] + xsi

    xs1 = wliq_soisno[0] - (pondmx + porsl[0] * dzmm[0] - wice_soisno[0])
    if xs1 > 0.:
        wliq_soisno[0] = pondmx + porsl[0] * dzmm[0] - wice_soisno[0]
    else:
        xs1 = 0.

    rsubst = rsubst + xs1 / deltim

    xs = 0.
    for j in range(nl_soil):
        if wliq_soisno[j] < 0.:
            xs = xs + wliq_soisno[j]
            wliq_soisno[j] = 0.

    errw_rsub = min(0., rsubst + xs / deltim)
    rsubst = max(0., rsubst + xs / deltim)
    return wice_soisno, wliq_soisno, zwt, wa, rsubst, errw_rsub


def WATER_2014 (nl_colm, const_physical,
                ipatch, patchtype, lb, nl_soil, deltim,
                z_soisno, dz_soisno, zi_soisno, bsw, porsl,
                psi0, hksati, theta_r, topostd,
                BVIC,
                rootr, rootflux, t_soisno,
                wliq_soisno, wice_soisno, smp, hk, pg_rain,
                sm, etr, qseva, qsdew, qsubl, qfros,
                qseva_soil, qsdew_soil, qsubl_soil, qfros_soil,
                qseva_snow, qsdew_snow, qsubl_snow, qfros_snow,
                fsno,
                rsur, rnof, qinfl, wtfact, pondmx,
                ssi, wimp, smpmin, zwt, wa,
                qcharge, errw_rsub,
                forc_aer                 ,
              mss_bcpho   ,mss_bcphi   ,mss_ocpho   ,mss_ocphi   ,
              mss_dst1    ,mss_dst2    ,mss_dst3    ,mss_dst4,
               flddepth=None, fldfrc=None, qinfl_fld=None):
    """
    determine net radiation.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         : this is the main SUBROUTINE to execute the calculation of hydrological processes   
    FLOW DIAGRAM FOR WATER_2014.F90

                    WATER_2014 ===> snowwater
                                    SurfaceRunoff_SIMTOP
                                    const_physical.tfrz
                                    SubsurfaceRunoff_SIMTOP
                      
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
    
    eff_porosity= np.zeros (nl_soil) # effective porosity = porosity - vol_ice
    dwat= np.zeros (nl_soil)    # change in soil water
    gwat = 0.0              # net water input from top (mm/s)
    rsubst = 0.0            # subsurface runoff (mm h2o/s)
    vol_liq= np.zeros (nl_soil) # partitial volume of liquid water in layer
    vol_ice= np.zeros (nl_soil) # partitial volume of ice lens in layer
    icefrac= np.zeros (nl_soil) # ice fraction (-)
    zmm = np.zeros (nl_soil)   # layer depth (mm)
    dzmm= np.zeros (nl_soil)   # layer thickness (mm)
    zimm= np.zeros (nl_soil)      # interface level below a "z" level (mm)

    err_solver = 0.0
    w_sum = 0.0
    gfld = 0.0
    rsur_fld = 0.0
    qinfl_fld_subgrid = 0.0 # inundation water input from top (mm/s)
    ps = 0
    pe = 0
    irrig_flag = 0  # 1 IF sprinker, 2 IF others
    qflx_irrig_drip = 0.0
    qflx_irrig_sprinkler = 0.0
    qflx_irrig_flood = 0.0
    qflx_irrig_paddy = 0.0
        
    #=======================================================================
    # [1] update the liquid water within snow layer and the water onto soil
    #=======================================================================

    if (not nl_colm['DEF_SPLIT_SOILSNOW']) or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):

        if lb >= 1:
            gwat = pg_rain + sm - qseva
        else:
            if (not nl_colm['DEF_USE_SNICAR']) or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):
                wice_soisno, wliq_soisno, gwat = snowwater(lb, deltim, ssi, wimp,
                                 pg_rain, qseva, qsdew, qsubl, qfros,
                                 dz_soisno[lb:0], wice_soisno[lb:0], wliq_soisno[lb:0])
            else:
                gwat = snow_water_snicar(lb, deltim, ssi, wimp,
                                        pg_rain, qseva, qsdew, qsubl, qfros,
                                        dz_soisno[lb:0], wice_soisno[lb:0], wliq_soisno[lb:0],
                                        forc_aer,
                                        mss_bcpho[lb:0], mss_bcphi[lb:0], mss_ocpho[lb:0], mss_ocphi[lb:0],
                                        mss_dst1[lb:0], mss_dst2[lb:0], mss_dst3[lb:0], mss_dst4[lb:0])

    else:

        if lb >= 1:
            gwat = pg_rain + sm - qseva_soil
        else:
            if not nl_colm['DEF_USE_SNICAR']:
                wice_soisno, wliq_soisno, gwat = snowwater(lb, deltim, ssi, wimp,
                                 pg_rain * fsno, qseva_snow, qsdew_snow, qsubl_snow, qfros_snow,
                                 dz_soisno[lb:0], wice_soisno[lb:0], wliq_soisno[lb:0])
            else:
                gwat = snow_water_snicar(lb, deltim, ssi, wimp,
                                        pg_rain * fsno, qseva_snow, qsdew_snow, qsubl_snow, qfros_snow,
                                        dz_soisno[lb:0], wice_soisno[lb:0], wliq_soisno[lb:0],
                                        forc_aer,
                                        mss_bcpho[lb:0], mss_bcphi[lb:0], mss_ocpho[lb:0], mss_ocphi[lb:0],
                                        mss_dst1[lb:0], mss_dst2[lb:0], mss_dst3[lb:0], mss_dst4[lb:0])
            gwat = gwat + pg_rain * (1 - fsno) - qseva_soil

    # if nl_colm['CROP']:
    #     if nl_colm['DEF_USE_IRRIGATION']:
    #         if patchtype == 0:
    #             ps = patch_pft_s[ipatch]
    #             pe = patch_pft_e[ipatch]
    #             CalIrrigationApplicationFluxes(ipatch, ps, pe, deltim, qflx_irrig_drip, qflx_irrig_sprinkler, qflx_irrig_flood, qflx_irrig_paddy, irrig_flag=2)
    #             gwat += qflx_irrig_drip + qflx_irrig_flood + qflx_irrig_paddy
    
    #=======================================================================
    # [2] surface runoff and infiltration
    #=======================================================================

    if patchtype <= 1:  # soil ground only

        # For water balance check, the sum of water in soil column before the calculation
        w_sum = sum(wliq_soisno[1:]) + sum(wice_soisno[1:]) + wa
        # porosity of soil, partial volume of ice and liquid
        for j in range(nl_soil):
            vol_ice[j] = min(porsl[j], wice_soisno[j] / (dz_soisno[j] * const_physical.denice))
            eff_porosity[j] = max(0.01, porsl[j] - vol_ice[j])
            vol_liq[j] = min(eff_porosity[j], wliq_soisno[j] / (dz_soisno[j] * const_physical.denh2o))
            
            if porsl[j] < 1.e-6:
                icefrac[j] = 0.0
            else:
                icefrac[j] = min(1.0, vol_ice[j] / porsl[j])
        
        # Surface runoff including water table and surface saturated area
        rsur = 0.0
        
        if gwat > 0.0:
            rsur = surfacerunoff(nl_soil, wtfact, wimp, porsl, psi0, hksati,
                                z_soisno, dz_soisno, zi_soisno, eff_porosity, icefrac, zwt, gwat)
        else:
            rsur = 0.0
        
        # Infiltration into surface soil layer
        qinfl = gwat - rsur

        # if nl_colm['CaMa_Flood']:
        #     if nl_colm['LWINFILT']:
        #         # re-infiltration [mm/s] calculation.
        #         # IF surface runoff is occurred (rsur != 0.), flood depth <1.e-6 and flood fraction <0.05,
        #         # the re-infiltration will not be calculated.
        #         if (flddepth > 1.e-6) and (fldfrc > 0.05) and (patchtype == 0):
        #             gfld = flddepth / deltim  # [mm/s]
        #             # surface runoff from inundation, this should not be added to the surface runoff from soil
        #             # otherwise, the surface runoff will be double counted.
        #             # only the re-infiltration is added to water balance calculation.
        #             SurfaceRunoff_SIMTOP(nl_soil, 1.0, wimp, porsl, psi0, hksati,
        #                                  z_soisno[1:], dz_soisno[1:], zi_soisno[0:],
        #                                  eff_porosity, icefrac, zwt, gfld, rsur_fld)
        #             # infiltration into surface soil layer
        #             qinfl_fld_subgrid = gfld - rsur_fld  # assume the re-infiltration occurs in whole patch area.
        #         else:
        #             qinfl_fld_subgrid = 0.0
        #             gfld = 0.0
        #             rsur_fld = 0.0

        #         qinfl_fld = qinfl_fld_subgrid * fldfrc  # [mm/s] re-infiltration in grid.
        #         qinfl = qinfl_fld + qinfl  # [mm/s] total infiltration in grid.
        #         flddepth = flddepth - deltim * qinfl_fld_subgrid  # renew flood depth [mm], the flood depth is reduced by re-infiltration but only in inundation area.
                
    #=======================================================================
    # [3] determine the change of soil water
    #=======================================================================

        # convert length units from m to mm
        zmm[:] = z_soisno[:] * 1000.0
        dzmm[:] = dz_soisno[:] * 1000.0
        zimm[:] = zi_soisno[:] * 1000.0
        
        soilwater(patchtype, nl_soil, deltim, wimp, smpmin, qinfl, etr, z_soisno, dz_soisno, zi_soisno, 
                t_soisno, vol_liq, vol_ice, smp, hk, icefrac, eff_porosity, porsl, hksati, bsw, 
                psi0, rootr, rootflux, zwt, dwat, qcharge)

        # Update the mass of liquid water
        for j in range(nl_soil):
            wliq_soisno[j] += dwat[j] * dzmm[j]

    #!=======================================================================
    #! [4] subsurface runoff and the corrections
    #!=======================================================================
        wice_soisno, wliq_soisno, zwt, wa, rsubst, errw_rsub = subsurfacerunoff(nl_soil, deltim, pondmx, eff_porosity, icefrac, dz_soisno, zi_soisno, wice_soisno, 
                        wliq_soisno, porsl, psi0, bsw, zwt, wa, qcharge, rsubst, errw_rsub)
        
        # Total runoff (mm/s)
        rnof = rsubst + rsur
        
        # Renew the ice and liquid mass due to condensation
        if not nl_colm['DEF_SPLIT_SOILSNOW'] or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):
            if lb >= 1:
                wliq_soisno[0] = max(0.0, wliq_soisno[0] + qsdew * deltim)
                wice_soisno[0] = max(0.0, wice_soisno[0] + (qfros - qsubl) * deltim)
            
            err_solver = (np.sum(wliq_soisno) + np.sum(wice_soisno) + wa) - w_sum - (gwat - etr - rnof - errw_rsub) * deltim
            
            if lb >= 1:
                err_solver -= (qsdew + qfros - qsubl) * deltim
        
        else:
            wliq_soisno[0] = max(0.0, wliq_soisno[0] + qsdew_soil * deltim)
            wice_soisno[0] = max(0.0, wice_soisno[0] + (qfros_soil - qsubl_soil) * deltim)
            
            err_solver = (np.sum(wliq_soisno) + np.sum(wice_soisno) + wa) - w_sum - (gwat - etr - rnof - errw_rsub) * deltim
            err_solver -= (qsdew_soil + qfros_soil - qsubl_soil) * deltim

        # Additional conditions based on CaMa_Flood and CoLMDEBUG definitions
        if nl_colm['CaMa_Flood']:
            if nl_colm['LWINFILT']:
                err_solver -= (gfld - rsur_fld) * fldfrc * deltim
        
        if nl_colm['CoLMDEBUG']:
            if abs(err_solver) > 1.e-3:
                print('Warning: water balance violation after all soilwater calculation', err_solver)
    else:
    
    #=======================================================================
    # [6] assumed hydrological scheme for the wetland and glacier
    #=======================================================================
        if patchtype == 2:  # WETLAND
            # 09/20/2019, by Chaoqun Li: a potential bug below
            # surface runoff could > total runoff
            # original CoLM: rsur=0., qinfl=gwat, rsubst=0., rnof=0.
            # i.e., all water to be infiltration
            qinfl = 0.
            rsur = max(0., gwat)
            rsubst = 0.
            rnof = 0.
            for j in range(nl_soil):
                if t_soisno[j] > const_physical.tfrz:
                    wice_soisno[j] = 0.0
                    wliq_soisno[j] = porsl[j] * dz_soisno[j] * 1000.

        elif patchtype == 3:  # LAND ICE
            rsur = max(0.0, gwat)
            qinfl = 0.
            rsubst = 0.
            rnof = rsur
            wice_soisno[:nl_soil] = [dz * 1000. for dz in dz_soisno[:nl_soil]]
            wliq_soisno[:nl_soil] = np.zeros(nl_soil)

        wa = 4800.
        zwt = 0.
        qcharge = 0.
        errw_rsub = 0.
    return flddepth, qinfl_fld, wice_soisno, wliq_soisno, smp, hk, zwt, wa, rsur, rnof, qinfl, qcharge, errw_rsub, 

def snow_water_snicar(nl_colm, lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros, dz_soisno, wice_soisno, wliq_soisno, qout_snowb, forc_aer, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4):
    denice = 917.0
    denh2o = 1000.0
    lb = lb-1

    scvng_fct_mlt_bcphi = 0.20
    scvng_fct_mlt_bcpho = 0.03
    scvng_fct_mlt_ocphi = 0.20
    scvng_fct_mlt_ocpho = 0.03
    scvng_fct_mlt_dst1 = 0.02
    scvng_fct_mlt_dst2 = 0.02
    scvng_fct_mlt_dst3 = 0.01
    scvng_fct_mlt_dst4 = 0.01

    wgdif = wice_soisno[lb] + (qfros - qsubl) * deltim
    wice_soisno[lb] = wgdif
    if wgdif < 0.0:
        wice_soisno[lb] = 0.0
        wliq_soisno[lb] += wgdif

    wliq_soisno[lb] += (pg_rain + qsdew - qseva) * deltim
    wliq_soisno[lb] = max(0.0, wliq_soisno[lb])
    vol_liq = np.zeros(lb)
    vol_ice = np.zeros(lb)
    eff_porosity = np.zeros(lb)

    for j in (lb, 1):
        vol_ice[j] = np.minimum(1.0, wice_soisno[j] / (dz_soisno[j] * denice))
        eff_porosity[j] = np.maximum(0.01, 1.0 - vol_ice[j])
        vol_liq[j] = np.minimum(eff_porosity[j], wliq_soisno[j] / (dz_soisno[j] * denh2o))

    qin = 0.0

    qin_bc_phi = 0.0
    qin_bc_pho = 0.0
    qin_oc_phi = 0.0
    qin_oc_pho = 0.0
    qin_dst1 = 0.0
    qin_dst2 = 0.0
    qin_dst3 = 0.0
    qin_dst4 = 0.0

    for j in range(lb, 1):
        wliq_soisno[j] += qin

        mss_bcphi[j] += qin_bc_phi
        mss_bcpho[j] += qin_bc_pho
        mss_ocphi[j] += qin_oc_phi
        mss_ocpho[j] += qin_oc_pho
        mss_dst1[j] += qin_dst1
        mss_dst2[j] += qin_dst2
        mss_dst3[j] += qin_dst3
        mss_dst4[j] += qin_dst4

        if j <= -1:
            if eff_porosity[j] < wimp or eff_porosity[j + 1] < wimp:
                qout = 0.0
            else:
                qout = max(0.0, (vol_liq[j] - ssi * eff_porosity[j]) * dz_soisno[j])
                qout = min(qout, (1.0 - vol_ice[j + 1] - vol_liq[j + 1]) * dz_soisno[j + 1])
        else:
            qout = max(0.0, (vol_liq[j] - ssi * eff_porosity[j]) * dz_soisno[j])

        qout *= 1000.0
        wliq_soisno[j] -= qout
        qin = qout

        mss_liqice = wliq_soisno[j] + wice_soisno[j]
        if mss_liqice < 1e-30:
            mss_liqice = 1e-30

        qout_bc_phi = qout * scvng_fct_mlt_bcphi * (mss_bcphi[j] / mss_liqice)
        if qout_bc_phi > mss_bcphi[j]:
            qout_bc_phi = mss_bcphi[j]
        mss_bcphi[j] -= qout_bc_phi
        qin_bc_phi = qout_bc_phi

        qout_bc_pho = qout * scvng_fct_mlt_bcpho * (mss_bcpho[j] / mss_liqice)
        if qout_bc_pho > mss_bcpho[j]:
            qout_bc_pho = mss_bcpho[j]
        mss_bcpho[j] -= qout_bc_pho
        qin_bc_pho = qout_bc_pho

        qout_oc_phi = qout * scvng_fct_mlt_ocphi * (mss_ocphi[j] / mss_liqice)
        if qout_oc_phi > mss_ocphi[j]:
            qout_oc_phi = mss_ocphi[j]
        mss_ocphi[j] -= qout_oc_phi
        qin_oc_phi = qout_oc_phi

        qout_oc_pho = qout * scvng_fct_mlt_ocpho * (mss_ocpho[j] / mss_liqice)
        if qout_oc_pho > mss_ocpho[j]:
            qout_oc_pho = mss_ocpho[j]
        mss_ocpho[j] -= qout_oc_pho
        qin_oc_pho = qout_oc_pho

        qout_dst1 = qout * scvng_fct_mlt_dst1 * (mss_dst1[j] / mss_liqice)
        if qout_dst1 > mss_dst1[j]:
            qout_dst1 = mss_dst1[j]
        mss_dst1[j] -= qout_dst1
        qin_dst1 = qout_dst1

        qout_dst2 = qout * scvng_fct_mlt_dst2 * (mss_dst2[j] / mss_liqice)
        if qout_dst2 > mss_dst2[j]:
            qout_dst2 = mss_dst2[j]
        mss_dst2[j] -= qout_dst2
        qin_dst2 = qout_dst2

        qout_dst3 = qout * scvng_fct_mlt_dst3 * (mss_dst3[j] / mss_liqice)
        if qout_dst3 > mss_dst3[j]:
            qout_dst3 = mss_dst3[j]
        mss_dst3[j] -= qout_dst3
        qin_dst3 = qout_dst3

        qout_dst4 = qout * scvng_fct_mlt_dst4 * (mss_dst4[j] / mss_liqice)
        if qout_dst4 > mss_dst4[j]:
            qout_dst4 = mss_dst4[j]
        mss_dst4[j] -= qout_dst4
        qin_dst4 = qout_dst4

    qout_snowb = qout / deltim
    if nl_colm['MODAL_AER']:
        flx_bc_dep_phi = forc_aer[2]
        flx_bc_dep_pho = forc_aer[0] + forc_aer[1]
        flx_bc_dep = forc_aer[0] + forc_aer[1] + forc_aer[2]

        flx_oc_dep_phi = forc_aer[5]
        flx_oc_dep_pho = forc_aer[3] + forc_aer[4]
        flx_oc_dep = forc_aer[3] + forc_aer[4] + forc_aer[5]

        flx_dst_dep_wet1 = forc_aer[6]
        flx_dst_dep_dry1 = forc_aer[7]
        flx_dst_dep_wet2 = forc_aer[8]
        flx_dst_dep_dry2 = forc_aer[9]
        flx_dst_dep_wet3 = forc_aer[10]
        flx_dst_dep_dry3 = forc_aer[11]
        flx_dst_dep_wet4 = forc_aer[12]
        flx_dst_dep_dry4 = forc_aer[13]
        flx_dst_dep = sum(forc_aer[6:14])
    else:
        # Assuming forc_aer is a numpy array or a list
        flx_bc_dep_phi = forc_aer[0] + forc_aer[2]
        flx_bc_dep_pho = forc_aer[1]
        flx_bc_dep = forc_aer[0] + forc_aer[1] + forc_aer[2]

        flx_oc_dep_phi = forc_aer[3] + forc_aer[5]
        flx_oc_dep_pho = forc_aer[4]
        flx_oc_dep = forc_aer[3] + forc_aer[4] + forc_aer[5]

        flx_dst_dep_wet1 = forc_aer[6]
        flx_dst_dep_dry1 = forc_aer[7]
        flx_dst_dep_wet2 = forc_aer[8]
        flx_dst_dep_dry2 = forc_aer[9]
        flx_dst_dep_wet3 = forc_aer[10]
        flx_dst_dep_dry3 = forc_aer[11]
        flx_dst_dep_wet4 = forc_aer[12]
        flx_dst_dep_dry4 = forc_aer[13]
        flx_dst_dep = (forc_aer[6] + forc_aer[7] + forc_aer[8] + forc_aer[9] +
                    forc_aer[10] + forc_aer[11] + forc_aer[12] + forc_aer[13])

    mss_bcphi[lb] += (flx_bc_dep_phi * deltim)
    mss_bcpho[lb] += (flx_bc_dep_pho * deltim)
    mss_ocphi[lb] += (flx_oc_dep_phi * deltim)
    mss_ocpho[lb] += (flx_oc_dep_pho * deltim)

    mss_dst1[lb] += (flx_dst_dep_dry1 + flx_dst_dep_wet1) * deltim
    mss_dst2[lb] += (flx_dst_dep_dry2 + flx_dst_dep_wet2) * deltim
    mss_dst3[lb] += (flx_dst_dep_dry3 + flx_dst_dep_wet3) * deltim
    mss_dst4[lb] += (flx_dst_dep_dry4 + flx_dst_dep_wet4) * deltim
    
    if nl_colm['MODAL_AER']:
        for j in range(lb, 1):
            if j >= lb:
                if j == lb:
                    # snow that has sublimated [kg/m2] (top layer only)
                    subsnow = max(0.0, (qsubl * deltim))

                    # fraction of layer mass that has sublimated:
                    if (wliq_soisno[j] + wice_soisno[j]) > 0.0:
                        frc_sub = subsnow / (wliq_soisno[j] + wice_soisno[j])
                    else:
                        frc_sub = 0.0
                else:
                    # prohibit sublimation effect to operate on sub-surface layers:
                    frc_sub = 0.0

                # fraction of layer mass transformed (sublimation only)
                frc_transfer = frc_sub

                # cap the fraction at 1
                if frc_transfer > 1.0:
                    frc_transfer = 1.0

                # transfer proportionate mass of BC and OC:
                dm_int = mss_bcphi[j] * frc_transfer
                mss_bcphi[j] -= dm_int
                mss_bcpho[j] += dm_int

                dm_int = mss_ocphi[j] * frc_transfer
                mss_ocphi[j] -= dm_int
                mss_ocpho[j] += dm_int


    return qout_snowb, mss_bcphi, mss_bcpho, mss_ocphi, mss_ocpho, mss_dst1, mss_dst2, mss_dst3, mss_dst4, wice_soisno, wliq_soisno

def water_vsf(nl_colm, const_physical, landpft, wetwatmax,
            ipatch,  patchtype,lb      ,nl_soil ,deltim ,
              z_soisno    ,dz_soisno   ,zi_soisno                    ,
              bsw         ,theta_r     ,topostd                      ,
              BVIC,                                                  
#ifdef vanGenuchten_Mualem_SOIL_MODEL
              alpha_vgm   ,n_vgm       ,L_vgm       ,sc_vgm  ,fc_vgm ,
#endif
              porsl       ,psi0        ,hksati      ,rootr   ,rootflux,
              t_soisno    ,wliq_soisno ,wice_soisno ,smp     ,hk     ,
              pg_rain     ,sm          ,                              
              etr         ,qseva       ,qsdew       ,qsubl   ,qfros  ,
              qseva_soil  ,qsdew_soil  ,qsubl_soil  ,qfros_soil      ,
              qseva_snow  ,qsdew_snow  ,qsubl_snow  ,qfros_snow      ,
              fsno                                                   ,
              rsur        ,rsur_se     ,rsur_ie     ,rnof            ,
              qinfl       ,wtfact      ,ssi         ,pondmx          ,
              wimp        ,zwt         ,wdsrf       ,wa      ,wetwat ,
#if(defined CaMa_Flood)
              # flddepth    ,fldfrc      ,qinfl_fld   ,
#endif
# ! SNICAR model variables
              forc_aer    ,
            vic_b_infilt, vic_Dsmax, vic_Ds, vic_Ws, vic_c, fevpg,
              mss_bcpho   ,mss_bcphi   ,mss_ocpho   ,mss_ocphi ,
              mss_dst1    ,mss_dst2    ,mss_dst3    ,mss_dst4 ):

    # Local variables
    eff_porosity = np.zeros(nl_soil)
    gwat = 0.0
    drainmax = 0.0
    rsubst = 0.0
    vol_liq = np.zeros(nl_soil)
    vol_ice = np.zeros(nl_soil)
    icefrac = np.zeros(nl_soil)
    err_solver = 0.0
    w_sum = 0.0
    wresi = np.zeros(nl_soil)
    qgtop = 0.0
    zwtmm = 0.0
    sp_zc = np.zeros(nl_soil)
    sp_zi = np.zeros(nl_soil + 1)
    sp_dz = np.zeros(nl_soil)
    is_permeable = [False] * nl_soil
    dzsum = 0.0
    dz = 0.0
    icefracsum = 0.0
    fracice_rsub = 0.0
    imped = 0.0

    ps = 0
    pe = 0
    irrig_flag = 1  # 1 if sprinker, 2 if others
    qflx_irrig_drip = 0.0
    qflx_irrig_sprinkler = 0.0
    qflx_irrig_flood = 0.0
    qflx_irrig_paddy = 0.0

    # theta_r = np.zeros(nl_soil)

    nprms = 1

    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        nprms = 5
    
    prms = np.zeros((nprms, nl_soil))

    gfld = 0.0
    qinfl_all = 0.0
    rsur_fld = 0.0
    qinfl_fld_subgrid = 0.0# inundation water input from top (mm/s)

    # [1] Update the liquid water within snow layer and the water onto soil
    if not nl_colm['DEF_SPLIT_SOILSNOW'] or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):
        if lb >= 1:
            gwat = pg_rain + sm - qseva
        else:
            if not nl_colm['DEF_USE_SNICAR'] or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):
                wice_soisno, wliq_soisno, gwat = snowwater(lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros, dz_soisno, wice_soisno, wliq_soisno)
            else:
                gwat = snow_water_snicar(lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros, dz_soisno, wice_soisno, wliq_soisno, forc_aer, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4)
    else:
        if lb >= 1:
            gwat = pg_rain + sm - qseva_soil
        else:
            if not nl_colm['DEF_USE_SNICAR']:
                wice_soisno, wliq_soisno, gwat = snowwater(lb, deltim, ssi, wimp, pg_rain * fsno, qseva_snow, qsdew_snow, qsubl_snow, qfros_snow, dz_soisno, wice_soisno, wliq_soisno)
            else:
                gwat = snow_water_snicar(lb, deltim, ssi, wimp, pg_rain * fsno, qseva_snow, qsdew_snow, qsubl_snow, qfros_snow, dz_soisno, wice_soisno, wliq_soisno, forc_aer, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4)
            gwat += pg_rain * (1 - fsno) - qseva_soil

    if nl_colm['CROP']:
        if nl_colm['DEF_USE_IRRIGATION']:
            if patchtype == 0:
                ps = landpft.patch_pft_s[ipatch]
                pe = landpft.patch_pft_e[ipatch]
                # CalIrrigationApplicationFluxes(ipatch, ps, pe, deltim, qflx_irrig_drip, qflx_irrig_sprinkler, qflx_irrig_flood, qflx_irrig_paddy, irrig_flag=2)
                # gwat += qflx_irrig_drip + qflx_irrig_flood + qflx_irrig_paddy

    # [2] Surface runoff and infiltration
    if patchtype <= 1:
        w_sum = sum(wliq_soisno[:]) + sum(wice_soisno[:]) + wa + wdsrf

        wresi = np.zeros(nl_soil)
        for j in range(nl_soil):
            vol_ice[j] = min(porsl[j], wice_soisno[j] / (dz_soisno[j] * const_physical.denice))
            icefrac[j] = 0.0 if porsl[j] < 1.e-6 else min(1.0, vol_ice[j] / porsl[j])
            eff_porosity[j] = max(wimp, porsl[j] - vol_ice[j])
            is_permeable[j] = eff_porosity[j] > max(wimp, theta_r[j])
            if is_permeable[j]:
                vol_liq[j] = wliq_soisno[j] / (dz_soisno[j] * const_physical.denh2o)
                vol_liq[j] = min(eff_porosity[j], max(0.0, vol_liq[j]))
                wresi[j] = wliq_soisno[j] - dz_soisno[j] * const_physical.denh2o * vol_liq[j]
            else:
                vol_liq[j] = 0.0

        if not nl_colm['CatchLateralFlow']:
            if nl_colm['DEF_Runoff_SCHEME'] == 0:
                if gwat > 0:
                    # 假设SurfaceRunoff_SIMTOP是已经定义好的对应Python函数，这里传入参数调用
                    rsur, rsur_se, rsur_ie = CoLM_Runoff.SurfaceRunoff_SIMTOP(nl_soil, wtfact, wimp, porsl, psi0, hksati,
                                                                  z_soisno[0:], dz_soisno[0:], zi_soisno[0:],
                                                                  eff_porosity, icefrac, zwt, gwat)
                else:
                    rsur = 0
                # 假设SubsurfaceRunoff_SIMTOP是已经定义好的对应Python函数，这里传入参数调用
                rsubst = CoLM_Runoff.SubsurfaceRunoff_SIMTOP(nl_soil, icefrac, dz_soisno[0:], zi_soisno[0:], zwt)

            elif nl_colm['DEF_Runoff_SCHEME'] == 1:
                # 假设vic_para是已经定义好的对应Python函数，这里传入参数调用
                soil_con, cell = CoLM_Hydro_VIC_Variables.vic_para(nl_colm, porsl, theta_r, hksati, bsw, wice_soisno, wliq_soisno, fevpg[ipatch], rootflux,
                         vic_b_infilt[ipatch], vic_Dsmax[ipatch], vic_Ds[ipatch], vic_Ws[ipatch], vic_c[ipatch])
                # 假设compute_vic_runoff是已经定义好的对应Python函数，这里传入参数调用
                cell = CoLM_Hydro_VIC.compute_vic_runoff(soil_con, gwat * deltim, soil_con.frost_fract, cell)

                if gwat > 0:
                    rsur = cell.runoff / deltim
                rsubst = cell.baseflow / deltim

            elif nl_colm['DEF_Runoff_SCHEME'] == 2:
                # 假设Runoff_XinAnJiang是已经定义好的对应Python函数，这里传入参数调用
                rsur, rsubst = CoLM_Runoff.Runoff_XinAnJiang(dz_soisno[0:nl_soil], eff_porosity[0:nl_soil],
                                                 vol_liq[0:nl_soil], topostd, gwat, deltim)
                rsur_se = rsur
                rsur_ie = 0

            elif nl_colm['DEF_Runoff_SCHEME'] == 3:
                # 假设Runoff_SimpleVIC是已经定义好的对应Python函数，这里传入参数调用
                rsur, rsubst = CoLM_Runoff.Runoff_SimpleVIC(dz_soisno[0:nl_soil], eff_porosity[0:nl_soil],
                                                vol_liq[0:nl_soil], BVIC, gwat, deltim)
                rsur_se = rsur
                rsur_ie = 0

            qgtop = gwat - rsur
        else:
            qgtop = gwat
            rsubst = 0

        #ifdef DataAssimilation
            # rsur = max(min(rsur * fslp_k(ipatch), gwat), 0.)
        #endif

            # infiltration into surface soil layer
            # qgtop = gwat - rsur
        #else
            # for lateral flow, "rsur" is calculated in HYDRO/MOD_Hydro_SurfaceFlow.F90
            # and is removed from surface water there.
            # qgtop = gwat
        #endif

        # if nl_colm['CaMa_Flood']:
        #     if nl_colm['LWINFILT']:
        #         # Re-infiltration calculation
        #         # If surface runoff occurs (rsur != 0.), flood depth < 1.e-6 and flood fraction < 0.05,
        #         # the re-infiltration will not be calculated.
        #         if (flddepth > 1.e-6) and (fldfrc > 0.05) and (patchtype == 0):
        #             gfld = flddepth / deltim  # [mm/s]
        #             # Surface runoff from inundation should not be added to the surface runoff from soil
        #             # Otherwise, the surface runoff will be double counted.
        #             # Only the re-infiltration is added to water balance calculation.
        #             rsur_fld = 0.0  # Initialize rsur_fld
        #             surfacerunoff(nl_soil, 1.0, wimp, porsl, psi0, hksati,
        #                         z_soisno[:], dz_soisno[:], zi_soisno[:],
        #                         eff_porosity, icefrac, zwt, gfld)
        #             # Infiltration into surface soil layer
        #             qinfl_fld_subgrid = gfld - rsur_fld  # Assume the re-infiltration occurs in whole patch area
        #         else:
        #             qinfl_fld_subgrid = 0.0
        #             gfld = 0.0
        #             rsur_fld = 0.0

        #         qinfl_fld = qinfl_fld_subgrid * fldfrc  # [mm/s] re-infiltration in grid
        #         qgtop = qinfl_fld + qgtop  # [mm/s] total infiltration in grid
        #         flddepth = flddepth - deltim * qinfl_fld_subgrid  # Renew flood depth [mm], the flood depth is reduced by re-infiltration but only in inundation area

        # [3] Determine the change of soil water
        zwtmm = zwt * 1000.0
        sp_zc = np.array([z * 1000.0 for z in z_soisno])
        sp_zi = np.array([zi * 1000.0 for zi in zi_soisno])

        for j in range(nl_soil):
            if vol_liq[j] < eff_porosity[j] - 1.e-8 and zwtmm <= sp_zi[j]:
                zwtmm = sp_zi[j+1]

        if nl_colm['Campbell_SOIL_MODEL']:
            prms[0,:] = bsw[:nl_soil]

        if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            prms[0, :] = alpha_vgm[:nl_soil]
            prms[1, :] = n_vgm[:nl_soil]
            prms[2, :] = L_vgm[:nl_soil]
            prms[3, :] = sc_vgm[:nl_soil]
            prms[4, :] = fc_vgm[:nl_soil]

        if zwtmm < sp_zi[nl_soil]:
            for j in range(nl_soil-1, -1, -1):
                if zwtmm >= sp_zi[j] and zwtmm < sp_zi[j+1]:
                    if zwtmm > sp_zi[j] and is_permeable[j]:
                        vol_liq[j] = (wliq_soisno[j] * 1000.0 / const_physical.denh2o - eff_porosity[j] * (sp_zi[j+1] - zwtmm)) / (zwtmm - sp_zi[j])
                        if vol_liq[j] < 0:
                            zwtmm = sp_zi[j+1]
                            vol_liq[j] = wliq_soisno[j] * 1000.0 / const_physical.denh2o / (sp_zi[j+1] - sp_zi[j])
                        vol_liq[j] = max(0.0, min(eff_porosity[j], vol_liq[j]))
                        wresi[j] = wliq_soisno[j] * 1000.0 / const_physical.denh2o - eff_porosity[j] * (sp_zi[j+1] - zwtmm) - vol_liq[j] * (zwtmm - sp_zi[j])
                    break

        wdsrf = max(0.0, wdsrf)

        if not is_permeable[0] and qgtop < 0:
            if wdsrf > 0:
                wdsrf += qgtop * deltim
                if wdsrf < 0:
                    wliq_soisno[0] = max(0.0, wliq_soisno[0] + wdsrf)
                    wdsrf = 0
            else:
                wliq_soisno[0] = max(0.0, wliq_soisno[0] + qgtop * deltim)
            qgtop = 0
        HSW = Hydro_SoilWater(nl_colm)
        qinfl, wdsrf, zwtmm, wa, vol_liq[:nl_soil], smp, hk = HSW.soil_water_vertical_movement(
            nl_soil, deltim, sp_zc[:nl_soil], sp_zi, is_permeable[:nl_soil],
            eff_porosity[:nl_soil], theta_r[:nl_soil], psi0[:nl_soil], hksati[:nl_soil],
            nprms, prms[:, :nl_soil], porsl[nl_soil-1], qgtop, etr, rootr[:nl_soil], rootflux[:nl_soil],
            rsubst, qinfl, wdsrf, zwtmm, wa, vol_liq[:nl_soil], smp[:nl_soil], hk[:nl_soil], 1e-3)

        for j in range(nl_soil-1, -1, -1):
            if is_permeable[j]:
                if zwtmm < sp_zi[j+1]:
                    if zwtmm >= sp_zi[j]:
                        wliq_soisno[j] = const_physical.denh2o * (eff_porosity[j] * (sp_zi[j+1] - zwtmm) + vol_liq[j] * (zwtmm - sp_zi[j])) / 1000.0
                    else:
                        wliq_soisno[j] = const_physical.denh2o * (eff_porosity[j] * (sp_zi[j+1] - sp_zi[j])) / 1000.0
                else:
                    wliq_soisno[j] = const_physical.denh2o * (vol_liq[j] * (sp_zi[j+1] - sp_zi[j])) / 1000.0
                wliq_soisno[j] += wresi[j]

        zwt = zwtmm / 1000.0

        if not nl_colm['DEF_SPLIT_SOILSNOW'] or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):
            if lb >= 1:
                wliq_soisno[0] = max(0.0, wliq_soisno[0] + qsdew * deltim)
                wice_soisno[0] = max(0.0, wice_soisno[0] + (qfros - qsubl) * deltim)
        else:
            wliq_soisno[0] = max(0.0, wliq_soisno[0] + qsdew_soil * deltim)
            wice_soisno[0] = max(0.0, wice_soisno[0] + (qfros_soil - qsubl_soil) * deltim)

        if not nl_colm['CatchLateralFlow']:
            if wdsrf > pondmx:
                rsur = rsur + (wdsrf - pondmx) / deltim
                rsur_ie = rsur_ie + (wdsrf - pondmx) / deltim
                wdsrf = pondmx

            if zwt<=0:
                rsur_ie = 0.
                rsur_se = rsur

            rnof = rsubst + rsur

        if not nl_colm['CatchLateralFlow']:
            err_solver = (np.sum(wliq_soisno) + np.sum(wice_soisno) + wa + wdsrf) - w_sum - (gwat - etr - rsur - rsubst) * deltim
        else:
            err_solver = (np.sum(wliq_soisno) + np.sum(wice_soisno) + wa + wdsrf) - w_sum - (
                        gwat - etr - rsur - rsubst) * deltim

        if not nl_colm['DEF_SPLIT_SOILSNOW'] or (patchtype == 1 and nl_colm['DEF_URBAN_RUN']):
            if lb >= 1:
                err_solver -= (qsdew + qfros - qsubl) * deltim
        else:
            err_solver -= (qsdew_soil + qfros_soil - qsubl_soil) * deltim

        # if nl_colm['CaMa_Flood']:
        #     LWINFILT = True  # Example value
        #     if LWINFILT:
        #         err_solver -= (gfld - rsur_fld) * fldfrc * deltim

        if nl_colm['CoLMDEBUG']:
            if abs(err_solver) > 1.e-3:
                print(f"Warning (WATER_VSF): water balance violation {err_solver:.5e}")
                # print(f"Warning (WATER_VSF): water balance violation {err_solver:.5e}, {landpatch.eindex[ipatch]}")
            if any(wliq_soisno < -1.e-3):
                print(f"Warning (WATER_VSF): negative soil water {wliq_soisno}")
    else:
        if patchtype == 2:  # WETLAND
            qinfl = 0.0
            zwt = 0.0

            if lb >= 1:
                wetwat = wdsrf + wa + wetwat + (gwat - etr + qsdew + qfros - qsubl) * deltim
            else:
                wetwat = wdsrf + wa + wetwat + (gwat - etr) * deltim

            wresi[:] = 0.0
            for j in range(nl_soil):
                if t_soisno[j] > const_physical.tfrz:
                    wresi[j] = max(wliq_soisno[j] - porsl[j] * dz_soisno[j] * 1000.0, 0.0)
                    wliq_soisno[j] -= wresi[j]

            wetwat += np.sum(wresi)

            if wetwat > wetwatmax:
                wdsrf = wetwat - wetwatmax
                wetwat = wetwatmax
                wa = 0.0
            elif wetwat < 0.0:
                wa = wetwat
                wdsrf = 0.0
                wetwat = 0.0
            else:
                wdsrf = 0.0
                wa = 0.0

            if not nl_colm['CatchLateralFlow']:
                if wdsrf > pondmx:
                    rsur = (wdsrf - pondmx) / deltim
                    wdsrf = pondmx
                else:
                    rsur = 0.0
                rnof = rsur
                rsur_se = rsur
                rsur_ie = 0.

    return  wice_soisno, wliq_soisno,smp,hk,zwt,wdsrf,wa, wetwat,rsur, rsur_se, rsur_ie, rnof, qinfl, mss_bcpho ,mss_bcphi ,mss_ocpho ,mss_ocphi ,mss_dst1 ,mss_dst2 ,mss_dst3 ,mss_dst4



