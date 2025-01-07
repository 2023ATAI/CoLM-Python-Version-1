import numpy as np
import math
from CoLM_WetBulb import wetbulb

def rain_snow_temp(nl_colm, const_physical, var_global,patchtype, forc_t, forc_q, forc_psrf, forc_prc, forc_prl, forc_us,forc_vs,tcrit,
              prc_rain,prc_snow,prl_rain,prl_snow,t_precip,bifall):
    """
    determine net radiation.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         : xxxxxxxxxxxxxxxxxxxxxxxxxxxx

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
    # wet-bulb temperature
    t_hydro = 0    # temperature of falling hydrometeor [deg C]
    t_precip = wetbulb(forc_t, forc_psrf, forc_q, const_physical.hvap, const_physical.cpair)

    if nl_colm['DEF_precip_phase_discrimination_scheme'] == 'I':
        # Wang, Y.H., Broxton, P., Fang, Y., Behrangi, A., Barlage, M., Zeng, X., & Niu, G.Y. (2019).
        # A Wet-Bulb Temperature Based Rain-Snow Partitioning Scheme Improves Snowpack Prediction
        # Over the Drier Western United States. Geophysical Research Letters, 46, 13,825-13,835.
        #
        # Behrangi et al. (2018) On distinguishing snowfall from rainfall
        # using near-surface atmospheric information: Comparative analysis,
        # uncertainties and hydrologic importance. Q J R Meteorol Soc. 144 (Suppl. 1):89-102
        if t_precip > const_physical.tfrz + 3.0:
            flfall = 1.0      # fraction of liquid water within falling precip
        elif t_precip >= const_physical.tfrz - 2.0:
            flfall = max(0.0, 1.0 - 1.0 / (1.0 + 5.00e-5 * np.exp(2.0 * (t_precip + 4.))))   # Figure 5c of Behrangi et al. (2018)
            # flfall = max(0.0, 1.0 - 1.0/(1.0+6.99e-5*exp(2.0*(t_precip+3.97)))) # Equation 1 of Wang et al. (2019)
        else:
            flfall = 0.0

    elif nl_colm['DEF_precip_phase_discrimination_scheme'] == 'II':
        # CLM5.0
        glaciers = False
        if patchtype == 3:
            glaciers = True

        if glaciers:
            all_snow_t_c = -2.0
            all_rain_t_c = 0.0
        else:
            all_snow_t_c = 0.0
            all_rain_t_c = 2.0

        all_snow_t = all_snow_t_c + const_physical.tfrz
        frac_rain_slope = 1.0 / (all_rain_t_c - all_snow_t_c)

        # Re-partition precipitation into rain/snow for a single column.
        # Rain and snow variables should be set initially, and are updated here
        flfall = min(1.0, max(0.0, (forc_t - all_snow_t) * frac_rain_slope))
        # bifall = min(169.0, 50. + 1.7 * (max(0.0, forc_t - const_physical.tfrz + 15.)) ** 1.5)
    elif nl_colm['DEF_precip_phase_discrimination_scheme'] == 'III':
        # Hydromet_Temp方法没找到，配置不执行
        pass

        # Hydromet_Temp(forc_psrf,(forc_t-273.15),forc_q,t_hydro)
        # if t_hydro > 3.0:
        #     flfall = 1.0     # fraction of liquid water within falling precip
        # elif (t_hydro >= -3.0) and (t_hydro <= 3.0):
        #     flfall = max(0.0, 1.0/(1.0+2.50286*0.125006**t_hydro))
        # else:
        #     flfall = 0.0

    else:
        # the upper limit of air temperature is set for snowfall, this cut-off
        # was selected based on Fig. 1, Plate 3-1, of Snow Hydrology (1956).
        # the percentage of liquid water by mass, which is arbitrarily set to
        # vary linearly with air temp, from 0% at 273.16 to 40% max at 275.16.
        if forc_t > const_physical.tfrz + 2.0:
            flfall = 1.0     # fraction of liquid water within falling precip.
            # bifall = 169.15  # (not used)
        else:
            flfall = max(0.0, -54.632 + 0.2 * forc_t)
            # bifall = 50. + 1.7 * (max(0.0, forc_t - const_physical.tfrz + 15.)) ** 1.5

    bifall = NewSnowBulkDensity(forc_t, forc_us, forc_vs, const_physical.tfrz)

    prc_rain = forc_prc * flfall        # convective rainfall (mm/s)
    prl_rain = forc_prl * flfall        # large scale rainfall (mm/s)
    prc_snow = forc_prc * (1. - flfall) # convective snowfall (mm/s)
    prl_snow = forc_prl * (1. - flfall) # large scale snowfall (mm/s)

    # temperature of rainfall or snowfall
    if forc_t > 275.65:
        if t_precip < const_physical.tfrz:
            t_precip = const_physical.tfrz
    else:
        t_precip = min(const_physical.tfrz, t_precip)
        if flfall > 1.0e-6:
            t_precip = const_physical.tfrz - np.sqrt((1.0 / flfall) - 1.0) / 100.0

    return prc_rain, prc_snow, prl_rain, prl_snow, t_precip, bifall

def NewSnowBulkDensity(forc_t,forc_us,forc_vs,tfrz):
    """
    Scheme for bulk density of newly fallen dry snow

    Parameters:
    forc_t (float): temperature at agcm reference height [kelvin]
    forc_us (float): wind speed in eastward direction [m/s]
    forc_vs (float): wind speed in northward direction [m/s]

    Returns:
    float: bulk density of newly fallen dry snow [kg/m3]
    """
    bifall = 0
    t_for_bifall_degC = 0
    forc_wind = 0

    if forc_t > tfrz + 2.0:
        bifall = 50.0 + 1.7 * (17.0) ** 1.5
    elif forc_t > tfrz - 15.0:
        bifall = 50.0 + 1.7 * (forc_t - tfrz + 15.0) ** 1.5
    else:
        if forc_t > tfrz - 57.55:
            t_for_bifall_degC = (forc_t - tfrz)
        else:
            t_for_bifall_degC = -57.55
        bifall = -(50.0 / 15.0 + 0.0333 * 15.0) * t_for_bifall_degC - 0.0333 * t_for_bifall_degC ** 2

    forc_wind = math.sqrt(forc_us ** 2 + forc_vs ** 2)
    if forc_wind > 0.1:
        # Density offset for wind-driven compaction, initial ideas based on Liston et. al (2007) J. Glaciology,
        # 53(181), 241-255. Modified for a continuous wind impact and slightly more sensitive to wind - Andrew Slater, 2016
        bifall = bifall + (266.861 * ((1.0 + math.tanh(forc_wind / 5.0)) / 2.0) ** 8.8)

    return bifall

