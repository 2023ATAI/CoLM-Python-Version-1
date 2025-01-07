import numpy as np
import math
import CoLM_Hydro_SoilFunction

def SoilSurfaceResistance(nl_colm, const_physical, nl_soil, forc_rhoair, hksati, porsl, psi0, bsw=None,
                          theta_r=None, alpha_vgm=None, n_vgm=None, L_vgm=None,
                          sc_vgm=None, fc_vgm=None, dz_soisno=None, t_soisno=None, wliq_soisno=None,
                          wice_soisno=None, fsno=None, qg=None):
    """
    Original author  : Qinghliang Li, 17/02/2024; Jinlong Zhu,   17/02/2024;
    software         :  Main SUBROUTINE to CALL soil resistance model
                    ! - Options for soil surface resistance schemes
                    !    1: SL14, Swenson and Lawrence (2014)
                    !    2: SZ09, Sakaguchi and Zeng (2009)
                    !    3: TR13, Tang and Riley (2013)
                    !    4: LP92, Lee and Pielke (1992)
                    !    5: S92,  Sellers et al (1992)
    Args:
        nl_soil (int): Upper bound of array.
        forc_rhoair (float): Density of air [kg/m^3].
        hksati (ndarray): Hydraulic conductivity at saturation [mm h2o/s].
        porsl (ndarray): Soil porosity [-].
        psi0 (ndarray): Saturated soil suction [mm] (NEGATIVE).
        bsw (ndarray, optional): Clapp and Hornbereger "b" parameter [-].
        theta_r (ndarray, optional): Residual moisture content [-].
        alpha_vgm (ndarray, optional): A parameter corresponding approximately to the inverse of the air-entry value.
        n_vgm (ndarray, optional): Pore-connectivity parameter [dimensionless].
        L_vgm (ndarray, optional): A shape parameter [dimensionless].
        sc_vgm (ndarray, optional): Saturation at the air entry value in the classical vanGenuchten model [-].
        fc_vgm (ndarray, optional): A scaling factor by using air entry value in the Mualem model [-].
        dz_soisno (ndarray): Layer thickness [m].
        t_soisno (ndarray): Soil/snow skin temperature [K].
        wliq_soisno (ndarray): Liquid water [kg/m^2].
        wice_soisno (ndarray): Ice lens [kg/m^2].
        fsno (float): Fractional snow cover [-].
        qg (float): Ground specific humidity [kg/kg].

    Returns:
        float: Soil surface resistance [s/m].

    """
    # Calculate the top soil volumetric water content (m3/m3), soil matrix potential,
    # and soil hydraulic conductivity
    rss = 0
    soil_gas_diffusivity_scheme = 0
    if nl_colm['Campbell_SOIL_MODEL']:
        soil_gas_diffusivity_scheme = 1
    elif nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        soil_gas_diffusivity_scheme = 6

    vol_liq = max(wliq_soisno[0], 1.0e-6) / (const_physical.denh2o * dz_soisno[0])
    s_node = min(1.0, vol_liq / porsl[0])

    # Calculate effective soil porosity
    eff_porosity = max(0.01, porsl[0] - min(porsl[0], wice_soisno[0] / (dz_soisno[0] * const_physical.denice)))

    if nl_colm['Campbell_SOIL_MODEL']:
        pass
    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        # Calculate soil matrix potential and soil hydraulic conductivity
        smp_node = CoLM_Hydro_SoilFunction.soil_psi_from_vliq(nl_colm, s_node * (porsl[0] - theta_r[0]) + theta_r[0], porsl[0], theta_r[0], psi0[0],5,
                                      [alpha_vgm[0], n_vgm[0], L_vgm[0], sc_vgm[0], fc_vgm[0]])
        hk = CoLM_Hydro_SoilFunction.soil_hk_from_psi(nl_colm, smp_node, psi0[0], hksati[0], 5, [alpha_vgm[0], n_vgm[0], L_vgm[0], sc_vgm[0], fc_vgm[0]])

        # Convert units from mm to m
        smp_node /= 1000.0
        hk /= 1000.0

        # Calculate air-free pore space
        aird = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(nl_colm, -1.0e7, porsl[0], theta_r[0], psi0[0], 5, [alpha_vgm[0], n_vgm[0], L_vgm[0], sc_vgm[0],
                                  fc_vgm[0]])

    # Calculate soil gas diffusivity (Dg) using the chosen scheme
    d0 = 2.12e-5 * (t_soisno[0] / 273.15) ** 1.75 #invalid value encountered in scalar power
    eps = porsl[0] - aird

    # Initialize tortuosity (tao)
    tao = 0.0
    case = soil_gas_diffusivity_scheme

    # Choose the soil gas diffusivity scheme
    if case == 1:
        if nl_colm['Campbell_SOIL_MODEL']:
        # BBC scheme
            tao = eps * eps * (eps / porsl[0]) ** (3.0 / max(3.0, bsw[0]))
    if case == 2:
        # P_WLR scheme
        tao = 0.66 * eps * (eps / porsl[0])
    elif case == 3:
        # MI_WLR scheme
        tao = eps ** (4.0 / 3.0) * (eps / porsl[0])
    elif case == 4:
        # MA_WLR scheme
        tao = eps ** (3.0 / 2.0) * (eps / porsl[0])
    elif case == 5:
        # M_Q scheme
        tao = eps ** (4.0 / 3.0) * (eps / porsl[0]) ** 2.0
    elif case == 6:
        if nl_colm['Campbell_SOIL_MODEL']:
        # 3POE scheme
            eps100 = porsl[0] - porsl[0] * (psi0[0] / -1000.0) ** (1.0 / bsw[0])
        if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            eps100 = porsl[0] - CoLM_Hydro_SoilFunction.soil_vliq_from_psi(nl_colm, -1000.0, porsl[0], theta_r[0], psi0[0], 5,[alpha_vgm[0], n_vgm[0],
                                                   L_vgm[0], sc_vgm[0], fc_vgm[0]])

        tao = porsl[0] * porsl[0] * (eps / porsl[0]) ** (2.0 + np.log(eps100 ** 0.25) / np.log(eps100 / porsl[0]))

    # Calculate gas and water diffusivity (dg and dw)
    dg = d0 * tao
    if nl_colm['Campbell_SOIL_MODEL']:
        # For TR13 scheme with Campbell_SOIL_MODEL
        # Calculate water diffusivity (dw) using Eq.(A5)
        dw = -hk * bsw[0] * smp_node / vol_liq
    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        # For TR13 scheme with vanGenuchten_Mualem_SOIL_MODEL
        # Calculate water diffusivity (dw) using Eqs. (A2), (A7), (A8), and (A10)
        m_vgm = 1.0 - 1.0 / n_vgm[0]  # Calculate m_vgm
        S = (1.0 + (-alpha_vgm[0] * smp_node) ** n_vgm[0]) ** (-m_vgm)  # Calculate S
        dw = -hk * (m_vgm - 1.0) / (alpha_vgm[0] * m_vgm * (porsl[0] - theta_r[0])) \
             * S ** (-1.0 / m_vgm) * (1.0 - S ** (1.0 / m_vgm)) ** (-m_vgm)  # Calculate dw

    # Assuming DEF_RSS_SCHEME is the variable holding the selected soil resistance scheme

    # CASE 1: SL14 scheme
    if nl_colm['DEF_RSS_SCHEME'] == 1:
        dsl = dz_soisno[0] * max(1.0e-6, (0.8 * eff_porosity - vol_liq)) / max(1.0e-6, (0.8 * porsl[0] - aird))
        dsl = max(dsl,0.0)
        dsl = min(dsl,0.2)
        rss = dsl / dg

    # CASE 2: SZ09 scheme
    elif nl_colm['DEF_RSS_SCHEME'] == 2:
        dsl = dz_soisno[0] * (math.exp((1.0 - vol_liq / porsl[0]) ** 5) - 1.0) / (math.exp(1.0) - 1.0)
        dsl = min(dsl,0.2)
        dsl = max(dsl,0.0)
        rss = dsl / dg

    # CASE 3: TR13 scheme
    elif nl_colm['DEF_RSS_SCHEME'] == 3:
        B = const_physical.denh2o / (qg * forc_rhoair)
        rg_1 = 2.0 * dg * eps / dz_soisno[0]
        rw_1 = 2.0 * dw * B * vol_liq / dz_soisno[0]
        rss_1 = rg_1 + rw_1
        rss = 1.0 / rss_1

    # CASE 4: LP92 beta scheme
    elif nl_colm['DEF_RSS_SCHEME'] == 4:
        wx = (max(wliq_soisno[0], 1.0e-6) / const_physical.denh2o + wice_soisno[0] / const_physical.denice) / dz_soisno[0]
        fac = min(1.0, wx/porsl(1))
        fac = max(fac , 0.001)
        if nl_colm['Campbell_SOIL_MODEL']:
            wfc = porsl[0] * (0.1 / (86400.0 * hksati[0])) ** (1.0 / (2.0 * bsw[0] + 3.0))
        if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            wfc = theta_r[0] + (porsl[0] - theta_r[0]) * (1 + (alpha_vgm[0] * 339.9) ** n_vgm[0]) ** (
                        (1.0 / n_vgm[0]) - 1)
        # For Lee and Pielke 1992 beta scheme
        if wx < wfc:  # when water content of the top layer is less than that at F.C.
            fac_fc = min(1.0, wx / wfc)
            fac_fc = max(fac_fc, 0.001)
            rss = 0.25 * (1.0 - np.cos(fac_fc * 3.1415926)) ** 2
        else:  # when water content of the top layer is more than that at F.C.
            rss = 1.0

    elif nl_colm['DEF_RSS_SCHEME'] == 5:
        # For Sellers 1992 scheme
        wx = (max(wliq_soisno(1), 1.0e-6) / const_physical.denh2o + wice_soisno(1) / const_physical.denice) / dz_soisno(1)
        fac = min(1.0, wx / porsl(1))
        fac = max(fac, 0.001)
        # Original Sellers (1992)
        # rss = exp(8.206 - 4.255 * fac)
        # Adjusted Sellers (1992) to decrease rss for wet soil according to Noah-MP v5
        rss = np.exp(8.206 - 6.0 * fac)

    # Account for snow fractional cover for rss

    if nl_colm['DEF_RSS_SCHEME'] != 4:
        # With 1/rss = fsno/rss_snow + (1-fsno)/rss_soil,
        # assuming rss_snow = 1, so rss is calibrated as:
        if (1.0 - fsno + fsno * rss) > 0.0:
            rss = rss / (1.0 - fsno + fsno * rss)
        else:
            rss = 0.0
        rss = min(1.0e6, rss)

    # Account for snow fractional cover for LP92 beta scheme
    # NOTE: rss here is for soil beta value
    if nl_colm['DEF_RSS_SCHEME'] == 4:
        # Modify soil beta by snow cover, assuming soil beta for snow surface is 1.
        rss = (1.0 - fsno) * rss + fsno
    return rss