import numpy as np
import math

def soil_hcap_cond(co_lm, const_physical, vf_gravels_s, vf_om_s, vf_sand_s, vf_pores_s, wf_gravels_s, wf_sand_s, k_solids,csol,kdry,ksat_u,ksat_f, BA_alpha,BA_beta, temperature,vf_water,vf_ice,hcap,thk):

    """
    Original author  : Qinghliang Li,  Jinlong Zhu, 17/02/2024;
    software         : Calculate bulk soil heat capacity and soil thermal conductivity with 8 optional schemes
                        ! The default soil thermal conductivity scheme is the fourth one (Balland V. and P. A. Arp, 2005)

    Parameters:
        vf_gravels_s : float
            Volumetric fraction of gravels within the soil solids.
        vf_om_s : float
            Volumetric fraction of organic matter within the soil solids.
        vf_sand_s : float
            Volumetric fraction of sand within soil solids.
        vf_pores_s : float
            Volumetric pore space of the soil.
        wf_gravels_s : float
            Gravimetric fraction of gravels.
        wf_sand_s : float
            Gravimetric fraction of sand within soil solids.
        k_solids : float
            Thermal conductivity of soil solids.
        csol : float
            Heat capacity of dry soil [J/(m3 K)].
        kdry : float
            Thermal conductivity for dry soil [W/m/K].
        ksat_u : float
            Thermal conductivity of unfrozen saturated soil [W/m/K].
        ksat_f : float
            Thermal conductivity of frozen saturated soil [W/m/K].
        BA_alpha : float
            Alpha in Balland and Arp(2005) thermal conductivity scheme.
        BA_beta : float
            Beta in Balland and Arp(2005) thermal conductivity scheme.
        temperature : float
            Soil temperature [K].
        vf_water : float
            Volumetric fraction of water.
        vf_ice : float
            Volumetric fraction of ice.

        Returns:
        hcap : float
            Bulk soil heat capacity [J/(m3 K)].
        thk : float
            Soil thermal conductivity [W/(m K)].
    """
    # ______________________________________________________________________
    # The heat capacity and thermal conductivity [J(m3 K)]
    # ______________________________________________________________________
    c_water = 4.188e6  # J/(m3 K)
    c_ice = 1.94153e6  # J/(m3 K)

    hcap = csol + vf_water * c_water + vf_ice * c_ice

    # ______________________________________________________________________
    #  Setting
    # ______________________________________________________________________
    k_air = 0.024  # (W/m/K)
    k_water = 0.57  # (W/m/K)
    k_ice = 2.29  # (W/m/K)

    # Calculate the sum of gravel and sand fractions
    a = vf_gravels_s + vf_sand_s

    # Calculate the degree of saturation
    sr = (vf_water + vf_ice) / vf_pores_s
    # Ensure sr is within the range [1.0e-6, 1.0]
    sr = min(1.0, sr)

    # Check if sr is greater than or equal to 1.0e-10
    if sr >= 1.0e-10:
        case = co_lm['DEF_THERMAL_CONDUCTIVITY_SCHEME']
        if case == 1:
            # Oleson et al., 2013: Technical Description of version 4.5 of the Community Land Model
            # (CLM). NCAR/TN-503+STR (Section 6.3: Soil and Snow Thermal Properties)
            if temperature > const_physical.tfrz:  # Unfrozen soil
                ke = np.log10(sr) + 1.0
            else:  # Frozen or partially frozen soils
                ke = sr

        elif case == 2:
            # Johansen O (1975): Thermal conductivity of soils. PhD Thesis. Trondheim, Norway:
            # University of Trondheim. US army Crops of Engineerings, CRREL English Translation 637.
            if temperature > const_physical.tfrz:  # Unfrozen soils
                if a > 0.4:  # coarse-grained
                    ke = 0.7 * np.log10(max(sr, 0.05)) + 1.0
                else:  # Fine-grained
                    ke = np.log10(max(sr, 0.1)) + 1.0
            else:  # Frozen or partially frozen soils
                ke = sr

        elif case == 3:
            # Cote, J., and J.-M. Konrad (2005), A generalized thermal conductivity model for soils
            # and construction materials. Canadian Geotechnical Journal, 42(2): 443-458.
            if temperature > const_physical.tfrz:  # Unfrozen soils
                if a > 0.40:
                    kappa = 4.60
                elif a > 0.25:
                    kappa = 3.55
                elif a > 0.01:
                    kappa = 1.90
                else:
                    kappa = 0.60
            else:  # Frozen or partially frozen soils
                if a > 0.40:
                    kappa = 1.70
                elif a > 0.25:
                    kappa = 0.95
                elif a > 0.01:
                    kappa = 0.85
                else:
                    kappa = 0.25
                ke = kappa * sr / (1.0 + (kappa - 1.0) * sr)
        elif case == 4:
            # [4] Balland V. and P. A. Arp, 2005: Modeling soil thermal conductivities over a wide
            # range of conditions. J. Environ. Eng. Sci. 4: 549-558.
            # be careful in specifying all k affecting fractions as VOLUME FRACTION,
            # whether these fractions are part of the bulk volume, the pore space, or the solid space.
            if temperature > const_physical.tfrz:  # Unfrozen soil
                ke = sr ** (0.5 * (1.0 + vf_om_s - BA_alpha * vf_sand_s - vf_gravels_s)) \
                     * ((1.0 / (1.0 + np.exp(-BA_beta * sr))) ** 3 - ((1.0 - sr) / 2.0) ** 3) ** (1.0 - vf_om_s)
            else:  # Frozen or partially frozen soils
                ke = sr ** (1.0 + vf_om_s)

        elif case == 5:
            # [5] Lu et al., 2007: An improved model for predicting soil thermal conductivity from
            # water content at room temperature. Soil Sci. Soc. Am. J. 71:8-14
            if a > 0.4:  # Coarse-textured soils = soils with sand fractions >40 (%)
                alpha = 0.728
                beta = 1.165
            else:  # Fine-textured soils = soils with sand fractions <40 (%)
                alpha = 0.37
                beta = 1.29
            if temperature > const_physical.tfrz:  # Unfrozen soils
                ke = np.exp(alpha * (1.0 - sr ** (alpha - beta)))
            else:  # Frozen or partially frozen soils
                ke = sr
    else:
        ke = 0.0

    # Calculate thermal conductivity factor based on specified scheme
    if co_lm['DEF_THERMAL_CONDUCTIVITY_SCHEME'] >= 1 and co_lm['DEF_THERMAL_CONDUCTIVITY_SCHEME'] <= 5:
        ke = max(ke, 0.0)
        ke = min(ke, 1.0)
        if temperature > const_physical.tfrz:  # Unfrozen soil
            thk = (ksat_u - kdry) * ke + kdry
        else:  # Frozen or partially frozen soils
            thk = (ksat_f - kdry) * ke + kdry

    if co_lm['DEF_THERMAL_CONDUCTIVITY_SCHEME'] == 6:
        # [6] Series-Parallel Models (Tarnawski and Leong, 2012)
        a = wf_gravels_s + wf_sand_s

        # a fitting parameter of the soil solid uniform passage
        aa = 0.0237 - 0.0175 * a ** 3

        # a fitting parameter of a minuscule portion of soil water (nw) plus a minuscule portion of soil air (na)
        nwm = 0.088 - 0.037 * a ** 3

        # the degree of saturation of the minuscule pore space
        x = 0.6 - 0.3 * a ** 3
        if sr < 1.0e-6:
            nw_nwm = 0.0
        else:
            nw_nwm = math.exp(1.0 - sr ** (-x))

        if temperature > const_physical.tfrz:  # Unfrozen soil
            thk = k_solids * aa + (1.0 - vf_pores_s - aa + nwm) ** 2 \
                  / ((1.0 - vf_pores_s - aa) / k_solids + nwm / (k_water * nw_nwm + k_air * (1.0 - nw_nwm))) \
                  + k_water * (vf_pores_s * sr - nwm * nw_nwm) \
                  + k_air * (vf_pores_s * (1.0 - sr) - nwm * (1.0 - nw_nwm))
        else:
            thk = k_solids * aa + (1.0 - vf_pores_s - aa + nwm) ** 2 \
                  / ((1.0 - vf_pores_s - aa) / k_solids + nwm / (k_ice * nw_nwm + k_air * (1.0 - nw_nwm))) \
                  + k_ice * (vf_pores_s * sr - nwm * nw_nwm) \
                  + k_air * (vf_pores_s * (1.0 - sr) - nwm * (1.0 - nw_nwm))

    if co_lm['DEF_THERMAL_CONDUCTIVITY_SCHEME'] == 7:
        # [7] Thermal properties of soils, in Physics of Plant Environment,
        # ed. by W.R. van Wijk (North-Holland, Amsterdam, 1963), pp. 210-235
        if sr * vf_pores_s <= 0.09:
            ga = 0.013 + 0.944 * sr * vf_pores_s
        else:
            ga = 0.333 - (1.0 - sr) * vf_pores_s / vf_pores_s * (0.333 - 0.035)
        gc = 1.0 - 2.0 * ga

        if temperature > const_physical.tfrz:  # Unfrozen soil
            aa = (2.0 / (1.0 + (k_air / k_water - 1.0) * ga) +
                  1.0 / (1.0 + (k_air / k_water - 1.0) * gc)) / 3.0
            aaa = (2.0 / (1.0 + (k_solids / k_water - 1.0) * 0.125) +
                   1.0 / (1.0 + (k_solids / k_water - 1.0) * (1.0 - 2.0 * 0.125))) / 3.0

            thk = (sr * vf_pores_s * k_water + (1.0 - sr) * vf_pores_s * aa * k_air + (
                        1.0 - vf_pores_s) * aaa * k_solids) \
                  / (sr * vf_pores_s + (1.0 - sr) * vf_pores_s * aa + (1.0 - vf_pores_s) * aaa)
        else:
            aa = (2.0 / (1.0 + (k_air / k_ice - 1.0) * ga) +
                  1.0 / (1.0 + (k_air / k_ice - 1.0) * gc)) / 3.0
            aaa = (2.0 / (1.0 + (k_solids / k_ice - 1.0) * 0.125) +
                   1.0 / (1.0 + (k_solids / k_ice - 1.0) * (1.0 - 2.0 * 0.125))) / 3.0

            thk = (sr * vf_pores_s * k_ice + (1.0 - sr) * vf_pores_s * aa * k_air + (1.0 - vf_pores_s) * aaa * k_solids) \
                  / (sr * vf_pores_s + (1.0 - sr) * vf_pores_s * aa + (1.0 - vf_pores_s) * aaa)

    if co_lm['DEF_THERMAL_CONDUCTIVITY_SCHEME'] == 8:
        # [8] Yan & He et al., 2019: A generalized model for estimating effective soil thermal conductivity
        # based on the Kasubuchi algorithm, Geoderma, Vol 353, 227-242
        beta = -0.303 * ksat_u - 0.201 * wf_sand_s + 1.532
        if vf_water > 0.01:
            ke = (1 + (vf_pores_s / beta) ** (-1.0 * beta)) / (1 + (vf_water / beta) ** (-1.0 * beta))
        else:
            ke = 0.0

        ke = max(ke, 0.0)
        ke = min(ke, 1.0)

        if temperature > const_physical.tfrz:  # Unfrozen soil
            thk = (ksat_u - kdry) * ke + kdry
        else:  # Frozen or partially frozen soils
            thk = (ksat_f - kdry) * ke + kdry

    return hcap, thk