import numpy as np
import CoLM_Qsadv

class CoLM_ForcingDownscaling(object):
    def __init__(self, nl_colm) -> None:
        self.nl_colm = nl_colm
        # Molecular weight dry air [kg/kmole]
        self.SHR_CONST_MWDAIR = 28.966
        # Molecular weight water vapor
        self.SHR_CONST_MWWV = 18.016
        # Avogadro's number [molecules/kmole]
        self.SHR_CONST_AVOGAD = 6.02214e26
        # Boltzmann's constant [J/K/molecule]
        self.SHR_CONST_BOLTZ = 1.38065e-23
        # Universal gas constant [J/K/kmole]
        self.SHR_CONST_RGAS = self.SHR_CONST_AVOGAD * self.SHR_CONST_BOLTZ
        # Dry air gas constant [J/K/kg]
        self.rair = self.SHR_CONST_RGAS / self.SHR_CONST_MWDAIR
        # Surface temperature lapse rate (deg m-1)
        self.lapse_rate = 0.006

    def downscale_forcings(self, glaciers,
                       forc_topo_g, forc_maxelv_g, forc_t_g, forc_th_g, forc_q_g,
                       forc_pbot_g, forc_rho_g, forc_prc_g, forc_prl_g, forc_lwrad_g,
                       forc_hgt_g, forc_swrad_g, forc_us_g, forc_vs_g,
                       slp_type_c, asp_type_c, area_type_c, svf_c, cur_c, sf_lut_c,
                       julian_day, coszen, cosazi, alb,
                       forc_topo_c, forc_t_c, forc_th_c, forc_q_c, forc_pbot_c,
                       forc_rho_c, forc_prc_c, forc_prl_c, forc_lwrad_c, forc_swrad_c,
                       forc_us_c, forc_vs_c, grav, cpair):

        self.num_gridcells = 0  # example value
        num_columns = 0  # example value
        self.begc = np.zeros(self.num_gridcells, dtype=int)
        self.endc = np.zeros(self.num_gridcells, dtype=int)

        glaciers = np.full(num_columns,False)
        wt_column = np.zeros(num_columns)

        forc_slp_c = np.zeros(num_columns)
        forc_asp_c = np.zeros(num_columns)
        forc_cur_c = np.zeros(num_columns)
        forc_svf_c = np.zeros(num_columns)
        forc_sf_c = np.zeros(num_columns)

        forc_t_g = np.zeros(self.num_gridcells)
        forc_th_g = np.zeros(self.num_gridcells)
        forc_q_g = np.zeros(self.num_gridcells)
        forc_pbot_g = np.zeros(self.num_gridcells)
        forc_rho_g = np.zeros(self.num_gridcells)
        forc_prc_g = np.zeros(self.num_gridcells)
        forc_prl_g = np.zeros(self.num_gridcells)
        forc_lwrad_g = np.zeros(self.num_gridcells)
        forc_topo_g = np.zeros(self.num_gridcells)
        forc_hgt_grc = np.zeros(self.num_gridcells)
        forc_topo_c = np.zeros(num_columns)
        forc_us_g = np.zeros(num_columns)
        forc_vs_g = np.zeros(num_columns)
        forc_swrad_g = np.zeros(num_columns)

        # Allocate arrays for column forcings
        forc_t_c = np.zeros(num_columns)
        forc_th_c = np.zeros(num_columns)
        forc_q_c = np.zeros(num_columns)
        forc_pbot_c = np.zeros(num_columns)
        forc_rho_c = np.zeros(num_columns)
        forc_prc_c = np.zeros(num_columns)
        forc_prl_c = np.zeros(num_columns)
        forc_lwrad_c = np.zeros(num_columns)
        forc_swrad_c = np.zeros(num_columns)
        forc_us_c = np.zeros(num_columns)
        forc_vs_c = np.zeros(num_columns)

        for g in range(self.num_gridcells):
            for c in range(self.begc[g], self.endc[g] + 1):
                forc_t_c[c] = forc_t_g[g]
                forc_th_c[c] = forc_th_g[g]
                forc_q_c[c] = forc_q_g[g]
                forc_pbot_c[c] = forc_pbot_g[g]
                forc_rho_c[c] = forc_rho_g[g]
                forc_prc_c[c] = forc_prc_g[g]
                forc_prl_c[c] = forc_prl_g[g]
                forc_lwrad_c[c] = forc_lwrad_g[g]

        for g in range(self.num_gridcells):
            hsurf_g = forc_topo_g[g]  # gridcell sfc elevation
            tbot_g = forc_t_g[g]  # atm sfc temp
            thbot_g = forc_th_g[g]  # atm sfc pot temp
            qbot_g = forc_q_g[g]  # atm sfc spec humid
            pbot_g = forc_pbot_g[g]  # atm sfc pressure
            rhos_g = forc_rho_g[g]  # atm density
            zbot = forc_hgt_grc[g]  # atm ref height

            max_elev_c = np.max(forc_topo_c[self.begc[g]:self.endc[g] + 1])  # maximum column level elevation value within the grid

            for c in range(self.begc[g], self.endc[g] + 1):
                hsurf_c = forc_topo_c[c]  # column sfc elevation
                tbot_c = tbot_g - self.lapse_rate * (hsurf_c - hsurf_g)  # adjust temp for column
                Hbot = self.rair * 0.5 * (tbot_g + tbot_c) / grav  # scale ht at avg temp
                pbot_c = pbot_g * np.exp(-(hsurf_c - hsurf_g) / Hbot)  # adjust press for column
                thbot_c = thbot_g + (tbot_c - tbot_g) * np.exp((zbot / Hbot) * (self.rair / cpair))  # adjust pot temp for column

                es_g, dum1, qs_g, dum2 = CoLM_Qsadv.qsadv(tbot_g, pbot_g)
                es_c, dum1, qs_c, dum2 = CoLM_Qsadv.qsadv(tbot_c, pbot_c)
                qbot_c = qbot_g * (qs_c / qs_g)  # adjust q for column

                rhos_c_estimate = self.rhos(qbot=qbot_c, pbot=pbot_c, tbot=tbot_c)
                rhos_g_estimate = self.rhos(qbot=qbot_g, pbot=pbot_g, tbot=tbot_g)
                rhos_c = rhos_g * (rhos_c_estimate / rhos_g_estimate)  # adjust density for column

                forc_t_c[c] = tbot_c
                forc_th_c[c] = thbot_c
                forc_q_c[c] = qbot_c
                forc_pbot_c[c] = pbot_c
                forc_rho_c[c] = rhos_c

                # adjust precipitation
                if self.nl_colm['DEF_DS_precipitation_adjust_scheme']== 'I':
                    delta_prc_c = forc_prc_g[g] * (forc_topo_c[c] - forc_topo_g[g]) / max_elev_c
                    forc_prc_c[c] = forc_prc_g[g] + delta_prc_c  # convective precipitation [mm/s]

                    delta_prl_c = forc_prl_g[g] * (forc_topo_c[c] - forc_topo_g[g]) / max_elev_c
                    forc_prl_c[c] = forc_prl_g[g] + delta_prl_c  # large scale precipitation [mm/s]

                elif self.nl_colm['DEF_DS_precipitation_adjust_scheme']== 'II':
                    delta_prc_c = forc_prc_g[g] * 2.0 * 0.27e-3 * (forc_topo_c[c] - forc_topo_g[g]) / \
                        (1.0 - 0.27e-3 * (forc_topo_c[c] - forc_topo_g[g]))
                    forc_prc_c[c] = forc_prc_g[g] + delta_prc_c  # large scale precipitation [mm/s]

                    delta_prl_c = forc_prl_g[g] * 2.0 * 0.27e-3 * (forc_topo_c[c] - forc_topo_g[g]) / \
                        (1.0 - 0.27e-3 * (forc_topo_c[c] - forc_topo_g[g]))
                    forc_prl_c[c] = forc_prl_g[g] + delta_prl_c  # large scale precipitation [mm/s]

                elif self.nl_colm['DEF_DS_precipitation_adjust_scheme']== 'III':
                    # Implement the logic for scheme III
                    pass

                if forc_prl_c[c] < 0:
                    print('negative prl', forc_prl_g[g], max_elev_c, forc_topo_c[c], forc_topo_g[g])

                if forc_prc_c[c] < 0:
                    print('negative prc', forc_prc_g[g], max_elev_c, forc_topo_c[c], forc_topo_g[g])

                forc_prc_c[c] = max(forc_prc_c[c], 0.0)
                forc_prl_c[c] = max(forc_prl_c[c], 0.0)

        forc_lwrad_c = self.downscale_longwave(self.num_gridcells, num_columns, self.begc, self.endc, glaciers, wt_column,
                        forc_topo_g, forc_t_g, forc_q_g, forc_pbot_g, forc_lwrad_g,
                        forc_topo_c, forc_t_c, forc_q_c, forc_pbot_c, forc_lwrad_c)
        
        return forc_t_c, forc_th_c, forc_q_c, forc_pbot_c, forc_rho_c, forc_prc_c, forc_prl_c, forc_lwrad_c, 
        forc_swrad_c, forc_us_c, forc_vs_c

    # PRIVATE MEMBER FUNCTIONS:
    def rhos(self, qbot, pbot, tbot):
        """
        -----------------------------------------------------------------------------
        DESCRIPTION:
        Compute atmospheric density (kg/m**3)
        -----------------------------------------------------------------------------

        ARGUMENTS:
        qbot : float
            Atmospheric specific humidity (kg/kg)
        pbot : float
            Atmospheric pressure (Pa)
        tbot : float
            Atmospheric temperature (K)

        RETURNS:
        float
            Atmospheric density (kg/m**3)
        """

        # LOCAL VARIABLES:
        wv_to_dair_weight_ratio = self.SHR_CONST_MWWV / self.SHR_CONST_MWDAIR

        egcm = qbot * pbot / (wv_to_dair_weight_ratio + (1.0 - wv_to_dair_weight_ratio) * qbot)
        rhos = (pbot - (1.0 - wv_to_dair_weight_ratio) * egcm) / (self.rair * tbot)

        return rhos

    def downscale_longwave(self, num_gridcells, num_columns, begc, endc, glaciers, wt_column, 
        forc_topo_g, forc_t_g, forc_q_g, forc_pbot_g, forc_lwrad_g, 
        forc_topo_c, forc_t_c, forc_q_c, forc_pbot_c, forc_lwrad_c):
        """
        Downscale longwave radiation.
        """
        # Local variables
        lapse_rate_longwave = 0.032
        longwave_downscaling_limit = 0.5
        # Define constants
        hsurf_c = 0.0  # column-level elevation (m)
        hsurf_g = 0.0  # gridcell-level elevation (m)
        num_gridcells = 10  # example value, should be set to the actual number of grid cells

        sum_lwrad_g = np.zeros(num_gridcells)  # weighted sum of column-level lwrad
        sum_wts_g = np.zeros(num_gridcells)  # sum of weights that contribute to sum_lwrad_g
        lwrad_norm_g = np.ones(num_gridcells)  # normalization factors
        newsum_lwrad_g = np.zeros(num_gridcells)  # weighted sum of column-level lwrad after normalization

        pv_g = 0.0  # water vapor pressure at grid cell (hPa)
        pv_c = 0.0  # water vapor pressure at column (hPa)
        emissivity_clearsky_g = 0.0  # clear-sky emissivity at grid cell
        emissivity_clearsky_c = 0.0  # clear-sky emissivity at grid column
        emissivity_allsky_g = 0.0  # all-sky emissivity at grid cell
        es_g = 0.0  # example value for es_g
        es_c = 0.0  # example value for es_c
        dum1 = 0.0  # example value for dum1
        dum2 = 0.0  # example value for dum2
        dum3 = 0.0  # example value for dum3

        # Example of initializing arrays with the size of num_gridcells
        # In practice, these arrays should be populated with actual data
        sum_lwrad_g = np.random.rand(num_gridcells)
        sum_wts_g = np.random.rand(num_gridcells)
        lwrad_norm_g = np.random.rand(num_gridcells)
        newsum_lwrad_g = np.random.rand(num_gridcells)

        # Initialize (needs to be done for ALL active columns)
        forc_lwrad_c = forc_lwrad_g
        for g in range (self.num_gridcells):
            for c in range(self.begc[g], self.endc[g]):
                hsurf_g = forc_topo_g[g]
                hsurf_c = forc_topo_c[c]

                if self.nl_colm['DEF_DS_longwave_adjust_scheme']== 'I':
                    # Fiddes and Gruber, 2014, TopoSCALE v.1.0: downscaling gridded climate data in
                    # complex terrain. Geosci. Model Dev., 7, 387-405. doi:10.5194/gmd-7-387-2014.
                    # Equation (1) (2) (3); here, the empirical parameters x1 and x2 are different from
                    # Konzelmann et al. (1994) where x1 = 0.443 and x2 = 8 (optimal for measurements on the Greenland ice sheet)

                    es_g, dum1, dum2, dum3 = CoLM_Qsadv.qsadv(forc_t_g[g], forc_pbot_g[g])
                    es_c, dum1, dum2, dum3 = CoLM_Qsadv.qsadv(forc_t_c[c], forc_pbot_c[c])
                    pv_g = forc_q_g[g] * es_g / 100.0
                    pv_c = forc_q_c[c] * es_c / 100.0

                    emissivity_clearsky_g = 0.23 + 0.43 * (pv_g / forc_t_g[g])**(1.0 / 5.7)
                    emissivity_clearsky_c = 0.23 + 0.43 * (pv_c / forc_t_c[c])**(1.0 / 5.7)
                    emissivity_allsky_g = forc_lwrad_g[g] / (5.67e-8 * forc_t_g[g]**4)

                    forc_lwrad_c[c] = (emissivity_clearsky_c + (emissivity_allsky_g - emissivity_clearsky_g)) * 5.67e-8 * forc_t_c[c]**4
                else:
                    # Longwave radiation is downscaled by assuming a linear decrease in downwelling longwave radiation
                    # with increasing elevation (0.032 W m-2 m-1, limited to 0.5 - 1.5 times the gridcell mean value,
                    # then normalized to conserve gridcell total energy) (Van Tricht et al., 2016, TC) Figure 6,
                    # doi:10.5194/tc-10-2379-2016

                    if glaciers:
                        forc_lwrad_c[c] = forc_lwrad_g[g] - lapse_rate_longwave * (hsurf_c - hsurf_g)
                    else:
                        forc_lwrad_c[c] = forc_lwrad_g[g] - 4.0 * forc_lwrad_g[g] / (0.5 * (forc_t_c[c] + forc_t_g[g])) * self.lapse_rate * (hsurf_c - hsurf_g)

                forc_lwrad_c[c] = min(forc_lwrad_c[c], forc_lwrad_g[g] * (1.0 + longwave_downscaling_limit))
                forc_lwrad_c[c] = max(forc_lwrad_c[c], forc_lwrad_g[g] * (1.0 - longwave_downscaling_limit))

                sum_lwrad_g[g] = sum_lwrad_g[g] + wt_column[c]*forc_lwrad_c[c]
                sum_wts_g[g] = sum_wts_g[g] + wt_column[c]
            if sum_wts_g[g] == 0.0:
                lwrad_norm_g[g] = 1.0
            elif sum_lwrad_g[g] == 0.0:
                lwrad_norm_g[g] = 1.0
            else:
                lwrad_norm_g[g] = forc_lwrad_g[g] / (sum_lwrad_g[g] / sum_wts_g[g])

            for c in range(self.begc[g], self.endc[g] + 1):
                forc_lwrad_c[c] *= lwrad_norm_g[g]
                newsum_lwrad_g[g] += wt_column[c] * forc_lwrad_c[c]

        for g in range(self.num_gridcells):
            if sum_wts_g[g] > 0.0:
                if abs((newsum_lwrad_g[g] / sum_wts_g[g]) - forc_lwrad_g[g]) > 1e-8:
                    print(f'g, newsum_lwrad_g, sum_wts_g, forc_lwrad_g: {g}, {newsum_lwrad_g[g]}, {sum_wts_g[g]}, {forc_lwrad_g[g]}')

        return forc_lwrad_c

# def downscale_wind(forc_us_g, forc_vs_g, cur_c, asp_type_c, slp_type_c, area_type_c):
#     """
#     Downscale wind speed.

#     Parameters:
#     forc_us_g : float
#         Eastward wind (m/s)
#     forc_vs_g : float
#         Northward wind (m/s)
#     cur_c : float
#         Curvature
#     asp_type_c : array_like
#         Topographic aspect of each character of one patch
#     slp_type_c : array_like
#         Topographic slope of each character of one patch
#     area_type_c : array_like
#         Area percentage of each character of one patch

#     Returns:
#     tuple of floats
#         Adjusted eastward wind (m/s) and adjusted northward wind (m/s)
#     """

#     # Local variables
#     num_type = len(asp_type_c)  # Assuming num_type is the length of asp_type_c
#     wind_dir = 0
#     ws_g = 0
#     wind_dir_slp = np.zeros(num_type)
#     ws_c_type = np.zeros(num_type)
#     ws_c = 0

#     # Calculate wind direction
#     if forc_us_g == 0.0:
#         wind_dir = 0
#     else:
#         wind_dir = math.atan(forc_vs_g / forc_us_g)

#     # Non-adjusted wind speed
#     ws_g = math.sqrt(forc_vs_g * forc_vs_g + forc_us_g * forc_us_g)

#     # Compute the slope in the direction of the wind
#     for i in range(num_type):
#         wind_dir_slp[i] = slp_type_c[i] * math.cos(wind_dir - asp_type_c[i] * math.pi / 180)

#     # Compute wind speed adjustment
#     for i in range(num_type):
#         ws_c_type[i] = ws_g * (1 + (0.58 * wind_dir_slp[i]) + 0.42 * cur_c) * area_type_c[i]

#     # Adjusted wind speed
#     ws_c = np.sum(ws_c_type)
#     forc_us_c = ws_c * math.cos(wind_dir)
#     forc_vs_c = ws_c * math.sin(wind_dir)

#     return forc_us_c, forc_vs_c

# # -----------------------------------------------------------------------------



# # -----------------------------------------------------------------------------

# def downscale_shortwave(forc_topo_g, forc_pbot_g, forc_swrad_g,
#                         forc_topo_c, forc_pbot_c, julian_day, coszen, cosazi, alb,
#                         slp_type_c, asp_type_c, svf_c, sf_lut_c, area_type_c):
#     """
#     Downscale shortwave radiation.

#     Rouf, T., Mei, Y., Maggioni, V., Houser, P., & Noonan, M. (2020). A Physically Based
#     Atmospheric Variables Downscaling Technique. Journal of Hydrometeorology,
#     21(1), 93–108. https://doi.org/10.1175/JHM-D-19-0109.1

#     Sisi Chen, Lu Li, Yongjiu Dai, et al. Exploring Topography Downscaling Methods for
#     Hyper-Resolution Land Surface Modeling. Authorea. April 25, 2024.
#     DOI: 10.22541/au.171403656.68476353/v1

#     Must be done after downscaling of surface pressure.
#     """
#     S = 1370
#     thr = 85 * math.pi / 180
#     shortwave_downscaling_limit = 0.5

#     # Local variables
#     zen_rad = math.acos(coszen)
#     azi_rad = math.acos(cosazi)
#     zen_deg = math.degrees(zen_rad)
#     azi_deg = math.degrees(azi_rad)

#     # Assuming num_type is the length of slp_type_c
#     num_type = len(slp_type_c)

#     idx_azi = int(azi_deg / 360 * num_azimuth)
#     idx_zen = int(zen_deg / 90 * num_zenith)
#     sf_c = sf_lut_c[idx_azi, idx_zen]

#     rt_R = 1 + 0.034 * math.cos(2 * math.pi * julian_day / 365)
#     toa_swrad = S * rt_R * coszen

#     clr_idx = forc_swrad_g / toa_swrad
#     diff_wgt = 1 - clr_idx

#     k_c = math.exp(-0.036 * forc_pbot_c / 100000)
#     opt_factor = 1 - math.exp(-k_c)

#     a_p = 1 / (1 + 1.68 * alb)

#     svf = svf_c
#     balb = alb

#     diff_swrad_g = diff_wgt * forc_swrad_g
#     beam_swrad_g = (1 - diff_wgt) * forc_swrad_g

#     diff_swrad_c = diff_swrad_g * svf
#     beam_swrad_c = beam_swrad_g * sf_c

#     refl_swrad_c = 0
#     beam_swrad_type = np.zeros(num_type)
#     refl_swrad_type = np.zeros(num_type)
#     tcf_type = np.zeros(num_type)
#     cosill_type = np.zeros(num_type)

#     for i in range(num_type):
#         cosill_type[i] = math.cos(zen_rad - slp_type_c[i] * math.pi / 180)
#         beam_swrad_type[i] = beam_swrad_c * area_type_c[i] * cosill_type[i]
#         refl_swrad_type[i] = beam_swrad_type[i] * balb

#     forc_swrad_c = diff_swrad_c + np.sum(beam_swrad_type) + np.sum(refl_swrad_type)
#     forc_swrad_c = min(forc_swrad_c, forc_swrad_g * (1 + shortwave_downscaling_limit))
#     forc_swrad_c = max(forc_swrad_c, forc_swrad_g * (1 - shortwave_downscaling_limit))

#     return forc_swrad_c

# def downscale_shortwave(coszen, cosazi, julian_day, forc_swrad_g, sf_lut_c, num_azimuth, num_zenith,
#                         forc_pbot_g, forc_pbot_c, thr, slp_type_c, asp_type_c, area_type_c,
#                         svf_c, alb, shortwave_downscaling_limit):

#     PI = math.pi
#     S = 1370  # Solar constant

#     #-----------------------------------------------------------------------------
#     # Calculate shadow factor according to sun zenith and azimuth angle
#     zen_rad = math.acos(coszen)
#     azi_rad = math.acos(cosazi)
#     zen_deg = zen_rad * 180 / PI  # Turn deg
#     azi_deg = azi_rad * 180.0 / PI  # Turn deg

#     idx_azi = int(azi_deg * num_azimuth / 360)
#     idx_zen = int(zen_deg * num_zenith / 90)
#     if idx_azi == 0: idx_azi = 1
#     if idx_zen == 0: idx_zen = 1

#     sf_c = sf_lut_c[idx_azi, idx_zen]
#     if sf_c < 0: sf_c = 0
#     if sf_c > 1: sf_c = 1

#     #-----------------------------------------------------------------------------
#     # Calculate top-of-atmosphere incident shortwave radiation
#     rt_R = 1 - 0.01672 * math.cos(0.9856 * (julian_day - 4))
#     toa_swrad = S * (rt_R ** 2) * coszen

#     #-----------------------------------------------------------------------------
#     # Calculate clearness index
#     if toa_swrad <= 0:
#         clr_idx = 1
#     else:
#         clr_idx = forc_swrad_g / toa_swrad
#     if clr_idx > 1: clr_idx = 1

#     #-----------------------------------------------------------------------------
#     # Calculate diffuse weight
#     # Ruiz-Arias, J. A., Alsamamra, H., Tovar-Pescador, J., & Pozo-Vázquez, D. (2010).
#     # Proposal of a regressive model for the hourly diffuse solar radiation under all sky
#     # conditions. Energy Conversion and Management, 51(5), 881–893.
#     # https://doi.org/10.1016/j.enconman.2009.11.024
#     diff_wgt = 0.952 - 1.041 * math.exp(-1 * math.exp(2.3 - 4.702 * clr_idx))
#     if diff_wgt > 1: diff_wgt = 1
#     if diff_wgt < 0: diff_wgt = 0

#     #-----------------------------------------------------------------------------
#     # Calculate diffuse and beam radiation
#     diff_swrad_g = forc_swrad_g * diff_wgt
#     beam_swrad_g = forc_swrad_g * (1 - diff_wgt)

#     #-----------------------------------------------------------------------------
#     # Calculate broadband attenuation coefficient [Pa^-1]
#     if clr_idx <= 0:
#         k_c = 0
#     else:
#         k_c = math.log(clr_idx) / forc_pbot_c

#     #-----------------------------------------------------------------------------
#     # Calculate factor to account for the difference of optical path length due to pressure difference
#     opt_factor = math.exp(k_c * (forc_pbot_g - forc_pbot_c))
#     # Control the boundary of optical path length
#     if opt_factor > 10000 or opt_factor < -10000: opt_factor = 0

#     #-----------------------------------------------------------------------------
#     # Adjust the zenith angle so that the range of zenith angles is less than 85°
#     if zen_rad > thr: zen_rad = thr

#     #-----------------------------------------------------------------------------
#     # Loop for four defined types to downscale beam radiation
#     num_type = len(slp_type_c)
#     cosill_type = np.zeros(num_type)
#     beam_swrad_type = np.zeros(num_type)
#     for i in range(num_type):
#         # Calculate the cosine of solar illumination angle, cos(θ),
#         # ranging between −1 and 1, indicates if the sun is below or
#         # above the local horizon (note that values lower than 0 are set to 0 indicate self shadow)
#         cosill_type[i] = math.cos(slp_type_c[i]) + math.tan(zen_rad) * math.sin(slp_type_c[i]) * math.cos(asp_type_c[i] * PI / 180)
#         if cosill_type[i] > 1: cosill_type[i] = 1
#         if cosill_type[i] < 0: cosill_type[i] = 0

#         # Downscaling beam radiation
#         a_p = area_type_c[i]
#         if a_p > 1.0: a_p = 1
#         if a_p < 0: a_p = 0
#         beam_swrad_type[i] = sf_c * cosill_type[i] * opt_factor * a_p * beam_swrad_g
#     beam_swrad_c = np.sum(beam_swrad_type)

#     #-----------------------------------------------------------------------------
#     # Downscaling diffuse radiation
#     svf = svf_c
#     if svf > 1: svf = 1
#     if svf < 0: svf = 0
#     diff_swrad_c = svf * diff_swrad_g

#     #-----------------------------------------------------------------------------
#     # Downscaling reflected radiation
#     balb = alb
#     tcf_type = np.zeros(num_type)
#     refl_swrad_type = np.zeros(num_type)
#     for i in range(num_type):
#         tcf_type[i] = (1 + math.cos(slp_type_c[i])) / 2 - svf
#         if tcf_type[i] < 0: tcf_type[i] = 0

#         if np.isnan(alb):
#             refl_swrad_type[i] = -1.0e36
#         else:
#             if balb < 0 or balb > 1: balb = 0
#             refl_swrad_type[i] = balb * tcf_type[i] * (beam_swrad_c * coszen + (1 - svf) * diff_swrad_c)
#     refl_swrad_c = np.sum(refl_swrad_type[refl_swrad_type != -1.0e36])
#     forc_swrad_c = beam_swrad_c + diff_swrad_c + refl_swrad_c

#     #-----------------------------------------------------------------------------
#     # Ensure that we don't depart too far from the atmospheric forcing value:
#     # negative values of swrad are certainly bad, but small positive values might
#     # also be bad. We can especially run into trouble due to the normalization: a
#     # small swrad value in one column can lead to a big normalization factor,
#     # leading to huge swrad values in other columns.
#     forc_swrad_c = min(forc_swrad_c, forc_swrad_g * (1.0 + shortwave_downscaling_limit))
#     forc_swrad_c = max(forc_swrad_c, forc_swrad_g * (1.0 - shortwave_downscaling_limit))
#     if forc_swrad_c == 0: forc_swrad_c = 0.0001

#     return forc_swrad_c
