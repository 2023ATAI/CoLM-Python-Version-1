import numpy as np


def soil_color_refl(L):

    """
    Original author  : Qinghliang Li,  Jinlong Zhu, 17/02/2024;
    software         : Guess the soil color (reflectance) based on the land cover types

    Args:
        L                     (ndarray):      land cover types (GLCC USGS/MODIS IGBP) [-]
    Returns:

        soil_s_v_alb          (float):         albedo of visible of the saturated soil [-]
        soil_d_v_alb           (float):        albedo of visible of the dry soil [-]
        soil_s_n_alb           (float):        albedo of near infrared of the saturated soil [-]
        soil_d_n_alb           (float):        albedo of near infrared of the dry soil [-]
    """
    # Define soil color reflectance arrays
    soil_s_v_refl = np.array([0.26, 0.24, 0.22, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
                              0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04])
    soil_d_v_refl = np.array([0.37, 0.35, 0.33, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25,
                              0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15])
    soil_s_n_refl = np.array([0.52, 0.48, 0.44, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28,
                              0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08])
    soil_d_n_refl = np.array([0.63, 0.59, 0.55, 0.51, 0.49, 0.47, 0.45, 0.43, 0.41, 0.39,
                              0.37, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19])
    if self.casename.nl_colm['LULC_USGS']:
        isc = {
            0: 1,  # Ocean
            1: 16,  # Urban and Built-Up Land
            2: 3,  # Dryland Cropland and Pasture
            3: 9,  # Irrigated Cropland and Pasture
            4: 10,  # Mixed Dryland/Irrigated Cropland and Pasture
            5: 4,  # Cropland/Grassland Mosaic
            6: 6,  # Cropland/Woodland Mosaic
            7: 2,  # Grassland
            8: 8,  # Shrubland
            9: 7,  # Mixed Shrubland/Grassland
            10: 5,  # Savanna
            11: 19,  # Deciduous Broadleaf Forest
            12: 20,  # Deciduous Needleleaf Forest
            13: 18,  # Evergreen Broadleaf Forest
            14: 17,  # Evergreen Needleleaf Forest
            15: 16,  # Mixed Forest
            16: 1,  # Water Bodies
            17: 15,  # Herbaceous Wetland
            18: 14,  # Wooded Wetland
            19: 1,  # Barren or Sparsely Vegetated
            20: 12,  # Herbaceous Tundra
            21: 12,  # Wooded Tundra
            22: 13,  # Mixed Tundra
            23: 11,  # Bare Ground Tundra
            24: 1  # Snow or Ice
        }
    if self.casename.nl_colm['LULC_IGBP']:
        isc = {
            0: 1,  # Ocean
            1: 17,  # Evergreen Needleleaf Forest
            2: 18,  # Evergreen Broadleaf Forest
            3: 20,  # Deciduous Needleleaf Forest
            4: 19,  # Deciduous Broadleaf Forest
            5: 13,  # Mixed Forest
            6: 9,  # Closed Shrublands
            7: 8,  # Open Shrublands
            8: 4,  # Woody Savannas
            9: 3,  # Savannas
            10: 2,  # Grasslands
            11: 15,  # Permanent Wetlands
            12: 6,  # Croplands
            13: 16,  # Urban and Built-Up
            14: 12,  # Cropland/Natural Vegetation Mosaic
            15: 1,  # Snow and Ice
            16: 1,  # Barren or Sparsely Vegetated
            17: 1  # Land Water Bodies
        }

    # Select soil albedo values based on indices
    soil_s_v_alb = soil_s_v_refl[isc[L]]
    soil_d_v_alb = soil_d_v_refl[isc[L]]
    soil_s_n_alb = soil_s_n_refl[isc[L]]
    soil_d_n_alb = soil_d_n_refl[isc[L]]

    return soil_s_v_alb, soil_d_v_alb, soil_s_n_alb, soil_d_n_alb