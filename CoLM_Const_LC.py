import numpy as np
import math

class CoLM_Const_LC(object):
    def __init__(self,N_land_classification, colm, nl_soil, z_soih) -> None:
        DEF_USE_PLANTHYDRAULICS = colm['DEF_USE_PLANTHYDRAULICS']
        self.patchtypes_igbp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 4]  # Patch types for IGBP classification

        self.htop0_igbp = [17.0, 35.0, 17.0, 20.0, 20.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Top canopy height

        self.hbot0_igbp = [8.5, 1.0, 8.5, 11.5, 10.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.3, 0.01, 0.01, 0.01, 0.01]  # Bottom canopy height

        self.fveg0_igbp = np.full(N_land_classification,1.0)

        self.sai0_igbp = [2.0, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]  # Specific leaf area index

        self.z0mr_igbp = np.full(N_land_classification,0.1)

        self.displar_igbp = np.full(N_land_classification,0.667)

        self.sqrtdi_igbp = np.full(N_land_classification,5.0)

        self.chil_igbp = [
            0.010, 0.100, 0.010, 0.250, 0.125, 0.010, 0.010, 0.010,
            0.010, -0.300, 0.100, -0.300, 0.010, -0.300, 0.010, 0.010, 0.010
        ]  # Leaf angle distribution parameter

        self.rhol_vis_igbp = [
            0.070, 0.100, 0.070, 0.100, 0.070, 0.105, 0.105, 0.105,
            0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105
        ]  # Leaf reflectance (visible)

        self.rhos_vis_igbp = [
            0.160, 0.160, 0.160, 0.160, 0.160, 0.160, 0.160, 0.160,
            0.160, 0.360, 0.160, 0.360, 0.160, 0.360, 0.160, 0.160, 0.160
        ]  # Soil reflectance (visible)

        self.rhol_nir_igbp = [
            0.350, 0.450, 0.350, 0.450, 0.400, 0.450, 0.450, 0.580,
            0.580, 0.580, 0.450, 0.580, 0.450, 0.580, 0.450, 0.450, 0.580
        ]  # Leaf reflectance (NIR)

        self.rhos_nir_igbp = [
            0.390, 0.390, 0.390, 0.390, 0.390, 0.390, 0.390, 0.390,
            0.390, 0.580, 0.390, 0.580, 0.390, 0.580, 0.390, 0.390, 0.580
        ]  # Soil reflectance (NIR)

        # Transmittance of green leaf in the visible band
        self.taul_vis_igbp = [0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050,
                        0.050, 0.070, 0.050, 0.070, 0.050, 0.070, 0.050, 0.050,
                        0.050]

        # Transmittance of dead leaf in the visible band
        self.taus_vis_igbp = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                        0.001, 0.220, 0.001, 0.220, 0.001, 0.220, 0.001, 0.001,
                        0.001]

        # Transmittance of green leaf in the near infrared band
        self.taul_nir_igbp = [0.100, 0.250, 0.100, 0.250, 0.150, 0.250, 0.250, 0.250,
                        0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250,
                        0.250]

        # Transmittance of dead leaf in the near infrared band
        self.taus_nir_igbp = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                        0.001, 0.380, 0.001, 0.380, 0.001, 0.380, 0.001, 0.001,
                        0.001]

        # Maximum carboxylation rate at 25°C at canopy top
        self.vmax25_igbp = [54.0, 72.0, 57.0, 52.0, 52.0, 52.0, 52.0, 52.0,
                    52.0, 52.0, 52.0, 57.0, 100.0, 57.0, 52.0, 52.0,
                    52.0]

        # Quantum efficiency
        self.effcon_igbp = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
          0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
          0.08]

        # Conductance-photosynthesis slope parameter
        g1_igbp = np.full(N_land_classification,9.0)

        # Conductance-photosynthesis intercept
        g0_igbp = np.full(N_land_classification,100)

        # Conductance-photosynthesis slope parameter
        self.gradm_igbp = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
          9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
          9.0 ]

        # Conductance-photosynthesis intercept
        self.binter_igbp = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
          0.01]

        # Respiration fraction
        self.respcp_igbp = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
          0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
          0.015]

        # Slope of high temperature inhibition function (s1)
        self.shti_igbp = np.full(N_land_classification,0.3)

        # Slope of low temperature inhibition function (s3)
        self.slti_igbp = np.full(N_land_classification,0.2)

        # Temperature coefficient in gs-a model (s5)
        self.trda_igbp = np.full(N_land_classification,1.3)

        # Temperature coefficient in gs-a model (s6)
        self.trdm_igbp = np.full(N_land_classification,328.0)

        # Temperature coefficient in gs-a model (273.16+25)
        self.trop_igbp = np.full(N_land_classification,298.0)

        # 1/2 point of high temperature inhibition function (s2)
        self.hhti_igbp = [303.0, 313.0, 303.0, 311.0, 307.0, 308.0, 313.0, 313.0,
                    313.0, 308.0, 313.0, 308.0, 308.0, 308.0, 303.0, 313.0,
                    308.0]

        # 1/2 point of low temperature inhibition function (s4)
        self.hlti_igbp = [278.0, 288.0, 278.0, 283.0, 281.0, 281.0, 288.0, 288.0,
                    288.0, 281.0, 283.0, 281.0, 281.0, 281.0, 278.0, 288.0,
                    281.0]

        # Coefficient of leaf nitrogen allocation
        self.extkn_igbp = np.full(N_land_classification,0.5)

        # depth at 50% roots
        self.d50_igbp =[15.0,  15.0,  16.0,  16.0,  15.5,  19.0,  28.0,  18.5,
                    28.0,   9.0,   9.0,  22.0,  23.0,  22.0,   1.0,   9.0,
                    1.0 ]

        # coefficient of root profile
        self.beta_igbp  =[-1.623, -1.623, -1.681, -1.681, -1.652, -1.336, -1.909, -1.582,
                    -1.798, -1.359, -1.359, -1.796, -1.757, -1.796, -1.000, -2.261, 
                    -1.000 ]

        self.roota_igbp =[6.706,  7.344,  7.066,  5.990,  4.453,  6.326,  7.718,  7.604,
                8.235, 10.740, 10.740,  5.558,  5.558,  5.558, 10.740,  4.372,
                10.740 ]

        self.rootb_igbp =[ 2.175,  1.303,  1.953,  1.955,  1.631,  1.567,  1.262,  2.300,
                1.627,  2.608,  2.608,  2.614,  2.614,  2.614,  2.608,  0.978,
                2.608 ]

        nsl = 0

        # Plant Hydraulics Parameters
        self.kmax_sun0_igbp = [2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008]
        self.kmax_sha0_igbp = [2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008]
        self.kmax_xyl0_igbp = [2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008]
        self.kmax_root0_igbp = [2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
          2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008]
        self.psi50_sun0_igbp = [-465000.0, -260000.0, -380000.0, -270000.0, -330000.0, -393333.3,
                        -393333.3, -340000.0, -340000.0, -340000.0, -343636.4, -340000.0,
                        -150000.0, -343636.4, -150000.0, -150000.0, -150000.0]
        self.psi50_sha0_igbp = [-465000.0, -260000.0, -380000.0, -270000.0, -330000.0, -393333.3,
          -393333.3, -340000.0, -340000.0, -340000.0, -343636.4, -340000.0,
          -150000.0, -343636.4, -150000.0, -150000.0, -150000.0]
        self.psi50_xyl0_igbp = [-465000.0, -260000.0, -380000.0, -270000.0, -330000.0, -393333.3,
          -393333.3, -340000.0, -340000.0, -340000.0, -343636.4, -340000.0,
          -200000.0, -343636.4, -200000.0, -200000.0, -200000.0]
        self.psi50_root0_igbp = [-465000.0, -260000.0, -380000.0, -270000.0, -330000.0, -393333.3, 
          -393333.3, -340000.0, -340000.0, -340000.0, -343636.4, -340000.0,
          -200000.0, -343636.4, -200000.0, -200000.0, -200000.0]
        # Shape-fitting parameter for vulnerability curve (-)
        self.ck0_igbp = [3.95, 3.95, 3.95, 3.95, 3.95, 3.95,
          3.95, 3.95, 3.95, 3.95, 3.95, 3.95,
          3.95, 3.95, 0.  , 0.  , 0.]

        patchtypes_usgs = [1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 4,
          2, 2, 0, 0, 0, 0, 0, 3]

        htop0_usgs = [1.0,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,
          0.5,   0.5,  20.0,  17.0,  35.0,  17.0,  20.0,   0.5,
          0.5,  17.0,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5]

        hbot0_usgs = [0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
          0.0,   0.0,   1.0,   1.0,   1.0,   1.0,   1.0,   0.0,
          0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]

        fveg0_usgs = np.full(N_land_classification,1.0)

        sai0_usgs = [0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 1.0, 0.5,
         1.0, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0,
         0.2, 2.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]
        z0mr_usgs = np.full(N_land_classification,0.1)
        displar_usgs = np.full(N_land_classification,0.667)
        sqrtdi_usgs = np.full(N_land_classification,5.0)
        chil_usgs = [-0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300,  0.010,
           0.010, -0.300,  0.250,  0.010,  0.100,  0.010,  0.125, -0.300,
          -0.300,  0.100,  0.010, -0.300, -0.300, -0.300, -0.300, -0.300]
        rhol_vis_usgs = [0.105,  0.105,  0.105,  0.105,  0.105,  0.105,  0.105,  0.100,
          0.100,  0.105,  0.100,  0.070,  0.100,  0.070,  0.070,  0.105,
          0.105,  0.100,  0.100,  0.105,  0.105,  0.105,  0.105,  0.105]
        rhos_vis_usgs = [0.360,  0.360,  0.360,  0.360,  0.360,  0.360,  0.360,  0.160,
          0.160,  0.360,  0.160,  0.160,  0.160,  0.160,  0.160,  0.360,
          0.360,  0.160,  0.160,  0.360,  0.360,  0.360,  0.360,  0.360]
        rhol_nir_usgs = [0.580,  0.580,  0.580,  0.580,  0.580,  0.580,  0.580,  0.450,
          0.450,  0.580,  0.450,  0.350,  0.450,  0.350,  0.400,  0.580,
          0.580,  0.450,  0.450,  0.580,  0.580,  0.580,  0.580,  0.580]
        rhos_nir_usgs = [0.580,  0.580,  0.580,  0.580,  0.580,  0.580,  0.580,  0.390,
          0.390,  0.580,  0.390,  0.390,  0.390,  0.390,  0.390,  0.580,
          0.580,  0.390,  0.390,  0.580,  0.580,  0.580,  0.580,  0.580]
        taul_vis_usgs = [0.070,  0.070,  0.070,  0.070,  0.070,  0.070,  0.070,  0.070,
          0.070,  0.070,  0.050,  0.050,  0.050,  0.050,  0.050,  0.070,
          0.070,  0.050,  0.070,  0.070,  0.070,  0.070,  0.070,  0.070]
        taus_vis_usgs= [0.220,  0.220,  0.220,  0.220,  0.220,  0.220,  0.220,  0.001,
          0.001,  0.220,  0.001,  0.001,  0.001,  0.001,  0.001,  0.220,
          0.220,  0.001,  0.001,  0.220,  0.220,  0.220,  0.220,  0.220]
        taul_nir_usgs = [0.250,  0.250,  0.250,  0.250,  0.250,  0.250,  0.250,  0.250,
          0.250,  0.250,  0.250,  0.100,  0.250,  0.100,  0.150,  0.250,
          0.250,  0.250,  0.250,  0.250,  0.250,  0.250,  0.250,  0.250]
        taus_nir_usgs = [0.380,  0.380,  0.380,  0.380,  0.380,  0.380,  0.380,  0.001,
          0.001,  0.380,  0.001,  0.001,  0.001,  0.001,  0.001,  0.380,
          0.380,  0.001,  0.001,  0.380,  0.380,  0.380,  0.380,  0.380]
        vmax25_usgs = [100.0, 57.0, 57.0, 57.0, 52.0, 52.0, 52.0, 52.0,
           52.0, 52.0, 52.0, 57.0, 72.0, 54.0, 52.0, 57.0,
           52.0, 52.0, 52.0, 52.0, 52.0, 52.0, 52.0, 52.0]
        effcon_usgs = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
          0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
          0.08, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05]
        g1_usgs = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
          4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
          4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        g0_usgs = [100, 100, 100, 100, 100, 100, 100, 100,
          100, 100, 100, 100, 100, 100, 100, 100,
          100, 100, 100, 100, 100, 100, 100, 100]
        gradm_usgs = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
          9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
          9.0, 9.0, 9.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        binter_usgs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.01, 0.04, 0.04, 0.04, 0.04, 0.04]
        respcp_usgs = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
          0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
          0.015, 0.015, 0.015, 0.025, 0.025, 0.025, 0.025, 0.025]
        shti_usgs= np.full(N_land_classification,0.3)
        slti_usgs= np.full(N_land_classification,0.2)
        trda_usgs= np.full(N_land_classification,1.3)
        trdm_usgs= np.full(N_land_classification,328.0)
        trop_usgs= np.full(N_land_classification,298.0)
        hhti_usgs = [308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 313.0,
         313.0, 308.0, 311.0, 303.0, 313.0, 303.0, 307.0, 308.0,
         308.0, 313.0, 313.0, 313.0, 313.0, 313.0, 313.0, 308.0]
        hlti_usgs = [281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 283.0,
         283.0, 281.0, 283.0, 278.0, 288.0, 278.0, 281.0, 281.0,
         281.0, 288.0, 283.0, 288.0, 288.0, 288.0, 288.0, 281.0]
        d50_usgs = [23.0,  21.0,  23.0,  22.0,  15.7,  19.0,   9.3,  47.0,
         28.2,  21.7,  16.0,  16.0,  15.0,  15.0,  15.5,   1.0,
          9.3,  15.5,  27.0,   9.0,   9.0,   9.0,   9.0,   1.0]
        extkn_usgs = np.full(N_land_classification,0.5)
        beta_usgs = [-1.757, -1.835, -1.757, -1.796, -1.577, -1.738, -1.359, -3.245,
         -2.302, -1.654, -1.681, -1.681, -1.632, -1.632, -1.656, -1.000,
         -1.359, -1.656, -2.051, -2.621, -2.621, -2.621, -2.621, -1.000]

        # 定义roota_usgs数组
        roota_usgs = [
            5.558, 5.558, 5.558, 5.558, 8.149, 5.558, 10.740, 7.022,
            8.881, 7.920, 5.990, 7.066, 7.344, 6.706, 4.453, 10.740,
            10.740, 4.453, 8.992, 8.992, 8.992, 8.992, 4.372, 10.740
        ]

        # 定义rootb_usgs数组
        rootb_usgs = [
            2.614, 2.614, 2.614, 2.614, 2.611, 2.614, 2.608, 1.415,
            2.012, 1.964, 1.955, 1.953, 1.303, 2.175, 1.631, 2.608,
            2.608, 1.631, 8.992, 8.992, 8.992, 8.992, 0.978, 2.608
        ]

        # 定义kmax_sun0_usgs数组
        kmax_sun0_usgs = [
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 0., 2.e-008, 2.e-008,
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008
        ]

        # 定义kmax_sha0_usgs数组
        kmax_sha0_usgs = [
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 0., 2.e-008, 2.e-008,
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008
        ]

        # 定义kmax_xyl0_usgs数组
        kmax_xyl0_usgs = [
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 0., 2.e-008, 2.e-008,
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008
        ]

        # 定义kmax_root0_usgs数组
        kmax_root0_usgs = [
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008,
            2.e-008, 2.e-008, 2.e-008, 0., 2.e-008, 2.e-008,
            0., 2.e-008, 2.e-008, 2.e-008, 2.e-008, 2.e-008
        ]

        # 定义psi50_sun0_usgs数组
        psi50_sun0_usgs = [
            -150000.0, -340000.0, -340000.0, -340000.0, -340000.0, -343636.4,
            -340000.0, -393333.3, -366666.7, -340000.0, -270000.0, -380000.0,
            -260000.0, -465000.0, -330000.0, -150000.0, -340000.0, -347272.7,
            -150000.0, -340000.0, -342500.0, -341250.0, -150000.0, -150000.0
        ]

        # 定义psi50_sha0_usgs数组
        psi50_sha0_usgs = [
            -150000.0, -340000.0, -340000.0, -340000.0, -340000.0, -343636.4,
            -340000.0, -393333.3, -366666.7, -340000.0, -270000.0, -380000.0,
            -260000.0, -465000.0, -330000.0, -150000.0, -340000.0, -347272.7,
            -150000.0, -340000.0, -342500.0, -341250.0, -150000.0, -150000.0
        ]

        # 定义psi50_xyl0_usgs数组
        psi50_xyl0_usgs = [
            -200000.0, -340000.0, -340000.0, -340000.0, -340000.0, -343636.4,
            -340000.0, -393333.3, -366666.7, -340000.0, -270000.0, -380000.0,
            -260000.0, -465000.0, -330000.0, -200000.0, -340000.0, -347272.7,
            -200000.0, -340000.0, -342500.0, -341250.0, -200000.0, -200000.0
        ]

        # 定义psi50_root0_usgs数组
        psi50_root0_usgs = [
            -200000.0, -340000.0, -340000.0, -340000.0, -340000.0, -343636.4,
            -340000.0, -393333.3, -366666.7, -340000.0, -270000.0, -380000.0,
            -260000.0, -465000.0, -330000.0, -200000.0, -340000.0, -347272.7,
            -200000.0, -340000.0, -342500.0, -341250.0, -200000.0, -200000.0
        ]

        # 定义ck0_usgs数组
        ck0_usgs = [
            0., 3.95, 3.95, 3.95, 3.95, 3.95,
            3.95, 3.95, 3.95, 3.95, 3.95, 3.95,
            3.95, 3.95, 3.95, 0., 3.95, 3.95,
            0., 3.95, 3.95, 3.95, 0., 0.
        ]

        # 定义各种数组
        self.patchtypes = np.zeros(N_land_classification)
        self.htop0 = np.zeros(N_land_classification)
        self.hbot0 = np.zeros(N_land_classification)
        self.fveg0 = np.zeros(N_land_classification)
        self.sai0 = np.zeros(N_land_classification)
        self.chil = np.zeros(N_land_classification)
        self.z0mr = np.zeros(N_land_classification)
        self.displar = np.zeros(N_land_classification)
        self.sqrtdi = np.zeros(N_land_classification)

        self.vmax25 = np.zeros(N_land_classification)
        self.effcon = np.zeros(N_land_classification)
        self.g1 = np.zeros(N_land_classification)
        self.g0 = np.zeros(N_land_classification)
        self.gradm = np.zeros(N_land_classification)
        self.binter = np.zeros(N_land_classification)
        self.respcp = np.zeros(N_land_classification)
        self.shti = np.zeros(N_land_classification)
        self.slti = np.zeros(N_land_classification)
        self.trda = np.zeros(N_land_classification)
        self.trdm = np.zeros(N_land_classification)
        self.trop = np.zeros(N_land_classification)
        self.hhti = np.zeros(N_land_classification)
        self.hlti = np.zeros(N_land_classification)
        self.extkn = np.zeros(N_land_classification)

        self.d50 = np.zeros(N_land_classification)
        self.beta = np.zeros(N_land_classification)

        self.kmax_sun = np.zeros(N_land_classification)
        self.kmax_sha = np.zeros(N_land_classification)
        self.kmax_xyl = np.zeros(N_land_classification)
        self.kmax_root = np.zeros(N_land_classification)
        self.psi50_sun = np.zeros(N_land_classification)
        self.psi50_sha = np.zeros(N_land_classification)
        self.psi50_xyl = np.zeros(N_land_classification)
        self.psi50_root = np.zeros(N_land_classification)
        self.ck = np.zeros(N_land_classification)

        self.roota = np.zeros(N_land_classification)
        self.rootb = np.zeros(N_land_classification)

        self.rho = np.zeros((2, 2, N_land_classification))
        self.tau = np.zeros((2, 2, N_land_classification))

        # 定义ROOTFR_SCHEME
        ROOTFR_SCHEME = 1
        # fraction of roots in each soil layer
        self.rootfr = np.zeros((nl_soil, N_land_classification))

        if colm['LULC_USGS']:
            self.patchtypes = patchtypes_usgs
            self.htop0 = htop0_usgs
            self.hbot0 = hbot0_usgs
            self.fveg0 = fveg0_usgs
            self.sai0 = sai0_usgs
            self.z0mr = z0mr_usgs
            self.displar = displar_usgs
            self.sqrtdi = sqrtdi_usgs
            self.chil = chil_usgs
            self.vmax25 = [val * 1.e-6 for val in vmax25_usgs]
            self.effcon = effcon_usgs
            self.g1 = g1_usgs
            self.g0 = g0_usgs
            self.gradm = gradm_usgs
            self.binter = binter_usgs
            self.respcp = respcp_usgs
            self.shti = shti_usgs
            self.slti = slti_usgs
            self.trda = trda_usgs
            self.trdm = trdm_usgs
            self.trop = trop_usgs
            self.hhti = hhti_usgs
            self.hlti = hlti_usgs
            self.extkn = extkn_usgs
            self.d50 = d50_usgs
            self.beta = beta_usgs
            if colm['DEF_USE_PLANTHYDRAULICS']:
                self.kmax_sun = kmax_sun0_usgs
                self.kmax_sha = kmax_sha0_usgs
                self.kmax_xyl = kmax_xyl0_usgs
                self.kmax_root = kmax_root0_usgs
                self.psi50_sun = psi50_sun0_usgs
                self.psi50_sha = psi50_sha0_usgs
                self.psi50_xyl = psi50_xyl0_usgs
                self.psi50_root = psi50_root0_usgs
                self.ck = ck0_usgs
            self.roota = roota_usgs
            self.rootb = rootb_usgs

            self.rho[0, 0, :] = rhol_vis_usgs
            self.rho[1, 0, :] = rhol_nir_usgs
            self.rho[0, 1, :] = rhol_vis_usgs
            self.rho[1, 1, :] = rhol_nir_usgs
            self.tau[0, 0, :] = rhol_vis_usgs
            self.tau[1, 0, :] = rhol_nir_usgs
            self.tau[0, 1, :] = rhol_vis_usgs
            self.tau[1, 1, :] = rhol_nir_usgs
        else:
            self.patchtypes = self.patchtypes_igbp
            self.htop0 = self.htop0_igbp
            self.hbot0 = self.hbot0_igbp
            self.fveg0 = self.fveg0_igbp
            self.sai0 = self.sai0_igbp
            self.z0mr = self.z0mr_igbp
            self.displar = self.displar_igbp
            self.sqrtdi = self.sqrtdi_igbp
            self.chil = self.chil_igbp
            # vmax25      = vmax25_igbp      * 1.e-6
            self.vmax25 = [val * 1.e-6 for val in self.vmax25_igbp]
            self.g1 = g1_igbp
            self.g0 = g0_igbp
            self.effcon = self.effcon_igbp
            self.gradm = self.gradm_igbp
            self.binter = self.binter_igbp
            self.respcp = self.respcp_igbp
            self.shti = self.shti_igbp
            self.slti = self.slti_igbp
            self.trda = self.trda_igbp
            self.trdm = self.trdm_igbp
            self.trop = self.trop_igbp
            self.hhti = self.hhti_igbp
            self.hlti = self.hlti_igbp
            self.extkn = self.extkn_igbp
            self.d50 = self.d50_igbp
            self.beta = self.beta_igbp

            if colm['DEF_USE_PLANTHYDRAULICS']:
                self.kmax_sun = self.kmax_sun0_igbp
                self.kmax_sha = self.kmax_sha0_igbp
                self.kmax_xyl = self.kmax_xyl0_igbp
                self.kmax_root = self.kmax_root0_igbp
                self.psi50_sun = self.psi50_sun0_igbp
                self.psi50_sha = self.psi50_sha0_igbp
                self.psi50_xyl = self.psi50_xyl0_igbp
                self.psi50_root = self.psi50_root0_igbp
                self.ck = self.ck0_igbp

            self.roota = self.roota_igbp
            self.rootb = self.rootb_igbp

            self.rho[0, 0, :] = self.rhol_vis_igbp
            self.rho[1, 0, :] = self.rhol_nir_igbp
            self.rho[0, 1, :] = self.rhos_vis_igbp
            self.rho[1, 1, :] = self.rhos_nir_igbp
            self.tau[0, 0, :] = self.taul_vis_igbp
            self.tau[1, 0, :] = self.taul_nir_igbp
            self.tau[0, 1, :] = self.taus_vis_igbp
            self.tau[1, 1, :] = self.taus_nir_igbp

        # ----------------------------------------------------------
        # The definition of global root distribution is based on
        # Schenk and Jackson, 2002: The Global Biogeography of Roots.
        # Ecological Monagraph 72(3): 311-328.
        # ----------------------------------------------------------
        if ROOTFR_SCHEME == 1:
            for i in range(N_land_classification):
                self.rootfr[0,i]=1./(1.+(z_soih[0]*100./self.d50[i])**self.beta[i])
                self.rootfr[nl_soil-1,i]=1.-1./(1.+(z_soih[nl_soil-1-1]*100./self.d50[i])**self.beta[i])

                for nsl in range (nl_soil-2):
                    self.rootfr[nsl+1,i]=1./(1.+(z_soih[nsl+1]*100./self.d50[i])**self.beta[i])-1./(1.+(z_soih[nsl+1-1]*100./self.d50[i])**self.beta[i])
        else:
            for i in range(N_land_classification):
                self.rootfr[0,i] = 1. - 0.5*( 
                    math.exp(-self.roota[i] * z_soih[0]) 
                    + math.exp(-self.rootb[i] * z_soih[0]) )

                self.rootfr[nl_soil-1,i] = 0.5*( 
                        math.exp(-self.roota[i] * z_soih[nl_soil]) 
                    + math.exp(-self.rootb[i] * z_soih[nl_soil]) )
                
                for nsl in range (nl_soil-2):
                    self.rootfr[nsl+1,i] = 0.5*( 
                        math.exp(-self.roota[i] * z_soih[nsl+1-1]) 
                        + math.exp(-self.rootb[i] * z_soih[nsl+1-1]) 
                        - math.exp(-self.roota[i] * z_soih[nsl+1]) 
                        - math.exp(-self.rootb[i] * z_soih[nsl+1]) )