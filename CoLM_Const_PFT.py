import numpy as np

class Const_PFT(object):
    def __init__(self, nl_colm, N_PFT, N_CFT, nl_soil) -> None:
        self.nl_colm = nl_colm
        self.N_PFT = N_PFT
        self.N_CFT = N_CFT
        self.nl_soil = nl_soil
        # Plant Functional Type classification
        #---------------------------
        # 0  not vegetated
        # 1  needleleaf evergreen temperate tree
        # 2  needleleaf evergreen boreal tree
        # 3  needleleaf deciduous boreal tree
        # 4  broadleaf evergreen tropical tree
        # 5  broadleaf evergreen temperate tree
        # 6  broadleaf deciduous tropical tree
        # 7  broadleaf deciduous temperate tree
        # 8  broadleaf deciduous boreal tree
        # 9  broadleaf evergreen shrub
        #10  broadleaf deciduous temperate shrub
        #11  broadleaf deciduous boreal shrub
        #12  c3 arctic grass
        #13  c3 non-arctic grass
        #14  c4 grass
        #15  c3 crop
        #16  c3_irrigated
        #17  temperate_corn
        #18  irrigated_temperate_corn
        #19  spring_wheat
        #20  irrigated_spring_wheat
        #21  winter_wheat
        #22  irrigated_winter_wheat
        #23  temperate_soybean
        #24  irrigated_temperate_soybean
        #25  barley
        #26  irrigated_barley
        #27  winter_barley
        #28  irrigated_winter_barley
        #29  rye
        #30  irrigated_rye
        #31  winter_rye
        #32  irrigated_winter_rye
        #33  cassava
        #34  irrigated_cassava
        #35  citrus
        #36  irrigated_citrus
        #37  cocoa
        #38  irrigated_cocoa
        #39  coffee
        #40  irrigated_coffee
        #41  cotton
        #42  irrigated_cotton
        #43  datepalm
        #44  irrigated_datepalm
        #45  foddergrass
        #46  irrigated_foddergrass
        #47  grapes
        #48  irrigated_grapes
        #49  groundnuts
        #50  irrigated_groundnuts
        #51  millet
        #52  irrigated_millet
        #53  oilpalm
        #54  irrigated_oilpalm
        #55  potatoes
        #56  irrigated_potatoes
        #57  pulses
        #58  irrigated_pulses
        #59  rapeseed
        #60  irrigated_rapeseed
        #61  rice
        #62  irrigated_rice
        #63  sorghum
        #64  irrigated_sorghum
        #65  sugarbeet
        #66  irrigated_sugarbeet
        #67  sugarcane
        #68  irrigated_sugarcane
        #69  sunflower
        #70  irrigated_sunflower
        #71  miscanthus
        #72  irrigated_miscanthus
        #73  switchgrass
        #74  irrigated_switchgrass
        #75  tropical_corn
        #76  irrigated_tropical_corn
        #77  tropical_soybean
        #78  irrigated_tropical_soybean

        # canopy layer number

        self.canlay_p = None
        if nl_colm['CROP']:
            self.canlay_p = [0, 2, 2, 2, 2, 2, 2, 2
                , 2, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1, 1
                , 1, 1, 1, 1, 1, 1, 1
            ]
        else:
            self.canlay_p = [0, 2, 2, 2, 2, 2, 2, 2
                , 2, 1, 1, 1, 1, 1, 1, 1
            ]

        # canopy top height
        self.htop0_p = None
        if nl_colm['CROP']:
            self.htop0_p = [0.5,  17.0,  17.0,  14.0,  35.0,  35.0,  18.0,  20.0
                ,20.0,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5]
        else:
            self.htop0_p = [0.5,  17.0,  17.0,  14.0,  35.0,  35.0,  18.0,  20.0
                ,20.0,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5,   0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ]

        # canopy bottom height
        # 01/06/2020, yuan: adjust htop: grass/shrub -> 0, tree->1
        self.hbot0_p = None
        if nl_colm['CROP']:
            self.hbot0_p = [0.00,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0
                , 1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        else:
            self.hbot0_p = [0.00,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0
                , 1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]

        # defulat vegetation fractional cover
        self.fveg0_p = np.ones(self.N_PFT+self.N_CFT-1)

        # default stem area index
        self.sai0_p = None
        if nl_colm['CROP']:
            self.sai0_p = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
            , 2.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
            ]
        else:
            self.sai0_p = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
            , 2.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2]

        # ratio to calculate roughness length z0m
        self.z0mr_p = np.full(self.N_PFT+self.N_CFT-1,0.1)

        # ratio to calculate displacement height d
        self.displar_p = np.full(self.N_PFT+self.N_CFT-1,0.667)

        # inversesqrt leaf specific dimension size 4 cm
        self.sqrtdi_p = np.full(self.N_PFT+self.N_CFT-1, 5.0)

        # leaf angle distribution parameter
        self.chil_p = None
        if nl_colm['CROP']:
            self.chil_p = [-0.300,  0.010,  0.010,  0.010,  0.100,  0.100,  0.010,  0.250
                , 0.250,  0.010,  0.250,  0.250, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
                , -0.300, -0.300, -0.300, -0.300, -0.300, -0.300, -0.300
            ]
        else:
            self.chil_p = [-0.300,  0.010,  0.010,  0.010,  0.100,  0.100,  0.010,  0.250
                , 0.250,  0.010,  0.250,  0.250, -0.300, -0.300, -0.300, -0.300]

        # reflectance of green leaf in virsible band
        self.rhol_vis_p = None
        if  nl_colm['LULC_IGBP_PC']:
        # Leaf optical properties adapted from measured data (Dong et al., 2021)
            if nl_colm['CROP']:
                self.rhol_vis_p = [0.110,  0.070,  0.070,  0.070,  0.100,  0.110,  0.100,  0.100
                    , 0.100,  0.070,  0.100,  0.100,  0.110,  0.110,  0.110,  0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                ]
            else:
                self.rhol_vis_p = [0.110,  0.070,  0.070,  0.070,  0.100,  0.110,  0.100,  0.100
                    , 0.100,  0.070,  0.100,  0.100,  0.110,  0.110,  0.110,  0.110]
        else:
            if nl_colm['CROP']:
                self.rhol_vis_p = [0.110,  0.070,  0.070,  0.070,  0.100,  0.100,  0.100,  0.100
                , 0.100,  0.070,  0.100,  0.100,  0.110,  0.110,  0.110,  0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                    , 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110
                ]
            else:
                self.rhol_vis_p = [0.110,  0.070,  0.070,  0.070,  0.100,  0.100,  0.100,  0.100
                , 0.100,  0.070,  0.100,  0.100,  0.110,  0.110,  0.110,  0.110]

        # reflectance of dead leaf in virsible band
        self.rhos_vis_p = None
        if nl_colm['CROP']:
            self.rhos_vis_p = [0.310,  0.160,  0.160,  0.160,  0.160,  0.160,  0.160,  0.160
                , 0.160,  0.160,  0.160,  0.160,  0.310,  0.310,  0.310,  0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
                , 0.310, 0.310, 0.310, 0.310, 0.310, 0.310, 0.310
            ]
        else:
            self.rhos_vis_p = [0.310,  0.160,  0.160,  0.160,  0.160,  0.160,  0.160,  0.160
                , 0.160,  0.160,  0.160,  0.160,  0.310,  0.310,  0.310,  0.310]

        # reflectance of green leaf in near infrared band
        self.rhol_nir_p = None
        if nl_colm['LULC_IGBP_PC']:
        # Leaf optical properties adapted from measured data (Dong et al., 2021)
            if nl_colm['CROP']:
                self.rhol_nir_p = [0.350,  0.360,  0.370,  0.360,  0.450,  0.460,  0.450,  0.420
                , 0.450,  0.350,  0.450,  0.450,  0.350,  0.350,  0.350,  0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                ]
            else:
                self.rhol_nir_p = [0.350,  0.360,  0.370,  0.360,  0.450,  0.460,  0.450,  0.420
                , 0.450,  0.350,  0.450,  0.450,  0.350,  0.350,  0.350,  0.350]
        else:
            if nl_colm['CROP']:
                self.rhol_nir_p = [0.350,  0.350,  0.350,  0.350,  0.450,  0.450,  0.450,  0.450
                , 0.450,  0.350,  0.450,  0.450,  0.350,  0.350,  0.350,  0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                    , 0.350, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350
                ]
            else:
                self.rhol_nir_p = [0.350,  0.350,  0.350,  0.350,  0.450,  0.450,  0.450,  0.450
                , 0.450,  0.350,  0.450,  0.450,  0.350,  0.350,  0.350,  0.350]

        # reflectance of dead leaf in near infrared band
        self.rhos_nir_p = None
        if nl_colm['CROP']:
            self.rhos_nir_p = [0.530,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390
                , 0.390,  0.390,  0.390,  0.390,  0.530,  0.530,  0.530,  0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
                , 0.530, 0.530, 0.530, 0.530, 0.530, 0.530, 0.530
            ]
        else:
            self.rhos_nir_p = [0.530,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390,  0.390
                , 0.390,  0.390,  0.390,  0.390,  0.530,  0.530,  0.530,  0.530]

        # transmittance of green leaf in visible band
        self.taul_vis_p = None
        if nl_colm['LULC_IGBP_PC']:
        # Leaf optical properties adpated from measured data (Dong et al., 2021)
            if nl_colm['CROP']:
                self.taul_vis_p = [0.050,  0.050,  0.050,  0.050,  0.050,  0.060,  0.050,  0.060
                , 0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                ]
            else:
                self.taul_vis_p = [0.050,  0.050,  0.050,  0.050,  0.050,  0.060,  0.050,  0.060
                , 0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050]
        else:
            if nl_colm['CROP']:
                self.taul_vis_p = [0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050
                , 0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                    , 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050
                ]
            else:
                self.taul_vis_p = [0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050
                , 0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050,  0.050]

        # transmittance of dead leaf in visible band
        self.taus_vis_p = None
        if nl_colm['CROP']:
            self.taus_vis_p = [0.120,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001
                , 0.001,  0.001,  0.001,  0.001,  0.120,  0.120,  0.120,  0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
                , 0.120, 0.120, 0.120, 0.120, 0.120, 0.120, 0.120
            ]
        else:
            self.taus_vis_p = [0.120,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001
                , 0.001,  0.001,  0.001,  0.001,  0.120,  0.120,  0.120,  0.120]

        # transmittance of green leaf in near infrared band
        self.taul_nir_p = None
        if nl_colm['LULC_IGBP_PC']:
        # Leaf optical properties adapted from measured data (Dong et al., 2021)
            if nl_colm['CROP']:
                self.taul_nir_p = [0.340,  0.280,  0.290,  0.380,  0.250,  0.330,  0.250,  0.430
                , 0.400,  0.100,  0.250,  0.250,  0.340,  0.340,  0.340,  0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                ]
            else:
                self.taul_nir_p = [0.340,  0.280,  0.290,  0.380,  0.250,  0.330,  0.250,  0.430
                , 0.400,  0.100,  0.250,  0.250,  0.340,  0.340,  0.340,  0.340]
        else:
            if nl_colm['CROP']:
                self.taul_nir_p = [0.340,  0.100,  0.100,  0.100,  0.250,  0.250,  0.250,  0.250
                , 0.250,  0.100,  0.250,  0.250,  0.340,  0.340,  0.340,  0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                    , 0.340, 0.340, 0.340, 0.340, 0.340, 0.340, 0.340
                ]
            else:
                self.taul_nir_p = [0.340,  0.100,  0.100,  0.100,  0.250,  0.250,  0.250,  0.250
                , 0.250,  0.100,  0.250,  0.250,  0.340,  0.340,  0.340,  0.340]

        # transmittance of dead leaf in near infrared band
        self.taus_nir_p = None
        if nl_colm['CROP']:
            self.taus_nir_p = [0.250,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001
                , 0.001,  0.001,  0.001,  0.001,  0.250,  0.250,  0.250,  0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
                , 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250
            ]
        else:
            self.taus_nir_p = [0.250,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001
                , 0.001,  0.001,  0.001,  0.001,  0.250,  0.250,  0.250,  0.250]

        # maximum carboxylation rate at 25 C at canopy top
        # /06/03/2014/ based on Bonan et al., 2011 (Table 2)
        #real(r8), parameter :: vmax25_p(0:self.N_PFT+self.N_CFT-1)
        #   = (/ 52.0, 61.0, 54.0, 57.0, 72.0, 72.0, 52.0, 52.0
        #      , 52.0, 72.0, 52.0, 52.0, 52.0, 52.0, 52.0, 57.0
        # /07/27/2022/ based on Bonan et al., 2011 (Table 2, VmaxF(N))
        self.vmax25_p = None
        if nl_colm['CROP']:
            self.vmax25_p = [52.0, 55.0, 42.0, 29.0, 41.0, 51.0, 36.0, 30.0
                , 40.0, 36.0, 30.0, 19.0, 21.0, 26.0, 25.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
                , 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0
            ]* 0.3 * 1.e-6
        else:
            self.vmax25_p = [52.0, 55.0, 42.0, 29.0, 41.0, 51.0, 36.0, 30.0
                , 40.0, 36.0, 30.0, 19.0, 21.0, 26.0, 25.0, 57.0]

        # quantum efficiency
        self.effcon_p = None
        if nl_colm['CROP']:
            self.effcon_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
            ]
        else:
            self.effcon_p = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08
                , 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05, 0.08]

        # conductance-photosynthesis slope parameter
        self.g1_p = None
        if nl_colm['CROP']:
            self.g1_p = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
            ]
        else:
            self.g1_p = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                , 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

        # conductance-photosynthesis intercept
        self.g0_p = None
        if nl_colm['CROP']:
            self.g0_p = [100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100
            ]
        else:
            self.g0_p = [100, 100, 100, 100, 100, 100, 100, 100
                , 100, 100, 100, 100, 100, 100, 100, 100]

        # conductance-photosynthesis slope parameter
        self.gradm_p = None
        if nl_colm['CROP']:
            self.gradm_p = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 4.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
            ]
        else:
            self.gradm_p = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0
                , 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 4.0, 9.0]

        # conductance-photosynthesis intercept
        self.binter_p = None
        if nl_colm['CROP']:
            self.binter_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
            ]
        else:
            self.binter_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
                , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.01]

        # respiration fraction
        self.respcp_p = None
        if nl_colm['CROP']:
            self.respcp_p = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.025, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
            ]
        else:
            self.respcp_p = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
                , 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.025, 0.015]

        # slope of high temperature inhibition FUNCTION (s1)
        self.shti_p = np.full(self.N_PFT+self.N_CFT-1, 0.3)

        # slope of low temperature inhibition FUNCTION (s3)
        self.slti_p = np.full(self.N_PFT+self.N_CFT-1, 0.2)

        # temperature coefficient in gs-a model (s5)
        self.trda_p = np.full(self.N_PFT+self.N_CFT-1,1.3)

        # temperature coefficient in gs-a model (s6)
        self.trdm_p = np.full(self.N_PFT+self.N_CFT-1,328.0)

        # temperature coefficient in gs-a model (273.16+25)
        self.trop_p = np.full(self.N_PFT+self.N_CFT-1,298.0)

        # 1/2 point of high temperature inhibition FUNCTION (s2)
        self.hhti_p = None
        if nl_colm['CROP']:
            self.hhti_p = [308.0, 303.0, 303.0, 303.0, 313.0, 313.0, 311.0, 311.0
                ,311.0, 313.0, 313.0, 303.0, 303.0, 308.0, 313.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
                , 308.0, 308.0, 308.0, 308.0, 308.0, 308.0, 308.0
            ]
        else:
            self.hhti_p = [308.0, 303.0, 303.0, 303.0, 313.0, 313.0, 311.0, 311.0
                ,311.0, 313.0, 313.0, 303.0, 303.0, 308.0, 313.0, 308.0]

        # 1/2 point of low temperature inhibition FUNCTION (s4)
        self.hlti_p = None
        if nl_colm['CROP']:
            self.hlti_p = [281.0, 278.0, 278.0, 278.0, 288.0, 288.0, 283.0, 283.0
                ,283.0, 283.0, 283.0, 278.0, 278.0, 281.0, 288.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
                , 281.0, 281.0, 281.0, 281.0, 281.0, 281.0, 281.0
            ]
        else:
            self.hlti_p = [281.0, 278.0, 278.0, 278.0, 288.0, 288.0, 283.0, 283.0
                ,283.0, 283.0, 283.0, 278.0, 278.0, 281.0, 288.0, 281.0]

        # coefficient of leaf nitrogen allocation
        self.extkn_p = np.full(self.N_PFT+self.N_CFT-1, 0.5)

        self.rho_p = None
        self.tau_p = None
        if nl_colm['CROP']:
            self.rho_p = np.zeros((2,2,self.N_PFT-1))#leaf reflectance
            self.tau_p = np.zeros((2,2,self.N_PFT-1))#leaf transmittance
        else:
            self.rho_p = np.zeros((2,2,self.N_PFT+self.N_CFT-1))#leaf reflectance
            self.tau_p = np.zeros((2,2,self.N_PFT+self.N_CFT-1))#leaf transmittance


        # depth at 50% roots
        self.d50_p = None
        if nl_colm['CROP']:
            self.d50_p = [27.0,  21.0,  12.0,  12.0,  15.0,  23.0,  16.0,  23.0
                ,12.0,  23.5,  23.5,  23.5,   9.0,   7.0,  16.0,  22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
                , 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0
            ]
        else:
            self.d50_p = [27.0,  21.0,  12.0,  12.0,  15.0,  23.0,  16.0,  23.0
                ,12.0,  23.5,  23.5,  23.5,   9.0,   7.0,  16.0,  22.0]

        # coefficient of root profile
        self.beta_p = None
        if nl_colm['CROP']:
            self.beta_p = [-2.051, -1.835, -1.880, -1.880, -1.632, -1.757, -1.681, -1.757
            , -1.880, -1.623, -1.623, -1.623, -2.621, -1.176, -1.452, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
                , -1.796, -1.796, -1.796, -1.796, -1.796, -1.796, -1.796
            ]
        else:
            self.beta_p = [-2.051, -1.835, -1.880, -1.880, -1.632, -1.757, -1.681, -1.757
            , -1.880, -1.623, -1.623, -1.623, -2.621, -1.176, -1.452, -1.796]

        # woody (1) or grass (0)
        self.woody = None
        if nl_colm['CROP']:
            self.woody = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        else:
            self.woody = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 ]

        # Set the root distribution parameters of PFT
        self.roota = None
        if nl_colm['CROP']:
            self.roota = [0.0,   7.0,   7.0,   7.0,   7.0,   7.0,   6.0,   6.0
            ,  6.0,   7.0,   7.0,   7.0,  11.0,  11.0,  11.0,   6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            , 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0
            ]
        else:
            self.roota = [0.0,   7.0,   7.0,   7.0,   7.0,   7.0,   6.0,   6.0
            ,  6.0,   7.0,   7.0,   7.0,  11.0,  11.0,  11.0,   6.0]


        self.rootb = None
        if nl_colm['CROP']:
            self.rootb = [0.0,   2.0,   2.0,   2.0,   1.0,   1.0,   2.0,   2.0
                ,  2.0,   1.5,   1.5,   1.5,   2.0,   2.0,   2.0,   3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                , 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
            ]
        else:
            self.rootb = [0.0,   2.0,   2.0,   2.0,   1.0,   1.0,   2.0,   2.0
                ,  2.0,   1.5,   1.5,   1.5,   2.0,   2.0,   2.0,   3.0]

        #   bgc PFT constants

        self.grperc = np.full(self.N_PFT+self.N_CFT-1, 0.11)
        self.grpnow = np.full(self.N_PFT+self.N_CFT-1, 1.0)
        self.lf_flab = np.full(self.N_PFT+self.N_CFT-1, 0.25)
        self.lf_fcel = np.full(self.N_PFT+self.N_CFT-1, 0.5)
        self.lf_flig = np.full(self.N_PFT+self.N_CFT-1, 0.25)
        self.fr_flab = np.full(self.N_PFT+self.N_CFT-1, 0.25)
        self.fr_fcel = np.full(self.N_PFT+self.N_CFT-1, 0.5)
        self.fr_flig = np.full(self.N_PFT+self.N_CFT-1, 0.25)

        self.isshrub = None
        if nl_colm['CROP']:
            self.isshrub = [False, False, False, False, False, False, False, False
            , False, True,  True,  True,  False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isshrub = [False, False, False, False, False, False, False, False
            , False, True,  True,  True,  False, False, False, False]

        self.isgrass = None
        if nl_colm['CROP']:
            self.isgrass = [False, False, False, False, False, False, False, False
            , False, False, False, False, True,  True,  True,  False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isgrass = [False, False, False, False, False, False, False, False
            , False, False, False, False, True,  True,  True,  False ]

        self.isbetr = None
        if nl_colm['CROP']:
            self.isbetr = [False, False, False, False, True,  False, False, False
            , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isbetr = [False, False, False, False, True,  False, False, False
            , False, False, False, False, False, False, False, False]

        self.isbdtr = None
        if nl_colm['CROP']:
            self.isbdtr = [False, False, False, False, False, False, True,  False
            , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isbdtr = [False, False, False, False, False, False, True,  False
            , False, False, False, False, False, False, False, False]

        self.isevg = None
        if nl_colm['CROP']:
            self.isevg = [False, True,  True,  False, True,  True,  False, False
            , False, True,  False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isevg = [False, True,  True,  False, True,  True,  False, False
            , False, True,  False, False, False, False, False, False]

        self.issed = None
        if nl_colm['CROP']:
            self.issed = [False, False, False, True,  False, False, False, True
            , True,  False, False, True,  True,  False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.issed = [False, False, False, True,  False, False, False, True
            , True,  False, False, True,  True,  False, False, False]

        self.isstd = None
        if nl_colm['CROP']:
            self.isstd = [False, False, False, False, False, False, True,  False
            , False, False, True,  False, False, True,  True,  True
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isstd = [False, False, False, False, False, False, True,  False
            , False, False, True,  False, False, True,  True,  True]

        self.isbare = None
        if nl_colm['CROP']:
            self.isbare = [True,  False, False, False, False, False, False, False
        , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
        ]
        else:
            self.isbare = [True,  False, False, False, False, False, False, False
        , False, False, False, False, False, False, False, False]

        self.iscrop = None
        if nl_colm['CROP']:
            self.iscrop = [False, False, False, False, False, False, False, False
            , False, False, False, False, False, False, False, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True, True
                , True, True, True, True, True, True, True
            ]
        else:
            self.iscrop = [False, False, False, False, False, False, False, False
            , False, False, False, False, False, False, False, True]

        self.isnatveg = None
        if nl_colm['CROP']:
            self.isnatveg = [False, True,  True,  True,  True,  True,  True,  True
            , True,  True,  True,  True,  True,  True,  True,  False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False, False
                , False, False, False, False, False, False, False
            ]
        else:
            self.isnatveg = [False, True,  True,  True,  True,  True,  True,  True
            , True,  True,  True,  True,  True,  True,  True,  False]

        self.fsr_pft = None
        if nl_colm['CROP']:
            self.fsr_pft = [ 0.,   0.26,   0.26,   0.26,   0.25,   0.25,   0.25,   0.25
            ,  0.25,   0.28,   0.28,   0.28,   0.33,   0.33,   0.33,   0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
                , 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33
            ]
        else:
            self.fsr_pft = [ 0.,   0.26,   0.26,   0.26,   0.25,   0.25,   0.25,   0.25
            ,  0.25,   0.28,   0.28,   0.28,   0.33,   0.33,   0.33,   0.33]

        self.fd_pft = None
        if nl_colm['CROP']:
            self.fd_pft = [0.,     24.,     24.,     24.,     24.,     24.,     24.,     24.
            ,   24.,     24.,     24.,     24.,     24.,     24.,     24.,     24.
                , 24., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0.
            ]
        else:
            self.fd_pft = [0.,     24.,     24.,     24.,     24.,     24.,     24.,     24.
            ,   24.,     24.,     24.,     24.,     24.,     24.,     24.,     24.]

        self.leafcn = None
        if nl_colm['CROP']:
            self.leafcn = [             1.,              58.,              58., 25.8131130614352
        ,  29.603315571344,  29.603315571344, 23.4521575984991, 23.4521575984991
        , 23.4521575984991, 36.4166059723234, 23.2558139534884, 23.2558139534884
        , 28.0269058295964, 28.0269058295964, 35.3606789250354, 28.0269058295964 ]
        else:
            self.leafcn = [             1.,              58.,              58., 25.8131130614352
        ,  29.603315571344,  29.603315571344, 23.4521575984991, 23.4521575984991
        , 23.4521575984991, 36.4166059723234, 23.2558139534884, 23.2558139534884
        , 28.0269058295964, 28.0269058295964, 35.3606789250354, 28.0269058295964
                , 25., 25., 25., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 20.
                , 20., 20., 20., 25.
                , 25., 20., 20., 20.
                , 20., 20., 20., 25.
                , 25., 20., 20.
        ]

        self.frootcn = None
        if nl_colm['CROP']:
            self.frootcn = [ 1.,     42.,     42.,     42.,     42.,     42.,     42.,     42.
            ,   42.,     42.,     42.,     42.,     42.,     42.,     42.,     42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42., 42.
                , 42., 42., 42., 42., 42., 42., 42.
            ]
        else:
            self.frootcn = [ 1.,     42.,     42.,     42.,     42.,     42.,     42.,     42.
            ,   42.,     42.,     42.,     42.,     42.,     42.,     42.,     42.]

        self.livewdcn = None
        if nl_colm['CROP']:
            self.livewdcn = [ 1.,     50.,     50.,     50.,     50.,     50.,     50.,     50.
            ,   50.,     50.,     50.,     50.,      0.,      0.,      0.,      0.
                , 0., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50.
            ]
        else:
            self.livewdcn = [ 1.,     50.,     50.,     50.,     50.,     50.,     50.,     50.
            ,   50.,     50.,     50.,     50.,      0.,      0.,      0.,      0.]

        self.deadwdcn = None
        if nl_colm['CROP']:
            self.deadwdcn = [ 1.,    500.,    500.,    500.,    500.,    500.,    500.,    500.
            ,  500.,    500.,    500.,    500.,      0.,      0.,      0.,      0.
                , 0., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500., 500.
                , 500., 500., 500., 500., 500., 500., 500.
            ]
        else:
            self.deadwdcn = [ 1.,    500.,    500.,    500.,    500.,    500.,    500.,    500.
            ,  500.,    500.,    500.,    500.,      0.,      0.,      0.,      0.]

        self.graincn = None
        if nl_colm['CROP']:
            self.graincn = [-999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.
            , -999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.
                , -999., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50., 50.
                , 50., 50., 50., 50., 50., 50., 50.
            ]
        else:
            self.graincn = [-999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.
            , -999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.,   -999.]

        self.lflitcn = None
        if nl_colm['CROP']:
            self.lflitcn = [ 1.,     70.,     80.,     50.,     60.,     60.,     50.,     50.
            ,   50.,     60.,     50.,     50.,     50.,     50.,     50.,     50.
                , 50., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25., 25.
                , 25., 25., 25., 25., 25., 25., 25.
                        ]
        else:
            self.lflitcn = [ 1.,     70.,     80.,     50.,     60.,     60.,     50.,     50.
            ,   50.,     60.,     50.,     50.,     50.,     50.,     50.,     50.]


        self.leaf_long = None
        if nl_colm['CROP']:
            self.leaf_long = [         0., 3.30916666666667, 3.30916666666667, 0.506666666666667
        ,            1.4025,           1.4025, 0.48333333333333, 0.483333333333333
        , 0.483333333333333, 1.32333333333333,             0.39,              0.39
        , 0.320833333333333, 0.32083333333333,             0.14, 0.320833333333333
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1., 1.
                , 1., 1., 1.
            ]
        else:
            self.leaf_long = [         0., 3.30916666666667, 3.30916666666667, 0.506666666666667
        ,            1.4025,           1.4025, 0.48333333333333, 0.483333333333333
        , 0.483333333333333, 1.32333333333333,             0.39,              0.39
        , 0.320833333333333, 0.32083333333333,             0.14, 0.320833333333333]


        self.cc_leaf = None
        if nl_colm['CROP']:
            self.cc_leaf = [0.,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8
            ,    0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                    ]
        else:
            self.cc_leaf = [0.,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8
            ,    0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8]


        self.cc_lstem = None
        if nl_colm['CROP']:
            self.cc_lstem = [0.,     0.3,     0.3,     0.3,    0.27,    0.27,    0.27,    0.27
            ,   0.27,    0.35,    0.35,    0.35,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                    ]
        else:
            self.cc_lstem = [0.,     0.3,     0.3,     0.3,    0.27,    0.27,    0.27,    0.27
            ,   0.27,    0.35,    0.35,    0.35,     0.8,     0.8,     0.8,     0.8]

        self.cc_dstem = None
        if nl_colm['CROP']:
            self.cc_dstem = [ 0.,     0.3,     0.3,     0.3,    0.27,    0.27,    0.27,    0.27
            ,   0.27,    0.35,    0.35,    0.35,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                        ]
        else:
            self.cc_dstem = [ 0.,     0.3,     0.3,     0.3,    0.27,    0.27,    0.27,    0.27
            ,   0.27,    0.35,    0.35,    0.35,     0.8,     0.8,     0.8,     0.8]

        self.cc_other = None
        if nl_colm['CROP']:
            self.cc_other = [ 0.,     0.5,     0.5,     0.5,    0.45,    0.45,    0.45,    0.45
            ,   0.45,    0.55,    0.55,    0.55,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                        ]
        else:
            self.cc_other = [ 0.,     0.5,     0.5,     0.5,    0.45,    0.45,    0.45,    0.45
            ,   0.45,    0.55,    0.55,    0.55,     0.8,     0.8,     0.8,     0.8]

        self.fm_leaf = None
        if nl_colm['CROP']:
            self.fm_leaf = [ 0.,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8
            ,    0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                        ]
        else:
            self.fm_leaf = [ 0.,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8
            ,    0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8,     0.8]

        self.fm_lstem = None
        if nl_colm['CROP']:
            self.fm_lstem = [ 0.,     0.5,     0.5,     0.5,    0.45,    0.45,    0.35,    0.35
            ,   0.45,    0.55,    0.55,    0.55,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                        ]
        else:
            self.fm_lstem = [ 0.,     0.5,     0.5,     0.5,    0.45,    0.45,    0.35,    0.35
            ,   0.45,    0.55,    0.55,    0.55,     0.8,     0.8,     0.8,     0.8]

        self.fm_lroot = None
        if nl_colm['CROP']:
            self.fm_lroot = [0.,    0.15,    0.15,    0.15,    0.13,    0.13,     0.1,     0.1
            ,   0.13,    0.17,    0.17,    0.17,     0.2,     0.2,     0.2,     0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                        ]
        else:
            self.fm_lroot = [0.,    0.15,    0.15,    0.15,    0.13,    0.13,     0.1,     0.1
            ,   0.13,    0.17,    0.17,    0.17,     0.2,     0.2,     0.2,     0.2]

        self.fm_root = None
        if nl_colm['CROP']:
            self.fm_root = [ 0.,    0.15,    0.15,    0.15,    0.13,    0.13,     0.1,     0.1
            ,   0.13,    0.17,    0.17,    0.17,     0.2,     0.2,     0.2,     0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                        ]
        else:
            self.fm_root = [ 0.,    0.15,    0.15,    0.15,    0.13,    0.13,     0.1,     0.1
            ,   0.13,    0.17,    0.17,    0.17,     0.2,     0.2,     0.2,     0.2]

        self.fm_droot = None
        if nl_colm['CROP']:
            self.fm_droot = [ 0.,    0.15,    0.15,    0.15,    0.13,    0.13,     0.1,     0.1
            ,   0.13,    0.17,    0.17,    0.17,     0.2,     0.2,     0.2,     0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                , 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                        ]
        else:
            self.fm_droot = [ 0.,    0.15,    0.15,    0.15,    0.13,    0.13,     0.1,     0.1
            ,   0.13,    0.17,    0.17,    0.17,     0.2,     0.2,     0.2,     0.2]


        self.fm_other = None
        if nl_colm['CROP']:
            self.fm_other = [ 0.,     0.5,     0.5,     0.5,    0.45,    0.45,    0.35,    0.35
            ,   0.45,    0.55,    0.55,    0.55,     0.8,     0.8,     0.8,     0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8
                        ]
        else:
            self.fm_other = [ 0.,     0.5,     0.5,     0.5,    0.45,    0.45,    0.35,    0.35
            ,   0.45,    0.55,    0.55,    0.55,     0.8,     0.8,     0.8,     0.8]


        self.froot_leaf = None
        if nl_colm['CROP']:
            self.froot_leaf = [0.,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5
            ,    1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5
                , 1., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2., 2.
                , 2., 2., 2., 2., 2., 2., 2.
                        ]
        else:
            self.froot_leaf = [0.,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5
            ,    1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5]

        self.croot_stem = None
        if nl_colm['CROP']:
            self.croot_stem = [0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3
            ,    0.3,     0.3,     0.3,     0.3,      0.,      0.,      0.,      0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0.
                        ]
        else:
            self.croot_stem = [0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3
            ,    0.3,     0.3,     0.3,     0.3,      0.,      0.,      0.,      0.]

        self.stem_leaf = None
        if nl_colm['CROP']:
            self.stem_leaf = [0.,     2.3,     2.3,      1.,     2.3,     1.5,      1.,     2.3
        ,    2.3,     1.4,    0.24,    0.24,      0.,      0.,      0.,      0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0.
                        ]
        else:
            self.stem_leaf = [0.,     2.3,     2.3,      1.,     2.3,     1.5,      1.,     2.3
        ,    2.3,     1.4,    0.24,    0.24,      0.,      0.,      0.,      0.]

        self.flivewd = None
        if nl_colm['CROP']:
            self.flivewd = [0.,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1
            ,    0.1,     0.5,     0.5,     0.1,      0.,      0.,      0.,      0.
                , 0., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1.
                    ]
        else:
            self.flivewd = [0.,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1
            ,    0.1,     0.5,     0.5,     0.1,      0.,      0.,      0.,      0.]


        self.fcur2 = None
        if nl_colm['CROP']:
            self.fcur2 = [0.,      1.,      1.,      0.,      1.,      1.,      0.,      0.
            ,     0.,      1.,      0.,      0.,      0.,      0.,      0.,      0.
                , 0., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1., 1.
                , 1., 1., 1., 1., 1., 1., 1.
                    ]
        else:
            self.fcur2 = [0.,      1.,      1.,      0.,      1.,      1.,      0.,      0.
            ,     0.,      1.,      0.,      0.,      0.,      0.,      0.,      0.]

        self.dsladlai = None
        if nl_colm['CROP']:
            self.dsladlai = [0., 0.00125,   0.001,   0.003, 0.00122,  0.0015,  0.0027,  0.0027
            , 0.0027,      0.,      0.,      0.,      0.,      0.,      0.,      0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0.
                        ]
        else:
            self.dsladlai = [0., 0.00125,   0.001,   0.003, 0.00122,  0.0015,  0.0027,  0.0027
            , 0.0027,      0.,      0.,      0.,      0.,      0.,      0.,      0.]

        self.slatop = None
        if nl_colm['CROP']:
            self.slatop = [ 0.,    0.01,    0.01, 0.02018,   0.019,   0.019,  0.0308,  0.0308
            , 0.0308, 0.01798, 0.03072, 0.03072, 0.04024, 0.04024, 0.03846, 0.04024
                , 0.035, 0.05, 0.05, 0.035, 0.035, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.05, 0.05, 0.035, 0.035, 0.035
                , 0.035, 0.035, 0.035, 0.05, 0.05, 0.035, 0.035
                    ]
        else:
            self.slatop = [ 0.,    0.01,    0.01, 0.02018,   0.019,   0.019,  0.0308,  0.0308
            , 0.0308, 0.01798, 0.03072, 0.03072, 0.04024, 0.04024, 0.03846, 0.04024]

        #--- crop variables ---

        self.manunitro = None
        if nl_colm['CROP']:
            self.manunitro = [0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.
            ,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.
                , 0., 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020
                , 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020
                , 0.0020, 0., 0., 0., 0., 0., 0., 0.
                , 0., 0.0020, 0.0020, 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0.0020, 0.0020, 0.
                , 0., 0., 0., 0.0020, 0.0020, 0., 0., 0.
                , 0., 0., 0., 0.0020, 0.0020, 0.0020, 0.0020
                        ]
        else:
            self.manunitro = [0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.
            ,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0. ]

        self.lfemerg = None
        if nl_colm['CROP']:
            self.lfemerg = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.11, 0.11, 0.07, 0.07, 0.03, 0.03, 0.15
                , 0.15, 0.07, 0.07, 0.03, 0.03, 0.07, 0.07, 0.03
                , 0.03, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.07, 0.07, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 0.12, 0.12, -999.9
                , -999.9, -999.9, -999.9, 0.11, 0.11, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 0.11, 0.11, 0.15, 0.15
                    ]
        else:
            self.lfemerg = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.mxmat = None
        if nl_colm['CROP']:
            self.mxmat = [-999, -999, -999, -999, -999, -99 , -999, -999
            ,   -999, -999, -999, -999, -999, -999, -999, -999
                , -999, 150, 150, 150, 150, 270, 270, 150
                , 150, 150, 150, 270, 270, 150, 150, 270
                , 270, -999, -999, -999, -999, -999, -999, -999
                , -999, 150, 150, -999, -999, -999, -999, -999
                , -999, -999, -999, -999, -999, -999, -999, -999
                , -999, -999, -999, -999, -999, 150, 150, -999
                , -999, -999, -999, 300, 300, -999, -999, -999
                , -999, -999, -999, 150, 150, 150, 150
                    ]
        else:
            self.mxmat = [-999, -999, -999, -999, -999, -99 , -999, -999
            ,   -999, -999, -999, -999, -999, -999, -999, -999]

        self.grnfill = None
        if nl_colm['CROP']:
            self.grnfill = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.64, 0.64, 0.6, 0.6, 0.67, 0.67, 0.69
                , 0.69, 0.6, 0.6, 0.67, 0.67, 0.6, 0.6, 0.67
                , 0.67, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.6, 0.6, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 0.68, 0.68, -999.9
                , -999.9, -999.9, -999.9, 0.64, 0.64, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 0.64, 0.64, 0.69, 0.69
                    ]
        else:
            self.grnfill = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.baset = None
        if nl_colm['CROP']:
            self.baset = [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.
            ,   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.
                , 0., 8., 8., 0., 0., 0., 0., 10.
                , 10., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 10., 10., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 0., 0., 0.
                , 0., 0., 0., 0., 0., 10., 10., 0.
                , 0., 0., 0., 10., 10., 0., 0., 0.
                , 0., 0., 0., 8., 8., 10., 10.
                    ]
        else:
            self.baset = [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.
            ,   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ]

        self.astemf = None
        if nl_colm['CROP']:
            self.astemf = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.3
                , 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05
                , 0.05, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.3, 0.3, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 0.05, 0.05, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 0.0, 0.0, 0.3, 0.3
                    ]
        else:
            self.astemf = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9 ]

        self.arooti = None
        if nl_colm['CROP']:
            self.arooti = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.2
                , 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.1, 0.1, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 0.1, 0.1, -999.9
                , -999.9, -999.9, -999.9, 0.4, 0.4, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 0.4, 0.4, 0.2, 0.2
                    ]
        else:
            self.arooti = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.arootf = None
        if nl_colm['CROP']:
            self.arootf = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.2
                , 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                , 0.0, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.2, 0.2, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 0.0, 0.0, -999.9
                , -999.9, -999.9, -999.9, 0.05, 0.05, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 0.05, 0.05, 0.2, 0.2
                    ]
        else:
            self.arootf = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.fleafi = None
        if nl_colm['CROP']:
            self.fleafi = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9
                , 0.9, 0.85, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9
                , 0.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.85, 0.85, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 0.75, 0.75, -999.9
                , -999.9, -999.9, -999.9, 0.8, 0.8, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 0.8, 0.8, 0.85, 0.85
                    ]
        else:
            self.fleafi = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.bfact = None
        if nl_colm['CROP']:
            self.bfact = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                    ]
        else:
            self.bfact = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.declfact = None
        if nl_colm['CROP']:
            self.declfact = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                , 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05
                        ]
        else:
            self.declfact = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.allconss = None
        if nl_colm['CROP']:
            self.allconss = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 2., 2., 1., 1., 1., 1., 5.
                , 5., 1., 1., 1., 1., 1., 1., 1.
                , 1., -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 5., 5., -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 1., 1., -999.9
                , -999.9, -999.9, -999.9, 2., 2., -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 2., 2., 5., 5
                        ]
        else:
            self.allconss = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.allconsl = None
        if nl_colm['CROP']:
            self.allconsl = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 5., 5., 3., 3., 3., 3., 2.
                , 2., 3., 3., 3., 3., 3., 3., 3.
                , 3., -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 2., 2., -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 3., 3., -999.9
                , -999.9, -999.9, -999.9, 5., 5., -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 5., 5., 2., 2.
                        ]
        else:
            self.allconsl = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
            ,   -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]


        self.fleafcn = None
        if nl_colm['CROP']:
            self.fleafcn = [999., 999., 999., 999., 999., 999., 999., 999.
            ,   999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65., 65.
                , 65., 65., 65., 65., 65., 65., 65.
                    ]
        else:
            self.fleafcn = [999., 999., 999., 999., 999., 999., 999., 999.
            ,   999., 999., 999., 999., 999., 999., 999., 999.]

        self.fstemcn = None
        if nl_colm['CROP']:
            self.fstemcn = [999., 999., 999., 999., 999., 999., 999., 999.
            ,   999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 120., 120., 100., 100., 100., 100., 130.
                , 130., 100., 100., 100., 100., 100., 100., 100.
                , 100., 999., 999., 999., 999., 999., 999., 999.
                , 999., 130., 130., 999., 999., 999., 999., 999.
                , 999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 999., 999., 999., 999., 100., 100., 999.
                , 999., 999., 999., 120., 120., 999., 999., 999.
                , 999., 999., 999., 120., 120., 130., 130.
                    ]
        else:
            self.fstemcn = [999., 999., 999., 999., 999., 999., 999., 999.
            ,   999., 999., 999., 999., 999., 999., 999., 999.]

        self.ffrootcn = None
        if nl_colm['CROP']:
            self.ffrootcn = [999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 0., 0., 40., 40., 40., 40., 0.
                , 0., 40., 40., 40., 40., 40., 40., 40.
                , 40., 999., 999., 999., 999., 999., 999., 999.
                , 999., 0., 0., 999., 999., 999., 999., 999.
                , 999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 999., 999., 999., 999., 40., 40., 999.
                , 999., 999., 999., 0., 0., 999., 999., 999.
                , 999., 999., 999., 0., 0., 0., 0.
                        ]
        else:
            self.ffrootcn = [999., 999., 999., 999., 999., 999., 999., 999.
                , 999., 999., 999., 999., 999., 999., 999., 999. ]

        self.laimx = None
        if nl_colm['CROP']:
            self.laimx = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 5., 5., 7., 7., 7., 7., 6.
                , 6., 7., 7., 7., 7., 7., 7., 7.
                , 7., -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, 6., 6., -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, 7., 7., -999.9
                , -999.9, -999.9, -999.9, 5., 5., -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, 5., 5., 6., 6.
                    ]
        else:
            self.laimx = [-999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9
                , -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9, -999.9]

        self.mergetoclmpft = None
        if nl_colm['CROP']:# merge crop functional types
            self.mergetoclmpft = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ,  19, 20, 21, 22, 23, 24, 19, 20, 21, 22, 19, 20, 21, 22, 61, 62, 19, 20, 61 
            ,  62, 61, 62, 41, 42, 41, 42, 19, 20, 19, 20, 61, 62, 75, 76, 61, 62, 19, 20 
            ,  19, 20, 19, 20, 61, 62, 75, 76, 19, 20, 67, 68, 19, 20, 75, 76, 75, 76, 75 
            ,  76, 77, 78]
        #   end bgc variables

        # Plant Hydraulics Paramters
        self.kmax_sun_p = None
        if nl_colm['CROP']:
            self.kmax_sun_p = [   0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                            ]
        else:
            self.kmax_sun_p = [   0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007]

        self.kmax_sha_p = None
        if nl_colm['CROP']:
            self.kmax_sha_p = [ 0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                        ]
        else:
            self.kmax_sha_p = [ 0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007]

        self.kmax_xyl_p = None
        if nl_colm['CROP']:
            self.kmax_xyl_p = [ 0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                        ]
        else:
            self.kmax_xyl_p = [ 0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007]

        self.kmax_root_p = None
        if nl_colm['CROP']:
            self.kmax_root_p = [ 0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                , 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007, 1.e-007
                            ]
        else:
            self.kmax_root_p = [ 0.,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007
                ,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007,1.e-007]

        # water potential at 50% loss of sunlit leaf tissue conductance (mmH2O)
        self.psi50_sun_p = None
        if nl_colm['CROP']:
            self.psi50_sun_p = [-150000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000
                        ]
        else:
            self.psi50_sun_p = [-150000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000]

        # water potential at 50% loss of shaded leaf tissue conductance (mmH2O)
        self.psi50_sha_p = None
        if nl_colm['CROP']:
            self.psi50_sha_p = [-150000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000
                        ]
        else:
            self.psi50_sha_p = [-150000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000]

        # water potential at 50% loss of xylem tissue conductance (mmH2O)
        self.psi50_xyl_p = None
        if nl_colm['CROP']:
            self.psi50_xyl_p = [-200000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000
                        ]
        else:
            self.psi50_xyl_p = [-200000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000]

        # water potential at 50% loss of root tissue conductance (mmH2O)
        self.psi50_root_p = None
        if nl_colm['CROP']:
            self.psi50_root_p = [-200000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000, -340000
                , -340000, -340000, -340000, -340000, -340000, -340000, -340000
                            ]
        else:
            self.psi50_root_p = [-200000, -530000, -400000, -380000, -250000, -270000, -340000, -270000
                ,-200000, -400000, -390000, -390000, -340000, -340000, -340000, -340000]

        # shape-fitting parameter for vulnerability curve (-)
        self.ck_p = None
        if nl_colm['CROP']:
            self.ck_p = [0.,  3.95, 3.95,  3.95, 3.95,  3.95, 3.95, 3.95
                ,3.95,  3.95, 3.95,  3.95, 3.95,  3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                , 3.95, 3.95, 3.95, 3.95, 3.95, 3.95, 3.95
                    ]
        else:
            self.ck_p = [0.,  3.95, 3.95,  3.95, 3.95,  3.95, 3.95, 3.95
                ,3.95,  3.95, 3.95,  3.95, 3.95,  3.95, 3.95, 3.95]

        #end plant hydraulic parameters

            # irrigation parameter for irrigated crop
        self.irrig_crop = None
        if nl_colm['CROP']:
            self.irrig_crop = [False, False, False, False, False, False, False, False
                    , False, False, False, False, False, False, False, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True, False
                , True, False, True, False, True, False, True
                        ]
        else:
            self.irrig_crop = [False, False, False, False, False, False, False, False
                    , False, False, False, False, False, False, False, False]
            
        if self.nl_colm['CROP']:
            self.rootfr_p = np.zeros((nl_soil, self.N_PFT + self.N_CFT))
        else:
            self.rootfr_p = np.zeros((nl_soil, self.N_PFT))

    def init_pft_const(self, zi_soi):
            ROOTFR_SCHEME = 1
            
            self.rho_p[0, 0, :] = self.rhol_vis_p[:]
            self.rho_p[1, 0, :] = self.rhol_nir_p[:]
            self.rho_p[0, 1, :] = self.rhos_vis_p[:]
            self.rho_p[1, 1, :] = self.rhos_nir_p[:]

            self.tau_p[0, 0, :] = self.taul_vis_p[:]
            self.tau_p[1, 0, :] = self.taul_nir_p[:]
            self.tau_p[0, 1, :] = self.taus_vis_p[:]
            self.tau_p[1, 1, :] = self.taus_nir_p[:]

            if self.ROOTFR_SCHEME == 1:
                if self.nl_colm['CROP']:
                    for i in range(self.N_PFT + self.N_CFT):
                        self.rootfr_p[0, i] = 1.0 / (1.0 + (zi_soi[0] * 100.0 / self.d50_p[i]) ** self.beta_p[i])
                        self.rootfr_p[self.nl_soil - 1, i] = 1.0 - 1.0 / (1.0 + (zi_soi[self.nl_soil - 2] * 100.0 / self.d50_p[i]) ** self.beta_p[i])

                        for nsl in range(1, self.nl_soil - 1):
                            self.rootfr_p[nsl, i] = (1.0 / (1.0 + (zi_soi[nsl] * 100.0 / self.d50_p[i]) ** self.beta_p[i]) -
                                                1.0 / (1.0 + (zi_soi[nsl - 1] * 100.0 / self.d50_p[i]) ** self.beta_p[i]))
                else:
                    for i in range(self.N_PFT):
                        self.rootfr_p[0, i] = 1.0 / (1.0 + (zi_soi[0] * 100.0 / self.d50_p[i]) ** self.beta_p[i])
                        self.rootfr_p[self.nl_soil - 1, i] = 1.0 - 1.0 / (1.0 + (zi_soi[self.nl_soil - 2] * 100.0 / self.d50_p[i]) ** self.beta_p[i])

                        for nsl in range(1, self.nl_soil - 1):
                            self.rootfr_p[nsl, i] = (1.0 / (1.0 + (zi_soi[nsl] * 100.0 / self.d50_p[i]) ** self.beta_p[i]) -
                                                1.0 / (1.0 + (zi_soi[nsl - 1] * 100.0 / self.d50_p[i]) ** self.beta_p[i]))
            else:
                if self.nl_colm['CROP']:
                    for i in range(self.N_PFT + self.N_CFT):
                        self.rootfr_p[0, i] = 1.0 - 0.5 * (np.exp(-self.roota[i] * zi_soi[0]) + np.exp(-self.rootb[i] * zi_soi[0]))

                        self.rootfr_p[self.nl_soil - 1, i] = 0.5 * (np.exp(-self.roota[i] * zi_soi[self.nl_soil - 1]) + np.exp(-self.rootb[i] * zi_soi[self.nl_soil - 1]))

                        for nsl in range(1, self.nl_soil - 1):
                            self.rootfr_p[nsl, i] = 0.5 * (np.exp(-self.roota[i] * zi_soi[nsl - 1]) +
                                                    np.exp(-self.rootb[i] * zi_soi[nsl - 1]) -
                                                    np.exp(-self.roota[i] * zi_soi[nsl]) -
                                                    np.exp(-self.rootb[i] * zi_soi[nsl]))
                else:
                    for i in range(self.N_PFT):
                        self.rootfr_p[0, i] = 1.0 - 0.5 * (np.exp(-self.roota[i] * zi_soi[0]) + np.exp(-self.rootb[i] * zi_soi[0]))

                        self.rootfr_p[self.nl_soil - 1, i] = 0.5 * (np.exp(-self.roota[i] * zi_soi[self.nl_soil - 1]) + np.exp(-self.rootb[i] * zi_soi[self.nl_soil - 1]))

                        for nsl in range(1, self.nl_soil - 1):
                            self.rootfr_p[nsl, i] = 0.5 * (np.exp(-self.roota[i] * zi_soi[nsl - 1]) +
                                                    np.exp(-self.rootb[i] * zi_soi[nsl - 1]) -
                                                    np.exp(-self.roota[i] * zi_soi[nsl]) -
                                                    np.exp(-self.rootb[i] * zi_soi[nsl]))

            return self.rho_p, self.tau_p, self.rootfr_p
