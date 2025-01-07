import numpy as np
import math


class CoLM_Vars_Global:
    def __init__(self, namelist) -> None:  # 初始化上面四个变量
        self.namelist = namelist

        # GLCC USGS number of land cover category
        #     0: "Ocean",
        # 1: "Evergreen Needleleaf Forests",
        # 2: "Evergreen Broadleaf Forests",
        # 3: "Deciduous Needleleaf Forests",
        # 4: "Deciduous Broadleaf Forests",
        # 5: "Mixed Forests",
        # 6: "Closed Shrublands",
        # 7: "Open Shrublands",
        # 8: "Woody Savannas",
        # 9: "Savannas",
        # 10: "Grasslands",
        # 11: "Permanent Wetlands",
        # 12: "Croplands",
        # 13: "Urban and Built-up Lands",
        # 14: "Cropland/Natural Vegetation Mosaics",
        # 15: "Permanent Snow and Ice",
        # 16: "Barren",
        # 17: "Water Bodies"
        if self.namelist.nl_colm['LULC_USGS']:
            # GLCC USGS number of land cover category
            self.N_land_classification = 24
            # GLCC USGS land cover named index (could be added IF needed)
            self.URBAN = 1
            self.WATERBODY = 16
        else:
            # MODIS IGBP number of land cover category
            self.N_land_classification = 17
            # GLCC USGS land cover named index (could be added IF needed)
            self.WETLAND = 11
            self.CROPLAND = 12
            self.URBAN = 13
            self.GLACIERS = 15
            self.WATERBODY = 17

        # number of plant functional types
        if self.namelist.nl_colm['CROP']:
            self.N_PFT = 16
            self.N_CFT = 0
        else:
            self.N_PFT = 15
            self.N_CFT = 64
        # endif

        # vertical layer number
        self.maxsnl = -5
        self.nl_soil = 10
        self.nl_soil_full = 15

        self.nl_lake = 10
        self.nl_roof = 10
        self.nl_wall = 10
        self.nvegwcs = 4  # number of vegetation water potential nodes

        # bgc variables
        self.ndecomp_pools = 7
        self.ndecomp_transitions = 10
        self.npcropmin = 17
        self.zmin_bedrock = 0.4
        self.nbedrock = 10
        self.ndecomp_pools_vr = self.ndecomp_pools * self.nl_soil

        # crop index
        self.noveg = 0
        self.nbrdlf_evr_shrub = 9
        self.nbrdlf_dcd_brl_shrub = 11
        self.nc3crop = 15
        self.nc3irrig = 16
        self.ntmp_corn = 17  # temperate_corn
        self.nirrig_tmp_corn = 18  # irrigated temperate corn
        self.nswheat = 19  # spring wheat
        self.nirrig_swheat = 20  # irrigated spring wheat
        self.nwwheat = 21  # winter wheat
        self.nirrig_wwheat = 22  # irrigated winter wheat
        self.ntmp_soybean = 23  # temperate soybean
        self.nirrig_tmp_soybean = 24  # irrigated temperate soybean
        self.ncotton = 41  # cotton
        self.nirrig_cotton = 42  # irrigated cotton
        self.nrice = 61  # rice
        self.nirrig_rice = 62  # irrigated rice
        self.nsugarcane = 67  # sugarcane
        self.nirrig_sugarcane = 68  # irrigated sugarcane
        self.nmiscanthus = 71  # miscanthus
        self.nirrig_miscanthus = 72  # irrigated miscanthus
        self.nswitchgrass = 73  # switchgrass
        self.nirrig_switchgrass = 74  # irrigated switchgrass
        self.ntrp_corn = 75  # tropical corn
        self.nirrig_trp_corn = 76  # irrigated tropical corn
        self.ntrp_soybean = 77  # tropical soybean
        self.nirrig_trp_soybean = 78  # irrigated tropical soybean

        # crop index
        self.z_soi = np.zeros(self.nl_soil)  # node depth [m]
        self.z_soih = np.zeros(self.nl_soil)  # interface level below a zsoi level [m]
        self.zi_soi = np.zeros(self.nl_soil)  # interface level below a zsoi level [m]
        self.dz_soi = np.zeros(self.nl_soil)  # soil node thickness [m]

        self.spval = -1.e36  # missing value
        self.spval_i4 = -9999  # missing value
        self.pi = 4 * math.atan(1.)  # pi value
        self.deg2rad = 1.745329251994330e-2

        self.irrig_start_time = 21600  # local time of irrigation start
        self.irrig_max_depth = 1.  # max irrigation depth
        self.irrig_threshold_fraction = 1.  # irrigation thershold
        self.irrig_min_cphase = 1.  # crop phenology when begin irrigation
        self.irrig_max_cphase = 4.  # crop phenology when end irrigation
        self.irrig_time_per_day = 14400  # irrigation last time

        nsl = 0
        N_URB = 0  # urban type number

        if self.namelist.nl_colm['DEF_URBAN_type_scheme'] == 1:
            N_URB = 3
        elif self.namelist.nl_colm['DEF_URBAN_type_scheme'] == 2:
            N_URB = 10

        for i in range(self.nl_soil):
            self.z_soi[i] = 0.025 * (math.exp(0.5 * (i + 1 - 0.5)) - 1.)  # node depths

        self.dz_soi[0] = 0.5 * (self.z_soi[0] + self.z_soi[1])  # =zsoih(1)
        self.dz_soi[self.nl_soil - 1] = self.z_soi[self.nl_soil - 1] - self.z_soi[self.nl_soil - 1 - 1]

        for i in range(self.nl_soil - 2):
            # thickness between two interfaces
            self.dz_soi[i + 1] = 0.5 * (self.z_soi[i + 1 + 1] - self.z_soi[i + 1 - 1])

        self.z_soih[self.nl_soil - 1] = self.z_soi[self.nl_soil - 1] + 0.5 * self.dz_soi[self.nl_soil - 1]

        for i in range(self.nl_soil - 1):
            self.z_soih[i] = 0.5 * (self.z_soi[i] + self.z_soi[i + 1])  # interface depths

        self.zi_soi[0] = self.dz_soi[0]

        for i in range(self.nl_soil - 1):
            self.zi_soi[i + 1] = self.zi_soi[i + 1 - 1] + self.dz_soi[i + 1]
