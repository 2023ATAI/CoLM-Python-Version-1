from CoLM_NetCDFSerial import NetCDFFile
import CoLM_Utils
import math
import sys
import os


class CoLM_SingleSrfdata(object):
    def __init__(self, config_namelist, var_global, mpi) -> None:
        self.SITE_lon_location = 0.
        self.SITE_lat_location = 0.
        self.SITE_landtype = 1
        self.SITE_lakedepth = 1.
        self.SITE_dbedrock = 0.
        self.SITE_topography = 0.
        self.namelist = config_namelist
        self.vars_global = var_global
        self.mpi = mpi
        self.netFile = NetCDFFile(mpi)

    def read_surface_data_single(self, path_file, mksrfdata):
        self.SITE_lat_location = self.netFile.ncio_read_serial(path_file, 'latitude')
        self.SITE_lon_location = self.netFile.ncio_read_serial(path_file, 'longitude')

        if self.namelist['LULC_USGS']:
            self.SITE_landtype = self.netFile.ncio_read_serial(path_file, 'USGS_classification')
        else:
            self.SITE_landtype = self.netFile.ncio_read_serial(path_file, 'IGBP_classification')

        self.SITE_lon_location = CoLM_Utils.normalize_longitude(self.SITE_lon_location)

        self.namelist['DEF_domain'].edges = math.floor(self.SITE_lat_location)
        self.namelist['DEF_domain'].edgen = self.namelist['DEF_domain'].edges + 1.0
        self.namelist['DEF_domain'].edgew = math.floor(self.SITE_lon_location)
        self.namelist['DEF_domain'].edgee = self.namelist['DEF_domain'].edgew + 1.0

        # if not isgreenwich: #CoLM_TimeManager
        #    LocalLongitude = self.SITE_lon_location

        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            if (not mksrfdata) or self.namelist['USE_SITE_pctpfts']:
                self.SITE_pfttyp = self.netFile.ncio_read_serial(path_file, 'pfttyp')
            # otherwise, retrieve from database by MOD_LandPFT.F90

        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            if (not mksrfdata) or self.namelist['USE_SITE_pctpfts']:
                self.SITE_pctpfts = self.netFile.ncio_read_serial(path_file, 'pctpfts')
            # otherwise, retrieve from database by Aggregation_PercentagesPFT.F90

        if self.namelist['CROP']:
            if (not mksrfdata) or self.namelist['USE_SITE_pctcrop']:
                if self.SITE_landtype == self.vars_global.CROPLAND:
                    self.SITE_croptyp = self.netFile.ncio_read_serial(path_file, 'croptyp', )
                    self.SITE_pctcrop = self.netFile.ncio_read_serial(path_file, 'pctcrop', )
                # otherwise, retrieve from database by MOD_LandPatch.F90

        if (not mksrfdata) or self.namelist['USE_SITE_htop']:
            # otherwise, retrieve from database by Aggregation_ForestHeight.F90
            if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
                self.SITE_htop_pfts = self.netFile.ncio_read_serial(path_file, 'canopy_height_pfts')
            else:
                self.SITE_htop = self.netFile.ncio_read_serial(path_file, 'canopy_height')

        if (not mksrfdata) or self.namelist['USE_SITE_LAI']:
            # otherwise, retrieve from database by Aggregation_LAI.F90
            self.SITE_LAI_year = self.netFile.ncio_read_serial(path_file, 'LAI_year')
            if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
                if self.namelist['DEF_LAI_MONTHLY']:
                    self.SITE_LAI_pfts_monthly = self.netFile.ncio_read_serial(path_file, 'LAI_pfts_monthly')
                    self.SITE_SAI_pfts_monthly = self.netFile.ncio_read_serial(path_file, 'SAI_pfts_monthly')
            else:
                if self.namelist['DEF_LAI_MONTHLY']:
                    self.SITE_LAI_monthly = self.netFile.ncio_read_serial(path_file, 'LAI_monthly')
                    self.SITE_SAI_monthly = self.netFile.ncio_read_serial(mksrfdata, 'SAI_monthly')
                else:
                    self.SITE_LAI_8day = self.netFile.ncio_read_serial(path_file, 'LAI_8day')

        if (not mksrfdata) or self.namelist['USE_SITE_lakedepth']:
            # otherwise, retrieve from database by Aggregation_LakeDepth.F90
            self.SITE_lakedepth = self.netFile.ncio_read_serial(path_file, 'lakedepth')

        if (not mksrfdata) or self.namelist['USE_SITE_soilreflectance']:
            # otherwise, retrieve from database by Aggregation_SoilBrightness.F90
            self.SITE_soil_s_v_alb = self.netFile.ncio_read_serial(path_file, 'soil_s_v_alb')
            self.SITE_soil_d_v_alb = self.netFile.ncio_read_serial(path_file, 'soil_d_v_alb')
            self.SITE_soil_s_n_alb = self.netFile.ncio_read_serial(path_file, 'soil_s_n_alb')
            self.SITE_soil_d_n_alb = self.netFile.ncio_read_serial(path_file, 'soil_d_n_alb')

        if (not mksrfdata) or self.namelist['USE_SITE_soilparameters']:
            # otherwise, retrieve from database by Aggregation_SoilParameters.F90
            self.SITE_soil_vf_quartz_mineral = self.netFile.ncio_read_serial(path_file, 'soil_vf_quartz_mineral')

            self.SITE_soil_vf_gravels = self.netFile.ncio_read_serial(path_file, 'soil_vf_gravels')
            self.SITE_soil_vf_sand = self.netFile.ncio_read_serial(path_file, 'soil_vf_sand')
            self.SITE_soil_vf_om = self.netFile.ncio_read_serial(path_file, 'soil_vf_om')
            self.SITE_soil_wf_gravels = self.netFile.ncio_read_serial(path_file, 'soil_wf_gravels')
            self.SITE_soil_wf_sand = self.netFile.ncio_read_serial(path_file, 'soil_wf_sand')
            self.SITE_soil_OM_density = self.netFile.ncio_read_serial(path_file, 'soil_OM_density')
            self.SITE_soil_BD_all = self.netFile.ncio_read_serial(path_file, 'soil_BD_all')
            self.SITE_soil_theta_s = self.netFile.ncio_read_serial(path_file, 'soil_theta_s')
            self.SITE_soil_k_s = self.netFile.ncio_read_serial(path_file, 'soil_k_s')
            self.SITE_soil_csol = self.netFile.ncio_read_serial(path_file, 'soil_csol')
            self.SITE_soil_tksatu = self.netFile.ncio_read_serial(path_file, 'soil_tksatu')
            self.SITE_soil_tksatf = self.netFile.ncio_read_serial(path_file, 'soil_tksatf')
            self.SITE_soil_tkdry = self.netFile.ncio_read_serial(path_file, 'soil_tkdry')
            self.SITE_soil_k_solids = self.netFile.ncio_read_serial(path_file, 'soil_k_solids')
            self.SITE_soil_psi_s = self.netFile.ncio_read_serial(path_file, 'soil_psi_s')
            self.SITE_soil_lambda = self.netFile.ncio_read_serial(path_file, 'soil_lambda')
            if self.namelist['vanGenuchten_Mualem_SOIL_MODEL']:
                self.SITE_soil_theta_r = self.netFile.ncio_read_serial(path_file, 'soil_theta_r')
                self.SITE_soil_alpha_vgm = self.netFile.ncio_read_serial(path_file, 'soil_alpha_vgm')
                self.SITE_soil_L_vgm = self.netFile.ncio_read_serial(path_file, 'soil_L_vgm')
                self.SITE_soil_n_vgm = self.netFile.ncio_read_serial(path_file, 'soil_n_vgm')
            self.SITE_soil_BA_alpha = self.netFile.ncio_read_serial(path_file, 'soil_BA_alpha')
            self.SITE_soil_BA_beta = self.netFile.ncio_read_serial(path_file, 'soil_BA_beta')

        if self.namelist['DEF_USE_BEDROCK']:
            if (not mksrfdata) or self.namelist['USE_SITE_dbedrock']:
                # otherwise, retrieve from database by Aggregation_DBedrock.F90
                self.SITE_dbedrock = self.netFile.ncio_read_serial(path_file, 'depth_to_bedrock')

        if (not mksrfdata) or self.namelist['USE_SITE_topography']:
            # otherwise, retrieve from database by Aggregation_Topography.F90
            self.SITE_topography = self.netFile.ncio_read_serial(path_file, 'elevation')

    def read_urban_surface_data_single(self, path_file, mksrfdata, mkrun=None):
        self.SITE_landtype = self.vars_global.URBAN
        self.SITE_lat_location = self.netFile.ncio_read_serial(path_file, 'latitude')
        self.SITE_lon_location = self.netFile.ncio_read_serial(path_file, 'longitude')

        self.namelist['DEF_domain'].edges = math.floor(self.SITE_lat_location)
        self.namelist['DEF_domain'].edgen = self.namelist['DEF_domain'].edges + 1.0
        self.namelist['DEF_domain'].edgew = math.floor(self.SITE_lon_location)
        self.namelist['DEF_domain'].edgee = self.namelist['DEF_domain'].edgew + 1.0

        # if not isgreenwich: #CoLM_TimeManager
        #    LocalLongitude = self.SITE_lon_location

        if mkrun is None:
            if not mksrfdata or self.namelist['USE_SITE_urban_paras']:
                self.SITE_fveg_urb = self.netFile.ncio_read_serial(path_file, 'tree_area_fraction')
                self.SITE_htop_urb = self.netFile.ncio_read_serial(path_file, 'tree_mean_height')
                self.SITE_flake_urb = self.netFile.ncio_read_serial(path_file, 'water_area_fraction')
                self.SITE_froof = self.netFile.ncio_read_serial(path_file, 'roof_area_fraction')
                self.SITE_hroof = self.netFile.ncio_read_serial(path_file, 'building_mean_height')
                self.SITE_fgimp = self.netFile.ncio_read_serial(path_file, 'impervious_area_fraction')
                self.SITE_hwr = self.netFile.ncio_read_serial(path_file, 'canyon_height_width_ratio')
                self.SITE_popden = self.netFile.ncio_read_serial(path_file, 'resident_population_density')

                self.SITE_fgper = 1 - (self.SITE_fgimp - self.SITE_froof) / (1 - self.SITE_froof - self.SITE_flake_urb)
                self.SITE_fveg_urb = self.SITE_fveg_urb * 100
                self.SITE_flake_urb = self.SITE_flake_urb * 100
        else:
            self.SITE_LAI_year = self.netFile.ncio_read_serial(path_file, 'LAI_year')
            self.SITE_LAI_monthly = self.netFile.ncio_read_serial(path_file, 'TREE_LAI')
            self.SITE_SAI_monthly = self.netFile.ncio_read_serial(path_file, 'TREE_SAI')

            self.SITE_urbtyp = self.netFile.ncio_read_serial(path_file, 'URBAN_TYPE')
            self.SITE_lucyid = self.netFile.ncio_read_serial(path_file, 'LUCY_id')
            self.SITE_fveg_urb = self.netFile.ncio_read_serial(path_file, 'PCT_Tree')
            self.SITE_htop_urb = self.netFile.ncio_read_serial(path_file, 'URBAN_TREE_TOP')
            self.SITE_flake_urb = self.netFile.ncio_read_serial(path_file, 'PCT_Water')
            self.SITE_froof = self.netFile.ncio_read_serial(path_file, 'WT_ROOF')
            self.SITE_hroof = self.netFile.ncio_read_serial(path_file, 'HT_ROOF')
            self.SITE_fgper = self.netFile.ncio_read_serial(path_file, 'WTROAD_PERV')
            self.SITE_hwr = self.netFile.ncio_read_serial(path_file, 'CANYON_HWR')
            self.SITE_popden = self.netFile.ncio_read_serial(path_file, 'POP_DEN')

            self.SITE_em_roof = self.netFile.ncio_read_serial(path_file, 'EM_ROOF')
            self.SITE_em_wall = self.netFile.ncio_read_serial(path_file, 'EM_WALL')
            self.SITE_em_gimp = self.netFile.ncio_read_serial(path_file, 'EM_IMPROAD')
            self.SITE_em_gper = self.netFile.ncio_read_serial(path_file, 'EM_PERROAD')
            self.SITE_t_roommax = self.netFile.ncio_read_serial(path_file, 'T_BUILDING_MAX')
            self.SITE_t_roommin = self.netFile.ncio_read_serial(path_file, 'T_BUILDING_MIN')
            self.SITE_thickroof = self.netFile.ncio_read_serial(path_file, 'THICK_ROOF')
            self.SITE_thickwall = self.netFile.ncio_read_serial(path_file, 'THICK_WALL')

            self.SITE_alb_roof = self.netFile.ncio_read_serial(path_file, 'ALB_ROOF')
            self.SITE_alb_wall = self.netFile.ncio_read_serial(path_file, 'ALB_WALL')
            self.SITE_alb_gimp = self.netFile.ncio_read_serial(path_file, 'ALB_IMPROAD')
            self.SITE_alb_gper = self.netFile.ncio_read_serial(path_file, 'ALB_PERROAD')

            self.SITE_cv_roof = self.netFile.ncio_read_serial(path_file, 'CV_ROOF')
            self.SITE_cv_wall = self.netFile.ncio_read_serial(path_file, 'CV_WALL')
            self.SITE_cv_gimp = self.netFile.ncio_read_serial(path_file, 'CV_IMPROAD')
            self.SITE_tk_roof = self.netFile.ncio_read_serial(path_file, 'TK_ROOF')
            self.SITE_tk_wall = self.netFile.ncio_read_serial(path_file, 'TK_WALL')
            self.SITE_tk_gimp = self.netFile.ncio_read_serial(path_file, 'TK_IMPROAD')

        if not mksrfdata or self.namelist['USE_SITE_lakedepth']:
            # otherwise retrieve from database by Aggregation_LakeDepth.F90
            self.SITE_lakedepth = self.netFile.ncio_read_serial(path_file, 'lakedepth')

        if not mksrfdata or self.namelist['USE_SITE_soilreflectance']:
            # otherwise retrieve from database by Aggregation_SoilBrightness.F90
            self.SITE_soil_s_v_alb = self.netFile.ncio_read_serial(path_file, 'soil_s_v_alb')
            self.SITE_soil_d_v_alb = self.netFile.ncio_read_serial(path_file, 'soil_d_v_alb')
            self.SITE_soil_s_n_alb = self.netFile.ncio_read_serial(path_file, 'soil_s_n_alb')
            self.SITE_soil_d_n_alb = self.netFile.ncio_read_serial(path_file, 'soil_d_n_alb')

        if not mksrfdata or self.namelist['USE_SITE_soilparameters']:
            # otherwise retrieve from database by Aggregation_SoilParameters.F90
            self.SITE_soil_vf_quartz_mineral = self.netFile.ncio_read_serial(path_file, 'soil_vf_quartz_mineral')
            self.SITE_soil_vf_gravels = self.netFile.ncio_read_serial(path_file, 'soil_vf_gravels')
            self.SITE_soil_vf_sand = self.netFile.ncio_read_serial(path_file, 'soil_vf_sand')
            self.SITE_soil_vf_om = self.netFile.ncio_read_serial(path_file, 'soil_vf_om')
            self.SITE_soil_wf_gravels = self.netFile.ncio_read_serial(path_file, 'soil_wf_gravels')
            self.SITE_soil_wf_sand = self.netFile.ncio_read_serial(path_file, 'soil_wf_sand')
            self.SITE_soil_OM_density = self.netFile.ncio_read_serial(path_file, 'soil_OM_density')
            self.SITE_soil_BD_all = self.netFile.ncio_read_serial(path_file, 'soil_BD_all')
            self.SITE_soil_theta_s = self.netFile.ncio_read_serial(path_file, 'soil_theta_s')
            self.SITE_soil_k_s = self.netFile.ncio_read_serial(path_file, 'soil_k_s')
            self.SITE_soil_csol = self.netFile.ncio_read_serial(path_file, 'soil_csol')
            self.SITE_soil_tksatu = self.netFile.ncio_read_serial(path_file, 'soil_tksatu')
            self.SITE_soil_tksatf = self.netFile.ncio_read_serial(path_file, 'soil_tksatf')
            self.SITE_soil_tkdry = self.netFile.ncio_read_serial(path_file, 'soil_tkdry')
            self.SITE_soil_k_solids = self.netFile.ncio_read_serial(path_file, 'soil_k_solids')
            self.SITE_soil_psi_s = self.netFile.ncio_read_serial(path_file, 'soil_psi_s')
            self.SITE_soil_lambda = self.netFile.ncio_read_serial(path_file, 'soil_lambda')
            if self.namelist['defvanGenuchten_Mualem_SOIL_MODEL']:
                self.SITE_soil_theta_r = self.netFile.ncio_read_serial(path_file, 'soil_theta_r')
                self.SITE_soil_alpha_vgm = self.netFile.ncio_read_serial(path_file, 'soil_alpha_vgm')
                self.SITE_soil_L_vgm = self.netFile.ncio_read_serial(path_file, 'soil_L_vgm')
                self.SITE_soil_n_vgm = self.netFile.ncio_read_serial(path_file, 'soil_n_vgm')
            # endif
            self.SITE_soil_BA_alpha = self.netFile.ncio_read_serial(path_file, 'soil_BA_alpha')
            self.SITE_soil_BA_beta = self.netFile.ncio_read_serial(path_file, 'soil_BA_beta')

        if self.namelist['DEF_USE_BEDROCK']:
            if not mksrfdata or self.namelist['USE_SITE_dbedrock']:
                # otherwise retrieve from database by Aggregation_DBedrock.F90
                self.SITE_dbedrock = self.netFile.ncio_read_serial(path_file, 'depth_to_bedrock')

        if not mksrfdata or self.namelist['USE_SITE_topography']:
            # otherwise retrieve from database by Aggregation_Topography.F90
            self.SITE_topography = self.netFile.ncio_read_serial(path_file, 'elevation')

    def write_surface_data_single(self, root_path, numpatch, mksrfdata, numpft=None):
        fsrfdata = os.path.join(root_path, self.namelist['DEF_dir_landdata'], 'srfdata.nc')
        if 'win' in sys.platform:
            fsrfdata = root_path + '\\' + self.namelist['DEF_dir_landdata'] + '\\' + 'srfdata.nc'

        if '/media/zjl/7C24CC1724CBD276' in fsrfdata:
            names = fsrfdata.split('/')
            s = ''
            for n in names:
                s = '/' + n
            fsrfdata = '/home/zjl' + s


        self.netFile.ncio_create_file(fsrfdata)

        self.netFile.ncio_define_dimension(fsrfdata, 'soil', self.vars_global.nl_soil)
        self.netFile.ncio_define_dimension(fsrfdata, 'patch', numpatch)
        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            self.netFile.ncio_define_dimension(fsrfdata, 'pft', numpft)

        self.netFile.ncio_define_dimension(fsrfdata, 'LAI_year', len(self.SITE_LAI_year))
        if self.namelist['DEF_LAI_MONTHLY']:
            self.netFile.ncio_define_dimension(fsrfdata, 'month', 12)
        else:
            self.netFile.ncio_define_dimension(fsrfdata, 'J8day', 46)

        self.netFile.ncio_write_serial(fsrfdata, 'latitude', self.SITE_lat_location)
        self.netFile.ncio_write_serial(fsrfdata, 'longitude', self.SITE_lon_location)

        if self.namelist['LULC_USGS']:
            self.netFile.ncio_write_serial(fsrfdata, 'USGS_classification', self.SITE_landtype)
        else:
            self.netFile.ncio_write_serial(fsrfdata, 'IGBP_classification', self.SITE_landtype)

        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            self.netFile.ncio_write_serial4(fsrfdata, 'pfttyp', self.SITE_pfttyp, 'pft')
            self.netFile.ncio_put_attr(fsrfdata, 'pfttyp', 'source', self.datasource(self.namelist['USE_SITE_pctpfts']))

        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            self.netFile.ncio_write_serial4(fsrfdata, 'pctpfts', self.SITE_pctpfts, 'pft')
            self.netFile.ncio_put_attr(fsrfdata, 'pctpfts', 'source',
                                       self.datasource(self.namelist['USE_SITE_pctpfts']))

        if self.namelist['CROP']:
            if self.SITE_landtype == self.vars_global.CROPLAND:
                self.netFile.ncio_write_serial4(fsrfdata, 'croptyp', self.SITE_croptyp, 'patch')
                self.netFile.ncio_write_serial4(fsrfdata, 'pctcrop', self.SITE_pctcrop, 'patch')
                self.netFile.ncio_put_attr(fsrfdata, 'croptyp', 'source',
                                           self.datasource(self.namelist['USE_SITE_pctcrop']))
                self.netFile.ncio_put_attr(fsrfdata, 'pctcrop', 'source',
                                           self.datasource(self.namelist['USE_SITE_pctcrop']))
        if (not mksrfdata) or self.namelist['USE_SITE_LAI']:
            if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
                self.netFile.ncio_write_serial4(fsrfdata, 'canopy_height_pfts', self.SITE_htop_pfts, 'pft')
                self.netFile.ncio_put_attr(fsrfdata, 'canopy_height_pfts', 'source',
                                           self.datasource(self.namelist['USE_SITE_htop']))
            else:
                self.netFile.ncio_write_serial4(fsrfdata, 'canopy_height', self.SITE_htop)
                self.netFile.ncio_put_attr(fsrfdata, 'canopy_height', 'source',
                                           self.datasource(self.namelist['USE_SITE_htop']))

        source = self.datasource(self.namelist['USE_SITE_LAI'])
        self.netFile.ncio_write_serial4(fsrfdata, 'LAI_year', self.SITE_LAI_year, 'LAI_year')
        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            if self.namelist['DEF_LAI_MONTHLY']:
                self.netFile.ncio_write_serial6(fsrfdata, 'LAI_pfts_monthly', self.SITE_LAI_pfts_monthly, 'pft',
                                                'month',
                                                'LAI_year')
                self.netFile.ncio_write_serial6(fsrfdata, 'SAI_pfts_monthly', self.SITE_SAI_pfts_monthly, 'pft',
                                                'month',
                                                'LAI_year')
                self.netFile.ncio_put_attr(fsrfdata, 'LAI_pfts_monthly', 'source', source)
                self.netFile.ncio_put_attr(fsrfdata, 'SAI_pfts_monthly', 'source', source)

        else:
            if self.namelist['DEF_LAI_MONTHLY']:
                self.netFile.ncio_write_serial5(fsrfdata, 'LAI_monthly', self.SITE_LAI_monthly, 'month', 'LAI_year')
                self.netFile.ncio_write_serial5(fsrfdata, 'SAI_monthly', self.SITE_SAI_monthly, 'month', 'LAI_year')
                self.netFile.ncio_put_attr(fsrfdata, 'LAI_monthly', 'source', source)
                self.netFile.ncio_put_attr(fsrfdata, 'SAI_monthly', 'source', source)
            else:
                self.netFile.ncio_write_serial5(fsrfdata, 'LAI_8day', self.SITE_LAI_8day, 'J8day', 'LAI_year')
                self.netFile.ncio_put_attr(fsrfdata, 'LAI_8day', 'source', source)

        self.netFile.ncio_write_serial_float(fsrfdata, 'lakedepth', self.SITE_lakedepth)
        self.netFile.ncio_put_attr(fsrfdata, 'lakedepth', 'source',
                                   self.datasource(self.namelist['USE_SITE_lakedepth']))

        source = self.datasource(self.namelist['USE_SITE_soilreflectance'])
        self.netFile.ncio_write_serial_float(fsrfdata, 'soil_s_v_alb', self.SITE_soil_s_v_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_s_v_alb', 'source', source)
        self.netFile.ncio_write_serial_float(fsrfdata, 'soil_d_v_alb', self.SITE_soil_d_v_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_d_v_alb', 'source', source)
        self.netFile.ncio_write_serial_float(fsrfdata, 'soil_s_n_alb', self.SITE_soil_s_n_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_s_n_alb', 'source', source)
        self.netFile.ncio_write_serial_float(fsrfdata, 'soil_d_n_alb', self.SITE_soil_d_n_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_d_n_alb', 'source', source)
        if not mksrfdata or self.namelist['USE_SITE_soilparameters']:
            source = self.datasource(self.namelist['USE_SITE_soilparameters'])
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_quartz_mineral', self.SITE_soil_vf_quartz_mineral, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_gravels       ', self.SITE_soil_vf_gravels, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_sand          ', self.SITE_soil_vf_sand, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_om            ', self.SITE_soil_vf_om, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_wf_gravels       ', self.SITE_soil_wf_gravels, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_wf_sand          ', self.SITE_soil_wf_sand, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_OM_density       ', self.SITE_soil_OM_density, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_BD_all           ', self.SITE_soil_BD_all, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_theta_s          ', self.SITE_soil_theta_s, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_k_s              ', self.SITE_soil_k_s, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_csol             ', self.SITE_soil_csol, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_tksatu           ', self.SITE_soil_tksatu, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_tksatf           ', self.SITE_soil_tksatf, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_tkdry            ', self.SITE_soil_tkdry, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_k_solids         ', self.SITE_soil_k_solids, 'soil')
            self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_quartz_mineral', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_gravels       ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_sand          ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_om            ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_wf_gravels       ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_wf_sand          ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_OM_density       ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_BD_all           ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_theta_s          ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_k_s              ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_csol             ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_tksatu           ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_tksatf           ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_tkdry            ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_k_solids         ', 'source', source)
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_psi_s ', self.SITE_soil_psi_s, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_lambda', self.SITE_soil_lambda, 'soil')
            self.netFile.ncio_put_attr(fsrfdata, 'soil_psi_s ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_lambda', 'source', source)
            if self.namelist['vanGenuchten_Mualem_SOIL_MODEL']:
                self.netFile.ncio_write_serial4(fsrfdata, 'soil_theta_r  ', self.SITE_soil_theta_r, 'soil')
                self.netFile.ncio_write_serial4(fsrfdata, 'soil_alpha_vgm', self.SITE_soil_alpha_vgm, 'soil')
                self.netFile.ncio_write_serial4(fsrfdata, 'soil_L_vgm    ', self.SITE_soil_L_vgm, 'soil')
                self.netFile.ncio_write_serial4(fsrfdata, 'soil_n_vgm    ', self.SITE_soil_n_vgm, 'soil')
                self.netFile.ncio_put_attr(fsrfdata, 'soil_theta_r  ', 'source', source)
                self.netFile.ncio_put_attr(fsrfdata, 'soil_alpha_vgm', 'source', source)
                self.netFile.ncio_put_attr(fsrfdata, 'soil_L_vgm    ', 'source', source)
                self.netFile.ncio_put_attr(fsrfdata, 'soil_n_vgm    ', 'source', source)
            #
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_BA_alpha', self.SITE_soil_BA_alpha, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_BA_beta ', self.SITE_soil_BA_beta, 'soil')
            self.netFile.ncio_put_attr(fsrfdata, 'soil_BA_alpha', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_BA_beta ', 'source', source)

        if self.namelist['DEF_USE_BEDROCK']:
            self.netFile.ncio_write_serial(fsrfdata, 'depth_to_bedrock', self.SITE_dbedrock)
            self.netFile.ncio_put_attr(fsrfdata, 'depth_to_bedrock', 'source',
                                       self.datasource(self.namelist['USE_SITE_dbedrock']))

        self.netFile.ncio_write_serial(fsrfdata, 'elevation', self.SITE_topography)
        self.netFile.ncio_put_attr(fsrfdata, 'elevation', 'source',
                                   self.datasource(self.namelist['USE_SITE_topography']))

    def write_urban_surface_data_single(self, root_path, numurban):
        fsrfdata = os.path.join(root_path, self.namelist['DEF_dir_landdata'], 'srfdata.nc')
        if 'win' in sys.platform:
            fsrfdata = root_path + '\\' + self.namelist['DEF_dir_landdata'] + '\\' + 'srfdata.nc'

        if '/media/zjl/7C24CC1724CBD276' in fsrfdata:
            names = fsrfdata.split('/')
            s = ''
            for n in names:
                s = '/' + n
            fsrfdata = '/home/zjl' + s

        self.netFile.ncio_create_file(fsrfdata)

        self.netFile.ncio_define_dimension(fsrfdata, 'soil', self.vars_global.nl_soil)
        self.netFile.ncio_define_dimension(fsrfdata, 'patch', numurban)

        self.netFile.ncio_define_dimension(fsrfdata, 'LAI_year', len(self.SITE_LAI_year))
        self.netFile.ncio_define_dimension(fsrfdata, 'month', 12)

        self.netFile.ncio_define_dimension(fsrfdata, 'ulev', 10)
        self.netFile.ncio_define_dimension(fsrfdata, 'numsolar', 2)
        self.netFile.ncio_define_dimension(fsrfdata, 'numrad', 2)

        self.netFile.ncio_write_serial(fsrfdata, 'latitude', self.SITE_lat_location)
        self.netFile.ncio_write_serial(fsrfdata, 'longitude', self.SITE_lon_location)

        source = self.datasource(self.namelist['USE_SITE_urban_LAI'])
        self.netFile.ncio_write_serial4(fsrfdata, 'LAI_year', self.SITE_LAI_year, 'LAI_year')
        self.netFile.ncio_write_serial5(fsrfdata, 'TREE_LAI', self.SITE_LAI_monthly, 'month', 'LAI_year')
        self.netFile.ncio_write_serial5(fsrfdata, 'TREE_SAI', self.SITE_SAI_monthly, 'month', 'LAI_year')
        self.netFile.ncio_put_attr(fsrfdata, 'TREE_LAI', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'TREE_SAI', 'source', source)

        self.netFile.ncio_write_serial(fsrfdata, 'lakedepth', self.SITE_lakedepth)
        self.netFile.ncio_put_attr(fsrfdata, 'lakedepth', 'source',
                                   self.datasource(self.namelist['USE_SITE_lakedepth']))

        self.netFile.ncio_write_serial4(fsrfdata, 'URBAN_TYPE', self.SITE_urbtyp, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'LUCY_id', self.SITE_lucyid, 'patch')
        source = self.datasource(self.namelist['USE_SITE_urban_paras'])
        self.netFile.ncio_write_serial4(fsrfdata, 'PCT_Tree', self.SITE_fveg_urb, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'URBAN_TREE_TOP', self.SITE_htop_urb, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'PCT_Water', self.SITE_flake_urb, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'WT_ROOF', self.SITE_froof, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'HT_ROOF', self.SITE_hroof, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'WTROAD_PERV', self.SITE_fgper, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'CANYON_HWR', self.SITE_hwr, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'POP_DEN', self.SITE_popden, 'patch')

        self.netFile.ncio_put_attr(fsrfdata, 'PCT_Tree', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'URBAN_TREE_TOP', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'PCT_Water', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'WT_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'HT_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'WTROAD_PERV', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'CANYON_HWR', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'POP_DEN', 'source', source)

        source = self.datasource(self.namelist['USE_SITE_thermal_paras'])
        self.netFile.ncio_write_serial4(fsrfdata, 'EM_ROOF', self.SITE_em_roof, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'EM_WALL', self.SITE_em_wall, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'EM_IMPROAD', self.SITE_em_gimp, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'EM_PERROAD', self.SITE_em_gper, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'T_BUILDING_MAX', self.SITE_t_roommax, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'T_BUILDING_MIN', self.SITE_t_roommin, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'THICK_ROOF', self.SITE_thickroof, 'patch')
        self.netFile.ncio_write_serial4(fsrfdata, 'THICK_WALL', self.SITE_thickwall, 'patch')

        self.netFile.ncio_put_attr(fsrfdata, 'EM_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'EM_WALL', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'EM_IMPROAD', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'EM_PERROAD', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'T_BUILDING_MAX', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'T_BUILDING_MIN', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'THICK_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'THICK_WALL', 'source', source)

        self.netFile.ncio_write_serial5(fsrfdata, 'ALB_ROOF', self.SITE_alb_roof, 'numrad', 'numsolar')
        self.netFile.ncio_write_serial5(fsrfdata, 'ALB_WALL', self.SITE_alb_wall, 'numrad', 'numsolar')
        self.netFile.ncio_write_serial5(fsrfdata, 'ALB_IMPROAD', self.SITE_alb_gimp, 'numrad', 'numsolar')
        self.netFile.ncio_write_serial5(fsrfdata, 'ALB_PERROAD', self.SITE_alb_gper, 'numrad', 'numsolar')

        self.netFile.ncio_put_attr(fsrfdata, 'ALB_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'ALB_WALL', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'ALB_IMPROAD', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'ALB_PERROAD', 'source', source)

        self.netFile.ncio_write_serial4(fsrfdata, 'CV_ROOF', self.SITE_cv_roof, 'ulev')
        self.netFile.ncio_write_serial4(fsrfdata, 'CV_WALL', self.SITE_cv_wall, 'ulev')
        self.netFile.ncio_write_serial4(fsrfdata, 'CV_IMPROAD', self.SITE_cv_gimp, 'ulev')
        self.netFile.ncio_write_serial4(fsrfdata, 'TK_ROOF', self.SITE_tk_roof, 'ulev')
        self.netFile.ncio_write_serial4(fsrfdata, 'TK_WALL', self.SITE_tk_wall, 'ulev')
        self.netFile.ncio_write_serial4(fsrfdata, 'TK_IMPROAD', self.SITE_tk_gimp, 'ulev')
        self.netFile.ncio_put_attr(fsrfdata, 'CV_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'CV_WALL', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'CV_IMPROAD', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'TK_ROOF', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'TK_WALL', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'TK_IMPROAD', 'source', source)

        source = self.datasource(self.namelist['USE_SITE_soilreflectance'])
        self.netFile.ncio_write_serial(fsrfdata, 'soil_s_v_alb', self.SITE_soil_s_v_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_s_v_alb', 'source', source)
        self.netFile.ncio_write_serial(fsrfdata, 'soil_d_v_alb', self.SITE_soil_d_v_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_d_v_alb', 'source', source)
        self.netFile.ncio_write_serial(fsrfdata, 'soil_s_n_alb', self.SITE_soil_s_n_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_s_n_alb', 'source', source)
        self.netFile.ncio_write_serial(fsrfdata, 'soil_d_n_alb', self.SITE_soil_d_n_alb)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_d_n_alb', 'source', source)

        source = self.datasource(self.namelist['USE_SITE_soilparameters'])
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_quartz_mineral', self.SITE_soil_vf_quartz_mineral, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_gravels       ', self.SITE_soil_vf_gravels, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_sand          ', self.SITE_soil_vf_sand, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_vf_om            ', self.SITE_soil_vf_om, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_wf_gravels       ', self.SITE_soil_wf_gravels, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_wf_sand          ', self.SITE_soil_wf_sand, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_OM_density       ', self.SITE_soil_OM_density, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_BD_all           ', self.SITE_soil_BD_all, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_theta_s          ', self.SITE_soil_theta_s, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_k_s              ', self.SITE_soil_k_s, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_csol             ', self.SITE_soil_csol, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_tksatu           ', self.SITE_soil_tksatu, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_tksatf           ', self.SITE_soil_tksatf, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_tkdry            ', self.SITE_soil_tkdry, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_k_solids         ', self.SITE_soil_k_solids, 'soil')
        self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_quartz_mineral', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_gravels       ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_sand          ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_vf_om            ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_wf_gravels       ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_wf_sand          ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_OM_density       ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_BD_all           ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_theta_s          ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_k_s              ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_csol             ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_tksatu           ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_tksatf           ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_tkdry            ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_k_solids         ', 'source', source)
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_psi_s ', self.SITE_soil_psi_s, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_lambda', self.SITE_soil_lambda, 'soil')
        self.netFile.ncio_put_attr(fsrfdata, 'soil_psi_s ', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_lambda', 'source', source)
        if self.namelist['vanGenuchten_Mualem_SOIL_MODEL']:
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_theta_r  ', self.SITE_soil_theta_r, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_alpha_vgm', self.SITE_soil_alpha_vgm, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_L_vgm    ', self.SITE_soil_L_vgm, 'soil')
            self.netFile.ncio_write_serial4(fsrfdata, 'soil_n_vgm    ', self.SITE_soil_n_vgm, 'soil')
            self.netFile.ncio_put_attr(fsrfdata, 'soil_theta_r  ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_alpha_vgm', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_L_vgm    ', 'source', source)
            self.netFile.ncio_put_attr(fsrfdata, 'soil_n_vgm    ', 'source', source)

        self.netFile.ncio_write_serial4(fsrfdata, 'soil_BA_alpha', self.SITE_soil_BA_alpha, 'soil')
        self.netFile.ncio_write_serial4(fsrfdata, 'soil_BA_beta ', self.SITE_soil_BA_beta, 'soil')
        self.netFile.ncio_put_attr(fsrfdata, 'soil_BA_alpha', 'source', source)
        self.netFile.ncio_put_attr(fsrfdata, 'soil_BA_beta ', 'source', source)

        if self.namelist['DEF_USE_BEDROCK']:
            self.netFile.ncio_write_serial(fsrfdata, 'depth_to_bedrock', self.SITE_dbedrock)
            self.netFile.ncio_put_attr(fsrfdata, 'depth_to_bedrock', 'source',
                                       self.datasource(self.namelist['USE_SITE_dbedrock']))

        self.netFile.ncio_write_serial(fsrfdata, 'elevation', self.SITE_topography)
        self.netFile.ncio_put_attr(fsrfdata, 'elevation', 'source',
                                   self.datasource(self.namelist['USE_SITE_topography']))

    def datasource(self, is_site):
        if is_site:
            return 'SITE'
        else:
            return 'DATABASE'

    def release(self, obj_var):
        if obj_var is not None:
            del obj_var

    def single_srfdata_final(self, mksrfdata):
        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            self.release(self.SITE_pfttyp)
            self.release(self.SITE_pctpfts)

        if self.namelist['CROP']:
            self.release(self.SITE_croptyp)
            self.release(self.SITE_pctcrop)

        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            self.release(self.SITE_htop_pfts)

        if hasattr(self, 'SITE_LAI_monthly') and self.SITE_LAI_monthly is not None:
            self.release(self.SITE_LAI_monthly)

        if hasattr(self, 'SITE_SAI_monthly') and self.SITE_SAI_monthly is not None:
            self.release(self.SITE_SAI_monthly)

        if self.namelist['LULC_IGBP_PFT'] or self.namelist['LULC_IGBP_PC']:
            self.release(self.SITE_LAI_pfts_monthly)
            self.release(self.SITE_SAI_pfts_monthly)

        self.release(self.SITE_LAI_year)
        self.release(self.SITE_LAI_8day)
        if not mksrfdata or self.namelist['USE_SITE_soilparameters']:
            self.release(self.SITE_soil_vf_quartz_mineral)
            self.release(self.SITE_soil_vf_gravels)
            self.release(self.SITE_soil_vf_sand)
            self.release(self.SITE_soil_vf_om)
            self.release(self.SITE_soil_wf_gravels)
            self.release(self.SITE_soil_wf_sand)
            self.release(self.SITE_soil_OM_density)
            self.release(self.SITE_soil_BD_all)
            self.release(self.SITE_soil_theta_s)
            self.release(self.SITE_soil_k_s)
            self.release(self.SITE_soil_csol)
            self.release(self.SITE_soil_tksatu)
            self.release(self.SITE_soil_tksatf)
            self.release(self.SITE_soil_tkdry)
            self.release(self.SITE_soil_k_solids)
            self.release(self.SITE_soil_psi_s)
            self.release(self.SITE_soil_lambda)
            self.release(self.SITE_soil_theta_r)
            if self.namelist['vanGenuchten_Mualem_SOIL_MODEL']:
                self.release(self.SITE_soil_alpha_vgm)
                self.release(self.SITE_soil_L_vgm)
                self.release(self.SITE_soil_n_vgm)
            self.release(self.SITE_soil_BA_alpha)
            self.release(self.SITE_soil_BA_beta)
