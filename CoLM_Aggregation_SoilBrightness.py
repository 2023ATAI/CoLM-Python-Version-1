# ----------------------------------------------------------------------
# DESCRIPTION:
# Aggregate lake depth of multiple pixels within a lake patch based on Global land cover types
# (updated with the specific dataset)
#
# Global Lake Coverage and Lake Depth (1km resolution)
#   (http://nwpi.krc.karelia.run/flake/)
#    Lake depth data legend
#    Value   Description
# 0       no lake indicated in this pixel
# 1       no any information about this lake and set the default value of 10 m
# 2       no information about depth for this lake and set the default value of 10 m
# 3       have the information about lake depth in this pixel
# 4       this is the river pixel according to our map, set the default value of 3 m
#
# REFERENCE:
# Kourzeneva, E., H. Asensio, E. Martin, and S. Faroux, 2012: Global gridded dataset of lake coverage and lake depth
# for USE in numerical weather prediction and climate modelling. Tellus A, 64, 15640.
#
# Created by Yongjiu Dai, 02/2014
#
# REVISIONS:
# Shupeng Zhang, 01/2022: porting codes to MPI parallel version
# ----------------------------------------------------------------------
import os
import numpy as np
from CoLM_DataType import DataType
from CoLM_NetCDFBlock import NetCDFBlock
from CoLM_AggregationRequestData import AggregationRequestData


class Aggregation_SoilBrightness(object):
    def __init__(self,nl_colm,mpi,gblock) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.gblock = gblock


    def Aggregation_SoilBrightness (self, gland, dir_rawdata, dir_model_landdata, lc_year):

        # ................................................................................................
        #   The soil color and reflectance is from the work:
        #   Peter J. Lawrence and Thomas N. Chase, 2007:
        #   Representing a MODIS consistent land surface in the Community Land Model (CLM 3.0):
        #   Part 1 generating MODIS consistent land surface parameters
        # ................................................................................................
        soil_s_v_refl = np.array([0.26, 0.24, 0.22, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
                                  0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04])

        soil_d_v_refl = np.array([0.37, 0.35, 0.33, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25,
                                  0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15])

        soil_s_n_refl = np.array([0.52, 0.48, 0.44, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28,
                                  0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08])

        soil_d_n_refl = np.array([0.63, 0.59, 0.55, 0.51, 0.49, 0.47, 0.45, 0.43, 0.41, 0.39,
                                  0.37, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19])

        cyear = '{:04d}'.format(lc_year)
        landdir = dir_model_landdata.strip() + '/soil/' + cyear

        if self.mpip_is_master:
            print('Aggregate Soil Brightness ...')
            os.system('mkdir -p ' + landdir.strip())

        if self.nl_colm['SinglePoint']:
            if self.nl_colm['USE_SITE_soilreflectance']:
                return

        # ................................................................................................
        #   aggregate the soil parameters from the resolution of raw data to modelling resolution
        # ................................................................................................

        lndname = dir_rawdata.strip() + '/soil_brightness.nc'

        if self.mpi.p_is_io:
            # 分配数据结构
            datayype = DataType(self.gblock)
            isc = datayype.allocate_block_data(gland)
            a_s_v_refl = datayype.allocate_block_data(gland)
            a_d_v_refl = datayype.allocate_block_data(gland)
            a_s_n_refl = datayype.allocate_block_data(gland)
            a_d_n_refl = datayype.allocate_block_data(gland)

            netCDFBlock = NetCDFBlock(lndname, 'soil_brightness', gland, isc)
            netCDFBlock.ncio_read_block(self.gblock)

            # 遍历所有地块和像素
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]

                for iy in range(gland.ycnt[jblk]):
                    for ix in range(gland.xcnt[iblk]):
                        ii = isc.blk[iblk, jblk].val[ix, iy]
                        if 1 <= ii <= 20:
                            a_s_v_refl.blk[iblk, jblk].val[ix, iy] = soil_s_v_refl[ii - 1]
                            a_d_v_refl.blk[iblk, jblk].val[ix, iy] = soil_d_v_refl[ii - 1]
                            a_s_n_refl.blk[iblk, jblk].val[ix, iy] = soil_s_n_refl[ii - 1]
                            a_d_n_refl.blk[iblk, jblk].val[ix, iy] = soil_d_n_refl[ii - 1]

        if self.mpi.p_is_worker:
            soil_s_v_alb = np.empty(numpatch)

            for ipatch in range(numpatch):
                L = landpatch.settyp[ipatch]

                if self.nl_colm['LULC_USGS']:
                    if L != 16 and L != 24:

                        soil_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_s_v_refl, data_r8_2d_out1=None)
                        soil_s_v_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_s_v_alb[ipatch] = -1.0e36
                else:
                    if L != 17 and L != 15:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_s_v_refl, data_r8_2d_out1=None)
                        soil_s_v_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_s_v_alb[ipatch] = -1.0e36

        if self.mpi.p_is_worker:
            soil_d_v_alb = np.empty(numpatch)

            for ipatch in range(numpatch):
                L = landpatch.settyp[ipatch]

                if self.nl_colm['LULC_USGS']:
                    if L != 16 and L != 24:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_d_v_refl, data_r8_2d_out1=None)
                        soil_d_v_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_d_v_alb[ipatch] = -1.0e36

                else:

                    if L != 17 and L != 15:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_d_v_refl, data_r8_2d_out1=None)
                        soil_d_v_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_d_v_alb[ipatch] = -1.0e36

        if self.mpi.p_is_worker:
            soil_s_n_alb = np.empty(numpatch)

            for ipatch in range(numpatch):
                L = landpatch.settyp[ipatch]

                if self.nl_colm['LULC_USGS']:
                    if L != 16 and L != 24:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_s_n_refl, data_r8_2d_out1=None)
                        soil_s_n_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_s_n_alb[ipatch] = -1.0e36

                else:

                    if L != 17 and L != 15:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_s_n_refl, data_r8_2d_out1=None)
                        soil_s_n_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_s_n_alb[ipatch] = -1.0e36

        if self.mpi.p_is_worker:
            soil_d_n_alb  = np.empty(numpatch)

            for ipatch in range(numpatch):
                L = landpatch.settyp[ipatch]

                if self.nl_colm['LULC_USGS']:
                    if L != 16 and L != 24:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland,
                                                            zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_d_n_refl, data_r8_2d_out1=None)
                        soil_d_n_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_d_n_alb[ipatch] = -1.0e36

                else:

                    if L != 17 and L != 15:
                        soil_one = aggregation_request_data(landpatch, ipatch, gland,
                                                            zip=self.nl_colm['USE_zip_for_aggregation'],
                                                            data_r8_2d_in1=a_d_n_refl, data_r8_2d_out1=None)
                        soil_d_n_alb[ipatch] = np.median(soil_one)
                    else:
                        soil_d_n_alb[ipatch] = -1.0e36

            if  self.nl_colm['SinglePoint']:
                self.nl_colm['SITE_soil_s_v_alb'] = soil_s_v_alb[0]
                self.nl_colm['SITE_soil_d_v_alb'] = soil_d_v_alb[0]
                self.nl_colm['SITE_soil_s_n_alb'] = soil_s_n_alb[0]
                self.nl_colm['SITE_soil_d_n_alb'] = soil_d_n_alb[0]


