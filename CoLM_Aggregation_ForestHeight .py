# ----------------------------------------------------------------------
#  Global Forest Height
#     (http://lidarradar.jpl.nasa.gov/)
#      Simard, M., N. Pinto, J. B. Fisher, and A. Baccini, 2011: Mapping
#      forest canopy height globally with spaceborne lidar.
#      J. Geophys. Res., 116, G04021.
#
#  Created by Yongjiu Dai, 02/2014
#
#  REVISIONS:
#  Hua Yuan,      ?/2020 : for land cover land use classifications
#  Shupeng Zhang, 01/2022: porting codes to MPI parallel version
# ----------------------------------------------------------------------
import os
import numpy as np
from CoLM_DataType import DataType
from CoLM_NetCDFBlock import NetCDFBlock


class Aggregation_ForestHeight(object):
    def __init__(self,nl_colm,mpi,gblock) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.gblock = gblock


    def Aggregation_ForestHeight (self, gland, dir_rawdata, dir_model_landdata, lc_year):

        cyear = str(lc_year)
        landdir = os.path.join(dir_model_landdata.strip(), 'htop', cyear)

        if self.nl_colm['LULC_IGBP']:
            if self.mpip_is_io:
                gland.allocate_block_data(htop)

            if self.p_is_io:
                dir_5x5 = os.path.join(dir_rawdata.strip(), 'plant_15s')
                suffix = 'CoLM' + cyear.strip()
                read_5x5_data(dir_5x5, suffix, gland, 'HTOP', htop)

            if self.p_is_worker:
                htop_patches = np.zeros(numpatch)

                for ipatch in range(numpatch):
                    if landpatch.settyp[ipatch] != 0:
                        htop_one = aggregation_request_data(landpatch, ipatch, gland, zip=self.nl_colm['USE_zip_for_aggregation'], area=area_one,
                                                 data_r8_2d_in1=htop)
                        htop_patches[ipatch] = np.sum(htop_one * area_one) / np.sum(area_one)

            SITE_htop = htop_patches[0]

            if self.p_is_worker:
                if 'htop_patches' in locals():
                    htop_patches = None
                if 'htop_one' in locals():
                    htop_one = None
                if 'area_one' in locals():
                    area_one = None

            SITE_htop_pfts = np.copy(htop_pfts)

