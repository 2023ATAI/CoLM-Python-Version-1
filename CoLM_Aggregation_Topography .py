# ----------------------------------------------------------------------
#  Global Topography data
#
#    Yamazaki, D., Ikeshima, D., Sosa, J.,Bates, P. D., Allen, G. H.,
#    Pavelsky, T. M. (2019).
#    MERIT Hydro: ahigh‐resolution global hydrographymap based on
#    latest topography dataset.Water Resources Research, 55, 5053–5073.
#
#  Created by Shupeng Zhang, 05/2023
# ----------------------------------------------------------------------
import os
import numpy as np
from CoLM_DataType import DataType
from CoLM_NetCDFBlock import NetCDFBlock


class Aggregation_Topography(object):
    def __init__(self,nl_colm,mpi,gblock) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.gblock = gblock


    def Aggregation_Topography (self, gland, dir_rawdata, dir_model_landdata, lc_year):

        cyear = str(lc_year)
        landdir = os.path.join(dir_model_landdata.strip(), 'topography', cyear)

        print('Aggregate Topography ...')
        os.system('mkdir -p ' + landdir.strip())

        if self.nl_colm['SinglePoint']:
            if self.nl_colm['USE_SITE_topography']:
                return

        lndname = os.path.join(dir_rawdata.strip(), 'elevation.nc')

        if self.mpi.p_is_io:
            datayype = DataType(self.gblock)
            topography = datayype.allocate_block_data(gtopo)

            netCDFBlock = NetCDFBlock(lndname, 'elevation', gland, topography)
            netCDFBlock.ncio_read_block(self.gblock)

    # ----------------------------------------------------------------------
    #   aggregate the elevation from the resolution of raw data to modelling resolution
    # ----------------------------------------------------------------------

        if self.mpi.p_is_worker:
            topography_patches = np.zeros(numpatch)
            topostd_patches = np.zeros(numpatch)

            for ipatch in range(numpatch):
                topography_one = aggregation_request_data(landpatch, ipatch, gtopo,
                                                    zip=self.nl_colm['USE_zip_for_aggregation'], area=area_one,
                                                    data_r8_2d_in1=topography)

                if np.any(topography_one != -9999.0):
                    topography_patches[ipatch] = np.sum(topography_one * area_one,
                                                            where=topography_one != -9999.0) / np.sum(area_one,
                                                                                                      where=topography_one != -9999.0)

                    topostd_patches[ipatch] = np.sum(
                        (topography_one - topography_patches[ipatch]) ** 2 * area_one,
                        where=topography_one != -9999.0) / np.sum(area_one, where=topography_one != -9999.0)
                    topostd_patches[ipatch] = np.sqrt(topostd_patches[ipatch])
                else:
                    topography_patches[ipatch] = -1.0e36
                    topostd_patches[ipatch] = -1.0e36

            self.nl_colm['SITE_topography'] = topography_patches[0]
            self.nl_colm['SITE_topostd'] = topostd_patches[0]