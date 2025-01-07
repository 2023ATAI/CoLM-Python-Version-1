# ----------------------------------------------------------------------
# DESCRIPTION:
# Global Topography data
#
#       Yamazaki, D., Ikeshima, D., Sosa, J.,Bates, P. D., Allen, G. H.,
#       Pavelsky, T. M. (2019).
#       MERIT Hydro: ahigh‐resolution global hydrographymap based on
#       latest topography dataset.Water Resources Research, 55, 5053–5073.
#
#     Created by Shupeng Zhang, 05/2023
# ----------------------------------------------------------------------
import os
import numpy as np
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_DataType import DataType
from CoLM_NetCDFBlock import NetCDFBlock
import CoLM_RangeCheck


def Aggregation_Topography (gtopo, dir_rawdata, dir_model_landdata, lc_year, \
                           nl_colm, mpi, gblock, gland, mesh,var_global, landpatch, srfdataDiag):

    area_one = np.empty(landpatch.numpatch)
    if nl_colm['SrfdataDiag']:
        typpatch = list(range(nl_colm['N_land_classification'] + 1))
        ityp = 0

    cyear = f"{lc_year:04d}"
    landdir = (dir_model_landdata.strip() + '/topography/' + cyear.strip())

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Aggregate topography ...')
        os.makedirs(landdir.strip(), exist_ok=True)

    if nl_colm['USEMPI']:
        pass

    if nl_colm['SinglePoint']:
        if nl_colm['USE_SITE_topography']:
            return

    lndname = os.path.join(dir_rawdata.strip(), 'elevation.nc')
    if mpi.p_is_io:
        dt = DataType(gblock)
        topography = dt.allocate_block_data(gtopo)
        netblock = NetCDFBlock(lndname, 'elevation', gtopo, topography, mpi)
        netblock.ncio_read_block(gblock)
        topography = netblock.rdata

        if nl_colm['USEMPI']:
            pass
    # ................................................................................................
    #  aggregate the elevation from the resolution of raw data to modelling resolution
    # ................................................................................................

    if mpi.p_is_worker:
        topography_patches = np.empty(landpatch.numpatch)
        for ipatch in range(landpatch.numpatch):
            ard = AggregationRequestData(mpi, mesh.mesh, gblock.pixel)
            topography_one = ard.aggregation_request_data(landpatch, ipatch, gridlai,
                                                   zip=nl_colm['USE_zip_for_aggregation'], \
                                                   data_r8_2d_in1=topography, area=area_one)
            if any(topography_one != -9999.0):
                topography_patches[ipatch] = np.sum(topography_one * area_one, where=(topography_one != -9999.0)) / np.sum(area_one, where=(topography_one != -9999.0))
            else:
                topography_patches[ipatch] = -1.0e36

        if nl_colm['USEMPI']:
            pass
    if nl_colm['RangeCheck']:
        CoLM_RangeCheck.check_vector_data('topography_patches', topography_patches)
    if not nl_colm['SinglePoint']:
        pass
    else:
        SITE_topography = topography_patches[0]

    if mpi.p_is_worker:
        del topography_patches







