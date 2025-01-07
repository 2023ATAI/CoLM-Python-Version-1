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
#
# REVISIONS:
# ----------------------------------------------------------------------
import os
import sys
import numpy as np
import CoLM_Utils
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
import CoLM_RangeCheck


def aggregation_lakedepth(gland, dir_rawdata, dir_model_landdata, lc_year,
                          nl_colm, mpi, gblock, mesh, pixel, var_global, landpatch, srfdataDiag, srfdata):
    lakedepth = None
    lakedepth_patches = None
    typlake = None
    numpatch = landpatch.numpatch
    if nl_colm['SrfdataDiag']:
        typlake = [17]
    cyear = lc_year

    landdir = os.path.join(dir_model_landdata, 'lakedepth', str(cyear))

    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'lakedepth' + str(cyear)

    if '/media/zjl/7C24CC1724CBD276' in landdir:
        names = landdir.split('/')
        s = ''
        for n in names:
            s = '/' + n
        landdir = '/home/zjl' + s

    if nl_colm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)

    if mpi.p_is_master:
        print('Aggregate lake depth ...')
        if not os.path.exists(landdir):
            os.makedirs(landdir)
    if nl_colm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)

    if nl_colm['SinglePoint']:
        if nl_colm['USE_SITE_lakedepth']:
            return

    # ................................................
    # global lake coverage and lake depth
    # ................................................
    lndname = os.path.join(dir_rawdata, 'lake_depth.nc')

    if 'win' in sys.platform:
        lndname = dir_rawdata + '\\' + 'lake_depth.nc'

    if mpi.p_is_io:
        dt = DataType(gblock)
        lakedepth = dt.allocate_block_data(gland)
        lakedepth = CoLM_NetCDFBlock.ncio_read_block(lndname, 'lake_depth', mpi, gblock, gland, lakedepth)
        lakedepth = dt.block_data_linear_transform(lakedepth, scl=0.1)

        if nl_colm['USEMPI']:
            pass
        # CALL aggregation_data_daemon (gland, data_r8_2d_in1 = lakedepth)

    #   ---------------------------------------------------------------
    #   aggregate the lake depth from the resolution of raw data to modelling resolution
    #   ---------------------------------------------------------------

    if mpi.p_is_worker:

        lakedepth_patches = np.zeros(numpatch)

        for ipatch in range(numpatch):
            L = landpatch.landpatch.settyp[ipatch]
            if L == var_global.WATERBODY:  # LAND WATER BODIES (17)
                ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                _,lakedepth_one,_,_,_,_,_,_,_,_,_ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                             zip=nl_colm['USE_zip_for_aggregation'],
                                                             data_r8_2d_in1=lakedepth)
                lakedepth_patches[ipatch] = CoLM_Utils.median(lakedepth_one, len(lakedepth_one))
            else:
                lakedepth_patches[ipatch] = -1.0e36

        if nl_colm['USEMPI']:
            pass
            # CALL aggregation_worker_done ()
    if nl_colm['USEMPI']:
        pass
    # CALL mpi_barrier (p_comm_glb, p_err)

    if nl_colm['RangeCheck']:
        CoLM_RangeCheck.check_vector_data('lakedepth_patches ', lakedepth_patches, mpi, nl_colm)

    if nl_colm['SinglePoint']:
        lndname = os.path.join(landdir, 'lakedepth_patches.nc')

        if 'win' in sys.platform:
            lndname = landdir + '\\' + 'lakedepth_patches.nc'
        vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
        vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
        vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
        vector_ones.ncio_write_vector(lndname, 'lakedepth_patches', 'patch', landpatch.landpatch, lakedepth_patches, 1)

    if nl_colm['SrfdataDiag']:
        lndname = os.path.join(dir_model_landdata, 'diag','lakedepth_'+str(cyear)+'.nc')

        if 'win' in sys.platform:
            lndname = dir_model_landdata + '\\' + '\\diag\\lakedepth_' + str(cyear) + '.nc'
        srfdataDiag.srfdata_map_and_write(lakedepth_patches, landpatch.landpatch.settyp, typlake, srfdataDiag.m_patch2diag,
                                          -1.0e36, lndname, 'lakedepth', compress=1, write_mode='one')
    else:
        srfdata.SITE_lakedepth = lakedepth_patches[0]
    # endif

    if mpi.p_is_worker:
        del lakedepth_patches
