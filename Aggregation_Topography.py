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
import sys
import numpy as np
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
import CoLM_RangeCheck
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector


def aggregation_topography(nl_colm, mpi, gblock, gtopo, dir_rawdata, dir_model_landdata, lc_year, landpatch,
                           mesh, pixel, sdd, srfdata):
    topography = None
    topography_patches = None
    numpatch = landpatch.numpatch
    if nl_colm['SrfdataDiag']:
        pass
        # integer:: typpatch(N_land_classification + 1), ityp
    cyear = str(lc_year)

    landdir = os.path.join(dir_model_landdata, 'topography', cyear)

    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'topography' + str(cyear)
    if '/media/zjl/7C24CC1724CBD276' in landdir:
        names = landdir.split('/')
        s = ''
        for n in names:
            s = '/' + n
        landdir = '/home/zjl' + s

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Aggregate Topography ...')
        os.system('mkdir -p ' + landdir.strip())

    if nl_colm['USEMPI']:
        pass

    if nl_colm['SinglePoint']:
        if nl_colm['USE_SITE_topography']:
            return

    lndname = os.path.join(dir_rawdata.strip(), 'elevation.nc')

    if mpi.p_is_io:
        datayype = DataType(gblock)
        topography = datayype.allocate_block_data(gtopo)

        lakedepth = CoLM_NetCDFBlock.ncio_read_block(lndname, 'elevation', mpi, gblock, gtopo, topography)
        if nl_colm['USEMPI']:
            pass

    # ----------------------------------------------------------------------
    #   aggregate the elevation from the resolution of raw data to modelling resolution
    # ----------------------------------------------------------------------

    if mpi.p_is_worker:
        topography_patches = np.zeros(numpatch)
        # topostd_patches = np.zeros(numpatch)
        ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
        area_one = 0
        for ipatch in range(numpatch):

            area_one, topography_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gtopo,
                                                                                               zip=nl_colm[
                                                                                                   'USE_zip_for_aggregation'],
                                                                                               area=area_one,
                                                                                               data_r8_2d_in1=topography)

            if np.any(topography_one != -9999.0):
                topography_patches[ipatch] = np.sum(topography_one * area_one,
                                                    where=topography_one != -9999.0) / np.sum(area_one,
                                                                                              where=topography_one != -9999.0)
            else:
                topography_patches[ipatch] = -1.0e36

        if nl_colm['USEMPI']:
            pass

    if nl_colm['USEMPI']:
        pass

    if nl_colm['RangeCheck']:
        CoLM_RangeCheck.check_vector_data('topography_patches ', topography_patches, mpi, nl_colm)

    if not nl_colm['SinglePoint']:
        lndname = os.path.join(landdir, 'topography_patches.nc')

        if 'win' in sys.platform:
            lndname = landdir + '\\' + 'topography_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'topography_patches', 'patch', landpatch.landpatch, topography_patches, nl_colm['DEF_Srfdata_CompressLevel'])

        if nl_colm['SrfdataDiag']:
            pass
            # typpatch = np.arange(N_land_classification)
            #
            #
            # lndname = os.path.join(dir_model_landdata.strip(), 'diag', 'topo_' + str(cyear) + '.nc')
            #
            # if 'win' in sys.platform:
            #     lndname = dir_model_landdata + '\\' + '\\diag\\topo_' + cyear + '.nc'
            #
            # srfdataDiag.srfdata_map_and_write(lakedepth_patches, landpatch.landpatch.settyp, typlake, srfdataDiag.m_patch2diag,
            #                                   -1.0e36, lndname, 'lakedepth', compress=1, write_mode='one')
            #
            # srfdataDiag.SITE_lakedepth = lakedepth_patches[0]
            # # Calculate and write element-wise topography data to a diagnostic NetCDF file
            # if p_is_worker:
            #     topo_elm = np.zeros(numelm)
            #     for i in range(numelm):
            #         ps = elm_patch.substt[i]
            #         pe = elm_patch.subend[i]
            #         topo_elm[i] = np.sum(topography_patches[ps:pe] * elm_patch.subfrc[ps:pe])
            #
            #     lndname = os.path.join(dir_model_landdata.strip(), 'diag', 'topo_elm_' + str(cyear) + '.nc')
            #     write_topography_diag_to_netcdf(lndname, topo_elm)
            #
            #     if topo_elm is not None:
            #         del topo_elm

        else:
            srfdata.SITE_topography = topography_patches[0]

    # Deallocate memory if applicable
    if mpi.p_is_worker:
        del topography_patches
