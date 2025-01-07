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
# ----------------------------------------------------------------------
import os
import sys
import numpy as np
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
from CoLM_AggregationRequestData import AggregationRequestData
import CoLM_RangeCheck
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector


def aggregation_soilbrightness(nl_colm, mpi, gblock, gland, dir_rawdata, dir_model_landdata, lc_year, landpatch, mesh,
                               pixel, srfdataDiag,srfdata):
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
    a_s_v_refl = None
    a_d_v_refl = None
    a_s_n_refl = None
    a_d_n_refl = None

    soil_s_v_alb = None
    soil_d_v_alb = None
    soil_s_n_alb = None
    numpatch = landpatch.numpatch

    landdir = os.path.join(dir_model_landdata, 'soil', str(cyear))

    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'soil' + '\\' + str(cyear)

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Aggregate Soil Brightness ...')
        if not os.path.exists(landdir):
            os.makedirs(landdir)

    if nl_colm['USEMPI']:
        pass

    if nl_colm['SinglePoint']:
        if nl_colm['USE_SITE_soilreflectance']:
            return

    # ................................................................................................
    #   aggregate the soil parameters from the resolution of raw data to modelling resolution
    # ................................................................................................
    lndname = os.path.join(dir_rawdata, 'soil_brightness.nc')

    if 'win' in sys.platform:
        lndname = dir_rawdata + '\\' + 'soil_brightness.nc'

    if mpi.p_is_io:
        # 分配数据结构
        datayype = DataType(gblock)
        isc = datayype.allocate_block_data(gland)
        a_s_v_refl = datayype.allocate_block_data(gland)
        a_d_v_refl = datayype.allocate_block_data(gland)
        a_s_n_refl = datayype.allocate_block_data(gland)
        a_d_n_refl = datayype.allocate_block_data(gland)

        isc = CoLM_NetCDFBlock.ncio_read_block(lndname, 'soil_brightness', mpi, gblock, gland, isc)

        # 遍历所有地块和像素
        for iblkme in range(gblock.nblkme):
            iblk = gblock.xblkme[iblkme]
            jblk = gblock.yblkme[iblkme]

            for iy in range(gland.ycnt[jblk]):
                for ix in range(gland.xcnt[iblk]):
                    ii = isc.blk[iblk, jblk].val[ix, iy]
                    if 1 <= ii < 20:
                        a_s_v_refl.blk[iblk, jblk].val[ix, iy] = soil_s_v_refl[ii]
                        a_d_v_refl.blk[iblk, jblk].val[ix, iy] = soil_d_v_refl[ii]
                        a_s_n_refl.blk[iblk, jblk].val[ix, iy] = soil_s_n_refl[ii]
                        a_d_n_refl.blk[iblk, jblk].val[ix, iy] = soil_d_n_refl[ii]
        if nl_colm['USEMPI']:
            pass
    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_worker:
        soil_s_v_alb = np.empty(numpatch)

        for ipatch in range(numpatch):
            L = landpatch.landpatch.settyp[ipatch]
            ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)

            if nl_colm['LULC_USGS']:
                if L != 16 and L != 24:

                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_s_v_refl)
                    soil_s_v_alb[ipatch] = np.median(soil_one)
                else:
                    soil_s_v_alb[ipatch] = -1.0e36
            else:
                if L != 17 and L != 15:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_s_v_refl)
                    soil_s_v_alb[ipatch] = np.median(soil_one)
                else:
                    soil_s_v_alb[ipatch] = -1.0e36

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_worker:
        soil_d_v_alb = np.empty(numpatch)

        for ipatch in range(numpatch):
            L = landpatch.landpatch.settyp[ipatch]
            ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)

            if nl_colm['LULC_USGS']:
                if L != 16 and L != 24:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_d_v_refl)
                    soil_d_v_alb[ipatch] = np.median(soil_one)
                else:
                    soil_d_v_alb[ipatch] = -1.0e36

            else:

                if L != 17 and L != 15:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_d_v_refl)
                    soil_d_v_alb[ipatch] = np.median(soil_one)
                else:
                    soil_d_v_alb[ipatch] = -1.0e36
        if nl_colm['USEMPI']:
            pass

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_worker:
        soil_s_n_alb = np.empty(numpatch)

        for ipatch in range(numpatch):
            L = landpatch.landpatch.settyp[ipatch]
            ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)

            if nl_colm['LULC_USGS']:
                if L != 16 and L != 24:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_s_n_refl)
                    soil_s_n_alb[ipatch] = np.median(soil_one)
                else:
                    soil_s_n_alb[ipatch] = -1.0e36

            else:

                if L != 17 and L != 15:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_s_n_refl)
                    soil_s_n_alb[ipatch] = np.median(soil_one)
                else:
                    soil_s_n_alb[ipatch] = -1.0e36

    if mpi.p_is_worker:
        soil_d_n_alb = np.empty(numpatch)
        ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)

        for ipatch in range(numpatch):
            L = landpatch.landpatch.settyp[ipatch]

            if nl_colm['LULC_USGS']:
                if L != 16 and L != 24:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_d_n_refl)
                    soil_d_n_alb[ipatch] = np.median(soil_one)
                else:
                    soil_d_n_alb[ipatch] = -1.0e36

            else:

                if L != 17 and L != 15:
                    _, soil_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                                          zip=nl_colm[
                                                                                              'USE_zip_for_aggregation'],
                                                                                          data_r8_2d_in1=a_d_n_refl)
                    soil_d_n_alb[ipatch] = np.median(soil_one)
                else:
                    soil_d_n_alb[ipatch] = -1.0e36
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('s_v_alb ', soil_s_v_alb, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('d_v_alb ', soil_d_v_alb, mpi, nl_colm, -1.e36)
            CoLM_RangeCheck.check_vector_data('s_n_alb ', soil_s_n_alb, mpi, nl_colm, -1.e36)
            CoLM_RangeCheck.check_vector_data('d_n_alb ', soil_d_n_alb, mpi, nl_colm, -1.e36)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'soil_s_v_alb_patches.nc')

            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'soil_s_v_alb_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'soil_s_v_alb', 'patch', landpatch.landpatch, soil_s_v_alb,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # Write-out the albedo of visible of the saturated soil to diagnostics file if enabled
            # (You need to implement the srfdata_map_and_write function or its equivalent)
            if nl_colm['SrfdataDiag']:
                typpatch = [(ityp, ityp) for ityp in range(landpatch.landpatch.N_land_classification)]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_brightness_patch_' + cyear + '.nc')

                srfdataDiag.srfdata_map_and_write(soil_s_v_alb, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36,
                                                  lndname, 'soil_s_v_alb', compress=1, write_mode='one')

            # (2) Write-out the albedo of visible of the dry soil
            lndname = os.path.join(landdir, 'soil_d_v_alb_patches.nc')

            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'soil_d_v_alb_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'soil_d_v_alb', 'patch', landpatch.landpatch, soil_d_v_alb,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # Write-out the albedo of visible of the dry soil to diagnostics file if enabled
            if nl_colm['SrfdataDiag']:
                typpatch = [(ityp, ityp) for ityp in range(landpatch.landpatch.N_land_classification)]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_brightness_patch_' + cyear + '.nc')
                srfdataDiag.srfdata_map_and_write(soil_d_v_alb, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36
                                                  , lndname, 'soil_d_v_alb', compress=1, write_mode='one')

            # (3) Write-out the albedo of near infrared of the saturated soil
            lndname = os.path.join(landdir, 'soil_s_n_alb_patches.nc')

            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'soil_s_n_alb_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'soil_s_n_alb', 'patch', landpatch.landpatch, soil_s_n_alb,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # Write-out the albedo of near infrared of the saturated soil to diagnostics file if enabled
            if nl_colm['SrfdataDiag']:
                typpatch = [(ityp, ityp) for ityp in range(landpatch.landpatch.N_land_classification)]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_brightness_patch_' + cyear + '.nc')
                srfdataDiag.srfdata_map_and_write(soil_s_n_alb, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36
                                                  , lndname, 'soil_s_n_alb', compress=1, write_mode='one')
            # endif

            # (4) Write-out the albedo of near infrared of the dry soil
            lndname = os.path.join(landdir, 'soil_d_n_alb_patches.nc')

            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'soil_d_n_alb_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'soil_d_n_alb', 'patch', landpatch.landpatch, soil_d_n_alb,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # Write-out the albedo of near infrared of the dry soil to diagnostics file if enabled
            if nl_colm['SrfdataDiag']:
                typpatch = [(ityp, ityp) for ityp in range(landpatch.landpatch.N_land_classification)]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_brightness_patch_' + cyear + '.nc')
                srfdataDiag.srfdata_map_and_write(soil_d_n_alb, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36
                                                  , lndname, 'soil_d_n_alb', compress=1, write_mode='one')
        else:
            srfdata.SITE_soil_s_v_alb = soil_s_v_alb[0]
            srfdata.SITE_soil_d_v_alb = soil_d_v_alb[0]
            srfdata.SITE_soil_s_n_alb = soil_s_n_alb[0]
            srfdata.SITE_soil_d_n_alb = soil_d_n_alb[0]

    if mpi.p_is_worker:
        del soil_s_v_alb
        del soil_d_v_alb
        del soil_s_n_alb
        del soil_d_n_alb
