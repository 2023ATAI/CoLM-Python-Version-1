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
import sys
import numpy as np
import CoLM_Utils
from CoLM_DataType import DataType
from CoLM_AggregationRequestData import AggregationRequestData
import CoLM_NetCDFBlock
import CoLM_RangeCheck
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
from CoLM_5x5DataReadin import CoLM_5x5DataReadin


def aggregation_forestheight(nl_colm, mpi, gblock, gland, dir_rawdata, dir_model_landdata, lc_year, landpatch, landpft,
                             mesh, pixel, srfdata, var_global):
    tree_height = None
    htop_one = None
    pct_one = None
    area_one = None
    pftPCT = None
    htop_pfts = None
    htop_patches = None
    htop = None
    tree_height_patches = None
    read_in = CoLM_5x5DataReadin(mpi, gblock)
    numpatch = landpatch.numpatch

    cyear = str(lc_year)
    landdir = os.path.join(dir_model_landdata.strip(), 'htop', cyear)
    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'htop' + '\\' + cyear

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Aggregate forest height ...')
        if not os.path.exists(landdir):
            print(landdir, 'landdir')
            os.makedirs(landdir.strip())

    if nl_colm['USEMPI']:
        pass

    # Single Point Check
    if nl_colm['SinglePoint']:
        if nl_colm['USE_SITE_htop']:
            return

    if nl_colm['LULC_USGS']:
        lndname = os.path.join(dir_rawdata, 'Forest_Height.nc')
        if 'win' in sys.platform:
            lndname = dir_rawdata + '\\' + 'Forest_Height.nc'
        print(lndname,'Forest_Height')

        if mpi.p_is_io:
            datayype = DataType(gblock)
            tree_height = datayype.allocate_block_data(gland)
            tree_height = CoLM_NetCDFBlock.ncio_read_block(lndname, 'lake_depth', mpi, gblock, gland, tree_height)

            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            tree_height_patches = np.zeros(numpatch)

            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0 and L != 1 and L != 16 and L != 24:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    _, tree_height_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                 gland, zip=nl_colm[
                            'USE_zip_for_aggregation'],
                                                                                                 data_r8_2d_in1=tree_height)
                    tree_height_patches[ipatch] = CoLM_Utils.median(tree_height_one)
                else:
                    tree_height_patches[ipatch] = -1.0e36

            if nl_colm['USEMPI']:
                pass

        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('htop_patches ', tree_height_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'htop_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'htop_patches.nc'
            print(lndname, 'htop_patches')
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'htop_patches', 'patch', landpatch.landpatch, tree_height_patches,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # SrfdataDiag
            if nl_colm['SrfdataDiag']:
                pass
                # typpatch = [ityp for ityp in range(N_land_classification + 1)]
                # lndname = dir_model_landdata.strip() + '/diag/htop_patch_' + cyear.strip() + '.nc'
                # srfdata_map_and_write(tree_height_patches, landpatch.landpatch.settyp, typpatch, m_patch2diag,
                #                       -1.0e36, lndname, 'htop', compress=1, write_mode='one')
        else:
            SITE_htop = tree_height_patches[0]

        if mpi.p_is_worker:
            del tree_height_patches

    if nl_colm['LULC_IGBP']:
        if mpi.p_is_io:
            dt = DataType(gblock)
            htop = dt.allocate_block_data(gland)
            dir_5x5 = os.path.join(dir_rawdata, 'plant_15s')
            suffix = 'MOD/' + cyear.strip()
            if 'win' in sys.platform:
                dir_5x5 = dir_rawdata + '\\' + 'plant_15s'
                suffix = 'MOD\\' + cyear

            htop = read_in.read_5x5_data(dir_5x5, suffix, gland, 'HTOP', htop)

            # MPI Aggregation
            if nl_colm['USEMPI']:
                pass
                # aggregation_data_daemon(gland, data_r8_2d_in1=htop)

        if mpi.p_is_worker:
            htop_patches = [0] * numpatch

            for ipatch in range(numpatch):
                if landpatch.landpatch.settyp[ipatch] != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one = 0
                    area_one, htop_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                 gland, zip=nl_colm[
                            'USE_zip_for_aggregation'],
                                                                                                 area=area_one,
                                                                                                 data_r8_2d_in1=htop)
                    htop_patches[ipatch] = sum(htop_one * area_one) / sum(area_one)

            if nl_colm['USEMPI']:
                pass
                # aggregation_worker_done()

        if nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb, p_err)

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('HTOP_patches ', htop_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'htop_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'htop_patches.nc'
            print(lndname, 'htop_patches.nc')
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'htop_patches', 'patch', landpatch.landpatch, htop_patches,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # SrfdataDiag
            if nl_colm['SrfdataDiag']:
                pass
                # typpatch = [ityp for ityp in range(N_land_classification + 1)]
                # lndname = dir_model_landdata.strip() + '/diag/htop_patch_' + cyear.strip() + '.nc'
                # srfdata_map_and_write(htop_patches, landpatch.landpatch.settyp, typpatch, m_patch2diag,
                #                       -1.0e36, lndname, 'htop', compress=1, write_mode='one')

        else:
            SITE_htop = htop_patches[0]

        # if mpi.p_is_worker:
        #     if htop_patches is not None:
        #         del htop_patches
        #     if htop_one is not None:
        #         del htop_one
        #     if area_one is not None:
        #         del area_one

    if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
        if mpi.p_is_io:
            datayype = DataType(gblock)
            htop = datayype.allocate_block_data(gland)
            pftPCT = datayype.allocate_block_data3(gland, read_in.N_PFT_modis, lb1=0)
        dir_5x5 = os.path.join(dir_rawdata, 'plant_15s')
        if 'win' in sys.platform:
            dir_5x5 = dir_rawdata + '\\' + 'plant_15s'
        suffix = 'MOD' + cyear.strip()

        if mpi.p_is_io:
            htop = read_in.read_5x5_data(dir_5x5, suffix, gland, 'HTOP', htop)
            pftPCT = read_in.read_5x5_data_pft(dir_5x5, suffix, gland, 'PCT_PFT', pftPCT)

            # MPI Aggregation
            if nl_colm['USEMPI']:
                pass
                # aggregation_data_daemon(gland, data_r8_2d_in1=htop, data_r8_3d_in1=pftPCT, n1_r8_3d_in1=16)

        if mpi.p_is_worker:
            htop_patches = np.zeros(numpatch)
            htop_pfts = np.zeros(landpft.numpft)

            for ipatch in range(numpatch):
                ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                area_one = 0
                area_one, htop_one, _, _, _, _, pct_one, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                gland, zip=nl_colm[
                        'USE_zip_for_aggregation'],
                                                                                                area=area_one,
                                                                                                data_r8_2d_in1=htop,
                                                                                                data_r8_3d_in1=pftPCT,
                                                                                                n1_r8_3d_in1=16,
                                                                                                lb1_r8_3d_in1=0)

                htop_patches[ipatch] = sum(htop_one * area_one) / sum(area_one)

                if nl_colm['CROP']:
                    if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0:
                        for ip in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch] + 1):
                            p = landpft.settyp[ip]
                            sumarea = sum(pct_one[p, :] * area_one)
                            if sumarea > 0:
                                htop_pfts[ip] = sum(htop_one * pct_one[p, :] * area_one) / sumarea
                            else:
                                htop_pfts[ip] = htop_patches[ipatch]
                    elif landpatch.landpatch.settyp[ipatch] == var_global.CROPLAND:
                        ip = landpft.patch_pft_s[ipatch]
                        htop_pfts[ip] = htop_patches[ipatch]
                else:
                    if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0 and landpatch.landpatch.settyp[
                        ipatch] != var_global.CROPLAND:
                        for ip in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch] + 1):
                            p = landpft.settyp[ip]
                            sumarea = sum(pct_one[p, :] * area_one)
                            if sumarea > 0:
                                htop_pfts[ip] = sum(htop_one * pct_one[p, :] * area_one) / sumarea
                            else:
                                htop_pfts[ip] = htop_patches[ipatch]

            # MPI Barrier
            if nl_colm['USEMPI']:
                pass
                # aggregation_worker_done()
        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('HTOP_patches ', htop_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('HTOP_pfts    ', htop_pfts, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'htop_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'htop_patches.nc'
            print(lndname, '11htop_patches.nc')
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'htop_patches', 'patch', landpatch.landpatch, htop_patches,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # SrfdataDiag
            if nl_colm['SrfdataDiag']:
                pass
                # typpatch = [ityp for ityp in range(N_land_classification + 1)]
                # lndname = dir_model_landdata.strip() + '/diag/htop_patch_' + cyear.strip() + '.nc'
                # srfdata_map_and_write(htop_patches, landpatch.landpatch.settyp, typpatch, m_patch2diag,
                #                       -1.0e36, lndname, 'htop', compress=1, write_mode='one')

            lndname = os.path.join(landdir, 'htop_pfts.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'htop_pfts.nc'
            print(lndname, 'htop_pfts.nc')

            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpft)
            vector_ones.ncio_define_dimension_vector(lndname, landpft, 'pft')
            vector_ones.ncio_write_vector(lndname, 'htop_pfts', 'pft', landpft, htop_pfts,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            # SrfdataDiag
            if nl_colm['SrfdataDiag']:
                pass
                # if 'CROP' not in globals():
                #     typpft = [ityp for ityp in range(N_PFT)]
                # else:
                #     typpft = [ityp for ityp in range(N_PFT + N_CFT)]
                # lndname = dir_model_landdata.strip() + '/diag/htop_pft_' + cyear.strip() + '.nc'
                # srfdata_map_and_write(htop_pfts, landpft.settyp, typpft, m_pft2diag,
                #                       -1.0e36, lndname, 'htop', compress=1, write_mode='one')
        else:
            srfdata.SITE_htop_pfts[:] = htop_pfts[:]

        if mpi.p_is_worker:
            if htop_patches is not None:
                del htop_patches
            if htop_pfts is not None:
                del htop_pfts
            if htop_one is not None:
                del htop_one
            if pct_one is not None:
                del pct_one
            if area_one is not None:
                del area_one
