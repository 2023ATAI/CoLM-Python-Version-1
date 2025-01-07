# ----------------------------------------------------------------------
# Percentage of Plant Function Types
#
# Original from Hua Yuan's OpenMP version.
#
# REVISIONS:
# Hua Yuan,      ?/2020 : for land cover land use classifications
# Shupeng Zhang, 01/2022: porting codes to MPI parallel version
# ----------------------------------------------------------------------
import os
import sys
import numpy as np
from CoLM_DataType import DataType
from CoLM_5x5DataReadin import CoLM_5x5DataReadin
from CoLM_AggregationRequestData import AggregationRequestData
import CoLM_RangeCheck
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector


def release(o):
    if o is not None:
        del o


def aggregation_percentagespft(var_global, mpi, nl_com, gblock, mesh, pixel, srfdiag, srfdata, crop, landpft, landpatch, gland, dir_rawdata,
                               dir_model_landdata,
                               lc_year):
    cyear = lc_year
    area_one = None
    pct_one = None
    readin = None
    pftPCT = None
    pct_pfts = None
    pct_pft_one =None

    landdir = os.path.join(dir_model_landdata, 'pctpft',str(cyear))

    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'pctpft' + '\\' + str(cyear)

    # ifdef USEMPI
    #    CALL mpi_barrier (p_comm_glb, p_err)
    # endif
    if mpi.p_is_master:
        print('Aggregate plant function type fractions ...')
        if not os.path.exists(landdir):
            os.makedirs(landdir)
    # ifdef USEMPI
    #    CALL mpi_barrier (p_comm_glb, p_err)
    # endif

    if nl_com['LULC_IGBP_PFT'] or nl_com['LULC_IGBP_PC']:
        if nl_com['SinglePoint']:
            if nl_com.nl_colm['USE_SITE_pctpfts']:
                return

        dir_5x5 = os.path.join(dir_rawdata, 'plant_15s')
        if 'win' in sys.platform:
            dir_5x5 = dir_rawdata + '\\' + 'plant_15s'
        # add parameter input for time year
        #    !write(cyear,'(i4.4)') lc_year
        cyear = str(lc_year)
        suffix = 'MOD' + cyear

        if mpi.p_is_io:
            dt = DataType(gblock)
            readin = CoLM_5x5DataReadin(mpi, gblock)
            pftPCT = dt.allocate_block_data2d(gland, readin.N_PFT_modis, lb1=0)

            readin.read_5x5_data_pft(dir_5x5, suffix, gland, 'PCT_PFT', pftPCT)
            if nl_com['USEMPI']:
                pass
                # CALL aggregation_data_daemon (gland, data_r8_3d_in1 = pftPCT, n1_r8_3d_in1 = N_PFT_modis)

        if mpi.p_is_worker:
            pct_pfts = np.zeros(landpft.numpft)

            for ipatch in range(landpatch.numpatch):
                ar = AggregationRequestData(nl_com['USEMPI'], mpi, mesh, pixel)
                if nl_com['CROP']:
                    if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0:
                        area_one = 0

                        area_one, _, _, _, _, _, _, pct_pft_one, _, _, _ = ar.aggregation_request_data(landpatch.landpatch,
                                                                                                       ipatch, gland,
                                                                                                       zip=nl_com[
                                                                                                           'USE_zip_for_aggregation'],
                                                                                                       area=area_one,
                                                                                                       data_r8_3d_in1=pftPCT,
                                                                                                       n1_r8_3d_in1=readin.N_PFT_modis,
                                                                                                       lb1_r8_3d_in1=0)
                        pct_one = sum(pct_pft_one)
                        pct_one = sum(pct_pft_one)
                        pct_one = max(pct_one, 1.0e-6)
                        sumarea = sum(area_one)

                        for ipft in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch]):
                            p = landpft.settyp(ipft)
                            pct_pfts[ipft] = sum(pct_pft_one[p, :] / pct_one * area_one) / sumarea

                        pct_pfts[landpft.patch_pft_s[ipatch]:landpft.patch_pft_e[ipatch]] = pct_pfts[
                                                                                            landpft.patch_pft_s[ipatch]:
                                                                                            landpft.patch_pft_e[ipatch]] \
                                                                                            / sum(
                            pct_pfts[landpft.patch_pft_s[ipatch]:landpft.patch_pft_e[ipatch]])
                    elif landpatch.landpatch.settyp[ipatch] == var_global.CROPLAND:
                        pct_pfts[landpft.patch_pft_s[ipatch]: landpft.patch_pft_e[ipatch]] = 1.
                else:
                    if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0 and landpatch.landpatch.settyp[
                            ipatch] != var_global.CROPLAND:
                        area_one = 0
                        area_one, _, _, _, _, _, _, pct_pft_one, _, _, _ = ar.aggregation_request_data(landpatch.landpatch,
                                                                                                       ipatch, gland,
                                                                                                       zip=nl_com[
                                                                                                           'USE_zip_for_aggregation'],
                                                                                                       area=area_one,
                                                                                                       data_r8_3d_in1=pftPCT,
                                                                                                       n1_r8_3d_in1=readin.N_PFT_modis,
                                                                                                       lb1_r8_3d_in1=0)

                        pct_one = sum(pct_pft_one)
                        pct_one = max(pct_one, 1.0e-6)
                        sumarea = sum(area_one)

                        for ipft in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch]):
                            p = landpft.settyp(ipft)
                            pct_pfts[ipft] = sum(pct_pft_one[p, :] / pct_one * area_one) / sumarea

                        pct_pfts[landpft.patch_pft_s[ipatch]:landpft.patch_pft_e[ipatch]] = pct_pfts[
                                                                                            landpft.patch_pft_s[ipatch]:
                                                                                            landpft.patch_pft_e[ipatch]] \
                                                                                            / sum(
                            pct_pfts[landpft.patch_pft_s[ipatch]:landpft.patch_pft_e[ipatch]])

            if nl_com['USEMPI']:
                pass
                # CALL aggregation_worker_done ()
            #    CALL mpi_barrier (p_comm_glb, p_err)

        if nl_com['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('PCT_PFTs ', pct_pfts,mpi, nl_com)
        # endif
        vectorones = CoLM_NetCDFVector(nl_com, mpi, gblock)
        if not nl_com['SinglePoint']:
            lndname = landdir + '/pct_pfts.nc'
            vectorones.ncio_create_file_vector(lndname, landpatch)
            vectorones.ncio_define_dimension_vector(lndname, landpft, 'pft')
            vectorones.ncio_write_vector(lndname, 'pct_pfts', 'pft', landpft, pct_pfts, 1)
            if nl_com['SrfdataDiag']:
                if nl_com['CROP']:
                    typpft = [ipft for ipft in range(var_global.N_PFT + var_global.N_CFT)]

                else:
                    typpft = list(range(var_global.N_PFT))

                lndname = dir_model_landdata + '/diag/pct_pfts_' + cyear + '.nc'
                srfdiag.srfdata_map_and_write(pct_pfts, landpft . settyp, typpft, srfdiag.m_pft2diag, -1.0e36, lndname, 'pctpfts', compress = 1, write_mode = 'one')
            else:
                srfdata.SITE_pctpfts = np.zeros(landpft.numpft)
                srfdata.SITE_pctpfts = pct_pfts

        if mpi.p_is_worker:
            release(pct_pfts)
            release(pct_one)
            release(area_one)
            release(pct_pft_one)

        if nl_com['CROP']:
            if not nl_com['SinglePoint']:
                lndname = landdir.strip() + '/pct_crops.nc'
                vectorones.ncio_create_file_vector(lndname, landpatch.landpatch)
                vectorones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                vectorones.ncio_write_vector(lndname, 'pct_crops', 'patch', landpatch.landpatch, crop.pctshrpch,
                                             nl_com['DEF_Srfdata_CompressLevel'])

                if nl_com['SrfdataDiag']:
                    typcrop = [ityp for ityp in range(1, var_global.N_CFT + 1)]
                    lndname = dir_model_landdata.strip() + '/diag/pct_crop_patch_' + cyear.strip() + '.nc'
                    srfdiag.srfdata_map_and_write(crop.pctshrpch, crop.cropclass, typcrop, srfdiag.m_patch2diag, -1.0e36, lndname,
                                          'pct_crop_patch',
                                          compress=1, write_mode='one')
            else:
                srfdata.SITE_croptyp = crop.cropclass.copy()
                srfdata.SITE_pctcrop = crop.pctshrpch.copy()