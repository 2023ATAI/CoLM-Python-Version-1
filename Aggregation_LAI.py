# ----------------------------------------------------------------------
# DESCRIPTION:
# 1. Global Plant Leaf Area Index
#     (http://globalchange.bnu.edu.cn)
#     Yuan H., et al., 2011:
#     Reprocessing the MODIS Leaf Area Index products for land surface
#     and climate modelling. Remote Sensing of Environment, 115: 1171-1187.
#
#  Created by Yongjiu Dai, 02/2014
#
#  REVISIONS:

# ----------------------------------------------------------------------
import os
import sys
import numpy as np
from CoLM_DataType import DataType
import CoLM_TimeManager
import CoLM_NetCDFBlock
import CoLM_RangeCheck
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_5x5DataReadin import CoLM_5x5DataReadin
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector


def aggregation_lai(nl_colm, mpi, gblock, gridlai, dir_rawdata, dir_model_landdata, lc_year, landpatch, landpft,
                    srfdata, mesh, pixel, var_global):
    if nl_colm['SrfdataDiag']:
        pass
    idate = np.zeros(3,dtype='int')
    simulation_lai_year_end = 0
    simulation_lai_year_start = 0
    numpatch = landpatch.numpatch
    start_year = 0
    end_year = 0
    ntime = 0
    Julian_day = 0
    c2 = ''
    c3 = ''
    datayype = None
    LAI = None
    SAI = None
    pftPCT = None

    area_one = None
    LAI_patches = None
    SAI_patches = None
    # srfdata.SITE_SAI_monthly = None
    lai_one = None
    sai_one = None
    pct_one = None
    pct_pft_one = None
    SAI_pfts = None
    LAI_pfts = None

    pftLSAI = None
    read_in = CoLM_5x5DataReadin(mpi, gblock)

    landdir = os.path.join(dir_model_landdata, 'LAI')
    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'LAI'

    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Aggregate LAI ...')
        if not os.path.exists(landdir):
            os.makedirs(landdir)

    if nl_colm['USEMPI']:
        pass

    if nl_colm['SinglePoint']:
        # 检查是否使用了 USE_SITE_LAI
        if nl_colm['USE_SITE_LAI']:
            return

    # idate[0] = nl_colm['DEF_simulation_time'].start_year
    idate[0] = nl_colm['start_year']
    # [nl_colm['DEF_simulation_time'].start_year, 0, 0]  # 初始化 idate 列表，第一个元素为年份，后两个元素为月份和日期（暂时置为 0）

    if not nl_colm['greenwich']:
        idate[2] = nl_colm['start_sec']
        # idate[2] = nl_colm['DEF_simulation_time'].start_sec  # 如果不是以 GMT 为基准时间，则将秒数存储在 idate 的第三个位置

        # 调用 monthday2julian 函数
        idate[0] = CoLM_TimeManager.monthday2julian(nl_colm['start_month'],
                                                    nl_colm['start_day'],
                                                    idate[1])
        # idate[0] = CoLM_TimeManager.monthday2julian(nl_colm['DEF_simulation_time'].start_year,
        #                                             nl_colm['DEF_simulation_time'].start_month,
        #                                             nl_colm['DEF_simulation_time'].start_day)

        # 调用 localtime2gmt 函数
        idate = CoLM_TimeManager.localtime2gmt(idate)

    simulation_lai_year_start = idate[0]
    idate[0] = nl_colm['end_year']
    # idate[1] = nl_colm['DEF_simulation_time'].end_year

    if not nl_colm['greenwich']:
        idate[2] = nl_colm['end_sec']
        # idate[2] = nl_colm['DEF_simulation_time'].end_sec

        # idate[0] = CoLM_TimeManager.monthday2julian(nl_colm['DEF_simulation_time'].end_month,
        #                                             nl_colm['DEF_simulation_time'].end_day, idate[2])
        idate[0] = CoLM_TimeManager.monthday2julian(nl_colm['end_month'],
                                                    nl_colm['end_day'], idate[1])
        idate = CoLM_TimeManager.localtime2gmt(idate)

    simulation_lai_year_end = idate[0]
    #  global plant leaf area index

    if nl_colm['LULC_USGS'] or nl_colm['LULC_IGBP']:

        if nl_colm['DEF_LAI_MONTHLY']:
            if nl_colm['LULCC']:
                start_year = lc_year
                end_year = lc_year
                ntime = 12

            if nl_colm['LULCC']:
                start_year = lc_year
                end_year = lc_year
                ntime = 12
            else:
                if nl_colm['DEF_LAI_CHANGE_YEARLY']:
                    start_year = simulation_lai_year_start
                    end_year = simulation_lai_year_end
                    ntime = 12
                else:
                    start_year = lc_year
                    end_year = lc_year
                    ntime = 12
        else:
            start_year = simulation_lai_year_start
            end_year = simulation_lai_year_end
            ntime = 46
        # -----LAI-----
        if mpi.p_is_io:
            datayype = DataType(gblock)
            LAI = datayype.allocate_block_data(gridlai)

        if mpi.p_is_worker:
            LAI_patches = np.empty(numpatch)

    if nl_colm['SinglePoint']:
        srfdata.SITE_LAI_year = [iy for iy in range(start_year, end_year + 1)]

        if nl_colm['DEF_LAI_MONTHLY']:
            srfdata.SITE_LAI_monthly = np.empty((12, end_year - start_year + 1))
        else:
            srfdata.SITE_LAI_8day = np.empty((46, end_year - start_year + 1))

    if not nl_colm['DEF_USE_LAIFEEDBACK']:
        for iy in range(start_year, end_year + 1):
            cyear = str(iy)
            if not os.path.exists(landdir):
                os.makedirs(landdir + cyear)

            for itime in range(ntime):
                if nl_colm['DEF_LAI_MONTHLY']:
                    c3 = str(itime).zfill(2)
                else:
                    Julian_day = 1 + itime * 8
                    c3 = str(Julian_day).zfill(3)

                if mpi.p_is_master:
                    print('Aggregate LAI :', iy, ':', itime, '/', ntime)

                if mpi.p_is_io:
                    if nl_colm['DEF_LAI_MONTHLY']:
                        dir_5x5 = os.path.join(dir_rawdata.strip(), 'plant_15s')
                        suffix = 'MOD' + cyear
                        read_in = CoLM_5x5DataReadin(mpi, gblock)
                        LAI = read_in.read_5x5_data_time(dir_5x5, suffix, gridlai, 'MONTHLY_LC_LAI', itime, LAI)
                    else:
                        lndname = os.path.join(dir_rawdata.strip(), 'lai_15s_8day',
                                               'lai_8-day_15s_' + cyear + '.nc')

                        # dt = DataType(gblock)
                        # lakedepth = dt.allocate_block_data(gland)

                        LAI = CoLM_NetCDFBlock.ncio_read_block_time(lndname, 'lai', gridlai, itime, LAI, mpi, gblock)
                        LAI = datayype.block_data_linear_transform(LAI, scl=0.1)

                    if nl_colm['USEMPI']:
                        pass

                # aggregate the plant leaf area index from the resolution of raw data to modelling resolution
                if mpi.p_is_worker:
                    for ipatch in range(numpatch):
                        ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                        area_one = 0
                        area_one, lai_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                    gridlai,
                                                                                                    zip=nl_colm[
                                                                                                        'USE_zip_for_aggregation'],
                                                                                                    area=area_one,
                                                                                                    data_r8_2d_in1=LAI)

                        # 计算 LAI_patches
                        LAI_patches[ipatch] = np.sum(lai_one * area_one) / np.sum(area_one)

                        if nl_colm['USEMPI']:
                            pass

                if nl_colm['RangeCheck']:
                    CoLM_RangeCheck.check_vector_data('LAI value ' + c3, LAI_patches, mpi, nl_colm)

                if nl_colm['USEMPI']:
                    pass

                # write out the plant leaf area index of grid patches
                if not nl_colm['SinglePoint']:
                    # 单点案例
                    # TODO: 时间年的参数输入
                    if nl_colm['DEF_LAI_MONTHLY']:
                        lndname = f"{landdir.strip()}{cyear.strip()}/LAI_patches{c3.strip()}.nc"
                    else:
                        # TODO: rename filename of 8-day LAI
                        lndname = f"{landdir.strip()}{cyear.strip()}/LAI_patches{c3.strip()}.nc"
                    vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                    vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
                    vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                    vector_ones.ncio_write_vector(lndname, 'LAI_patches', 'patch', landpatch.landpatch, LAI_patches,
                                                  nl_colm['DEF_Srfdata_CompressLevel'])

                    if nl_colm['SrfdataDiag']:
                        pass
                        # typpatch = np.arange(0, N_land_classification + 1)
                        #
                        # lndname = f"{dir_model_landdata.strip()}/diag/LAI_patch_{cyear.strip()}.nc"
                        #
                        # if DEF_LAI_MONTHLY:
                        #     varname = 'LAI'
                        # else:
                        #     # TODO: rename filename of 8-day LAI
                        #     varname = 'LAI_8-day'
                        #
                        # srfdata_map_and_write(LAI_patches, landpatch.landpatch.settyp, typpatch, m_patch2diag,
                        #                       -1.0e36, lndname, varname.strip(), compress=0, write_mode='one',
                        #                       lastdimname='Itime', lastdimvalue=itime)
                else:
                    if nl_colm['DEF_LAI_MONTHLY']:
                        srfdata.SITE_LAI_monthly[itime, iy - start_year] = LAI_patches[0]
                    else:
                        srfdata.SITE_LAI_8day[itime, iy - start_year] = LAI_patches[0]

    if nl_colm['DEF_LAI_MONTHLY']:
        if mpi.p_is_io:
            dt = DataType(gblock)
            SAI = dt.allocate_block_data(gridlai)

        if mpi.p_is_worker:
            SAI_patches = np.empty(numpatch)

        if not nl_colm['SinglePoint']:
            srfdata.SITE_SAI_monthly = np.empty((12, end_year - start_year + 1))

        dir_5x5 = os.path.join(dir_rawdata.strip(), 'plant_15s')

        for iy in range(start_year, end_year + 1):
            cyear = str(iy).zfill(4)
            suffix = 'MOD' + cyear

            for itime in range(0, 12):
                # print(itime,'==============')
                c3 = str(itime).zfill(2)

                if mpi.p_is_master:
                    print('Aggregate SAI :', iy, ':', itime + 1, '/', ntime)

                if mpi.p_is_io:
                    read_in = CoLM_5x5DataReadin(mpi, gblock)
                    SAI = read_in.read_5x5_data_time(dir_5x5, suffix, gridlai, 'MONTHLY_LC_SAI', itime, SAI)

                    if nl_colm['USEMPI']:
                        pass
                # -------------------------------------------------------------------------------------------
                # aggregate the plant stem area index from the resolution of raw data to modelling resolution
                # -------------------------------------------------------------------------------------------
                if mpi.p_is_worker:
                    for ipatch in range(numpatch):
                        ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                        area_one = 0
                        area_one, sai_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                    gridlai,
                                                                                                    zip=nl_colm[
                                                                                                        'USE_zip_for_aggregation'],
                                                                                                    area=area_one,
                                                                                                    data_r8_2d_in1=SAI)
                        SAI_patches[ipatch] = np.sum(sai_one * area_one) / np.sum(area_one)

                        if nl_colm['USEMPI']:
                            pass

                if nl_colm['RangeCheck']:
                    CoLM_RangeCheck.check_vector_data('LAI value ' + c3, LAI_patches, mpi, nl_colm)
                if nl_colm['USEMPI']:
                    pass
                # write out the plant leaf area index of grid patches
                if not nl_colm['SinglePoint']:
                    # 单点案例
                    # TODO: 时间年的参数输入
                    lndname = f"{landdir.strip()}{cyear.strip()}/SAI_patches{c3.strip()}.nc"
                    vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                    vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
                    vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                    vector_ones.ncio_write_vector(lndname, 'SAI_patches', 'patch', landpatch.landpatch, SAI_patches,
                                                  nl_colm['DEF_Srfdata_CompressLevel'])

                    if nl_colm['SrfdataDiag']:
                        pass
                else:
                    srfdata.SITE_SAI_monthly[itime, iy - start_year] = SAI_patches[1]

    if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
        if nl_colm['LULCC']:
            start_year = lc_year
            end_year = lc_year
            ntime = 12
        else:
            if nl_colm['DEF_LAI_CHANGE_YEARLY']:
                start_year = simulation_lai_year_start
                end_year = simulation_lai_year_end
                ntime = 12
            else:
                start_year = lc_year
                end_year = lc_year
                ntime = 12

        if mpi.p_is_io:
            dt = DataType(gblock)
            pftLSAI = dt.allocate_block_data3(gridlai, read_in.N_PFT_modis, lb1=0)
            dt = DataType(gblock)
            pftPCT = dt.allocate_block_data3(gridlai, read_in.N_PFT_modis, lb1=0)

        if mpi.p_is_worker:
            LAI_patches = np.empty(numpatch)
            LAI_pfts = np.empty(landpft.numpft)
            SAI_patches = np.empty(numpatch)
            SAI_pfts = np.empty(landpft.numpft)

        if nl_colm['SinglePoint']:
            SITE_LAI_year = np.arange(start_year, end_year + 1)

            # TODO-yuan-done: for multiple years
            srfdata.SITE_LAI_pfts_monthly = np.empty((landpft.numpft, 12, end_year - start_year + 1))
            srfdata.SITE_SAI_pfts_monthly = np.empty((landpft.numpft, 12, end_year - start_year + 1))

        dir_5x5 = os.path.join(dir_rawdata, 'plant_15s')
        if 'win' in sys.platform:
            dir_5x5 = dir_rawdata + '\\' + 'plant_15s'

        for iy in range(start_year, end_year + 1):
            cyear = str(iy)
            suffix = 'MOD' + cyear
            if not os.path.exists(landdir + cyear):
                os.makedirs(landdir + cyear)

            if mpi.p_is_io:
                pftPCT = read_in.read_5x5_data_pft(dir_5x5, suffix, gridlai, 'PCT_PFT', pftPCT)

            if not nl_colm['DEF_USE_LAIFEEDBACK']:
                for month in range(1, 13):
                    if mpi.p_is_io:
                        pftLSAI = read_in.read_5x5_data_pft_time(dir_5x5, suffix, gridlai, 'MONTHLY_PFT_LAI', month,
                                                                 pftLSAI)
                        if nl_colm['USEMPI']:
                            pass
                            # aggregation_data_daemon(gridlai, data_r8_3d_in1=pftPCT, n1_r8_3d_in1=16,
                            #                     data_r8_3d_in2=pftLSAI, n1_r8_3d_in2=16)

                    if mpi.p_is_worker:
                        for ipatch in range(numpatch):
                            ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                            area_one = 0
                            area_one, pct_pft_one, lai_pft_one, _, _, _, _, _, _, _ = ard.aggregation_request_data(
                                landpatch.landpatch, ipatch, gridlai, zip=nl_colm['USE_zip_for_aggregation'],
                                area=area_one,
                                data_r8_3d_in1=pftPCT,
                                n1_r8_3d_in1=16, lb1_r8_3d_in1=0,
                                data_r8_3d_in2=pftLSAI,
                                n1_r8_3d_in2=16, lb1_r8_3d_in2=0)

                            lai_one = np.empty(len(area_one))
                            pct_one = np.empty(len(area_one))

                            pct_one = np.sum(pct_pft_one, axis=0)
                            pct_one = np.maximum(pct_one, 1.0e-6)

                            lai_one = np.sum(lai_pft_one * pct_pft_one, axis=0) / pct_one
                            LAI_patches[ipatch] = np.sum(lai_one * area_one) / np.sum(area_one)
                            if not nl_colm['CROP']:
                                if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0:
                                    for ip in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch] + 1):
                                        p = landpft.settyp[ip]
                                        sumarea = np.sum(pct_pft_one[p, :] * area_one)
                                        if sumarea > 0:
                                            LAI_pfts[ip] = np.sum(
                                                lai_pft_one[p, :] * pct_pft_one[p, :] * area_one) / sumarea
                                        else:
                                            LAI_pfts[ip] = 0.0
                                # #ifdef CROP
                                elif landpatch.landpatch.settyp[ipatch] == var_global.CROPLAND:
                                    ip = landpft.patch_pft_s[ipatch]
                                    LAI_pfts[ip] = LAI_patches[ipatch]
                            else:
                                if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0 and landpatch.landpatch.settyp[
                                    ipatch] != var_global.CROPLAND:
                                    for ip in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch] + 1):
                                        p = landpft.settyp[ip]
                                        sumarea = np.sum(pct_pft_one[p, :] * area_one)
                                        if sumarea > 0:
                                            LAI_pfts[ip] = np.sum(
                                                lai_pft_one[p, :] * pct_pft_one[p, :] * area_one) / sumarea
                                        else:
                                            LAI_pfts[ip] = 0.0
                                    # #ifdef CROP
                                elif landpatch.landpatch.settyp[ipatch] == var_global.CROPLAND:
                                    ip = landpft.patch_pft_s[ipatch]
                                    LAI_pfts[ip] = LAI_patches[ipatch]

                            # #endif
                        if nl_colm['USEMPI']:
                            pass
                            # aggregation_worker_done()
                    c2 = str(month)
                    if nl_colm['RangeCheck']:
                        CoLM_RangeCheck.check_vector_data('LAI_patches ' + c2, LAI_patches, mpi, nl_colm)
                        CoLM_RangeCheck.check_vector_data('LAI_pfts    ' + c2, LAI_pfts, mpi, nl_colm)
                    # #endif
                    if nl_colm['USEMPI']:
                        pass
                        # mpi_barrier(p_comm_glb, p_err)

                    if not nl_colm['SinglePoint']:
                        lndname = os.path.join(landdir+cyear, 'LAI_patches'+c2+'.nc')
                        if 'win' in sys.platform:
                            lndname = landdir+cyear + '\\' + 'LAI_patches'+c2+'.nc'
                        vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                        vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
                        vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                        vector_ones.ncio_write_vector(lndname, 'LAI_patches', 'patch', landpatch.landpatch, LAI_patches,
                                                      nl_colm['DEF_Srfdata_CompressLevel'])

                        if nl_colm['SrfdataDiag']:
                            pass
                            # typpatch = np.arange(0, N_land_classification)
                            # lndname = dir_model_landdata + '/diag/LAI_patch_' + cyear + '.nc'
                            # varname = 'LAI'
                            # srfdata_map_and_write(LAI_patches, landpatch.landpatch.settyp, typpatch, m_patch2diag, -1.0e36, lndname,
                            #                       varname, compress=0, write_mode='one', lastdimname='Itime',
                            #                       lastdimvalue=month)

                        lndname = os.path.join(landdir + cyear, 'LAI_pfts' + c2 + '.nc')
                        if 'win' in sys.platform:
                            lndname = landdir + cyear + '\\' + 'LAI_pfts' + c2 + '.nc'
                        vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                        vector_ones.ncio_create_file_vector(lndname, landpft)
                        vector_ones.ncio_define_dimension_vector(lndname, landpft, 'pft')
                        vector_ones.ncio_write_vector(lndname, 'LAI_pfts', 'pft', landpft, LAI_pfts,
                                                      nl_colm['DEF_Srfdata_CompressLevel'])

                    if nl_colm['SrfdataDiag']:
                        pass
                    # # #ifndef CROP
                    # typpft = np.arange(0, N_PFT)
                    # # #else
                    # # typpft = np.arange(0, N_PFT + N_CFT)
                    # # #endif
                    # lndname = dir_model_landdata + '/diag/LAI_pft_' + cyear + '.nc'
                    # varname = 'LAI_pft'
                    # srfdata_map_and_write(LAI_pfts, landpft.settyp, typpft, m_pft2diag, -1.0e36, lndname,
                    #                       varname, compress=0, write_mode='one', lastdimname='Itime',
                    #                       lastdimvalue=month)
                    else:
                        srfdata.SITE_LAI_pfts_monthly[:, month, iy - start_year] = LAI_pfts[:]

            for month in range(1, 13):
                if mpi.p_is_io:
                    pftLSAI = read_in.read_5x5_data_pft_time(dir_5x5, suffix, gridlai, 'MONTHLY_PFT_SAI', month,
                                                             pftLSAI)
                    if nl_colm['USEMPI']:
                        pass
                    # aggregation_data_daemon(gridlai, data_r8_3d_in1=pftPCT, n1_r8_3d_in1=16,
                    #                         data_r8_3d_in2=pftLSAI, n1_r8_3d_in2=16)

                # -------------------------------------------------------------------------------------------
                # aggregate the plant leaf area index from the resolution of raw data to modelling resolution
                # -------------------------------------------------------------------------------------------

                if mpi.p_is_worker:
                    for ipatch in range(numpatch):
                        ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                        area_one = 0
                        area_one, _, _, _, _, _, _, pct_pft_one, sai_pft_one, _, _ = ard.aggregation_request_data(
                            landpatch.landpatch, ipatch, gridlai, zip=nl_colm['USE_zip_for_aggregation'], area=area_one,
                            data_r8_3d_in1=pftPCT, n1_r8_3d_in1=16,
                            lb1_r8_3d_in1=0,
                            data_r8_3d_in2=pftLSAI, n1_r8_3d_in2=16,
                            lb1_r8_3d_in2=0)

                        sai_one = np.empty(len(area_one))
                        pct_one = np.empty(len(area_one))

                        pct_one = np.sum(pct_pft_one, axis=0)
                        pct_one = np.maximum(pct_one, 1.0e-6)

                        sai_one = np.sum(sai_pft_one * pct_pft_one, axis=0) / pct_one
                        SAI_patches[ipatch] = np.sum(sai_one * area_one) / np.sum(area_one)
                        if nl_colm['CROP']:
                            if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0:
                                for ip in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch] + 1):
                                    p = landpft.settyp[ip]
                                    sumarea = np.sum(pct_pft_one[p, :] * area_one)
                                    if sumarea > 0:
                                        SAI_pfts[ip] = np.sum(
                                            sai_pft_one[p, :] * pct_pft_one[p, :] * area_one) / sumarea
                                    else:
                                        SAI_pfts[ip] = 0.0
                        else:
                            if landpft.const_lc.patchtypes[landpatch.landpatch.settyp[ipatch]] == 0 and landpatch.landpatch.settyp(
                                    ipatch) != var_global.CROPLAND:
                                for ip in range(landpft.patch_pft_s[ipatch], landpft.patch_pft_e[ipatch] + 1):
                                    p = landpft.settyp[ip]
                                    sumarea = np.sum(pct_pft_one[p, :] * area_one)
                                    if sumarea > 0:
                                        SAI_pfts[ip] = np.sum(
                                            sai_pft_one[p, :] * pct_pft_one[p, :] * area_one) / sumarea
                                    else:
                                        SAI_pfts[ip] = 0.0
                            elif landpatch.landpatch.settyp[ipatch] == var_global.CROPLAND:
                                ip = landpft.patch_pft_s[ipatch]
                                SAI_pfts[ip] = SAI_patches[ipatch]

                    if nl_colm['USEMPI']:
                        pass
                    # aggregation_worker_done()

                c2 = str(month)
                if nl_colm['RangeCheck']:
                    CoLM_RangeCheck.check_vector_data('SAI_patches ' + c2, SAI_patches, mpi, nl_colm)
                    CoLM_RangeCheck.check_vector_data('SAI_pfts    ' + c2, SAI_pfts, mpi, nl_colm)

                if nl_colm['USEMPI']:
                    pass

                # ---------------------------------------------------
                # write out the plant stem area index of grid patches
                # ---------------------------------------------------
                if not nl_colm['SinglePoint']:
                    lndname = os.path.join(landdir + cyear, 'SAI_patches' + c2 + '.nc')
                    if 'win' in sys.platform:
                        lndname = landdir + cyear + '\\' + 'SAI_patches' + c2 + '.nc'
                    vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                    vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
                    vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                    vector_ones.ncio_write_vector(lndname, 'SAI_patches', 'patch', landpatch.landpatch, SAI_patches,
                                                  nl_colm['DEF_Srfdata_CompressLevel'])

                    if nl_colm['SrfdataDiag']:
                        pass
                        # typpatch = np.arange(0, N_land_classification)
                        # lndname = dir_model_landdata.strip() + '/diag/SAI_patch_' + cyear.strip() + '.nc'
                        # varname = 'SAI'
                        # srfdata_map_and_write(SAI_patches, landpatch.landpatch.settyp, typpatch, m_patch2diag, -1.0e36, lndname,
                        #                       varname, compress=0, write_mode='one', lastdimname='Itime', lastdimvalue=month)

                    lndname = os.path.join(landdir + cyear, 'SAI_pfts' + c2 + '.nc')
                    if 'win' in sys.platform:
                        lndname = landdir + cyear + '\\' + 'SAI_pfts' + c2 + '.nc'
                    vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                    vector_ones.ncio_create_file_vector(lndname, landpft)
                    vector_ones.ncio_define_dimension_vector(lndname, landpft, 'pft')
                    vector_ones.ncio_write_vector(lndname, 'SAI_pfts', 'pft', landpft, SAI_pfts,
                                                  nl_colm['DEF_Srfdata_CompressLevel'])

                    if nl_colm['SrfdataDiag']:
                        pass
                        # #ifndef CROP
                        # typpft = np.arange(0, N_PFT - 1)
                        # # #else
                        # # typpft = np.arange(0, N_PFT + N_CFT - 1)
                        # # #endif
                        # lndname = dir_model_landdata.strip() + '/diag/SAI_pft_' + cyear.strip() + '.nc'
                        # varname = 'SAI_pft'
                        # srfdata_map_and_write(SAI_pfts, landpft.settyp, typpft, m_pft2diag, -1.0e36, lndname,
                        #                       varname, compress=0, write_mode='one', lastdimname='Itime', lastdimvalue=month)
                else:
                    srfdata.SITE_SAI_pfts_monthly[:, month, iy - start_year] = SAI_pfts[:]

        if mpi.p_is_worker:
            if LAI_patches is not None:
                del LAI_patches
            if LAI_pfts is not None:
                del LAI_pfts
            if lai_one is not None:
                del lai_one
            if SAI_patches is not None:
                del SAI_patches
            if SAI_pfts is not None:
                del SAI_pfts
            if sai_one is not None:
                del sai_one
            if pct_one is not None:
                del pct_one
            if pct_pft_one is not None:
                del pct_pft_one
            if area_one is not None:
                del area_one
