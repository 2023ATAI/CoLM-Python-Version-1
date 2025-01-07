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
#  Hua Yuan,      ?/2020 : for land cover land use classifications
#  Shupeng Zhang, 01/2022: porting codes to MPI parallel version
#  Hua Yuan,      05/2023: TODO
# ----------------------------------------------------------------------
import os
import numpy as np
from CoLM_DataType import DataType
from CoLM_NetCDFBlock import NetCDFBlock
from CoLM_AggregationRequestData import AggregationRequestData


class Aggregation_LAI (object):
    def __init__(self,nl_colm,mpi,gblock) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.gblock = gblock


    def Aggregation_LAI (self, gridlai, dir_rawdata, dir_model_landdata, lc_year):
        landdir = dir_model_landdata.strip() + '/LAI/'

        if self.mpi.p_is_master:
            print('Aggregate LAI ...')
            os.system('mkdir -p ' + landdir.strip())

        if self.nl_colm['SinglePoint']:
            # 检查是否使用了 USE_SITE_LAI
            if self.nl_colm['USE_SITE_LAI']:
                return

        idate = [self.nl_colm['DEF_simulation_time'].start_year, 0, 0]  # 初始化 idate 列表，第一个元素为年份，后两个元素为月份和日期（暂时置为 0）

        if not self.nl_colm['isgreenwich']:
            idate[2] = self.nl_colm['DEF_simulation_time'].start_sec  # 如果不是以 GMT 为基准时间，则将秒数存储在 idate 的第三个位置

            # 调用 monthday2julian 函数
            idate[1] = monthday2julian(self.nl_colm['DEF_simulation_time'].start_year, self.nl_colm['DEF_simulation_time'].start_month,
                                       self.nl_colm['DEF_simulation_time'].start_day)

            # 调用 localtime2gmt 函数
            idate = localtime2gmt(idate)

            simulation_lai_year_end = idate[0]

        #  global plant leaf area index

        if self.nl_colm['LULC_USGS'] or self.nl_colm['LULC_IGBP']:
            if self.nl_colm['DEF_LAI_MONTHLY']:
                if self.nl_colm['LULCC_defined']:
                    start_year = lc_year
                    end_year = lc_year
                    ntime = 12
                else:
                    if self.nl_colm['DEF_LAI_CHANGE_YEARLY']:
                        start_year = self.nl_colm['simulation_lai_year_start']
                        end_year = self.nl_colm['simulation_lai_year_end']
                        ntime = 12
                    else:
                        start_year = lc_year
                        end_year = lc_year
                        ntime = 12
            else:
                start_year = self.nl_colm['simulation_lai_year_start']
                end_year = self.nl_colm['simulation_lai_year_end']
                ntime = 46
            # -----LAI-----
            if self.mpi.p_is_io:
                datayype = DataType(self.gblock)
                LAI = datayype.allocate_block_data(gridlai)

            if self.mpi.p_is_worker:
                LAI_patches = np.empty(numpatch)

        if self.nl_colm['SinglePoint']:
            SITE_LAI_year = [iy for iy in range(start_year, end_year + 1)]

            if self.nl_colm['DEF_LAI_MONTHLY']:
                SITE_LAI_monthly = np.empty((12, end_year - start_year + 1))
            else:
                SITE_LAI_8day = np.empty((46, end_year - start_year + 1))

        if not self.nl_colm['DEF_USE_LAIFEEDBACK']:
            for iy in range(start_year, end_year + 1):
                cyear = str(iy)
                os.system('mkdir -p ' + landdir.strip() + cyear)

                for itime in range(ntime):
                    if self.nl_colm['DEF_LAI_MONTHLY']:
                        c3 = str(itime).zfill(2)
                    else:
                        Julian_day = 1 + (itime) * 8
                        c3 = str(Julian_day).zfill(3)

                    if self.mpi.p_is_master:
                        print('Aggregate LAI :', iy, ':', itime, '/', ntime)

                    if self.mpi.p_is_io:
                        if self.nl_colm['DEF_LAI_MONTHLY']:
                            dir_5x5 = os.path.join(dir_rawdata.strip(), 'plant_15s')
                            suffix = 'MOD' + cyear
                            read_5x5_data_time(dir_5x5, suffix, gridlai, 'MONTHLY_LC_LAI', itime, LAI)
                        else:
                            lndname = os.path.join(dir_rawdata.strip(), 'lai_15s_8day',
                                                   'lai_8-day_15s_' + cyear + '.nc')
                            ncio_read_block_time(lndname, 'lai', gridlai, itime, LAI)
                            block_data_linear_transform(LAI, scl=0.1)

        #aggregate the plant leaf area index from the resolution of raw data to modelling resolution
        if self.mpi.p_is_worker:
            for ipatch in range(numpatch):
                # 调用 aggregation_request_data 函数请求数据
                lai_one = aggregation_request_data(landpatch, ipatch, gridlai, zip=self.nl_colm['USE_zip_for_aggregation'], area=area_one,
                                         data_r8_2d_in1=LAI)

                # 计算 LAI_patches
                LAI_patches[ipatch] = np.sum(lai_one * area_one) / np.sum(area_one)

        #write out the plant leaf area index of grid patches
        if self.nl_colm['SinglePoint']:
            # 单点案例
            # TODO: 时间年的参数输入
            if self.nl_colm['DEF_LAI_MONTHLY']:
                self.nl_colm['SITE_LAI_monthly'][itime - 1, iy - start_year] = LAI_patches[0]
            else:
                self.nl_colm['SITE_LAI_8day'][itime - 1, iy - start_year] = LAI_patches[0]
        


