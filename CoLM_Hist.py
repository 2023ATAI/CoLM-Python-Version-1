from CoLM_Vars_1DAccFluxes import CoLM_Vars_1DAccFluxes
# from CoLM_Hydro_Hist import CoLM_Hydro_Hist
from CoLM_HistGridded import CoLM_HistGridded
import CoLM_TimeManager
import numpy as np
from CoLM_DataType import DataType

HistForm= 'Gridded'

def hist_init(nl_colm,  mpi, var_global, landpatch, landelm, gblock, pixel, mesh, numelm, spval, gforc, pctshrpch,dir_hist, varfluxes):
    # Allocate and flush accumulated fluxes
    varfluxes.allocate_acc_fluxes(landelm)
    varfluxes.flush_acc_fluxes(var_global.maxsnl,var_global.nl_soil,var_global.nvegwcs,var_global.nl_lake)

    # Determine history format
    HistForm = 'Gridded'
    if nl_colm['UNSTRUCTURED'] or nl_colm['CATCHMENT']:
        if nl_colm['DEF_HISTORY_IN_VECTOR']:
            HistForm = 'Vector'

    if nl_colm['SinglePoint']:
        HistForm = 'Single'

    gridded = None

    # Initialize based on the history format
    if HistForm == 'Gridded':
        gridded = CoLM_HistGridded(nl_colm, mpi, pixel, mesh, landpatch.landpatch, gblock)
        gridded.hist_gridded_init(gforc, pctshrpch)
        return gridded
    elif HistForm == 'Single':
        pass
        # hist_single_init()
    
    # if nl_colm['CatchLateralFlow']:
    #     hydro_hist = CoLM_Hydro_Hist(mpi, numelm, numhru, spval)
    #     hydro_hist.hist_basin_init()
    return gridded

def hist_final (nl_colm, varaccfluxes):
    varaccfluxes.deallocate_acc_fluxes ()

    if nl_colm['SinglePoint']:
        pass
    #   CALL hist_single_final ()

    if nl_colm['CatchLateralFlow']:
        pass
    #   CALL hist_basin_final ()

def write_history_variable_2d(is_hist, acc_vec, file_hist, varname, itime_in_file, sumarea, filter, longname, units, gridded, nac, spval):
        
    if not is_hist:
        return

    if HistForm == 'Gridded':
        gridded.flux_map_and_write_2d(acc_vec, file_hist, varname, itime_in_file, sumarea, filter, longname, units, nac, spval)
    
    # elif HistForm == 'Vector':
    #     aggregate_to_vector_and_write_2d(acc_vec, file_hist, varname, itime_in_file, filter, longname, units)
    
    # elif HistForm == 'Single':
    #     single_write_2d(acc_vec, file_hist, varname, itime_in_file, longname, units)

def write_history_variable_3d(is_hist, acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, sumarea, filter, longname, units, gridded, nac, spval ):
    if not is_hist:
        return

    if HistForm == 'Gridded':
        gridded.flux_map_and_write_3d(acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, sumarea, filter, longname, units, nac, spval)
    
    # elif HistForm == 'Vector':
    #     aggregate_to_vector_and_write_3d(acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, filter, longname, units)
    
    # elif HistForm == 'Single':
    #     single_write_3d(acc_vec, file_hist, varname, itime_in_file, dim1name, ndim1, longname, units)

def write_history_variable_4d(is_hist, acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, dim2name, lb2, ndim2, sumarea, filter, longname, units, gridded, nac, spval):
    if not is_hist:
        return

    if HistForm == 'Gridded':
        gridded.flux_map_and_write_4d(acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, dim2name, lb2, ndim2, sumarea, filter, longname, units, nac, spval)
    
    # elif HistForm == 'Vector':
    #     aggregate_to_vector_and_write_4d(acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, dim2name, lb2, ndim2, filter, longname, units)
    
    # elif HistForm == 'Single':
    #     single_write_4d(acc_vec, file_hist, varname, itime_in_file, dim1name, ndim1, dim2name, ndim2, longname, units)


def write_history_variable_ln(is_hist, acc_vec, file_hist, varname, itime_in_file, sumarea, filter, longname, units, gridded, spval, nac_ln):
    if not is_hist:
        return

    if HistForm == 'Gridded':
        gridded.flux_map_and_write_ln(acc_vec, file_hist, varname, itime_in_file, sumarea, filter, longname, units, spval, nac_ln)
    
    # elif HistForm == 'Vector':
    #     aggregate_to_vector_and_write_ln(acc_vec, file_hist, varname, itime_in_file, filter, longname, units)
    
    # elif HistForm == 'Single':
    #     single_write_ln(acc_vec, file_hist, varname, itime_in_file, longname, units)


def hist_write_time(filename, dataname, time, itime,gridded):
    
    itime = 0  # Initialize itime; will be updated by the appropriate function

    if HistForm == 'Gridded':
        itime = gridded.hist_gridded_write_time(filename, dataname, time, itime)
    
    # elif HistForm == 'Vector':
    #     itime = hist_vector_write_time(filename, dataname, time)
    
    # elif HistForm == 'Single':
    #     itime = hist_single_write_time(filename, dataname, time)
    
    return itime

def hist_out(nl_colm, nl_colm_forcing, nl_colm_history, mpi, const_physical,var_global, numelm, var_accfluxes, spval, landpatch, gblock, vtv, vti, forcing,
             varfluxes, var_1forcing, histgrid, nl_soil, maxsnl, nvegwcs, nl_lake, idate, deltim, itstamp,
             etstamp, ptstamp, dir_hist, site, gridded):
    itime_in_file = 0
    vecacc = None

    if itstamp <= ptstamp:
        var_accfluxes.flush_acc_fluxes(spval)
        return
    else:
        var_accfluxes.accumulate_fluxes(numelm, nl_colm_forcing, forcing, var_1forcing, varfluxes, vtv, vti, const_physical)

    lwrite = False
    def_hist_freq = nl_colm['DEF_HIST_FREQ']

    if def_hist_freq == 'TIMESTEP':
        lwrite = True
    elif def_hist_freq == 'HOURLY':
        lwrite = CoLM_TimeManager.isendofhour(idate, deltim) or not (itstamp < etstamp)
    elif def_hist_freq == 'DAILY':
        lwrite = CoLM_TimeManager.isendofday(idate, deltim) or not (itstamp < etstamp)
    elif def_hist_freq == 'MONTHLY':
        lwrite = CoLM_TimeManager.isendofmonth(idate, deltim) or not (itstamp < etstamp)
    elif def_hist_freq == 'YEARLY':
        lwrite = CoLM_TimeManager.isendofyear(idate, deltim) or not (itstamp < etstamp)
    else:
        print("Warning: Please USE one of TIMESTEP/HOURLY/DAILY/MONTHLY/YEARLY for history frequency.")

    if lwrite:
        year, julian_day = idate.year, idate.day
        month, day = CoLM_TimeManager.julian2monthday(year, julian_day)

        days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if CoLM_TimeManager.isleapyear(year):
            days_month[1] = 29

        def_hist_groupby = nl_colm['DEF_HIST_groupby']
        cdate=''
        
        if def_hist_groupby == 'YEAR':
            cdate = str(year)
        elif def_hist_groupby == 'MONTH':
            cdate = str(year)+'-'+str(month)
        elif def_hist_groupby == 'DAY':
            cdate = str(year)+'-'+str(month)+'-'+str(day)
        else:
            print("Warning: Please USE one of DAY/MONTH/YEAR for history group.")

        file_hist = f"{dir_hist.strip()}/{site.strip()}_hist_{cdate.strip()}.nc"

        hist_write_time(file_hist, 'time', idate, itime_in_file,gridded)

        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                filter = np.full( landpatch.numpatch,False)
                VecOnes = np.full(landpatch.numpatch,1.0)
                vecacc = np.zeros(landpatch.numpatch)

        datatype = DataType(gblock)
        sumarea = None
        if HistForm == 'Gridded':
            if mpi.p_is_io:
                sumarea = datatype.allocate_block_data(histgrid.ghist)

        if mpi.p_is_worker and landpatch.numpatch > 0:
            filter = [pt < 99 for pt in vti.patchtype]

            if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                filter = [f and m for f, m in zip(filter, forcing.forcmask)]

            filter = [f and pm for f, pm in zip(filter, vti.patchmask)]

        if HistForm == 'Gridded':
            histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

        if HistForm == 'Gridded' and itime_in_file == 1:
            gridded.hist_write_var_real8_2d(file_hist, 'landarea', histgrid.ghist, 1, sumarea, compress=1, longname='land area', units='km2')

        # Calls to write_history_variable_2d functions for various variables follow the same pattern:
        write_history_variable_2d(nl_colm_history['xy_us'], var_accfluxes.a_us, file_hist, 'f_xy_us', itime_in_file, sumarea, filter, 'wind in eastward direction', 'm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_vs'], var_accfluxes.a_vs, file_hist, 'f_xy_vs', itime_in_file, sumarea, filter, 'wind in northward direction', 'm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_t'], var_accfluxes.a_t, file_hist, 'f_xy_t', itime_in_file, sumarea, filter, 'temperature at reference height', 'kelvin',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_q'], var_accfluxes.a_q, file_hist, 'f_xy_q', itime_in_file, sumarea, filter, 'specific humidity at reference height', 'kg/kg',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_prc'], var_accfluxes.a_prc, file_hist, 'f_xy_prc', itime_in_file, sumarea, filter, 'convective precipitation', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_prl'], var_accfluxes.a_prl, file_hist, 'f_xy_prl', itime_in_file, sumarea, filter, 'large scale precipitation', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_pbot'], var_accfluxes.a_pbot, file_hist, 'f_xy_pbot', itime_in_file, sumarea, filter, 'atmospheric pressure at the surface', 'pa',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_frl'], var_accfluxes.a_frl, file_hist, 'f_xy_frl', itime_in_file, sumarea, filter, 'atmospheric infrared (longwave,gridded, var_accfluxes.nac, var_global.spval) radiation', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_solarin'], var_accfluxes.a_solarin, file_hist, 'f_xy_solarin', itime_in_file, sumarea, filter, 'downward solar radiation at surface', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_rain'], var_accfluxes.a_rain, file_hist, 'f_xy_rain', itime_in_file, sumarea, filter, 'rain', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['xy_snow'], var_accfluxes.a_snow, file_hist, 'f_xy_snow', itime_in_file, sumarea, filter, 'snow', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)

        if nl_colm['DEF_USE_CBL_HEIGHT']:
            write_history_variable_2d(nl_colm_history['xy_hpbl'], var_accfluxes.a_hpbl, file_hist, 'f_xy_hpbl', itime_in_file, sumarea, filter, 'boundary layer height', 'm',gridded, var_accfluxes.nac, var_global.spval)

        # Mapping the fluxes and state variables at patch [landpatch.numpatch] to grid
        if mpi.p_is_worker and landpatch.numpatch > 0:
            filter = [pt < 99 for pt in vti.patchtype]

            if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                filter = [f and m for f, m in zip(filter, forcing.forcmask)]

            filter = [f and pm for f, pm in zip(filter, vti.patchmask)]

        if HistForm == 'Gridded':
            histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

        write_history_variable_2d(nl_colm_history['taux'], var_accfluxes.a_taux, file_hist, 'f_taux', itime_in_file, sumarea, filter, 'wind stress: E-W', 'kg/m/s2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['tauy'], var_accfluxes.a_tauy, file_hist, 'f_tauy', itime_in_file, sumarea, filter, 'wind stress: N-S', 'kg/m/s2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fsena'], var_accfluxes.a_fsena, file_hist, 'f_fsena', itime_in_file, sumarea, filter, 'sensible heat from canopy height to atmosphere', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['lfevpa'], var_accfluxes.a_lfevpa, file_hist, 'f_lfevpa', itime_in_file, sumarea, filter, 'latent heat flux from canopy height to atmosphere', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fevpa'], var_accfluxes.a_fevpa, file_hist, 'f_fevpa', itime_in_file, sumarea, filter, 'evapotranspiration from canopy height to atmosphere', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fsenl'], var_accfluxes.a_fsenl, file_hist, 'f_fsenl', itime_in_file, sumarea, filter, 'sensible heat from leaves', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fevpl'], var_accfluxes.a_fevpl, file_hist, 'f_fevpl', itime_in_file, sumarea, filter, 'evaporation+transpiration from leaves','mm/s',gridded, var_accfluxes.nac, var_global.spval)

        # Call equivalent Python functions for each variable
        write_history_variable_2d(nl_colm_history['etr'], var_accfluxes.a_etr, file_hist, 'f_etr', itime_in_file, sumarea, filter, 'transpiration rate', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fseng'], var_accfluxes.a_fseng, file_hist, 'f_fseng', itime_in_file, sumarea, filter, 'sensible heat flux from ground', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fevpg'], var_accfluxes.a_fevpg, file_hist, 'f_fevpg', itime_in_file, sumarea, filter, 'evaporation heat flux from ground', 'mm/s',gridded, var_accfluxes.nac, var_global.spval)
        write_history_variable_2d(nl_colm_history['fgrnd'], var_accfluxes.a_fgrnd, file_hist, 'f_fgrnd', itime_in_file, sumarea, filter, 'ground heat flux', 'W/m2',gridded, var_accfluxes.nac, var_global.spval)

        # 假设 write_history_variable_2d 函数已经正确定义好，参数顺序和功能与Fortran版本相对应

        # 太阳能被阳光照射的冠层吸收 [W/m2]
        write_history_variable_2d(nl_colm_history["sabvsun"], var_accfluxes.a_sabvsun, file_hist, 'f_sabvsun', itime_in_file, sumarea,
                                  filter,
                                  'solar absorbed by sunlit canopy', 'W/m2', gridded, var_accfluxes.nac, spval)

        # 太阳能被阴影部分吸收 [W/m2]
        write_history_variable_2d(nl_colm_history["sabvsha"], var_accfluxes.a_sabvsha, file_hist, 'f_sabvsha', itime_in_file, sumarea,
                                  filter,
                                  'solar absorbed by shaded', 'W/m2', gridded, var_accfluxes.nac, spval)

        # 太阳能被地面吸收 [W/m2]
        write_history_variable_2d(nl_colm_history["sabg"], var_accfluxes.a_sabg, file_hist, 'f_sabg', itime_in_file, sumarea, filter,
                                  'solar absorbed by ground', 'W/m2', gridded, var_accfluxes.nac, spval)

        # 地面和冠层向外长波辐射 [W/m2]
        write_history_variable_2d(nl_colm_history["olrg"], var_accfluxes.a_olrg, file_hist, 'f_olrg', itime_in_file, sumarea, filter,
                                  'outgoing long-wave radiation from ground+canopy', 'W/m2', gridded, var_accfluxes.nac, spval)

        # 净辐射 [W/m2]
        write_history_variable_2d(nl_colm_history["rnet"], var_accfluxes.a_rnet, file_hist, 'f_rnet', itime_in_file, sumarea, filter,
                                  'net radiation', 'W/m2', gridded, var_accfluxes.nac, spval)

        # 水分平衡误差 [mm/s]
        write_history_variable_2d(nl_colm_history["xerr"], var_accfluxes.a_xerr, file_hist, 'f_xerr', itime_in_file, sumarea, filter,
                                  'the error of water banace', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 能量平衡误差 [W/m2]
        write_history_variable_2d(nl_colm_history["zerr"], var_accfluxes.a_zerr, file_hist, 'f_zerr', itime_in_file, sumarea, filter,
                                  'the error of energy balance', 'W/m2', gridded, var_accfluxes.nac, spval)

        # 地表径流 [mm/s]
        write_history_variable_2d(nl_colm_history["rsur"], var_accfluxes.a_rsur, file_hist, 'f_rsur', itime_in_file, sumarea, filter,
                                  'surface runoff', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 饱和超量地表径流 [mm/s]
        write_history_variable_2d(nl_colm_history["rsur_se"], var_accfluxes.a_rsur_se, file_hist, 'f_rsur_se', itime_in_file, sumarea,
                                  filter,
                                  'saturation excess surface runoff', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 入渗超量地表径流 [mm/s]
        write_history_variable_2d(nl_colm_history["rsur_ie"], var_accfluxes.a_rsur_ie, file_hist, 'f_rsur_ie', itime_in_file, sumarea,
                                  filter,
                                  'infiltration excess surface runoff', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 地下径流 [mm/s]
        write_history_variable_2d(nl_colm_history["rsub"], var_accfluxes.a_rsub, file_hist, 'f_rsub', itime_in_file, sumarea, filter,
                                  'subsurface runoff', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 总径流 [mm/s]
        write_history_variable_2d(nl_colm_history["rnof"], var_accfluxes.a_rnof, file_hist, 'f_rnof', itime_in_file, sumarea, filter,
                                  'total runoff', 'W/m2', gridded, var_accfluxes.nac, spval)

        # Special handling for instantaneous total water storage
        if nl_colm['DataAssimilation']:
            if mpi.p_is_worker:
                vecacc = var_accfluxes.a_wat.copy()  # Assuming copy behavior similar to Fortran's array assignment
                vecacc[vecacc != spval] *= var_accfluxes.nac
                write_history_variable_2d(nl_colm_history['wat_inst'], vecacc, file_hist, 'f_wat_inst', itime_in_file, sumarea, filter, 'instantaneous total water storage', '-',gridded, var_accfluxes.nac, spval)

        # 判断是否有CatchLateralFlow这个宏定义对应的功能（在Python中可以用条件判断来模拟这种情况，不过具体如何判断需要根据实际背景来定，这里简单示例一种可能的方式）
        # 假设这里有个变量catch_lateral_flow来模拟#ifdef CatchLateralFlow的条件判断，你可以根据实际情况修改这个判断逻辑
        if nl_colm['CatchLateralFlow']:
            # 地表水深变化率 [mm/s]
            write_history_variable_2d(nl_colm_history["xwsur"], var_accfluxes.a_xwsur, file_hist, 'f_xwsur', itime_in_file, sumarea,
                                      filter,
                                      'rate of surface water depth change', 'mm/s', gridded, var_accfluxes.nac, spval)

            # 地下水变化率 [mm/s]
            write_history_variable_2d(nl_colm_history["xwsub"], var_accfluxes.a_xwsub, file_hist, 'f_xwsub', itime_in_file, sumarea,
                                      filter,
                                      'rate of ground water change', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 截留量 [mm/s]
        write_history_variable_2d(nl_colm_history["qintr"], var_accfluxes.a_qintr, file_hist, 'f_qintr', itime_in_file, sumarea, filter,
                                  'interception', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 入渗量 [mm/s]
        write_history_variable_2d(nl_colm_history["qinfl"], var_accfluxes.a_qinfl, file_hist, 'f_qinfl', itime_in_file, sumarea, filter,
                                  'f_qinfl', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 穿透降水量 [mm/s]
        write_history_variable_2d(nl_colm_history["qdrip"], var_accfluxes.a_qdrip, file_hist, 'f_qdrip', itime_in_file, sumarea, filter,
                                  'total throughfall', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 总储水量 [mm]
        write_history_variable_2d(nl_colm_history["wat"], var_accfluxes.a_wat, file_hist, 'f_wat', itime_in_file, sumarea, filter,
                                  'total water storage', 'mm', gridded, var_accfluxes.nac, spval)

        # 瞬时总储水量 [mm]
        if mpi.p_is_worker:
            vecacc = var_accfluxes.a_wat.copy()  # 假设wat是类似数组或列表的结构，这里进行拷贝（具体根据实际数据类型调整）
            # 以下WHERE语句的功能不太明确，假设是对vecacc中的非特殊值（spval对应的情况）进行操作，这里简单示例一种可能的模拟逻辑，具体根据真实含义调整
            for index in range(len(vecacc)):
                if vecacc[index] != spval:
                    vecacc[index] = vecacc[index] * var_accfluxes.nac
        write_history_variable_2d(nl_colm_history["wat_inst"], vecacc, file_hist, 'f_wat_inst', itime_in_file, sumarea,
                                  filter,
                                  'instantaneous total water storage', 'mm', gridded, var_accfluxes.nac, spval)

        # 冠层同化速率 [mol m-2 s-1]
        write_history_variable_2d(nl_colm_history["assim"], var_accfluxes.a_assim, file_hist, 'f_assim', itime_in_file, sumarea, filter,
                                  'canopy assimilation rate', 'mol m-2 s-1', gridded, var_accfluxes.nac, spval)

        # 呼吸作用（植物 + 土壤）[mol m-2 s-1]
        write_history_variable_2d(nl_colm_history["respc"], var_accfluxes.a_respc, file_hist, 'f_respc', itime_in_file, sumarea, filter,
                                  'respiration (plant+soil, gridded, var_accfluxes.nac, spval)', 'mol m-2 s-1', gridded, var_accfluxes.nac, spval)

        # 地下水补给速率 [mm/s]
        write_history_variable_2d(nl_colm_history["qcharge"], var_accfluxes.a_qcharge, file_hist, 'f_qcharge', itime_in_file, sumarea,
                                  filter,
                                  'groundwater recharge rate', 'mm/s', gridded, var_accfluxes.nac, spval)

        # 地表温度 [K]
        write_history_variable_2d(nl_colm_history["t_grnd"], var_accfluxes.a_t_grnd, file_hist, 'f_t_grnd', itime_in_file, sumarea,
                                  filter,
                                  'ground surface temperature', 'K', gridded, var_accfluxes.nac, spval)

        # 叶片温度 [K]
        write_history_variable_2d(nl_colm_history["tleaf"], var_accfluxes.a_tleaf, file_hist, 'f_tleaf', itime_in_file, sumarea, filter,
                                  'leaf temperature', 'K', gridded, var_accfluxes.nac, spval)

        # 叶片上的水深度 [mm]
        write_history_variable_2d(nl_colm_history["ldew"], var_accfluxes.a_ldew, file_hist, 'f_ldew', itime_in_file, sumarea, filter,
                                  'depth of water on foliage', 'mm', gridded, var_accfluxes.nac, spval)

        # 积雪覆盖，水当量 [mm]
        write_history_variable_2d(nl_colm_history["scv"], var_accfluxes.a_scv, file_hist, 'f_scv', itime_in_file, sumarea, filter,
                                  'snow cover, water equivalent', 'mm', gridded, var_accfluxes.nac, spval)

        # 积雪深度 [meter]
        write_history_variable_2d(nl_colm_history["snowdp"], var_accfluxes.a_snowdp, file_hist, 'f_snowdp', itime_in_file, sumarea,
                                  filter,
                                  'snow depth', 'meter', gridded, var_accfluxes.nac, spval)

        # 地面雪覆盖比例
        write_history_variable_2d(nl_colm_history["fsno"], var_accfluxes.a_fsno, file_hist, 'f_fsno', itime_in_file, sumarea, filter,
                                  'fraction of snow cover on ground', '-', gridded, var_accfluxes.nac, spval)

        # 植被覆盖比例（不包括被雪覆盖的植被）[-]
        write_history_variable_2d(nl_colm_history["sigf"], var_accfluxes.a_sigf, file_hist, 'f_sigf', itime_in_file, sumarea, filter,
                                  'fraction of veg cover, excluding snow-covered veg', '-', gridded, var_accfluxes.nac, spval)

        # 叶片绿度
        write_history_variable_2d(nl_colm_history["green"], var_accfluxes.a_green, file_hist, 'f_green', itime_in_file, sumarea, filter,
                                  'leaf greenness', '-', gridded, var_accfluxes.nac, spval)

        # 叶面积指数
        write_history_variable_2d(nl_colm_history["lai"], var_accfluxes.a_lai, file_hist, 'f_lai', itime_in_file, sumarea, filter,
                                  'leaf area index', 'm2/m2', gridded, var_accfluxes.nac, spval)

        # 阳生叶面积指数
        write_history_variable_2d(nl_colm_history["laisun"], var_accfluxes.a_laisun, file_hist, 'f_laisun', itime_in_file, sumarea,
                                  filter,
                                  'sunlit leaf area index', 'm2/m2', gridded, var_accfluxes.nac, spval)

        # 阴生叶面积指数
        write_history_variable_2d(nl_colm_history["laisha"], var_accfluxes.a_laisha, file_hist,
                                  "f_laisha", itime_in_file, sumarea, filter,
                                  'shaded leaf area index', 'm2/m2', gridded, var_accfluxes.nac, spval)

        # 茎面积指数
        write_history_variable_2d(nl_colm_history["sai"], var_accfluxes.a_sai, file_hist, 'f_sai', itime_in_file, sumarea, filter,
                                  'stem area index', 'm2/m2', gridded, var_accfluxes.nac, spval)

        # Example for a 4D variable
        write_history_variable_4d(nl_colm_history['alb'], var_accfluxes.a_alb, file_hist, 'f_alb', itime_in_file, 'band', 1, 2, 'rtyp', 1, 2, sumarea, filter, 'averaged albedo direct', '%',gridded, var_accfluxes.nac, spval)

        # 平均整体表面发射率
        write_history_variable_2d(nl_colm_history["emis"], var_accfluxes.a_emis, file_hist, 'f_emis', itime_in_file, sumarea, filter,
                                  'averaged bulk surface emissivity', '-', gridded, var_accfluxes.nac, spval)

        # 有效粗糙度 [m]
        write_history_variable_2d(nl_colm_history["z0m"], var_accfluxes.a_z0m, file_hist, 'f_z0m', itime_in_file, sumarea, filter,
                                  'effective roughness', 'm', gridded, var_accfluxes.nac, spval)

        # 地表辐射温度 [K]
        write_history_variable_2d(nl_colm_history["trad"], var_accfluxes.a_trad, file_hist, 'f_trad', itime_in_file, sumarea, filter,
                                  'radiative temperature of surface', 'kelvin', gridded, var_accfluxes.nac, spval)

        # 2米高度空气温度 [kelvin]
        write_history_variable_2d(nl_colm_history["tref"], var_accfluxes.a_tref, file_hist, 'f_tref', itime_in_file, sumarea, filter,
                                  '2 m height air temperature', 'kelvin', gridded, var_accfluxes.nac, spval)

        # 2米高度空气比湿度 [kg/kg]
        write_history_variable_2d(nl_colm_history["qref"], var_accfluxes.a_qref, file_hist, 'f_qref', itime_in_file, sumarea, filter,'2 m height air specific humidity', 'kg/kg', gridded, var_accfluxes.nac, spval)

        # 模拟Fortran中根据条件对filter进行处理的逻辑（具体需根据实际情况进一步核对和完善，这里是大致模拟）
        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                # 假设patchtype、forcmask、patchmask都是类似数组或可迭代结构，进行相应的逻辑判断，这里简单用列表推导式示例一种可能的实现方式，需根据真实数据类型调整
                filter[:] = np.full(len(filter),vti.patchtype == 2)  # 对应patchtype == 2的逻辑
                if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                    filter = [f and fm for f, fm in zip(filter, forcing.forcmask)]  # 对应filter = filter.and. forcmask的逻辑
                filter = [f and pm for f, pm in zip(filter, vti.patchmask)]  # 对应filter = filter.and. patchmask的逻辑

        # 湿地储水量 [mm]
        write_history_variable_2d(nl_colm_history["wetwat"], var_accfluxes.a_wetwat, file_hist, 'f_wetwat', itime_in_file, sumarea,filter,'wetland water storage', 'mm', gridded, var_accfluxes.nac, spval)
        # Wetland water storage example
        if mpi.p_is_worker:
            vecacc = vtv.wat.copy()
            vecacc[vecacc != spval] *= var_accfluxes.nac
        write_history_variable_2d(nl_colm_history['wetwat_inst'], vecacc, file_hist, 'f_wetwat_inst', itime_in_file, sumarea, filter, 'instantaneous wetland water storage', 'mm',gridded, var_accfluxes.nac, spval)
        #------------------------------------------------------------------------------------------
        #Mapping the fluxes and state variables at patch [landpatch.numpatch] to grid
        #------------------------------------------------------------------------------------------

        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                filter = [pt < 99 for pt in vti.patchtype]
                if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                    filter = [f and pm for f, pm in zip(filter, forcing.forcmask)]

        if HistForm == 'Gridded':
            histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

        # 1: assimsun enf temperate
        write_history_variable_2d(nl_colm_history['assimsun'], var_accfluxes.a_assimsun, file_hist, 'f_assimsun', itime_in_file, sumarea, filter,
                                'Photosynthetic assimilation rate of sunlit leaf for needleleaf evergreen temperate tree',
                                'mol m-2 s-1',gridded, var_accfluxes.nac, spval)

        # 1: assimsha enf temperate
        write_history_variable_2d(nl_colm_history['assimsha'], var_accfluxes.a_assimsha, file_hist, 'f_assimsha', itime_in_file, sumarea, filter,
                                'Photosynthetic assimilation rate of shaded leaf for needleleaf evergreen temperate tree',
                                'mol m-2 s-1',gridded, var_accfluxes.nac, spval)

        # 1: etrsun enf temperate
        write_history_variable_2d(nl_colm_history['etrsun'], var_accfluxes.a_etrsun, file_hist, 'f_etrsun', itime_in_file, sumarea, filter,
                                'Transpiration rate of sunlit leaf for needleleaf evergreen temperate tree', 'mm s-1',gridded, var_accfluxes.nac, spval)

        # 1: etrsha enf temperate
        write_history_variable_2d(nl_colm_history['etrsha'], var_accfluxes.a_etrsha, file_hist, 'f_etrsha', itime_in_file, sumarea, filter,
                                'Transpiration rate of shaded leaf for needleleaf evergreen temperate tree', 'mm s-1',gridded, var_accfluxes.nac, spval)

        # rstfacsun
        write_history_variable_2d(nl_colm_history['rstfacsun'], var_accfluxes.a_rstfacsun, file_hist, 'f_rstfacsun', itime_in_file, sumarea, filter,
                                'Ecosystem level Water stress factor on sunlit canopy', 'unitless',gridded, var_accfluxes.nac, spval)

        # rstfacsha
        write_history_variable_2d(nl_colm_history['rstfacsha'], var_accfluxes.a_rstfacsha, file_hist, 'f_rstfacsha', itime_in_file, sumarea, filter,
                                'Ecosystem level Water stress factor on shaded canopy', 'unitless',gridded, var_accfluxes.nac, spval)

        # gssun
        write_history_variable_2d(nl_colm_history['gssun'], var_accfluxes.a_gssun, file_hist, 'f_gssun', itime_in_file, sumarea, filter,
                                'Ecosystem level canopy conductance on sunlit canopy', 'mol m-2 s-1',gridded, var_accfluxes.nac, spval)

        # gssha
        write_history_variable_2d(nl_colm_history['gssha'], var_accfluxes.a_gssha, file_hist, 'f_gssha', itime_in_file, sumarea, filter,
                                'Ecosystem level canopy conductance on shaded canopy', 'mol m-2 s-1',gridded, var_accfluxes.nac, spval)

        # soil resistance [m/s]
        write_history_variable_2d(nl_colm_history['rss'], var_accfluxes.a_rss, file_hist, 'f_rss', itime_in_file, sumarea, filter,
                                'soil surface resistance', 's/m',gridded, var_accfluxes.nac, spval)

        # # grain to crop seed carbon
        # write_history_variable_2d(nl_colm_history['ndep_to_sminn'], var_accfluxes.a_ndep_to_sminn, file_hist, 'f_ndep_to_sminn', itime_in_file, sumarea, filter,
        #                         'nitrogen deposition', 'gN/m2/s',gridded, var_accfluxes.nac, spval)
        #
        # if nl_colm['DEF_USE_OZONESTRESS']:
        #     # ozone concentration
        #     write_history_variable_2d(nl_colm_history['xy_ozone'], var_accfluxes.a_ozone, file_hist, 'f_xy_ozone', itime_in_file, sumarea, filter,
        #                             'Ozone concentration', 'mol/mol',gridded, var_accfluxes.nac, spval)
        #
        # # litter 1 carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['litr1c_vr'], var_accfluxes.a_litr1c_vr, file_hist, 'f_litr1c_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'litter 1 carbon density in soil layers', 'gC/m3')
        #
        # # litter 2 carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['litr2c_vr'], var_accfluxes.a_litr2c_vr, file_hist, 'f_litr2c_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'litter 2 carbon density in soil layers', 'gC/m3')
        #
        # # litter 3 carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['litr3c_vr'], var_accfluxes.a_litr3c_vr, file_hist, 'f_litr3c_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'litter 3 carbon density in soil layers', 'gC/m3')
        #
        # # soil 1 carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['soil1c_vr'], var_accfluxes.a_soil1c_vr, file_hist, 'f_soil1c_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'soil 1 carbon density in soil layers', 'gC/m3')
        #
        # # soil 2 carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['soil2c_vr'], var_accfluxes.a_soil2c_vr, file_hist, 'f_soil2c_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'soil 2 carbon density in soil layers', 'gC/m3')
        #
        # # soil 3 carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['soil3c_vr'], var_accfluxes.a_soil3c_vr, file_hist, 'f_soil3c_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'soil 3 carbon density in soil layers', 'gC/m3')
        #
        # # coarse woody debris carbon density in soil layers
        # write_history_variable_3d(nl_colm_history['cwdc_vr'], var_accfluxes.a_cwdc_vr, file_hist, 'f_cwdc_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'coarse woody debris carbon density in soil layers', 'gC/m3')
        #
        # # litter 1 nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['litr1n_vr'], var_accfluxes.a_litr1n_vr, file_hist, 'f_litr1n_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'litter 1 nitrogen density in soil layers', 'gN/m3')
        #
        # # litter 2 nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['litr2n_vr'], var_accfluxes.a_litr2n_vr, file_hist, 'f_litr2n_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'litter 2 nitrogen density in soil layers', 'gN/m3')
        #
        # # litter 3 nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['litr3n_vr'], var_accfluxes.a_litr3n_vr, file_hist, 'f_litr3n_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'litter 3 nitrogen density in soil layers', 'gN/m3')
        #
        # # soil 1 nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['soil1n_vr'], var_accfluxes.a_soil1n_vr, file_hist, 'f_soil1n_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'soil 1 nitrogen density in soil layers', 'gN/m3')
        #
        # # soil 2 nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['soil2n_vr'], var_accfluxes.a_soil2n_vr, file_hist, 'f_soil2n_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'soil 2 nitrogen density in soil layers', 'gN/m3')
        #
        # # soil 3 nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['soil3n_vr'], var_accfluxes.a_soil3n_vr, file_hist, 'f_soil3n_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'soil 3 nitrogen density in soil layers', 'gN/m3')
        #
        # # coarse woody debris nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['cwdn_vr'], var_accfluxes.a_cwdn_vr, file_hist, 'f_cwdn_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'coarse woody debris nitrogen density in soil layers', 'gN/m3')
        #
        # # Mineral nitrogen density in soil layers
        # write_history_variable_3d(nl_colm_history['sminn_vr'], var_accfluxes.a_sminn_vr, file_hist, 'f_sminn_vr', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'mineral nitrogen density in soil layers', 'gN/m3')
        #
        # # Bulk density in soil layers
        # write_history_variable_3d(nl_colm_history['BD_all'], var_accfluxes.a_BD_all, file_hist, 'f_BD_all', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'bulk density in soil layers', 'kg/m3')
        #
        # # Field capacity in soil layers
        # write_history_variable_3d(nl_colm_history['wfc'], var_accfluxes.a_wfc, file_hist, 'f_wfc', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'field capacity in soil layers', 'm3/m3')
        #
        # # Organic matter density in soil layers
        # write_history_variable_3d(nl_colm_history['OM_density'], var_accfluxes.a_OM_density, file_hist, 'f_OM_density', itime_in_file, 'soil', 1, nl_soil,
        #                         sumarea, filter, 'organic matter density in soil layers', 'kg/m3')
        #
        # if nl_colm['DEF_USE_NITRIF']:
        #     # O2 soil Concentration for non-inundated area
        #     write_history_variable_3d(nl_colm_history['CONC_O2_UNSAT'], var_accfluxes.a_conc_o2_unsat, file_hist, 'f_CONC_O2_UNSAT', itime_in_file, 'soil', 1, nl_soil,
        #                             sumarea, filter, 'O2 soil Concentration for non-inundated area', 'mol/m3')
        #
        #     # O2 consumption from HR and AR for non-inundated area
        #     write_history_variable_3d(nl_colm_history['O2_DECOMP_DEPTH_UNSAT'], var_accfluxes.a_o2_decomp_depth_unsat, file_hist, 'f_O2_DECOMP_DEPTH_UNSAT', itime_in_file, 'soil', 1, nl_soil,
        #                             sumarea, filter, 'O2 consumption from HR and AR for non-inundated area', 'mol/m3/s')
        #
        # if nl_colm['DEF_USE_FIRE']:
        #     write_history_variable_2d(nl_colm_history['abm'], vecacc, file_hist, 'f_abm', itime_in_file, sumarea, filter,
        #                             'peak crop fire month', 'unitless',gridded, var_accfluxes.nac, spval)
        #
        #     write_history_variable_2d(nl_colm_history['gdp'], vecacc, file_hist, 'f_gdp', itime_in_file, sumarea, filter,
        #                             'gdp', 'unitless',gridded, var_accfluxes.nac, spval)
        #
        #     write_history_variable_2d(nl_colm_history['peatf'], vecacc, file_hist, 'f_peatf', itime_in_file, sumarea, filter,
        #                             'peatf', 'unitless',gridded, var_accfluxes.nac, spval)
        #
        #     write_history_variable_2d(nl_colm_history['hdm'], vecacc, file_hist, 'f_hdm', itime_in_file, sumarea, filter,
        #                             'hdm', 'unitless',gridded, var_accfluxes.nac, spval)
        #
        #     write_history_variable_2d(nl_colm_history['lnfm'], vecacc, file_hist, 'f_lnfm', itime_in_file, sumarea, filter,
        #                             'lnfm', 'unitless',gridded, var_accfluxes.nac, spval)
        #
        # if mpi.p_is_worker and landpatch.numpatch > 0:
        #     for i in range(landpatch.numpatch):
        #         filter[i] = vti.patchclass[i] != 12 and vti.patchtype[i] == 0
        #
        # if HistForm == 'Gridded':
        #     histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)
        #
        # # List of 2D history variables with corresponding function calls
        # write_history_variable_2d(nl_colm_history['gpp_enftemp'], var_accfluxes.a_gpp_enftemp, file_hist, 'f_gpp_enftemp', itime_in_file, sumarea, filter, 'gross primary productivity for needleleaf evergreen temperate tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_enftemp'], var_accfluxes.a_leafc_enftemp, file_hist, 'f_leafc_enftemp', itime_in_file, sumarea, filter, 'leaf carbon display pool for needleleaf evergreen temperate tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_enfboreal'], var_accfluxes.a_gpp_enfboreal, file_hist, 'f_gpp_enfboreal', itime_in_file, sumarea, filter, 'gross primary productivity for needleleaf evergreen boreal tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_enfboreal'], var_accfluxes.a_leafc_enfboreal, file_hist, 'f_leafc_enfboreal', itime_in_file, sumarea, filter, 'leaf carbon display pool for needleleaf evergreen boreal tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_dnfboreal'], var_accfluxes.a_gpp_dnfboreal, file_hist, 'f_gpp_dnfboreal', itime_in_file, sumarea, filter, 'gross primary productivity for needleleaf deciduous boreal tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_dnfboreal'], var_accfluxes.a_leafc_dnfboreal, file_hist, 'f_leafc_dnfboreal', itime_in_file, sumarea, filter, 'leaf carbon display pool for needleleaf deciduous boreal tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_ebftrop'], var_accfluxes.a_gpp_ebftrop, file_hist, 'f_gpp_ebftrop', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf evergreen tropical tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_ebftrop'], var_accfluxes.a_leafc_ebftrop, file_hist, 'f_leafc_ebftrop', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf evergreen tropical tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_ebftemp'], var_accfluxes.a_gpp_ebftemp, file_hist, 'f_gpp_ebftemp', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf evergreen temperate tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_ebftemp'], var_accfluxes.a_leafc_ebftemp, file_hist, 'f_leafc_ebftemp', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf evergreen temperate tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_dbftrop'], var_accfluxes.a_gpp_dbftrop, file_hist, 'f_gpp_dbftrop', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf deciduous tropical tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_dbftrop'], var_accfluxes.a_leafc_dbftrop, file_hist, 'f_leafc_dbftrop', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf deciduous tropical tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_dbftemp'], var_accfluxes.a_gpp_dbftemp, file_hist, 'f_gpp_dbftemp', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf deciduous temperate tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_dbftemp'], var_accfluxes.a_leafc_dbftemp, file_hist, 'f_leafc_dbftemp', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf deciduous temperate tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_dbfboreal'], var_accfluxes.a_gpp_dbfboreal, file_hist, 'f_gpp_dbfboreal', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf deciduous boreal tree', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_dbfboreal'], var_accfluxes.a_leafc_dbfboreal, file_hist, 'f_leafc_dbfboreal', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf deciduous boreal tree', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_ebstemp'], var_accfluxes.a_gpp_ebstemp, file_hist, 'f_gpp_ebstemp', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf evergreen temperate shrub', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_ebstemp'], var_accfluxes.a_leafc_ebstemp, file_hist, 'f_leafc_ebstemp', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf evergreen temperate shrub', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_dbstemp'], var_accfluxes.a_gpp_dbstemp, file_hist, 'f_gpp_dbstemp', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf deciduous temperate shrub', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_dbstemp'], var_accfluxes.a_leafc_dbstemp, file_hist, 'f_leafc_dbstemp', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf deciduous temperate shrub', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_dbsboreal'], var_accfluxes.a_gpp_dbsboreal, file_hist, 'f_gpp_dbsboreal', itime_in_file, sumarea, filter, 'gross primary productivity for broadleaf deciduous boreal shrub', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['leafc_dbsboreal'], var_accfluxes.a_leafc_dbsboreal, file_hist, 'f_leafc_dbsboreal', itime_in_file, sumarea, filter, 'leaf carbon display pool for broadleaf deciduous boreal shrub', 'gC/m2',gridded, var_accfluxes.nac, spval),
        # write_history_variable_2d(nl_colm_history['gpp_c3arcgrass'], var_accfluxes.a_gpp_c3arcgrass, file_hist, 'f_gpp_c3arcgrass', itime_in_file, sumarea, filter, 'gross primary productivity for c3 arctic grass', 'gC/m2/s',gridded, var_accfluxes.nac, spval),
        #
        # # 12: Leaf carbon display pool for C3 grass
        # write_history_variable_2d(
        #     nl_colm_history['leafc_c3grass'],
        #     var_accfluxes.a_leafc_c3grass, file_hist, 'f_leafc_c3grass', itime_in_file, sumarea, filter,
        #     'leaf carbon display pool for C3 grass', 'gC/m2'
        # ,gridded, var_accfluxes.nac, spval)
        #
        # # 13: GPP C3 grass
        # write_history_variable_2d(
        #     nl_colm_history['gpp_c3grass'],
        #     var_accfluxes.a_gpp_c3grass, file_hist, 'f_gpp_c3grass', itime_in_file, sumarea, filter,
        #     'gross primary productivity for C3 grass', 'gC/m2/s'
        # ,gridded, var_accfluxes.nac, spval)
        #
        # # 13: Leaf carbon display pool arctic C3 grass
        # write_history_variable_2d(
        #     nl_colm_history['leafc_c3grass'],
        #     var_accfluxes.a_leafc_c3grass, file_hist, 'f_leafc_c3grass', itime_in_file, sumarea, filter,
        #     'leaf carbon display pool for C3 arctic grass', 'gC/m2'
        # ,gridded, var_accfluxes.nac, spval)
        #
        # # 14: GPP C4 grass
        # write_history_variable_2d(
        #     nl_colm_history['gpp_c4grass'],
        #     var_accfluxes.a_gpp_c4grass, file_hist, 'f_gpp_c4grass', itime_in_file, sumarea, filter,
        #     'gross primary productivity for C4 grass', 'gC/m2/s'
        # ,gridded, var_accfluxes.nac, spval)
        #
        # # 14: Leaf carbon display pool arctic C4 grass
        # write_history_variable_2d(
        #     nl_colm_history['leafc_c4grass'],
        #     var_accfluxes.a_leafc_c4grass, file_hist, 'f_leafc_c4grass', itime_in_file, sumarea, filter,
        #     'leaf carbon display pool for C4 arctic grass', 'gC/m2'
        # ,gridded, var_accfluxes.nac, spval)

        # Temperature and water (excluding land water bodies and ocean patches)
        if mpi.p_is_worker and landpatch.numpatch > 0:
            filter = [pt <=3 for pt in vti.patchtype]
            if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                filter = filter & forcing.forcmask
            filter = filter & vti.patchmask

        if HistForm == 'Gridded':
            histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

        # Soil temperature [K]
        write_history_variable_3d(
            nl_colm_history['t_soisno'], 
            var_accfluxes.a_t_soisno, file_hist, 'f_t_soisno', itime_in_file, 'soilsnow', maxsnl+1, nl_soil-maxsnl, 
            sumarea, filter, 'soil temperature', 'K',gridded, var_accfluxes.nac, spval
        )

        # Liquid water in soil layers [kg/m2]
        write_history_variable_3d(
            nl_colm_history['wliq_soisno'], 
            var_accfluxes.a_wliq_soisno, file_hist, 'f_wliq_soisno', itime_in_file, 'soilsnow', maxsnl+1, nl_soil-maxsnl, 
            sumarea, filter, 'liquid water in soil layers', 'kg/m2',gridded, var_accfluxes.nac, spval
        )

        # Ice lens in soil layers [kg/m2]
        write_history_variable_3d(
            nl_colm_history['wice_soisno'], 
            var_accfluxes.a_wice_soisno, file_hist, 'f_wice_soisno', itime_in_file, 'soilsnow', maxsnl+1, nl_soil-maxsnl, 
            sumarea, filter, 'ice lens in soil layers', 'kg/m2',gridded, var_accfluxes.nac, spval
        )

        # Additional diagnostic variables for output (vegetated land only <= 2)
        if mpi.p_is_worker and landpatch.numpatch > 0:
            filter = [pt <=2 for pt in vti.patchtype]
            if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                filter = filter & forcing.forcmask
            filter = filter & vti.patchmask

        if HistForm == 'Gridded':
            histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

        # Volumetric soil water in layers [m3/m3]
        write_history_variable_3d(
            nl_colm_history['h2osoi'], 
            var_accfluxes.a_h2osoi, file_hist, 'f_h2osoi', itime_in_file, 'soil', 1, nl_soil, 
            sumarea, filter, 'volumetric water in soil layers', 'm3/m3',gridded, var_accfluxes.nac, spval
        )

        # Root water uptake [mm h2o/s]
        write_history_variable_3d(
            nl_colm_history['rootr'], 
            var_accfluxes.a_rootr, file_hist, 'f_rootr', itime_in_file, 'soil', 1, nl_soil, 
            sumarea, filter, 'root water uptake', 'mm h2o/s',gridded, var_accfluxes.nac, spval
        )

        if nl_colm['DEF_USE_PLANTHYDRAULICS']:
            # Vegetation water potential [mm]
            write_history_variable_3d(
                nl_colm_history['vegwp'], 
                var_accfluxes.a_vegwp, file_hist, 'f_vegwp', itime_in_file, 'vegnodes', 1, nvegwcs, 
                sumarea, filter, 'vegetation water potential', 'mm',gridded, var_accfluxes.nac, spval
            )

        # Water table depth [m]
        write_history_variable_2d(
            nl_colm_history['zwt'], 
            var_accfluxes.a_zwt, file_hist, 'f_zwt', itime_in_file, sumarea, filter, 
            'the depth to water table', 'm',gridded, var_accfluxes.nac, spval
        )

        # --------------------------------------------------------------------
        # depth of surface water (including land ice and ocean patches)
        # --------------------------------------------------------------------

        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                filter = [pt <=4 for pt in vti.patchtype]

                if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                    filter = np.logical_and(filter, forcing.forcmask)

                filter = np.logical_and(filter, vti.patchmask)

            if HistForm == 'Gridded':
                histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

            # Water storage in aquifer [mm]
            write_history_variable_2d(nl_colm_history['wa'],
                                    var_accfluxes.a_wa, file_hist, 'f_wa', itime_in_file, sumarea, filter,
                                    'water storage in aquifer', 'mm',gridded, var_accfluxes.nac, spval)

            # Instantaneous water storage in aquifer [mm]
            if mpi.p_is_worker:
                vecacc = vtv.wa.copy()
                vecacc[vecacc != spval] *= var_accfluxes.nac

            write_history_variable_2d(nl_colm_history['wa_inst'],
                                    vecacc, file_hist, 'f_wa_inst', itime_in_file, sumarea, filter,
                                    'instantaneous water storage in aquifer', 'mm',gridded, var_accfluxes.nac, spval)

            # Depth of surface water [mm]
            write_history_variable_2d(nl_colm_history['wdsrf'],
                                    var_accfluxes.a_wdsrf, file_hist, 'f_wdsrf', itime_in_file, sumarea, filter,
                                    'depth of surface water', 'mm',gridded, var_accfluxes.nac, spval)

            # Instantaneous depth of surface water [mm]
            if mpi.p_is_worker:
                vecacc = vtv.wdsrf.copy()
                vecacc[vecacc != spval] *= var_accfluxes.nac

            write_history_variable_2d(nl_colm_history['wdsrf_inst'],
                                    vecacc, file_hist, 'f_wdsrf_inst', itime_in_file, sumarea, filter,
                                    'instantaneous depth of surface water', 'mm',gridded, var_accfluxes.nac, spval)

            # Land water bodies' ice fraction and temperature
            if mpi.p_is_worker:
                if landpatch.numpatch > 0:
                    filter = vti.patchtype == 4
                    if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                        filter = np.logical_and(filter, forcing.forcmask)

            if HistForm == 'Gridded':
                histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

            # Lake temperature [K]
            write_history_variable_3d(nl_colm_history['t_lake'],
                                    var_accfluxes.a_t_lake, file_hist, 'f_t_lake', itime_in_file, 'lake', 1, nl_lake, sumarea, filter,
                                    'lake temperature', 'K',gridded, var_accfluxes.nac, spval)

            # Lake ice fraction cover [0-1]
            write_history_variable_3d(nl_colm_history['lake_icefrac'],
                                    var_accfluxes.a_lake_icefrac, file_hist, 'f_lake_icefrac', itime_in_file, 'lake', 1, nl_lake,
                                    sumarea, filter, 'lake ice fraction cover', '0-1',gridded, var_accfluxes.nac, spval)

            # Retrieve through averaged fluxes
            if mpi.p_is_worker:
                if landpatch.numpatch > 0:
                    filter = [pt < 99 for pt in vti.patchtype]

                    if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                        filter = np.logical_and(filter, forcing.forcmask)

                    filter = np.logical_and(filter, vti.patchmask)

            if HistForm == 'Gridded':
                histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

            # u* in similarity theory [m/s]
            write_history_variable_2d(nl_colm_history['ustar'],
                                    var_accfluxes.a_ustar, file_hist, 'f_ustar', itime_in_file, sumarea, filter,
                                    'u* in similarity theory based on patch', 'm/s',gridded, var_accfluxes.nac, spval)

            # u* in similarity theory [m/s]
            write_history_variable_2d(nl_colm_history['ustar2'],
                                    var_accfluxes.a_ustar2, file_hist, 'f_ustar2', itime_in_file, sumarea, filter,
                                    'u* in similarity theory based on grid', 'm/s',gridded, var_accfluxes.nac, spval)

            # t* in similarity theory [K]
            write_history_variable_2d(nl_colm_history['tstar'],
                                    var_accfluxes.a_tstar, file_hist, 'f_tstar', itime_in_file, sumarea, filter,
                                    't* in similarity theory', 'K',gridded, var_accfluxes.nac, spval)

            # q* in similarity theory [kg/kg]
            write_history_variable_2d(nl_colm_history['qstar'],
                                    var_accfluxes.a_qstar, file_hist, 'f_qstar', itime_in_file, sumarea, filter,
                                    'q* in similarity theory', 'kg/kg',gridded, var_accfluxes.nac, spval)

            # Dimensionless height (z/L,gridded, var_accfluxes.nac, spval) used in Monin-Obukhov theory
            write_history_variable_2d(nl_colm_history['zol'],
                                    var_accfluxes.a_zol, file_hist, 'f_zol', itime_in_file, sumarea, filter,
                                    'dimensionless height (z/L,gridded, var_accfluxes.nac, spval) used in Monin-Obukhov theory', '-',gridded, var_accfluxes.nac, spval)

            # Bulk Richardson number in surface layer
            write_history_variable_2d(nl_colm_history['rib'],
                                    var_accfluxes.a_rib, file_hist, 'f_rib', itime_in_file, sumarea, filter,
                                    'bulk Richardson number in surface layer', '-',gridded, var_accfluxes.nac, spval)

            # Integral of profile FUNCTION for momentum
            write_history_variable_2d(nl_colm_history['fm'],
                                    var_accfluxes.a_fm, file_hist, 'f_fm', itime_in_file, sumarea, filter,
                                    'integral of profile FUNCTION for momentum', '-',gridded, var_accfluxes.nac, spval)

            # Integral of profile FUNCTION for heat
            write_history_variable_2d(nl_colm_history['fh'],
                                    var_accfluxes.a_fh, file_hist, 'f_fh', itime_in_file, sumarea, filter,
                                    'integral of profile FUNCTION for heat', '-',gridded, var_accfluxes.nac, spval)

            # Integral of profile FUNCTION for moisture
            write_history_variable_2d(nl_colm_history['fq'],
                                    var_accfluxes.a_fq, file_hist, 'f_fq', itime_in_file, sumarea, filter,
                                    'integral of profile FUNCTION for moisture', '-',gridded, var_accfluxes.nac, spval)

            # 10m u-velocity [m/s]
            write_history_variable_2d(nl_colm_history['us10m'],
                                    var_accfluxes.a_us10m, file_hist, 'f_us10m', itime_in_file, sumarea, filter,
                                    '10m u-velocity', 'm/s',gridded, var_accfluxes.nac, spval)

            # 10m v-velocity [m/s]
            write_history_variable_2d(nl_colm_history['vs10m'],
                                    var_accfluxes.a_vs10m, file_hist, 'f_vs10m', itime_in_file, sumarea, filter,
                                    '10m v-velocity', 'm/s',gridded, var_accfluxes.nac, spval)

            # Integral of profile FUNCTION for momentum at 10m [-]
            write_history_variable_2d(nl_colm_history['fm10m'],
                                    var_accfluxes.a_fm10m, file_hist, 'f_fm10m', itime_in_file, sumarea, filter,
                                    'integral of profile FUNCTION for momentum at 10m', '-',gridded, var_accfluxes.nac, spval)

            # Total reflected solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['sr'],
                                    var_accfluxes.a_sr, file_hist, 'f_sr', itime_in_file, sumarea, filter,
                                    'reflected solar radiation at surface [W/m2]', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Incident direct beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['solvd'],
                                    var_accfluxes.a_solvd, file_hist, 'f_solvd', itime_in_file, sumarea, filter,
                                    'incident direct beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Incident diffuse beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['solvi'],
                                    var_accfluxes.a_solvi, file_hist, 'f_solvi', itime_in_file, sumarea, filter,
                                    'incident diffuse beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Incident direct beam nir solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['solnd'],
                                    var_accfluxes.a_solnd, file_hist, 'f_solnd', itime_in_file, sumarea, filter,
                                    'incident direct beam nir solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Incident diffuse beam nir solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['solni'],
                                    var_accfluxes.a_solni, file_hist, 'f_solni', itime_in_file, sumarea, filter,
                                    'incident diffuse beam nir solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Reflected direct beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['srvd'],
                                    var_accfluxes.a_srvd, file_hist, 'f_srvd', itime_in_file, sumarea, filter,
                                    'reflected direct beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Reflected diffuse beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['srvi'],
                                    var_accfluxes.a_srvi, file_hist, 'f_srvi', itime_in_file, sumarea, filter,
                                    'reflected diffuse beam vis solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Reflected direct beam nir solar radiation (W/m2,gridded, var_accfluxes.nac, spval)
            write_history_variable_2d(nl_colm_history['srnd'],
                                    var_accfluxes.a_srnd, file_hist, 'f_srnd', itime_in_file, sumarea, filter,
                                    'reflected direct beam nir solar radiation (W/m2,gridded, var_accfluxes.nac, spval)', 'W/m2',gridded, var_accfluxes.nac, spval)

            write_history_variable_2d(nl_colm_history["srni"], var_accfluxes.a_srni, file_hist, 'f_srni', itime_in_file, sumarea, filter,
                           'reflected diffuse beam nir solar radiation (W/m2)', 'W/m2',gridded, var_accfluxes.nac, spval)

            # Local noon fluxes
            if mpi.p_is_worker:
                if landpatch.numpatch > 0:
                    filter = (var_accfluxes.nac_ln > 0)
                    if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                        filter = filter & forcing.forcmask

            if HistForm == 'Gridded':
                histgrid.mp2g_hist.map(VecOnes, sumarea, spv=spval, msk=filter)

            # incident direct beam vis solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['solvdln'], var_accfluxes.a_solvdln, file_hist, 'f_solvdln', itime_in_file, sumarea, filter,
                                    'incident direct beam vis solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # incident diffuse beam vis solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['solviln'], var_accfluxes.a_solviln, file_hist, 'f_solviln', itime_in_file, sumarea, filter,
                                    'incident diffuse beam vis solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # incident direct beam nir solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['solndln'], var_accfluxes.a_solndln, file_hist, 'f_solndln', itime_in_file, sumarea, filter,
                                    'incident direct beam nir solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # incident diffuse beam nir solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['solniln'], var_accfluxes.a_solniln, file_hist, 'f_solniln', itime_in_file, sumarea, filter,
                                    'incident diffuse beam nir solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # reflected direct beam vis solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['srvdln'], var_accfluxes.a_srvdln, file_hist, 'f_srvdln', itime_in_file, sumarea, filter,
                                    'reflected direct beam vis solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # reflected diffuse beam vis solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['srviln'], var_accfluxes.a_srviln, file_hist, 'f_srviln', itime_in_file, sumarea, filter,
                                    'reflected diffuse beam vis solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # reflected direct beam nir solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['srndln'], var_accfluxes.a_srndln, file_hist, 'f_srndln', itime_in_file, sumarea, filter,
                                    'reflected direct beam nir solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            # reflected diffuse beam nir solar radiation at local noon (W/m2)
            write_history_variable_ln(nl_colm_history['srniln'], var_accfluxes.a_srniln, file_hist, 'f_srniln', itime_in_file, sumarea, filter,
                                    'reflected diffuse beam nir solar radiation at local noon (W/m2)', 'W/m2', gridded, spval, var_accfluxes.nac_ln)

            if  nl_colm['CatchLateralFlow']:
                pass
                # hist_basin_out(file_hist, idate)

            # Deallocate if allocated
            if filter is not None:
                del filter
            if VecOnes is not None:
                del VecOnes

            # Call function to flush accumulated fluxes
            var_accfluxes.flush_acc_fluxes(maxsnl, nl_soil, nvegwcs,nl_lake)


