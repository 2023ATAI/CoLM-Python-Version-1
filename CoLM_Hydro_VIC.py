import numpy as np
import CoLM_Hydro_VIC_Variables

def compute_vic_runoff(nl_colm, soil_con, ppt, frost_fract, cell):
    """
    模拟Fortran中compute_vic_runoff子例程的功能，用于计算VIC模型相关的径流等变量
    """
    # 假设dltime从合适的地方获取（原Fortran代码中从DEF_simulation_time%timestep获取）
    runoff_steps_per_dt = 0
    dltime = nl_colm['DEF_simulation_time%timestep']  # 这里先简单赋值，实际需要正确获取对应的值
    runoff_steps_per_day = 86400 // dltime
    model_steps_per_day = 86400 // dltime
    # runoff_steps_per_dt = runoff_steps_per_day // model_steps_per_day
    org_moist = np.zeros(CoLM_Hydro_VIC_Variables.MAX_LAYERS)

    # 设置临时变量，对应Fortran代码中的循环设置操作
    resid_moist = [soil_con.resid_moist[lindex] for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer)]
    max_moist = [soil_con.max_moist[lindex] for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer)]
    Ksat = [soil_con.Ksat[lindex] / runoff_steps_per_day for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer)]

    layer = cell.layer
    cell.runoff = 0
    cell.baseflow = 0
    cell.asat = 0
    A = 0
    Q12 = np.zeros(CoLM_Hydro_VIC_Variables.MAX_LAYERS - 1)
    evap = np.zeros(CoLM_Hydro_VIC_Variables.MAX_LAYERS, CoLM_Hydro_VIC_Variables.MAX_FROST_AREAS)
    avail_liq = np.zeros(CoLM_Hydro_VIC_Variables.MAX_LAYERS, CoLM_Hydro_VIC_Variables.MAX_FROST_AREAS)
    runoff = np.zeros(CoLM_Hydro_VIC_Variables.MAX_FROST_AREAS)

    # 初始化baseflow列表
    baseflow = np.zeros(CoLM_Hydro_VIC_Variables.MAX_FROST_AREAS)

    for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer):
        # 注意Python索引从0开始，这里对应Fortran代码中的索引转换
        evap[lindex, 0] = layer[lindex].evap / runoff_steps_per_dt
        org_moist[lindex] = layer[lindex].moist
        layer[lindex].moist = 0.0

        if evap[lindex, 0] > 0:
            sum_liq = 0.0
            for fidx in range(CoLM_Hydro_VIC_Variables.Nfrost):
                avail_liq[lindex, fidx] = org_moist[lindex] - layer[lindex].ice[fidx] - resid_moist[lindex]
                if avail_liq[lindex, fidx] < 0:
                    avail_liq[lindex, fidx] = 0
                sum_liq += avail_liq[lindex, fidx] * frost_fract[fidx]

            if sum_liq > 0:
                evap_fraction = evap[lindex, 0] / sum_liq
            else:
                evap_fraction = 1.0

            evap_sum = evap[lindex, 0]
            for fidx in range(CoLM_Hydro_VIC_Variables.Nfrost - 1, -1, -1):
                evap[lindex, fidx] = avail_liq[lindex, fidx] * evap_fraction
                avail_liq[lindex, fidx] = avail_liq[lindex, fidx] - evap[lindex, fidx]
                evap_sum -= evap[lindex, fidx] * frost_fract[fidx]
        else:
            for fidx in range(CoLM_Hydro_VIC_Variables.Nfrost - 1, 0, -1):
                evap[lindex, fidx] = evap[lindex, 0]

    for fidx in range(CoLM_Hydro_VIC_Variables.Nfrost):
        inflow = ppt

        liq = [org_moist[lindex] - layer[lindex].ice[fidx] for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer)]
        ice = [layer[lindex].ice[fidx] for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer)]

        tmp_moist_for_runoff = [liq[lindex] + ice[lindex] for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer)]

        compute_runoff_and_asat(soil_con, tmp_moist_for_runoff, inflow, A, runoff[fidx])
        tmp_dt_runoff = [runoff[0] / runoff_steps_per_dt]

        dt_inflow = inflow / runoff_steps_per_dt
        Dsmax = soil_con.Dsmax / runoff_steps_per_day

        for time_step in range(runoff_steps_per_dt):
            inflow = dt_inflow

            for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer - 1):
                tmp_liq = liq[lindex] - evap[lindex, fidx]
                if tmp_liq < resid_moist[lindex]:
                    tmp_liq = resid_moist[lindex]
                if tmp_liq > resid_moist[lindex]:
                    Q12[lindex] = calc_Q12(Ksat[lindex], tmp_liq, resid_moist[lindex], max_moist[lindex],
                                          soil_con.expt[lindex])
                else:
                    Q12[lindex] = 0.0

            last_index = 0.0
            for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer - 1):
                if lindex == 0:
                    dt_runoff = tmp_dt_runoff[fidx]
                else:
                    dt_runoff = 0.0

                tmp_inflow = 0.0

                liq[lindex] = liq[lindex] + (inflow - dt_runoff) - (Q12[lindex] + evap[lindex, fidx])

                if (liq[lindex] + ice[lindex]) > max_moist[lindex]:
                    tmp_inflow = (liq[lindex] + ice[lindex]) - max_moist[lindex]
                    liq[lindex] = max_moist[lindex] - ice[lindex]

                    if lindex == 0:
                        Q12[lindex] += tmp_inflow
                        tmp_inflow = 0.0
                    else:
                        tmplayer = lindex
                        while tmp_inflow > 0:
                            tmplayer -= 1
                            if tmplayer < 0:
                                runoff[fidx] += tmp_inflow
                                tmp_inflow = 0.0
                            else:
                                liq[tmplayer] += tmp_inflow
                                if (liq[tmplayer] + ice[tmplayer]) > max_moist[tmplayer]:
                                    tmp_inflow = (liq[tmplayer] + ice[tmplayer]) - max_moist[tmplayer]
                                    liq[tmplayer] = max_moist[tmplayer] - ice[tmplayer]
                                else:
                                    tmp_inflow = 0.0

                if liq[lindex] < 0:
                    Q12[lindex] += liq[lindex]
                    liq[lindex] = 0.0
                if (liq[lindex] + ice[lindex]) < resid_moist[lindex]:
                    Q12[lindex] += (liq[lindex] + ice[lindex]) - resid_moist[lindex]
                    liq[lindex] = resid_moist[lindex] - ice[lindex]

                inflow = Q12[lindex] + tmp_inflow
                Q12[lindex] += tmp_inflow

                last_index += 1

            # 计算Baseflow
            lindex = CoLM_Hydro_VIC_Variables.Nlayer - 1
            rel_moist = (liq[lindex] - resid_moist[lindex]) / (max_moist[lindex] - resid_moist[lindex])
            frac = Dsmax * soil_con.Ds / soil_con.Ws
            dt_baseflow = frac * rel_moist
            if rel_moist > soil_con.Ws:
                frac = (rel_moist - soil_con.Ws) / (1 - soil_con.Ws)
                dt_baseflow += Dsmax * (1 - soil_con.Ds / soil_con.Ws) * frac ** soil_con.c
            if dt_baseflow < 0:
                dt_baseflow = 0.0

            liq[lindex] = liq[lindex] + Q12[lindex -1] - (evap[lindex, fidx] + dt_baseflow)

            tmp_moist = 0.0
            if (liq[lindex] + ice[lindex]) < resid_moist[lindex]:
                dt_baseflow += (liq[lindex] + ice[lindex]) - resid_moist[lindex]
                liq[lindex] = resid_moist[lindex] - ice[lindex]
            if (liq[lindex] + ice[lindex]) > max_moist[lindex]:
                tmp_moist = (liq[lindex] + ice[lindex]) - max_moist[lindex]
                liq[lindex] = max_moist[lindex] - ice[lindex]
                tmplayer = lindex
                while tmp_moist > 0:
                    tmplayer -= 1
                    if tmplayer < 0:
                        runoff[fidx] += tmp_moist
                        tmp_moist = 0.0
                    else:
                        liq[tmplayer] += tmp_moist
                        if (liq[tmplayer] + ice[tmplayer]) > max_moist[tmplayer]:
                            tmp_moist = (liq[tmplayer] + ice[tmplayer]) - max_moist[tmplayer]
                            liq[tmplayer] = max_moist[tmplayer] - ice[tmplayer]
                        else:
                            tmp_moist = 0

            baseflow[fidx] += dt_baseflow

        if baseflow[fidx] < 0:
            baseflow[fidx] = 0.0

        for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer):
            tmp_moist_for_runoff[lindex] = liq[lindex] + ice[lindex]
        tmp_runoff = compute_runoff_and_asat(soil_con, tmp_moist_for_runoff, 0, A)

        for lindex in range(CoLM_Hydro_VIC_Variables.Nlayer):
            layer[lindex].moist += (liq[lindex] + ice[lindex]) * frost_fract[fidx]
        cell.asat += A * frost_fract[fidx]
        cell.runoff += runoff[fidx] * frost_fract[fidx]
        cell.baseflow += baseflow[fidx] * frost_fract[fidx]

    return cell


def compute_runoff_and_asat(soil_con, moist, inflow, A, runoff):
    """
    对应Fortran中的compute_runoff_and_asat子例程，计算饱和区域和径流相关量
    """
    top_moist = 0.0
    top_max_moist = 0.0
    top_moist += sum(moist[:CoLM_Hydro_VIC_Variables.Nlayer - 1])
    top_max_moist += sum(soil_con.max_moist[:CoLM_Hydro_VIC_Variables.Nlayer - 1])
    if top_moist > top_max_moist:
        top_moist = top_max_moist

    ex = soil_con.b_infilt / (1.0 + soil_con.b_infilt)
    A = 1.0 - (1.0 - top_moist / top_max_moist) ** ex

    max_infil = (1.0 + soil_con.b_infilt) * top_max_moist
    i_0 = max_infil * (1.0 - (1.0 - A) ** (1.0 / soil_con.b_infilt))

    if inflow == 0.0:
        runoff = 0.0
    elif max_infil == 0.0:
        runoff = inflow
    elif (i_0 + inflow) > max_infil:
        runoff = inflow - top_max_moist + top_moist
    else:
        basis = 1.0 - (i_0 + inflow) / max_infil
        runoff = (inflow - top_max_moist + top_moist +
                  top_max_moist * basis ** (1.0 * (1.0 + soil_con.b_infilt)))
    if runoff < 0.0:
        runoff = 0.0
    return A, runoff


def calc_Q12(Ksat, init_moist, resid_moist, max_moist, expt):
    """
    对应Fortran中的calc_Q12子例程，计算两层之间的排水情况
    """
    Q12 = init_moist - ((init_moist - resid_moist) ** (1.0 - expt) - Ksat /
                       (max_moist - resid_moist) ** expt * (1.0 - expt)) ** (1.0 / (1.0 - expt)) - resid_moist
    return Q12


def compute_zwt(soil_con, lindex, moist):
    """
    对应Fortran中的compute_zwt子例程，计算地下水位（zwt）
    """
    MISSING = -99999
    zwt = MISSING

    i = CoLM_Hydro_VIC_Variables.MAX_ZWTVMOIST - 1
    while i >= 1 and moist > soil_con.zwtvmoist_moist[lindex, i]:
        i -= 1

    if i == CoLM_Hydro_VIC_Variables.MAX_ZWTVMOIST - 1:
        if moist < soil_con.zwtvmoist_moist[lindex, i]:
            zwt = 999.0
        elif moist == soil_con.zwtvmoist_moist[lindex, i]:
            zwt = soil_con.zwtvmoist_zwt[lindex, i]
    else:
        zwt = soil_con.zwtvmoist_zwt[lindex, i + 1] + \
            (soil_con.zwtvmoist_zwt[lindex, i] - soil_con.zwtvmoist_zwt[lindex, i + 1]) * \
            (moist - soil_con.zwtvmoist_moist[lindex, i + 1]) /  \
            (soil_con.zwtvmoist_moist[lindex, i] - soil_con.zwtvmoist_moist[lindex, i + 1])

    return zwt