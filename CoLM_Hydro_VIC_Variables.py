import numpy as np
# 假设相关变量（如dz_soi、nl_soil等）已经在合适的作用域内正确定义

# 定义一些常量，对应Fortran代码中的参数定义
Nlayer = 3
Nfrost = 1
MAX_LAYERS = 3
MAX_FROST_AREAS = 3
MAX_ZWTVMOIST = 11
colm2vic_lay = [3, 6, 10]


# 定义layer_data_struct结构体对应的Python类
class LayerDataStruct:
    def __init__(self):
        self.ice = np.zeros(MAX_FROST_AREAS)
        self.moist = 0
        self.evap = 0
        self.zwt = 0


# 定义cell_data_struct结构体对应的Python类
class CellDataStruct:
    def __init__(self):
        self.asat = 0
        self.baseflow = 0
        self.runoff = 0
        self.layer = [LayerDataStruct() for _ in range(MAX_LAYERS)]
        self.zwt = 0
        self.zwt_lumped = 0


# 定义soil_con_struct结构体对应的Python类
class SoilConStruct:
    def __init__(self):
        self.frost_fract = np.zeros(MAX_FROST_AREAS)
        self.max_moist = np.zeros(MAX_LAYERS)
        self.resid_moist = np.zeros(MAX_LAYERS)
        self.Ksat = np.zeros(MAX_LAYERS)
        self.expt = np.zeros(MAX_LAYERS)
        self.b_infilt = 0
        self.Ds = 0
        self.Ws = 0
        self.Dsmax = 0
        self.c = 0
        self.depth = np.zeros(MAX_LAYERS)
        self.bubble = np.zeros(MAX_LAYERS)
        self.zwtvmoist_zwt = np.zeros((MAX_LAYERS + 2, MAX_ZWTVMOIST))
        self.zwtvmoist_moist = np.zeros((MAX_LAYERS + 2, MAX_ZWTVMOIST))


def vic_para(nl_colm, porsl, theta_r, hksati, bsw, wice_soisno, wliq_soisno, fevpg, rootflux,
             b_infilt, Dsmax, Ds, Ws, c, soil_con, cell, dz_soi):
    """
    对应Fortran中的vic_para子例程，处理VIC相关参数设置
    """
    dltime = nl_colm['DEF_simulation_time%timestep']
    soil_tmp = np.zeros(Nlayer)
    ice_tmp = np.zeros(Nlayer)

    # CoLM2VIC函数调用转换
    soil_tmp = CoLM2VIC(dz_soi)
    soil_con.depth = soil_tmp

    # CoLM2VIC_weight函数调用转换（处理max_moist）
    soil_tmp = CoLM2VIC_weight(porsl, dz_soi)
    soil_con.max_moist = [s * soil_con.depth[i] * 1000 for i, s in enumerate(soil_tmp)]

    # CoLM2VIC_weight函数调用转换（处理resid_moist）
    soil_tmp = CoLM2VIC_weight(theta_r, dz_soi)
    soil_con.resid_moist = [s * soil_con.depth[i] * 1000 for i, s in enumerate(soil_tmp)]

    # CoLM2VIC_weight函数调用转换（处理Ksat）
    soil_tmp = CoLM2VIC_weight(hksati, dz_soi)
    soil_con.Ksat = [s * 86400 for s in soil_tmp]

    # CoLM2VIC_weight函数调用转换（处理expt）
    soil_tmp = CoLM2VIC_weight(bsw, dz_soi)
    soil_con.expt = [s * 2 + 3 for s in soil_tmp]

    soil_con.b_infilt = b_infilt
    soil_con.Dsmax = Dsmax
    soil_con.Ds = Ds
    soil_con.Ws = Ws
    soil_con.c = c

    soil_con.frost_fract[:] = 1
    if sum(wice_soisno) > 0:
        Nfrost = 3
        for k in range(1, Nfrost + 1):
            if Nfrost == 1:
                soil_con.frost_fract[k - 1] = 1.0
            elif Nfrost == 2:
                soil_con.frost_fract[k - 1] = 0.5
            else:
                val = 1.0 / (Nfrost - 1)
                if k == 1 or k == Nfrost:
                    val /= 2.0
                soil_con.frost_fract[k - 1] = val

    # CoLM2VIC函数调用转换（处理cell层的moist）
    CoLM2VIC(wliq_soisno, soil_tmp)
    for ilay in range(1, Nlayer + 1):
        cell.layer[ilay - 1].moist = soil_tmp[ilay - 1]

    if sum(wice_soisno) > 0:
        for ilay in range(1, Nlayer + 1):
            lp = colm2vic_lay[ilay - 1]
            if ilay == 1:
                lb = 1
            else:
                lb = colm2vic_lay[ilay - 2] + 1
            vic_ice = VIC_IceLay(lb, lp, wice_soisno[lb - 1:lp])
            cell.layer[ilay - 1].ice = vic_ice

    # CoLM2VIC函数调用转换（处理cell层的evap）
    CoLM2VIC(rootflux, soil_tmp)
    for ilay in range(1, Nlayer + 1):
        cell.layer[ilay - 1].evap = soil_tmp[ilay - 1] * dltime
    cell.layer[0].evap = cell.layer[0].evap + fevpg * dltime
    return soil_con, cell

def VIC_IceLay(lb, lp, colm_ice):
    """
    对应Fortran中的VIC_IceLay子例程，处理冰相关分层数据转换
    """
    colm_lay = lp - lb + 1
    ice_tmp = colm_ice
    totalSum = sum(ice_tmp)
    vic_ice = np.zeros(3)
    vic_lay = 3

    if colm_lay == 1:
        vic_ice = totalSum / vic_lay
    elif colm_lay == 2:
        vic_ice[0] = ice_tmp[0] * 2.0 / vic_lay
        vic_ice[2] = ice_tmp[1] * 2.0 / vic_lay
    elif colm_lay == 3:
        vic_ice = ice_tmp
    else:
        idx = 1
        while idx <= min(int((colm_lay - 1) / vic_lay), vic_lay):
            multiplier = 1 if colm_lay > idx * vic_lay else 0
            vic_ice[0] += ice_tmp[idx - 1] * multiplier
            vic_ice[2] += ice_tmp[colm_lay - idx] * multiplier
            idx += 1
        multiplier = (colm_lay - idx * vic_lay) / vic_lay if colm_lay <= (idx + 1) * vic_lay else 0
        vic_ice[0] += ice_tmp[idx] * multiplier
        vic_ice[2] += ice_tmp[colm_lay - idx -1] * multiplier
    vic_ice[1] = totalSum - vic_ice[0] - vic_ice[2]
    return vic_ice


def CoLM2VIC(colm_water):
    """
    对应Fortran中的CoLM2VIC子例程，进行从CoLM到VIC的水量转换
    """
    vic_water = np.zeros(Nlayer)
    for i_vic in range(1, Nlayer + 1):
        vic_water[i_vic - 1] = 0
        if i_vic == 1:
            for i_colm in range(1, colm2vic_lay[i_vic - 1] + 1):
                vic_water[i_vic - 1] += colm_water[i_colm-1]
        else:
            for i_colm in range(colm2vic_lay[i_vic - 2] + 1, colm2vic_lay[i_vic - 1] + 1):
                vic_water[i_vic - 1] += colm_water[i_colm-1]
    return vic_water


def CoLM2VIC_weight(colm_water, dz_soi):
    """
    对应Fortran中的CoLM2VIC_weight子例程，进行带权重的从CoLM到VIC的水量转换
    """
    vic_water = np.zeros(Nlayer)
    for i_vic in range(1, Nlayer + 1):
        vic_water[i_vic - 1] = 0
        if i_vic == 1:
            sum_dz_soi = sum(dz_soi[1:colm2vic_lay[i_vic - 1] + 1])
            for i_colm in range(1, colm2vic_lay[i_vic - 1] + 1):
                vic_water[i_vic - 1] += colm_water[i_colm-1] * dz_soi[i_colm-1]
            vic_water[i_vic - 1] /= sum_dz_soi
        else:
            sum_dz_soi = sum(dz_soi[colm2vic_lay[i_vic - 2] + 1:colm2vic_lay[i_vic - 1]])
            for i_colm in range(colm2vic_lay[i_vic - 2] + 1, colm2vic_lay[i_vic - 1] + 1):
                vic_water[i_vic - 1] += colm_water[i_colm-1] * dz_soi[i_colm-1]
            vic_water[i_vic - 1] /= sum_dz_soi
    return vic_water


def VIC2CoLM(colm_water, dz_soi):
    """
    对应Fortran中的VIC2CoLM子例程，进行从VIC到CoLM的水量转换
    """
    vic_water = np.zeros(Nlayer)
    for i_vic in range(1, Nlayer + 1):
        if i_vic == 1:
            sum_dz_soi = sum(dz_soi[1:colm2vic_lay[i_vic - 1] + 1])
            for i_colm in range(1, colm2vic_lay[i_vic - 1] + 1):
                colm_water[i_colm-1] = vic_water[i_vic - 1] * (dz_soi[i_colm-1] / sum_dz_soi)
        else:
            sum_dz_soi = sum(dz_soi[colm2vic_lay[i_vic - 2] + 1:colm2vic_lay[i_vic - 1]])
            for i_colm in range(colm2vic_lay[i_vic - 2] + 1, colm2vic_lay[i_vic - 1] + 1):
                colm_water[i_colm-1] = vic_water[i_vic - 1] * (dz_soi[i_colm-1] / sum_dz_soi)