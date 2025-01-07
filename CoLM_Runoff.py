import numpy as np

def SurfaceRunoff_SIMTOP(nl_soil, wtfact, wimp, porsl, psi0, hksati, z_soisno, dz_soisno, zi_soisno,
                         eff_porosity, icefrac, zwt, gwat):
    """
    模拟地表径流的函数，对应Fortran中的SurfaceRunoff_SIMTOP子例程
    """
    rsur =0.0
    rsur_se = 0.0
    rsur_ie= 0.0

    # 计算饱和区域的比例
    fff = 0.5  # 径流衰减因子 (m-1)，对应Fortran代码中的参数定义
    fsat = wtfact * min(1.0, np.exp(-0.5 * fff * zwt))

    # 计算最大入渗能力
    slice_end = min(3, nl_soil)
    qinmax = np.min(10. ** (-6.0 * icefrac[0:slice_end]) * hksati[0:slice_end])
    if eff_porosity[0] < wimp:
        qinmax = 0.

    # 计算地表径流
    rsur = fsat * max(0.0, gwat) + (1. - fsat) * max(0., gwat - qinmax)

    if rsur_se is not None:
        rsur_se = fsat * max(0.0, gwat)
    if rsur_ie is not None:
        rsur_ie = (1. - fsat) * max(0., gwat - qinmax)

    return rsur, rsur_se, rsur_ie


def SubsurfaceRunoff_SIMTOP(nl_soil, icefrac, dz_soisno, zi_soisno, zwt):
    """
    模拟地下径流的函数，对应Fortran中的SubsurfaceRunoff_SIMTOP子例程
    """
    # 将土层厚度单位转换为mm
    dzmm = np.array([dz * 1000 for dz in dz_soisno])
    jwt = nl_soil -1
    # 确定水表层上方的土层索引
    for j in range(1, nl_soil + 1):
        if zwt <= zi_soisno[j]:
            jwt = j - 1
            break

    # 计算地形径流相关参数
    dzsum = 0.
    icefracsum = 0.
    for j in range(max(jwt, 0), nl_soil):
        dzsum += dzmm[j]
        icefracsum += icefrac[j] * dzmm[j]

    # 计算含冰阻抗因子及地下径流
    fracice_rsub = max(0., np.exp(-3. * (1. - (icefracsum / dzsum))) - np.exp(-3.)) / (1.0 - np.exp(-3.))
    imped = max(0., 1. - fracice_rsub)
    rsubst = imped * 5.5e-3 * np.exp(-2.5 * zwt)

    return rsubst


def Runoff_XinAnJiang(dz_soisno, eff_porosity, vol_liq, topostd, gwat, deltim):
    """
    对应Fortran中的Runoff_XinAnJiang子例程，计算径流相关量
    """
    watin = gwat * deltim / 1000.
    if watin <= 0:
        rsur = 0
        rsubst = 0
    else:
        sigmin = 100.
        sigmax = 1000.
        btopo = (topostd - sigmin) / (topostd - sigmax)
        btopo = min(max(btopo, 0.01), 0.5)

        w_int = np.sum(vol_liq[0:6] * dz_soisno[0:6])
        wsat_int = np.sum(eff_porosity[0:6] * dz_soisno[0:6])

        wtmp = (1 - w_int / wsat_int) ** (1 / (btopo + 1)) - watin / ((btopo + 1) * wsat_int)
        infil = wsat_int - w_int - wsat_int * (max(0., wtmp)) ** (btopo + 1)
        infil = min(infil, watin)

        rsur = (watin - infil) * 1000. / deltim
        rsubst = 0

    return rsur, rsubst


def Runoff_SimpleVIC(dz_soisno, eff_porosity, vol_liq, BVIC, gwat, deltim):
    """
    对应Fortran中的Runoff_SimpleVIC子例程，计算径流相关量
    """
    watin = gwat * deltim / 1000.  # 将单位转换为m
    if watin <= 0:
        rsur = 0
        rsubst = 0
    else:
        w_int = np.sum(vol_liq[0:6] * dz_soisno[0:6])
        wsat_int = np.sum(eff_porosity[0:6] * dz_soisno[0:6])

        InfilExpFac = BVIC / (1.0 + BVIC)
        SoilSaturateFrac = 1.0 - (max(0.0, (1.0 - (w_int / wsat_int)))) ** InfilExpFac
        SoilSaturateFrac = max(0.0, SoilSaturateFrac)
        SoilSaturateFrac = min(1.0, SoilSaturateFrac)

        WaterDepthMax = (1.0 + BVIC) * wsat_int
        WaterDepthInit = WaterDepthMax * (1.0 - (1.0 - SoilSaturateFrac) ** (1.0 / BVIC))

        if WaterDepthMax <= 0.0:
            RunoffSurface = watin
        elif (WaterDepthInit + watin) > WaterDepthMax:
            RunoffSurface = (WaterDepthInit + w_int) - WaterDepthMax
        else:
            InfilVarTmp = 1.0 - ((WaterDepthInit + watin) / WaterDepthMax)
            RunoffSurface = watin - wsat_int + w_int + wsat_int * (InfilVarTmp ** (1.0 + BVIC))

        if RunoffSurface < 0.0:
            RunoffSurface = 0.0
        if RunoffSurface > watin:
            RunoffSurface = watin

        infil = watin - RunoffSurface
        rsur = RunoffSurface * 1000. / deltim
        rsubst = 0

    return rsur, rsubst