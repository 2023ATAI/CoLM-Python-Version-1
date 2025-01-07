import numpy as np

def snowfraction(lai, sai, z0m, zlnd, scv, snowdp):
    """
    提供雪覆盖分数

    原作者: Yongjiu Dai, /09/1999/, /04/2014/
    修订: Hua Yuan, 10/2019: 删除sigf以与PFT分类兼容
    """
    
    # 定义常量
    m = 1.0  # CLM4.5中使用的m值为1.0。Niu等人（2007）给出的m值为1.6。Niu（2012）建议为3.0。
    
    # 定义输出变量
    wt = 0.0   # 被雪覆盖的植被部分 [-]
    sigf = 0.0 # 不包括被雪覆盖的植被部分的植被覆盖分数 [-]
    fsno = 0.0 # 被雪覆盖的土壤部分 [-]

    if lai + sai > 1e-6:
        # 被雪掩埋（覆盖）的植被部分
        wt = 0.1 * snowdp / z0m
        wt = wt / (1.0 + wt)

        # 不包括被雪覆盖的植被覆盖分数
        sigf = 1.0 - wt
    else:
        wt = 0.0
        sigf = 0.0

    # 土壤被雪覆盖的部分
    if snowdp > 0.0:
        fmelt = (scv / snowdp / 100.0) ** m
        fsno = np.tanh(snowdp / (2.5 * zlnd * fmelt))

    return wt, sigf, fsno

