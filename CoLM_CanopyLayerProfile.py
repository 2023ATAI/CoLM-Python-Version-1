import numpy as np
import CoLM_FrictionVelocity

def cal_z0_displa(lai, h, fc
                  ##USE MOD_Const_Physical, only: vonkar
                  ,vonkar):
    # Constants
    Cd = 0.2
    cd1 = 7.5
    psih = 0.193

    # Calculate sqrtdragc
    sqrtdragc = -vonkar / (np.log(0.01 / h) - psih)
    sqrtdragc = max(sqrtdragc, 0.0031 ** 0.5)

    if sqrtdragc <= 0.3:
        fai = (sqrtdragc ** 2 - 0.003) / 0.3
        fai = min(fai, fc * (1 - np.exp(-20)))
    else:
        fai = 0.29
        print("z0m, displa error!")

    # Calculate delta
    lai0 = -np.log(1 - fai / fc) / 0.5
    temp1 = (2 * cd1 * fai) ** 0.5
    delta = -h * (fc * 1.1 * np.log(1 + (Cd * lai0 * fc) ** 0.25) +
                  (1 - fc) * (1 - (1 - np.exp(-temp1)) / temp1))

    # Calculate z0m, displa
    fai = fc * (1 - np.exp(-0.5 * lai))
    sqrtdragc = min((0.003 + 0.3 * fai) ** 0.5, 0.3)
    temp1 = (2 * cd1 * fai) ** 0.5

    if lai > lai0:
        displa = delta + h * ((fc) * 1.1 * np.log(1 + (Cd * lai * fc) ** 0.25) +
                              (1 - fc) * (1 - (1 - np.exp(-temp1)) / temp1))
    else:
        displa = h * ((fc) * 1.1 * np.log(1 + (Cd * lai * fc) ** 0.25) +
                      (1 - fc) * (1 - (1 - np.exp(-temp1)) / temp1))

    displa = max(displa, 0)
    z0 = (h - displa) * np.exp(-vonkar / sqrtdragc + psih)

    if z0 < 0.01:
        z0 = 0.01
        displa = 0

    return z0, displa


def ueffect(utop, htop, hbot, z0mg, alpha, bee, fc):
    roots = np.zeros(2)
    rootn = 0
    uint = 0.0

    # The dichotomy method to find the root satisfies a certain accuracy,
    # assuming that there are at most 2 roots
    roots, rootn = ufindroots(htop, hbot, (htop + hbot) / 2.0, utop, htop, hbot, z0mg, alpha, roots, rootn)

    if rootn == 0:  # no root
        uint += fuint(utop, htop, hbot, htop, hbot, z0mg, alpha, bee, fc)
    
    if rootn == 1:
        uint += fuint(utop, htop, roots[0], htop, hbot, z0mg, alpha, bee, fc)
        uint += fuint(utop, roots[0], hbot, htop, hbot, z0mg, alpha, bee, fc)
    
    if rootn == 2:
        uint += fuint(utop, htop, roots[0], htop, hbot, z0mg, alpha, bee, fc)
        uint += fuint(utop, roots[0], roots[1], htop, hbot, z0mg, alpha, bee, fc)
        uint += fuint(utop, roots[1], hbot, htop, hbot, z0mg, alpha, bee, fc)

    ueffect = uint / (htop - hbot)
    return ueffect

def ufindroots(ztop, zbot, zmid, utop, htop, hbot, z0mg, alpha, roots=None, rootn=None):
    udiff_ub = udiff(ztop, utop, htop, hbot, z0mg, alpha)
    udiff_lb = udiff(zmid, utop, htop, hbot, z0mg, alpha)
    
    if udiff_ub * udiff_lb == 0:
        if udiff_lb == 0:
            rootn += 1
            if rootn > 2:
                raise ValueError("U root number > 2, abort!")
            roots[rootn - 1] = zmid
    elif udiff_ub * udiff_lb < 0:
        if ztop - zmid < 0.01:
            rootn += 1
            if rootn > 2:
                raise ValueError("U root number > 2, abort!")
            roots[rootn - 1] = (ztop + zmid) / 2.0
        else:
            roots, rootn = ufindroots(ztop, zmid, (ztop + zmid) / 2.0, utop, htop, hbot, z0mg, alpha, roots, rootn)
    
    udiff_ub = udiff(zmid, utop, htop, hbot, z0mg, alpha)
    udiff_lb = udiff(zbot, utop, htop, hbot, z0mg, alpha)
    
    if udiff_ub * udiff_lb == 0:
        if udiff_ub == 0:
            rootn += 1
            if rootn > 2:
                raise ValueError("U root number > 2, abort!")
            roots[rootn - 1] = zmid
    elif udiff_ub * udiff_lb < 0:
        if zmid - zbot < 0.01:
            rootn += 1
            if rootn > 2:
                raise ValueError("U root number > 2, abort!")
            roots[rootn - 1] = (zmid + zbot) / 2.0
        else:
            roots, rootn = ufindroots(zmid, zbot, (zmid + zbot) / 2.0, utop, htop, hbot, z0mg, alpha, roots, rootn)
    
    return roots, rootn

def fuint(utop, ztop, zbot, htop, hbot, z0mg, alpha, bee, fc):
    # Calculate fulogint
    fulogint = utop / np.log(htop / z0mg) * (
        ztop * np.log(ztop / z0mg) - zbot * np.log(zbot / z0mg) + zbot - ztop
    )

    # Determine which integral to use based on the value of udiff
    if udiff((ztop + zbot) / 2.0, utop, htop, hbot, z0mg, alpha) <= 0:
        # uexp is smaller
        fuexpint = utop * (htop - hbot) / alpha * (
            np.exp(-alpha * (htop - ztop) / (htop - hbot)) - 
            np.exp(-alpha * (htop - zbot) / (htop - hbot))
        )
        fuint = bee * fc * fuexpint + (1.0 - bee * fc) * fulogint
    else:
        # ulog is smaller
        fuint = fulogint

    return fuint

def udiff(z, utop, htop, hbot, z0mg, alpha):
    # Calculate uexp
    uexp = utop * np.exp(-alpha * (htop - z) / (htop - hbot))
    
    # Calculate ulog
    ulog = utop * np.log(z / z0mg) / np.log(htop / z0mg)
    
    # Calculate udiff
    udiff = uexp - ulog
    
    return udiff

def fkint(const_physical, ktop, ztop, zbot, htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc):
    fkcobint = fac * htop / ktop * (np.log(ztop) - np.log(zbot)) + \
               (1.0 - fac) * CoLM_FrictionVelocity.kintmoninobuk(const_physical, 0.0, z0h, obu, ustar, ztop, zbot)
    
    if kdiff(const_physical, (ztop + zbot) / 2.0, ktop, htop, hbot, obu, ustar, fac, alpha) <= 0:
        # kexp is smaller
        if alpha > 0:
            fkexpint = -(htop - hbot) / alpha / ktop * (
                np.exp(alpha * (htop - ztop) / (htop - hbot)) -
                np.exp(alpha * (htop - zbot) / (htop - hbot))
            )
        else:
            fkexpint = (ztop - zbot) / ktop
        
        fkint_value = bee * fc * fkexpint + (1.0 - bee * fc) * fkcobint
    else:
        # kcob is smaller
        fkint_value = fkcobint
    
    return fkint_value

def frd(const_physical, ktop, htop, hbot, ztop, zbot, displah, z0h, obu, ustar, z0mg, alpha, bee, fc):
    # Constants
    com1 = 0.4
    com2 = 0.08

    # Initialize variables
    roots = np.zeros(2)
    kint = 0.0

    # Calculate fac
    fac = 1.0 / (1.0 + np.exp(-(displah - com1) / com2))
    
    # Initialize root count
    rootn = 0

    # Call the kfindroots function
    roots, rootn = kfindroots(const_physical, ztop, zbot, (ztop + zbot) / 2.0, ktop, htop, hbot, obu, ustar, fac, alpha, roots, rootn)
    
    # Integrate based on the number of roots found
    if rootn == 0:
        kint += fkint(const_physical,ktop, ztop, zbot, htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc)
    elif rootn == 1:
        kint += fkint(const_physical,ktop, ztop, roots[0], htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc)
        kint += fkint(const_physical,ktop, roots[0], zbot, htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc)
    elif rootn == 2:
        kint += fkint(const_physical,ktop, ztop, roots[0], htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc)
        kint += fkint(const_physical,ktop, roots[0], roots[1], htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc)
        kint += fkint(const_physical,ktop, roots[1], zbot, htop, hbot, z0h, obu, ustar, fac, alpha, bee, fc)

    return kint

def kdiff(const_physical, z, ktop, htop, hbot, obu, ustar, fac, alpha):
    kexp = ktop * np.exp(-alpha * (htop - z) / (htop - hbot))
    klin = ktop * z / htop
    kcob = 1.0 / (fac / klin + (1.0 - fac) / CoLM_FrictionVelocity.kmoninobuk(const_physical, 0.0, obu, ustar, z))
    
    kdiff_value = kexp - kcob
    return kdiff_value

def kfindroots(const_physical, ztop, zbot, zmid, ktop, htop, hbot, obu, ustar, fac, alpha, roots=None, rootn=None):
    if roots is None:
        roots = np.zeros(2)
    if rootn is None:
        rootn = 0  # Using a list to allow modification within the function
    
    kdiff_ub = kdiff(const_physical, ztop, ktop, htop, hbot, obu, ustar, fac, alpha)
    kdiff_lb = kdiff(const_physical, zmid, ktop, htop, hbot, obu, ustar, fac, alpha)
    
    if kdiff_ub * kdiff_lb == 0:
        if kdiff_lb == 0:
            rootn += 1
            if rootn > 2:
                raise ValueError("K root number > 2, abort!")
            roots[rootn-1] = zmid
    elif kdiff_ub * kdiff_lb < 0:
        if ztop - zmid < 0.01:
            rootn += 1
            if rootn > 2:
                raise ValueError("K root number > 2, abort!")
            roots[rootn -1] = (ztop + zmid) / 2.0
        else:
            roots, rootn = kfindroots(const_physical, ztop, zmid, (ztop + zmid) / 2.0, ktop, htop, hbot, obu, ustar, fac, alpha, roots, rootn)
    
    kdiff_ub = kdiff(const_physical,zmid, ktop, htop, hbot, obu, ustar, fac, alpha)
    kdiff_lb = kdiff(const_physical,zbot, ktop, htop, hbot, obu, ustar, fac, alpha)
    
    if kdiff_ub * kdiff_lb == 0:
        if kdiff_ub == 0:
            rootn += 1
            if rootn > 2:
                raise ValueError("K root number > 2, abort!")
            roots[rootn - 1] = zmid
    elif kdiff_ub * kdiff_lb < 0:
        if zmid - zbot < 0.01:
            rootn += 1
            if rootn > 2:
                raise ValueError("K root number > 2, abort!")
            roots[rootn - 1] = (zmid + zbot) / 2.0
        else:
            roots, rootn = kfindroots(const_physical, zmid, zbot, (zmid + zbot) / 2.0, ktop, htop, hbot, obu, ustar, fac, alpha, roots, rootn)
    
    return roots, rootn


