import numpy as np
import math

def moninobukini(ur, th, thm, thv, dth, dqh, dthv, zldis, z0m, grav):

    """
    Original author  : Qinghliang Li,  Jinlong Zhu, 17/02/2024;
    software         : initialzation of Monin-Obukhov length,
                    ! the scheme is based on the work of Zeng et al. (1998):
                    ! Intercomparison of bulk aerodynamic algorithms for the computation
                    ! of sea surface fluxes using TOGA CORE and TAO data. J. Climate, Vol. 11: 2628-2644
    Args:
        ur (float): Wind speed at reference height [m/s].
        thm (float): Intermediate variable (tm+0.0098*ht).
        th (float): Potential temperature [kelvin].
        thv (float): Virtual potential temperature [kelvin].
        dth (float): Difference of virtual temperature between reference height and surface.
        dthv (float): Difference of virtual potential temperature between reference height and surface.
        dqh (float): Difference of humidity between reference height and surface.
        zldis (float): Reference height minus zero displacement height [m].
        z0m (float): Roughness length, momentum [m].

    Returns:
        um (float): Wind speed including the stability effect [m/s].
        obu (float): Monin-Obukhov length (m).
    """
    # Initial values of u* and convective velocity
    wc = 0.5
    if dthv >= 0.:
        um = max(ur, 0.1)
    else:
        um = (ur ** 2 + wc ** 2) ** 0.5

    rib = grav * zldis * dthv / (thv * um ** 2)

    if rib >= 0.:  # Neutral or stable
        zeta = rib * np.log(zldis / z0m) / (1. - 5. * min(rib, 0.19))
        zeta = min(2., max(zeta, 1.e-6))
    else:  # Unstable
        zeta = rib * np.log(zldis / z0m)
        zeta = max(-100., min(zeta, -1.e-6))

    obu = zldis / zeta

    return um, obu

def psi(k, zeta):
    """
    Stability function for unstable case (rib < 0)
    
    Parameters:
    k (int): An integer parameter
    zeta (float): Dimensionless height used in Monin-Obukhov theory
    
    Returns:
    float: The stability function value
    """
    chik = (1.0 - 16.0 * zeta) ** 0.25
    
    if k == 1:
        psi_val = (2.0 * math.log((1.0 + chik) * 0.5) +
                   math.log((1.0 + chik**2) * 0.5) -
                   2.0 * math.atan(chik) +
                   2.0 * math.atan(1.0))
    else:
        psi_val = 2.0 * math.log((1.0 + chik**2) * 0.5)
    
    return psi_val

# Example usage
k = 1
zeta = -0.5
print(psi(k, zeta))


def moninobuk(const_physical, hu,ht,hq,displa,z0m,z0h,z0q,obu,um):
    # Initial calculations
    zldis = hu - displa
    zeta = zldis / obu
    zetam = 1.574

    # Compute fm and ustar based on zeta
    if zeta < -zetam:  # zeta < -1
        fm = math.log(-zetam * obu / z0m)
        a = - psi(1, -zetam) + psi(1, z0m / obu) + 1.14 * ((-zeta) ** 0.333 - (zetam) ** 0.333)
        ustar = const_physical.vonkar * um / fm
    elif zeta < 0.0:  # -1 <= zeta < 0
        fm = math.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
        ustar = const_physical.vonkar * um / fm
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fm = math.log(zldis / z0m) + 5.0 * zeta - 5.0 * z0m / obu
        ustar = const_physical.vonkar * um / fm
    else:  # 1 < zeta, phi=5+zeta
        fm = math.log(obu / z0m) + 5.0 - 5.0 * z0m / obu + (5.0 * math.log(zeta) + zeta - 1.0)
        ustar = const_physical.vonkar * um / fm

    # For 10 meter wind-velocity
    zldis = 10.0 + z0m
    zeta = zldis / obu

    if zeta < -zetam:  # zeta < -1
        fm10m = math.log(-zetam * obu / z0m) - psi(1, -zetam) + psi(1, z0m / obu) + 1.14 * ((-zeta) ** 0.333 - (zetam) ** 0.333)
    elif zeta < 0.0:  # -1 <= zeta < 0
        fm10m = math.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fm10m = math.log(zldis / z0m) + 5.0 * zeta - 5.0 * z0m / obu
    else:  # 1 < zeta, phi=5+zeta
        fm10m = math.log(obu / z0m) + 5.0 - 5.0 * z0m / obu + (5.0 * math.log(zeta) + zeta - 1.0)
    
    # Temperature profile
    zldis = ht - displa
    zeta = zldis / obu
    zetat=0.465

    if zeta < -zetat:  # zeta < -1
        fh = math.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * ((zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fh = math.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fh = math.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fh = math.log(obu / z0h) + 5.0 - 5.0 * z0h / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # For 2 meter screen temperature
    zldis = 2.0 + z0h
    zeta = zldis / obu

    if zeta < -zetat:  # zeta < -1
        fh2m = math.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * ((zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fh2m = math.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fh2m = math.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fh2m = math.log(obu / z0h) + 5.0 - 5.0 * z0h / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # Humidity profile
    zldis = hq - displa
    zeta = zldis / obu

    if zeta < -zetat:  # zeta < -1
        fq = math.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * ((zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fq = math.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fq = math.log(zldis / z0q) + 5.0 * zeta - 5.0 * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq = math.log(obu / z0q) + 5.0 - 5.0 * z0q / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # For 2 meter screen humidity
    zldis = 2.0 + z0h
    zeta = zldis / obu

    if zeta < -zetat:  # zeta < -1
        fq2m = math.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * ((zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fq2m = math.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fq2m = math.log(zldis / z0q) + 5.0 * zeta - 5.0 * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq2m = math.log(obu / z0q) + 5.0 - 5.0 * z0q / obu + (5.0 * math.log(zeta) + zeta - 1.0)
    return ustar,fh2m,fq2m,fm10m,fm,fh,fq


def moninobukm(const_physical, hu, ht, hq, displa, z0m, z0h, z0q, obu, um,
displat, z0mt, htop):
              # displat, z0mt,ustar, fh2m, fq2m, htop, hmtop, fm, fh, fq, fht,fqt,phih):
    # Initial calculations
    zldis = hu-displa
    zeta = zldis / obu
    zetam = 1.574

    # Compute fm and ustar based on zeta
    if zeta < -zetam:  # zeta < -1
        fm = math.log(-zetam * obu / z0m)- psi(1, -zetam) + psi(1, z0m / obu) + 1.14 * ((-zeta) ** 0.333 - (zetam) ** 0.333)
        ustar = const_physical.vonkar * um / fm
    elif zeta < 0.0:  # -1 <= zeta < 0
        fm = math.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
        ustar = const_physical.vonkar * um / fm
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fm = math.log(zldis / z0m) + 5.0 * zeta - 5.0 * z0m / obu
        ustar = const_physical.vonkar * um / fm
    else:  # 1 < zeta, phi=5+zeta
        fm = math.log(obu / z0m) + 5.0 - 5.0 * z0m / obu + (5.0 * math.log(zeta) + zeta - 1.0)
        ustar = const_physical.vonkar * um / fm

    # For 10 meter wind-velocity
    zldis = htop - displa
    zeta = zldis / obu
    zetam = 1.574

    if zeta < -zetam:  # zeta < -1
        fmtop = math.log(-zetam * obu / z0m) - psi(1, -zetam) + psi(1, z0m / obu) + 1.14 * (
                    (-zeta) ** 0.333 - (zetam) ** 0.333)
    elif zeta < 0.0:  # -1 <= zeta < 0
        fmtop = math.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fmtop = math.log(zldis / z0m) + 5.0 * zeta - 5.0 * z0m / obu
    else:  # 1 < zeta, phi=5+zeta
        fmtop = math.log(obu / z0m) + 5.0 - 5.0 * z0m / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # Temperature profile
    zldis = ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        fh = math.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fh = math.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fh = math.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fh = math.log(obu / z0h) + 5.0 - 5.0 * z0h / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # For 2 meter screen temperature
    zldis = 2.0 + z0h
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        fh2m = math.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fh2m = math.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fh2m = math.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fh2m = math.log(obu / z0h) + 5.0 - 5.0 * z0h / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # Humidity profile
    zldis = displat + z0mt - displa  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        fht = math.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fht = math.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fht = math.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fht = math.log(obu / z0h) + 5.0 - 5.0 * z0h / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # For 2 meter screen humidity
    zldis = htop - displa  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        phih = 0.9 * const_physical.vonkar** (1.333)* (-zeta) ** (-0.333)
    elif zeta < 0.0:  # -1 <= zeta < 0
        phih = (1. - 16.*zeta)**(-0.5)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        phih = 1. + 5.*zeta
    else:  # 1 < zeta, phi=5+zeta
        phih = 5. + zeta

    zldis = hq - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        fq = math.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fq = math.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fq = math.log(zldis / z0q) + 5.0 * zeta - 5.0 * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq = math.log(obu / z0q) + 5.0 - 5.0 * z0q / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # for 2 meter screen humidity
    zldis = 2. + z0h
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        fq2m = math.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fq2m = math.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fq2m = math.log(zldis / z0q) + 5.0 * zeta - 5.0 * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq2m = math.log(obu / z0q) + 5.0 - 5.0 * z0q / obu + (5.0 * math.log(zeta) + zeta - 1.0)

    # for top layer humidity
    zldis = displat + z0mt - displa  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:  # zeta < -1
        fqt = math.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.0:  # -1 <= zeta < 0
        fqt = math.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.0:  # 0 <= zeta <= 1
        fqt = math.log(zldis / z0q) + 5.0 * zeta - 5.0 * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fqt = math.log(obu / z0q) + 5.0 - 5.0 * z0q / obu + (5.0 * math.log(zeta) + zeta - 1.0)


    return ustar, fh2m, fq2m, fmtop, fm, fh, fq, fht, fqt, phih


def kintmoninobuk(const_physical, displa, z0h, obu, ustar, ztop, zbot):
    # zldis is the reference height "minus" zero displacement height
    zldis = ztop - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        # zeta < -1
        fh_top = (np.log(-zetat * obu / z0h) - psi(2, -zetat) +
                  psi(2, z0h / obu) + 0.8 * (zetat**(-0.333) - (-zeta)**(-0.333)))
    elif zeta < 0:
        # -1 <= zeta < 0
        fh_top = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1:
        # 0 <= zeta <= 1
        fh_top = np.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:
        # 1 < zeta, phi=5+zeta
        fh_top = (np.log(obu / z0h) + 5.0 - 5.0 * z0h / obu +
                  (5.0 * np.log(zeta) + zeta - 1.0))

    zldis = zbot - displa
    zeta = zldis / obu

    if zeta < -zetat:
        # zeta < -1
        fh_bot = (np.log(-zetat * obu / z0h) - psi(2, -zetat) +
                  psi(2, z0h / obu) + 0.8 * (zetat**(-0.333) - (-zeta)**(-0.333)))
    elif zeta < 0:
        # -1 <= zeta < 0
        fh_bot = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1:
        # 0 <= zeta <= 1
        fh_bot = np.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu
    else:
        # 1 < zeta, phi=5+zeta
        fh_bot = (np.log(obu / z0h) + 5.0 - 5.0 * z0h / obu +
                  (5.0 * np.log(zeta) + zeta - 1.0))

    kintmoninobuk_value = 1.0 / (const_physical.vonkar / (fh_top - fh_bot) * ustar)
    return kintmoninobuk_value

def kmoninobuk(const_physical, displa, obu, ustar, z):
    # zldis is the reference height "minus" zero displacement height
    if z <= displa:
        return 0.0

    zldis = z - displa  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        # zeta < -1
        phih = 0.9 * const_physical.vonkar**(1.333) * (-zeta)**(-0.333)
    elif zeta < 0:
        # -1 <= zeta < 0
        phih = (1.0 - 16.0 * zeta)**(-0.5)
    elif zeta <= 1:
        # 0 <= zeta <= 1
        phih = 1.0 + 5.0 * zeta
    else:
        # 1 < zeta, phi = 5 + zeta
        phih = 5.0 + zeta

    kmoninobuk_value = const_physical.vonkar * (z - displa) * ustar / phih
    return kmoninobuk_value

