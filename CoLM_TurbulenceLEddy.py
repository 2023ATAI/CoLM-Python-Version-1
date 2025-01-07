import numpy as np


def moninobuk_leddy(hu, ht, hq, displa, z0m, z0h, z0q, obu, um, hpbl, vonkar):

    """
    Original author  : Qinghliang Li,  Jinlong Zhu, 17/02/2024;
    software         Implement the LZD2022 scheme (Liu et al., 2022), which accounts for large
                    ! eddy effects by inlcuding the boundary layer height in the phim FUNCTION,
                    ! to compute friction velocity, relation for potential temperature and
                    ! humidity profiles of surface boundary layer.

                    ! References:
                    ! [1] Zeng et al., 1998: Intercomparison of bulk aerodynamic algorithms
                    ! for the computation of sea surface fluxes using TOGA CORE and TAO data.
                    ! J. Climate, 11: 2628-2644.
                    ! [2] Liu et al., 2022: A surface flux estimation scheme accounting for
                    ! large-eddy effects for land surface modeling. GRL, 49, e2022GL101754.
                    ! Created by Shaofeng Liu, May 5, 2023
    """
    # Local variables
    zldis = hu - displa  # Reference height "minus" zero displacement height [m]
    zeta = zldis / obu  # Dimensionless height used in Monin-Obukhov theory

    # Adjustment factors for unstable (moz < 0) or stable (moz > 0) conditions
    # Wind profile
    zetazi = max(5. * hu, hpbl) / obu
    if zetazi >= 0.:  # Stable
        zetazi = min(200., max(zetazi, 1.e-5))
    else:  # Unstable
        zetazi = max(-1.e4, min(zetazi, -1.e-5))
    Bm = 0.0047 * (-zetazi) + 0.1854
    zetam = 0.5 * Bm ** 4 * (-16. - np.sqrt(256. + 4. / Bm ** 4))
    Bm2 = max(Bm, 0.2722)
    zetam2 = min(zetam, -0.13)

    if zeta < zetam2:  # zeta < zetam2
        fm = np.log(zetam2 * obu / z0m) - psi(1, zetam2) + psi(1, z0m / obu) - 2. * Bm2 * (
                    (-zeta) ** (-0.5) - (-zetam2) ** (-0.5))
        ustar = vonkar * um / fm
    elif zeta < 0.:  # zetam2 <= zeta < 0
        fm = np.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
        ustar = vonkar * um / fm
    elif zeta <= 1.:  # 0 <= zeta <= 1
        fm = np.log(zldis / z0m) + 5. * zeta - 5. * z0m / obu
        ustar = vonkar * um / fm
    else:  # 1 < zeta, phi=5+zeta
        fm = np.log(obu / z0m) + 5. - 5. * z0m / obu + (5. * np.log(zeta) + zeta - 1.)
        ustar = vonkar * um / fm

    # For 10 meter wind-velocity
    zldis = 10. + z0m
    zeta = zldis / obu
    if zeta < zetam2:  # zeta < zetam2
        fm10m = np.log(zetam2 * obu / z0m) - psi(1, zetam2) + psi(1, z0m / obu) - 2. * Bm2 * (
                    (-zeta) ** (-0.5) - (-zetam2) ** (-0.5))
    elif zeta < 0.:  # zetam2 <= zeta < 0
        fm10m = np.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
    elif zeta <= 1.:  # 0 <= zeta <= 1
        fm10m = np.log(zldis / z0m) + 5. * zeta - 5. * z0m / obu
    else:  # 1 < zeta, phi=5+zeta
        fm10m = np.log(obu / z0m) + 5. - 5. * z0m / obu + (5. * np.log(zeta) + zeta - 1.)

    # Temperature profile
    zldis = ht - displa
    zeta = zldis / obu
    zetat = 0.465
    if zeta < -zetat:  # zeta < -1
        fh = np.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.:  # -1 <= zeta < 0
        fh = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.:  # 0 <= zeta <= 1
        fh = np.log(zldis / z0h) + 5. * zeta - 5. * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fh = np.log(obu / z0h) + 5. - 5. * z0h / obu + (5. * np.log(zeta) + zeta - 1.)

    # For 2 meter screen temperature
    zldis = 2. + z0h  # ht-displa
    zeta = zldis / obu
    zetat = 0.465
    if zeta < -zetat:  # zeta < -1
        fh2m = np.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.:  # -1 <= zeta < 0
        fh2m = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1.:  # 0 <= zeta <= 1
        fh2m = np.log(zldis / z0h) + 5. * zeta - 5. * z0h / obu
    else:  # 1 < zeta, phi=5+zeta
        fh2m = np.log(obu / z0h) + 5. - 5. * z0h / obu + (5. * np.log(zeta) + zeta - 1.)

    # Humidity profile
    zldis = hq - displa
    zeta = zldis / obu
    zetat = 0.465
    if zeta < -zetat:  # zeta < -1
        fq = np.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.:  # -1 <= zeta < 0
        fq = np.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.:  # 0 <= zeta <= 1
        fq = np.log(zldis / z0q) + 5. * zeta - 5. * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq = np.log(obu / z0q) + 5. - 5. * z0q / obu + (5. * np.log(zeta) + zeta - 1.)

    # For 2 meter screen humidity
    zldis = 2. + z0h
    zeta = zldis / obu
    zetat = 0.465
    if zeta < -zetat:  # zeta < -1
        fq2m = np.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0.:  # -1 <= zeta < 0
        fq2m = np.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1.:  # 0 <= zeta <= 1
        fq2m = np.log(zldis / z0q) + 5. * zeta - 5. * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq2m = np.log(obu / z0q) + 5. - 5. * z0q / obu + (5. * np.log(zeta) + zeta - 1.)

    return ustar, fh2m, fq2m, fm10m, fm, fh, fq


def moninobukm_leddy(hu, ht, hq, displa, z0m, z0h, z0q, obu, um, displat, z0mt, hpbl,htop):
    # Constants
    vonkar = 0.4  # Von Karman constant

    # Local variables
    zldis = hu - displa
    zeta = zldis / obu

    # Wind profile
    zetazi = max(5 * hu, hpbl) / obu
    if zetazi >= 0:
        zetazi = min(200, max(zetazi, 1e-5))
    else:
        zetazi = max(-1e4, min(zetazi, -1e-5))

    Bm = 0.0047 * (-zetazi) + 0.1854
    zetam = 0.5 * Bm ** 4 * (-16. - np.sqrt(256. + 4. / Bm ** 4))
    Bm2 = max(Bm, 0.2722)
    zetam2 = min(zetam, -0.13)

    if zeta < zetam2:
        fm = np.log(zetam2 * obu / z0m) - psi(1, zetam2) + psi(1, z0m / obu) - 2. * Bm2 * (
                    (-zeta) ** (-0.5) - (-zetam2) ** (-0.5))
        ustar = vonkar * um / fm
    elif zeta < 0:
        fm = np.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
        ustar = vonkar * um / fm
    elif zeta <= 1:
        fm = np.log(zldis / z0m) + 5. * zeta - 5. * z0m / obu
        ustar = vonkar * um / fm
    else:
        fm = np.log(obu / z0m) + 5. - 5. * z0m / obu + (5. * np.log(zeta) + zeta - 1.)
        ustar = vonkar * um / fm

    # For canopy top wind-velocity
    zldis = ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < zetam2:
        fmtop = np.log(zetam2 * obu / z0m) - psi(1, zetam2) + psi(1, z0m / obu) - 2. * Bm2 * (
                    (-zeta) ** (-0.5) - (-zetam2) ** (-0.5))
    elif zeta < 0:
        fmtop = np.log(zldis / z0m) - psi(1, zeta) + psi(1, z0m / obu)
    elif zeta <= 1:
        fmtop = np.log(zldis / z0m) + 5. * zeta - 5. * z0m / obu
    else:
        fmtop = np.log(obu / z0m) + 5. - 5. * z0m / obu + (5. * np.log(zeta) + zeta - 1.)

    # Temperature profile
    zldis = ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        fh = np.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0:
        fh = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1:
        fh = np.log(zldis / z0h) + 5. * zeta - 5. * z0h / obu
    else:
        fh = np.log(obu / z0h) + 5. - 5. * z0h / obu + (5. * np.log(zeta) + zeta - 1.)

    # For 2-meter screen temperature
    zldis = 2. + z0h  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        fh2m = np.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0:
        fh2m = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1:
        fh2m = np.log(zldis / z0h) + 5. * zeta - 5. * z0h / obu
    else:
        fh2m = np.log(obu / z0h) + 5. - 5. * z0h / obu + (5. * np.log(zeta) + zeta - 1.)

    # For top layer temperature
    zldis = displat + z0mt - displa  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        fht = np.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0:
        fht = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1:
        fht = np.log(zldis / z0h) + 5. * zeta - 5. * z0h / obu
    else:
        fht = np.log(obu / z0h) + 5. - 5. * z0h / obu + (5. * np.log(zeta) + zeta - 1.)

    # For canopy top phi(h)
    # CESM TECH NOTE EQ. (5.31)
    zldis = htop - displa  # ht - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        phih = 0.9 * vonkar ** (1.333) * (-zeta) ** (-0.333)
    elif zeta < 0:
        phih = (1. - 16. * zeta) ** (-0.5)
    elif zeta <= 1:
        phih = 1. + 5. * zeta
    else:
        phih = 5. + zeta

    # Humidity profile
    zldis = hq - displa
    zeta = zldis / obu
    zetat = 0.465

    if zeta < -zetat:
        fq = np.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (
                    (zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0:
        fq = np.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1:
        fq = np.log(zldis / z0q) + 5. * zeta - 5. * z0q / obu
    else:
        fq = np.log(obu / z0q) + 5. - 5. * z0q / obu + (5. * np.log(zeta) + zeta - 1.)

    # for 2 meter screen humidity
    zldis = 2. + z0h
    zeta = zldis / obu
    zetat = 0.465
    if (zeta < -zetat):  # zeta < -1
        fq2m = np.log(-zetat * obu / z0q) - psi(2, -zetat) + \
               psi(2, z0q / obu) + 0.8 * ((zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif (zeta < 0.):  # -1 <= zeta < 0
        fq2m = np.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif (zeta <= 1.):  # 0 <= zeta <= 1
        fq2m = np.log(zldis / z0q) + 5. * zeta - 5. * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fq2m = np.log(obu / z0q) + 5. - 5. * z0q / obu + (5. * np.log(zeta) + zeta - 1.)

    # for top layer humidity
    zldis = displat + z0mt - displa  # ht-displa
    zeta = zldis / obu
    zetat = 0.465
    if (zeta < -zetat):  # zeta < -1
        fqt = np.log(-zetat * obu / z0q) - psi(2, -zetat) + \
              psi(2, z0q / obu) + 0.8 * ((zetat) ** (-0.333) - (-zeta) ** (-0.333))
    elif (zeta < 0.):  # -1 <= zeta < 0
        fqt = np.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif (zeta <= 1.):  # 0 <= zeta <= 1
        fqt = np.log(zldis / z0q) + 5. * zeta - 5. * z0q / obu
    else:  # 1 < zeta, phi=5+zeta
        fqt = np.log(obu / z0q) + 5. - 5. * z0q / obu + (5. * np.log(zeta) + zeta - 1.)

    return ustar, fmtop, fh



def psi(k, zeta):
    #  stability FUNCTION for unstable CASE (rib < 0)
    # Function psi for stability cases
    chik = (1. - 16. * zeta) ** 0.25
    if k == 1:
        return 2. * np.log((1. + chik) * 0.5) + np.log((1. + chik ** 2) * 0.5) - 2. * np.arctan(chik) + 2. * np.arctan(1.)
    else:
        return 2. * np.log((1. + chik ** 2) * 0.5)