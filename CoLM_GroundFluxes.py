import numpy as np
import CoLM_FrictionVelocity
import CoLM_TurbulenceLEddy

def GroundFluxes(const_physical, zlnd, zsno, hu, ht, hq, hpbl, us, vs, tm, qm, rhoair, psrf,
                 ur, thm, th, thv, t_grnd, qg, rss, dqgdT, htvp,
                 fsno, cgrnd, cgrndl, cgrnds,
                 t_soil, t_snow, q_soil, q_snow,
                 taux, tauy, fseng, fseng_soil, fseng_snow,
                 fevpg, fevpg_soil, fevpg_snow, tref, qref,
                 z0m, z0hg, zol, rib, ustar, qstar, tstar, fm, fh, fq,
                 co_lm, cpair, vonkar, grav):

    # ----------------------- Dummy argument --------------------------------
    # initial roughness length
    z0mg = (1.0 - fsno) * zlnd + fsno * zsno
    z0hg = z0mg
    z0qg = z0mg
    # potential temperatur at the reference height
    beta = 1.0
    zii = 1000.0
    z0m = z0mg
    # --------------------------------------------------------------------------------------------------------------
    # Compute sensible and latent fluxes and their derivatives with respect
    # to ground temperature using ground temperatures from previous time step.
    # --------------------------------------------------------------------------------------------------------------
    # Initialization variables
    nmozsgn = 0
    obuold = 0.0

    dth = thm - t_grnd
    dqh = qm - qg
    dthv = dth * (1.0 + 0.61 * qm) + 0.61 * th * dqh
    zldis = hu - 0.0
    fh2m = 0.0       #! relation for temperature at 2m
    fq2m = 0.0       #! relation for specific humidity at 2m
    fm10m = 0.0      #! integral of profile FUNCTION for momentum at 10m
    um = 0.0
    obu = 0.0
    zeta = 0.0

    um, obu = CoLM_FrictionVelocity.moninobukini(ur, th, thm, thv, dth, dqh, dthv, zldis, z0mg, grav)
    # Evaluated stability-dependent variables using moz from prior iteration
    niters = 6

    # begin stability iteration
    for iter in range(niters):
        displax = 0.
        if co_lm['DEF_USE_CBL_HEIGHT']:
            ustar, fh2m, fq2m, fm10m, fm, fh, fq = CoLM_TurbulenceLEddy.moninobuk_leddy(hu, ht, hq, displax, z0mg, z0hg, z0qg, obu, um, hpbl, ustar, fm, fh, fq)
        else:
            ustar,fh2m,fq2m,fm10m,fm,fh,fq = CoLM_FrictionVelocity.moninobuk(const_physical, hu, ht, hq, displax, z0mg, z0hg, z0qg, obu, um)

        tstar = vonkar / fh * dth
        qstar = vonkar / fq * dqh

        z0hg = z0mg / np.exp(0.13 * (ustar * z0mg / 1.5e-5) ** 0.45)
        z0qg = z0hg

        # 2023.04.06, weinan
        thvstar = tstar * (1.0 + 0.61 * qm) + 0.61 * th * qstar
        zeta = zldis * vonkar * grav * thvstar / (ustar ** 2 * thv)
        if zeta >= 0.0:     # stable
            zeta = min(2.0, max(zeta, 1e-6))
        else:               # unstable
            zeta = max(-100.0, min(zeta, -1e-6))
        obu = zldis / zeta

        if zeta >= 0.0:
            um = max(ur, 0.1)
        else:
            if co_lm['DEF_USE_CBL_HEIGHT']:
                zii = max(5.0 * hu, hpbl)
            wc = (-grav * ustar * thvstar * zii / thv) ** (1.0 / 3.0)
            wc2 = beta * beta * (wc ** 2)
            um = np.sqrt(ur ** 2 + wc2)

        if obuold * obu < 0.0:
            nmozsgn += 1
            if nmozsgn >= 4:
                break
        obuold = obu

    # Get derivative of fluxes with repect to ground temperature
    ram = 1.0 / (ustar ** 2 / um)
    rah = 1.0 / (vonkar / fh * ustar)
    raw = 1.0 / (vonkar / fq * ustar)

    # 08/23/2019, yuan:
    raih = rhoair * cpair / rah

    if dqh < 0.0:
        raiw = rhoair / raw  #dew case. no soil resistance
    else:
        if co_lm['DEF_RSS_SCHEME'] == 4:
            raiw = rss * rhoair / raw
        else:
            raiw = rhoair / (raw + rss)

    cgrnds = raih
    cgrndl = raiw * dqgdT
    cgrnd = cgrnds + htvp * cgrndl

    zol = zeta
    rib = min(5.0, zol * ustar ** 2 / (vonkar ** 2 / fh * um ** 2))

    # surface fluxes of momentum, sensible and latent
    # using ground temperatures from previous time step
    taux = -rhoair * us / ram
    tauy = -rhoair * vs / ram
    fseng = -raih * dth
    fevpg = -raiw * dqh

    fseng_soil = -raih * (thm - t_soil)
    fseng_snow = -raih * (thm - t_snow)
    fevpg_soil = -raiw * (qm - q_soil)
    fevpg_snow = -raiw * (qm - q_snow)

    # 2 m height air temperature
    tref = thm + vonkar / fh * dth * (fh2m / vonkar - fh / vonkar)
    qref = qm + vonkar / fq * dqh * (fq2m / vonkar - fq / vonkar)

    return taux, tauy, fseng, fseng_soil, fseng_snow, fevpg, fevpg_soil, fevpg_snow, cgrnd, cgrndl, cgrnds, tref, qref, z0m, z0hg, zol, rib, ustar, tstar, qstar, fm, fh, fq



