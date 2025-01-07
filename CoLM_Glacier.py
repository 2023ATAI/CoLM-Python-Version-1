import numpy as np
import CoLM_Qsadv
import CoLM_FrictionVelocity
import CoLM_TurbulenceLEddy
import CoLM_Utils
import CoLM_PhaseChange
import CoLM_SnowLayersCombineDivide
import CoLM_SoilSnowHydrology

def groundtem_glacier(nl_colm, const_physical, patchtype, lb, nl_ice, deltim, capr, cnfac, dz_icesno,
                      z_icesno, zi_icesno, t_icesno, wice_icesno, wliq_icesno,
                      scv, snowdp, forc_frl, sabg, sabg_snow_lyr, fseng,
                      fevpg, cgrnd, htvp, emg, imelt, snofrz, sm, xmf, fact,
                      pg_rain, pg_snow, t_precip):
    cv = np.zeros(nl_ice-lb) #heat capacity [J/(m2 K)]
    thk= np.zeros(nl_ice-lb) #thermal conductivity of layer
    tk= np.zeros(nl_ice-lb) #thermal conductivity [W/(m K)]
    at= np.zeros(nl_ice-lb) #"a" vector for tridiagonal matrix
    bt= np.zeros(nl_ice-lb) #"b" vector for tridiagonal matrix
    ct= np.zeros(nl_ice-lb) #"c" vector for tridiagonal matrix
    rt= np.zeros(nl_ice-lb) #"r" vector for tridiagonal solution
    fn= np.zeros(nl_ice-lb) #heat diffusion through the layer interface [W/m2]
    fn1= np.zeros(nl_ice-lb) #heat diffusion through the layer interface [W/m2]
    t_icesno_bef = np.zeros(nl_ice-lb)#snow/ice temperature before update
    brr = np.zeros(nl_ice-lb)#temporay set

    porsl = np.zeros(nl_ice) 
    psi0 = np.zeros(nl_ice) 
    bsw = np.zeros(nl_ice) 
    theta_r = np.zeros(nl_ice) 
    alpha_vgm = np.zeros(nl_ice) 
    n_vgm = np.zeros(nl_ice) 
    L_vgm = np.zeros(nl_ice) 
    sc_vgm = np.zeros(nl_ice) 
    fc_vgm = np.zeros(nl_ice) 

    # SNOW and LAND ICE heat capacity
    cv[:] = wice_icesno[:] * const_physical.cpice + wliq_icesno[:] * const_physical.cpliq
    if lb == 1 and scv > 0:
        cv[0] += const_physical.cpice * scv

    if lb <= 0:
        cv[lb:] = const_physical.cpliq * wliq_icesno[lb:] + const_physical.cpice * wice_icesno[lb:]

    # SNOW and LAND ICE thermal conductivity [W/(m K)]
    for j in range(lb, nl_ice + 1):
        thk[j] = const_physical.tkwat
        if t_icesno[j] <= const_physical.tfrz:
            thk[j] = 9.828 * np.exp(-0.0057 * t_icesno[j])

    if lb < 1:
        for j in range(lb, 1):
            rhosnow = (wice_icesno[j] + wliq_icesno[j]) / dz_icesno[j]

            # presently option [1] is the default option
            # [1] Jordan (1991) pp. 18
            thk[j] = const_physical.tkair + (7.75e-5 * rhosnow + 1.105e-6 * rhosnow * rhosnow) * (const_physical.tkice - const_physical.tkair)

            # [2] Sturm et al (1997)
            # thk[j] = 0.0138 + 1.01e-3 * rhosnow + 3.233e-6 * rhosnow**2
            # [3] Ostin and Andersson presented in Sturm et al., (1997)
            # thk[j] = -0.871e-2 + 0.439e-3 * rhosnow + 1.05e-6 * rhosnow**2
            # [4] Jansson(1901) presented in Sturm et al. (1997)
            # thk[j] = 0.0293 + 0.7953e-3 * rhosnow + 1.512e-12 * rhosnow**2
            # [5] Douville et al., (1995)
            # thk[j] = 2.2 * (rhosnow / const_physical.denice)**1.88
            # [6] van Dusen (1992) presented in Sturm et al. (1997)
            # thk[j] = 0.021 + 0.42e-3 * rhosnow + 0.22e-6 * rhosnow**2
    # Thermal conductivity at the layer interface
    for j in range(lb-1, nl_ice):
        # the following consideration is try to avoid the snow conductivity
        # to be dominant in the thermal conductivity of the interface.
        # Because when the distance of bottom snow node to the interface
        # is larger than that of interface to top ice node,
        # the snow thermal conductivity will be dominant, and the result is that
        # less heat transfer between snow and ice
        if j == 0 and (z_icesno[j + 1] - zi_icesno[j] < zi_icesno[j] - z_icesno[j]):
            tk[j] = 2.0 * thk[j] * thk[j + 1] / (thk[j] + thk[j + 1])
            tk[j] = max(0.5 * thk[j + 1], tk[j])
        else:
            tk[j] = thk[j] * thk[j + 1] * (z_icesno[j + 1] - z_icesno[j]) \
                    / (thk[j] * (z_icesno[j + 1] - zi_icesno[j]) + thk[j + 1] * (zi_icesno[j] - z_icesno[j]))

    # Set the thermal conductivity at the top ice layer interface to 0
    tk[nl_ice] = 0.0

    # net ground heat flux into the surface and its temperature derivative
    if nl_colm['DEF_USE_SNICAR']:
        hs = sabg_snow_lyr[lb] + emg * forc_frl - emg * const_physical.stefnc * t_icesno[lb] ** 4 - (fseng + fevpg * htvp) + \
             const_physical.cpliq * pg_rain * (t_precip - t_icesno[lb]) + const_physical.cpice * pg_snow * (t_precip - t_icesno[lb])
    else:
        hs = sabg + emg * forc_frl - emg * const_physical.stefnc * t_icesno[lb] ** 4 - (fseng + fevpg * htvp) + \
             const_physical.cpliq * pg_rain * (t_precip - t_icesno[lb]) + const_physical.cpice * pg_snow * (t_precip - t_icesno[lb])

    dhsdT = -cgrnd - 4.0 * emg * const_physical.stefnc * t_icesno[lb] ** 3 - const_physical.cpliq * pg_rain - const_physical.cpice * pg_snow
    t_icesno_bef[lb:] = t_icesno[lb:]

    j = lb
    fact[j] = deltim / cv[j] * dz_icesno[j] / (
                0.5 * (z_icesno[j] - zi_icesno[j - 1] + capr * (z_icesno[j + 1] - zi_icesno[j - 1])))

    for j in range(lb + 1, nl_ice + 1):
        fact[j] = deltim / cv[j]

    for j in range(lb, nl_ice):
        fn[j] = tk[j] * (t_icesno[j + 1] - t_icesno[j]) / (z_icesno[j + 1] - z_icesno[j])
    fn[nl_ice] = 0.0

    # set up vector r and vectors a, b, c that define tridiagonal matrix
    j = lb
    dzp = z_icesno[j + 1] - z_icesno[j]
    at[j] = 0.0
    bt[j] = 1 + (1 - cnfac) * fact[j] * tk[j] / dzp - fact[j] * dhsdT
    ct[j] = -(1 - cnfac) * fact[j] * tk[j] / dzp
    rt[j] = t_icesno[j] + fact[j] * (hs - dhsdT * t_icesno[j] + cnfac * fn[j])

    # Hua Yuan, January 12, 2023
    if lb <= 0:
        for j in range(lb + 1, 2):
            dzm = z_icesno[j] - z_icesno[j - 1]
            dzp = z_icesno[j + 1] - z_icesno[j]
            at[j] = -(1 - cnfac) * fact[j] * tk[j - 1] / dzm
            bt[j] = 1 + (1 - cnfac) * fact[j] * (tk[j] / dzp + tk[j - 1] / dzm)
            ct[j] = -(1 - cnfac) * fact[j] * tk[j] / dzp
            rt[j] = t_icesno[j] + fact[j] * sabg_snow_lyr[j] + cnfac * fact[j] * (fn[j] - fn[j - 1])

    for j in range(2, nl_ice):
        dzm = z_icesno[j] - z_icesno[j - 1]
        dzp = z_icesno[j + 1] - z_icesno[j]
        at[j] = -(1 - cnfac) * fact[j] * tk[j - 1] / dzm
        bt[j] = 1 + (1 - cnfac) * fact[j] * (tk[j] / dzp + tk[j - 1] / dzm)
        ct[j] = -(1 - cnfac) * fact[j] * tk[j] / dzp
        rt[j] = t_icesno[j] + cnfac * fact[j] * (fn[j] - fn[j - 1])

    j = nl_ice
    dzm = z_icesno[j] - z_icesno[j - 1]
    at[j] = -(1 - cnfac) * fact[j] * tk[j - 1] / dzm
    bt[j] = 1 + (1 - cnfac) * fact[j] * tk[j - 1] / dzm
    ct[j] = 0
    rt[j] = t_icesno[j] - cnfac * fact[j] * fn[j - 1]

    i= len[at]

    t_icesno  = CoLM_Utils.tridia(i, at, bt, ct, rt, t_icesno)

    # ------------------------------------------------------------------------------------
    # melting or freezing
    # ------------------------------------------------------------------------------------
    for j in range(lb, nl_ice):
        fn1[j] = tk[j] * (t_icesno[j + 1] - t_icesno[j]) / (z_icesno[j + 1] - z_icesno[j])
    fn1[nl_ice] = 0

    j = lb
    brr[j] = cnfac * fn[j] + (1. - cnfac) * fn1[j]

    for j in range(lb + 1, nl_ice + 1):
        brr[j] = cnfac * (fn[j] - fn[j - 1]) + (1. - cnfac) * (fn1[j] - fn1[j - 1])

    if nl_colm['DEF_USE_SNICAR']:
        pass
    else:
        t_icesno[lb:], wice_icesno[lb:], wliq_icesno[lb:], scv, snowdp, sm, xmf, imelt = CoLM_PhaseChange.meltf(patchtype, lb, nl_ice, deltim, fact[lb:], brr[lb:], hs, hs, hs, 1.0, dhsdT,
              t_icesno_bef[lb:], t_icesno[lb:], wliq_icesno[lb:], wice_icesno[lb:], imelt[lb:],
              scv, snowdp, sm, xmf, porsl, psi0,
              # ifdef Campbell_SOIL_MODEL
              bsw,
              # endif
              # ifdef vanGenuchten_Mualem_SOIL_MODEL
              theta_r, alpha_vgm, n_vgm, L_vgm, sc_vgm, fc_vgm,
              # endif
              dz_icesno[0:])

def groundfluxes_glacier(nl_colm, const_physical, zlnd, zsno, hu, ht, hq,
                         us, vs, tm, qm, rhoair, psrf,
                         ur, thm, th, thv, t_grnd, qg, dqgdT, htvp,
                         hpbl,
                         fsno, cgrnd, cgrndl, cgrnds,
                         taux, tauy, fsena, fevpa, fseng, fevpg, tref, qref,
                         z0m, zol, rib, ustar, tstar, qstar, fm, fh, fq):
    # Initialize roughness length
    if fsno > 0:
        z0mg = 0.002  # Table 1 of Brock et al., (2006)
    else:
        z0mg = 0.001  # Table 1 of Brock et al., (2006)

    z0hg = z0mg
    z0qg = z0mg

    # Potential temperature at the reference height
    beta = 1.0  # (in computing W_*)
    zii = 1000.0  # m (pbl height)
    z0m = z0mg

    # ------------------------------------------------------------------------------------
    # Compute sensible and latent fluxes and their derivatives with respect
    # !     to ground temperature using ground temperatures from previous time step.
    # ------------------------------------------------------------------------------------
    # Initialization variables
    nmozsgn = 0  # Number of times Moz changes sign
    obuold = 0.0  # Monin-Obukhov length from previous iteration

    dth = thm - t_grnd  # Difference in potential temperature between reference height and ground
    dqh = qm - qg  # Difference in specific humidity between reference height and ground
    dthv = dth * (1.0 + 0.61 * qm) + 0.61 * th * dqh  # Difference in virtual potential temperature
    zldis = hu - 0.0  # Reference height minus zero displacement height

    # Call Monin-Obukhov Initialization
    um, obu = CoLM_FrictionVelocity.moninobukini(ur, th, thm, thv, dth, dqh, dthv, zldis, z0mg)

    # Evaluate stability-dependent variables using Moz from prior iteration
    niters = 6  # Maximum number of iterations

    # Iteration loop for stability
    for iter in range(1, niters + 1):
        displax = 0.0
        if nl_colm['DEF_USE_CBL_HEIGHT']:
            ustar, fh2m, fq2m, fm10m, fm, fh, fq = CoLM_TurbulenceLEddy.moninobuk_leddy(hu, ht, hq, displax, z0mg, z0hg, z0qg, obu, um, hpbl, ustar, fh2m, fq2m, fm10m, fm, fh, fq)
        else:
            ustar,fh2m,fq2m,fm10m,fm,fh,fq = CoLM_FrictionVelocity.moninobuk(hu, ht, hq, displax, z0mg, z0hg, z0qg, obu, um, ustar, fh2m, fq2m, fm10m, fm, fh, fq, const_physical.vonkar)

        tstar = const_physical.vonkar / fh * dth
        qstar = const_physical.vonkar / fq * dqh

        z0hg = z0mg / np.exp(0.13 * (ustar * z0mg / 1.5e-5) ** 0.45)
        z0qg = z0hg

        thvstar = tstar * (1.0 + 0.61 * qm) + 0.61 * th * qstar
        zeta = zldis * const_physical.vonkar * const_physical.grav * thvstar / (ustar ** 2 * thv)

        # Adjust zeta based on stability
        if zeta >= 0.0:
            zeta = min(2.0, max(zeta, 1e-6))  # stable
        else:
            zeta = max(-100.0, min(zeta, -1e-6))  # unstable

        obu = zldis / zeta

        # Calculate um based on stability
        if zeta >= 0.0:
            um = max(ur, 0.1)
        else:
            if nl_colm['DEF_USE_CBL_HEIGHT']:
                zii = max(5.0 * hu, hpbl)
            wc = (-const_physical.grav * ustar * thvstar * zii / thv) ** (1.0 / 3.0)
            wc2 = beta * beta * (wc * wc)
            um = np.sqrt(ur * ur + wc2)

        # Check for sign change in obu
        if obuold * obu < 0.0:
            nmozsgn += 1
        if nmozsgn >= 4:
            break  # Exit loop if sign change occurs more than 4 times

        obuold = obu

    # Calculate aerodynamic resistances
    ram = 1.0 / (ustar * ustar / um)
    rah = 1.0 / (const_physical.vonkar / fh * ustar)
    raw = 1.0 / (const_physical.vonkar / fq * ustar)

    # Calculate surface flux derivatives with respect to ground temperature
    raih = rhoair * const_physical.cpair / rah
    raiw = rhoair / raw
    cgrnds = raih
    cgrndl = raiw * dqgdT
    cgrnd = cgrnds + htvp * cgrndl

    # Calculate dimensionless height and bulk Richardson number
    zol = zeta
    rib = min(5.0, zol * ustar ** 2 / (const_physical.vonkar ** 2 / fh * um ** 2))

    # Calculate surface fluxes of momentum, sensible and latent heat
    taux = -rhoair * us / ram
    tauy = -rhoair * vs / ram
    fseng = -raih * dth
    fevpg = -raiw * dqh

    fsena = fseng
    fevpa = fevpg

    # Calculate 2 m height air temperature
    tref = thm + const_physical.vonkar / fh * dth * (fh2m / const_physical.vonkar - fh / const_physical.vonkar)
    qref = qm + const_physical.vonkar / fq * dqh * (fq2m / const_physical.vonkar - fq / const_physical.vonkar)
    return  taux, tauy, fsena, fevpa, fseng, fevpg, cgrnd, cgrndl, cgrnds,  tref, qref,  z0m, zol, rib, ustar, tstar, qstar, fm, fh, fq

def GLACIER_TEMP(nl_colm, const_physical, patchtype, lb, nl_ice, deltim, zlnd, zsno, capr, cnfac,
                 forc_hgt_u, forc_hgt_t, forc_hgt_q, forc_us, forc_vs, forc_t, forc_q,
                 forc_hpbl, forc_rhoair, forc_psrf, coszen, sabg, forc_frl, fsno,
                 dz_icesno, z_icesno, zi_icesno, t_icesno, wice_icesno, wliq_icesno,
                 scv, snowdp, imelt, taux, tauy, fsena, fevpa, lfevpa, fseng, fevpg,
                 olrg, fgrnd, qseva, qsdew, qsubl, qfros, sm, tref, qref, trad, errore,
                 emis, z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq, pg_rain, pg_snow,
                 t_precip, snofrz, sabg_snow_lyr):
    """
        This is the main function to execute the calculation of thermal processes and surface fluxes of the land ice (glacier and ice sheet).

        Original authors: Yongjiu Dai and Nan Wei, September 2014
        Modified by Nan Wei, July 2017, to account for the interaction between precipitation and land ice.

        The function takes various atmospheric and land ice-related parameters as inputs and computes the surface fluxes, ground temperature, and other related variables.
        """
    
        
    # ------------------------------------------------------------------------------------
    # [1] Initial set and propositional variables
    # ------------------------------------------------------------------------------------

    # temperature and water mass from previous time step
    cgrnd = 0
    cgrndl = 0
    cgrnds = 0
    xmf = 0
    t_icesno_bef = np.zeros(nl_ice-lb)
    fact = np.zeros(nl_ice-lb)
    t_grnd = t_icesno[lb]
    t_icesno_bef[lb:] = t_icesno[lb:]

    # emissivity
    emg = 0.97

    # latent heat, assumed that the sublimation occurred only as wliq_icesno=0
    htvp = const_physical.hvap
    if wliq_icesno[lb] <= 0 and wice_icesno[lb] > 0:
        htvp = const_physical.hsub

    # potential temperature at the reference height
    thm = forc_t + 0.0098 * forc_hgt_t  # intermediate variable equivalent to
    # forc_t*(pgcm/forc_psrf)**(const_physical.rgas/const_physical.cpair)
    th = forc_t * (100000. / forc_psrf) ** (const_physical.rgas / const_physical.cpair)  # potential T
    thv = th * (1. + 0.61 * forc_q)  # virtual potential T
    ur = max(0.1, np.sqrt(forc_us * forc_us + forc_vs * forc_vs))  # limit set to 0.1

    # ------------------------------------------------------------------------------------
    # [2] specific humidity and its derivative at ground surface
    # ------------------------------------------------------------------------------------

    qred = 1.0
    eg, degdT, qsatg, qsatgdT = CoLM_Qsadv.qsadv(t_grnd, forc_psrf)
    qg = qred * qsatg
    dqgdT = qred * qsatgdT

    # ------------------------------------------------------------------------------------
    # [3] Compute sensible and latent fluxes and their derivatives with respect
    # !     to ground temperature using ground temperatures from previous time step.
    # ------------------------------------------------------------------------------------

    groundfluxes_glacier(nl_colm, const_physical, zlnd, zsno, forc_hgt_u, forc_hgt_t, forc_hgt_q,
                         forc_us, forc_vs, forc_t, forc_q, forc_rhoair, forc_psrf,
                         ur, thm, th, thv, t_grnd, qg, dqgdT, htvp,
                         forc_hpbl,
                         fsno, cgrnd, cgrndl, cgrnds,
                         z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq)

    # ------------------------------------------------------------------------------------
    # [4] Gound temperature
    # ------------------------------------------------------------------------------------

    groundtem_glacier(patchtype, lb, nl_ice, deltim,
                      capr, cnfac, dz_icesno, z_icesno, zi_icesno,
                      t_icesno, wice_icesno, wliq_icesno, scv, snowdp,
                      forc_frl, sabg, sabg_snow_lyr, fseng, fevpg, cgrnd, htvp, emg,
                      imelt, snofrz, sm, xmf, fact, pg_rain, pg_snow, t_precip)

    # ------------------------------------------------------------------------------------
    # [5] Correct fluxes to present ice temperature
    # ------------------------------------------------------------------------------------

    t_grnd = t_icesno[lb]
    tinc = t_icesno[lb] - t_icesno_bef[lb]
    fseng += tinc * cgrnds
    fevpg += tinc * cgrndl

    # calculation of evaporative potential; flux in kg m-2 s-1.
    # egidif holds the excess energy IF all water is evaporated
    # during the timestep. this energy is later added to the sensible heat flux.
    egsmax = (wice_icesno[lb] + wliq_icesno[lb]) / deltim
    egidif = max(0., fevpg - egsmax)
    fevpg = min(fevpg, egsmax)
    fseng += htvp * egidif

    # total fluxes to atmosphere
    fsena = fseng
    fevpa = fevpg
    lfevpa = htvp * fevpg  # W/m^2 (accounting for sublimation)

    qseva = 0.
    qsubl = 0.
    qfros = 0.
    qsdew = 0.

    if fevpg >= 0:
        qseva = min(wliq_icesno[lb] / deltim, fevpg)
        qsubl = fevpg - qseva
    else:
        if t_grnd < const_physical.tfrz:
            qfros = abs(fevpg)
        else:
            qsdew = abs(fevpg)

    # ground heat flux
    fgrnd = sabg + emg * forc_frl \
            - emg * const_physical.stefnc * t_icesno_bef[lb] ** 3 * (t_icesno_bef[lb] + 4. * tinc) \
            - (fseng + fevpg * htvp) \
            + const_physical.cpliq * pg_rain * (t_precip - t_icesno[lb]) \
            + const_physical.cpice * pg_snow * (t_precip - t_icesno[lb])

    # outgoing long-wave radiation from ground
    olrg = (1. - emg) * forc_frl + emg * const_physical.stefnc * t_icesno_bef[lb] ** 4 \
           + 4. * emg * const_physical.stefnc * t_icesno_bef[lb] ** 3 * tinc

    # averaged bulk surface emissivity
    emis = emg

    # radiative temperature
    trad = (olrg / const_physical.stefnc) ** 0.25

    errore = (sabg + forc_frl - olrg - fsena - lfevpa - xmf +
          const_physical.cpliq * pg_rain * (t_precip - t_icesno[lb]) +
          const_physical.cpice * pg_snow * (t_precip - t_icesno[lb]))

    # Adjust errore in a loop
    for j in range(lb, nl_ice + 1):
        errore -= (t_icesno[j] - t_icesno_bef[j]) / fact[j]

    # Conditional debug check
    if nl_colm['CoLMDEBUG']:
        if abs(errore) > 0.2:
            print('GLACIER_TEMP.py : energy balance violation')
            print(f'{errore:.3f}, {sabg:.3f}, {forc_frl:.3f}, {olrg:.3f}, {fsena:.3f}, {lfevpa:.3f}, {xmf:.3f}, {t_precip:.3f}, {t_icesno[lb]:.3f}')
            raise ValueError('Energy balance violation')
        
    return t_icesno, wice_icesno, wliq_icesno, scv, snowdp, snofrz, imelt, taux, tauy, fsena, lfevpa, fseng, fevpg, olrg, fgrnd, fevpa, qseva, qsdew, qsubl, qfros, sm, tref, qref, trad, emis, z0m, zol,rib,ustar,qstar,tstar,fm,fh,fq

def GLACIER_WATER(nl_soil, maxsnl, deltim, z_icesno, dz_icesno, zi_icesno, t_icesno, wliq_icesno, wice_icesno,  pg_rain, pg_snow, sm, scv, snowdp, imelt, fiold, snl, qseva, qsdew, qsubl, qfros, gwat, ssi, wimp, forc_us, forc_vs):

    """
        Subroutine to update glacier water properties.
    """
    # ------------------------------------------------------------------------------------
    # [1] update the liquid water within snow layer and the water onto the ice surface
    # !
    # ! Snow melting is treated in a realistic fashion, with meltwater
    # ! percolating downward through snow layers as long as the snow is unsaturated.
    # ! Once the underlying snow is saturated, any additional meltwater runs off.
    # ! When glacier ice melts, however, the meltwater is assumed to remain in place until it refreezes.
    # ! In warm parts of the ice sheet, the meltwater does not refreeze, but stays in place indefinitely.
    # ------------------------------------------------------------------------------------

    lb = snl + 1
    if lb >= 1:
        gwat = pg_rain + sm - qseva
    else:
        gwat = CoLM_SoilSnowHydrology.snowwater(lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros,
                  dz_icesno[lb:], wice_icesno[lb:], wliq_icesno[lb:], gwat)

    # ------------------------------------------------------------------------------------
    #  [2] surface runoff and infiltration
    # ------------------------------------------------------------------------------------

    if snl < 0:
        # Compaction rate for snow
        # Natural compaction and metamorphosis. The compaction rate
        # is recalculated for every new timestep
        lb = snl + 1  # lower bound of array
        dz_icesno[lb:] = CoLM_SnowLayersCombineDivide.snowcompaction(lb, deltim, imelt[lb:], fiold[lb:], t_icesno[lb:], wliq_icesno[lb:], wice_icesno[lb:],
                       forc_us, forc_vs, dz_icesno[lb:])

        # Combine thin snow elements
        lb = maxsnl + 1
        wliq_icesno[lb:], wice_icesno[lb:], t_icesno[lb:], dz_icesno[lb:], zi_icesno[lb:], snowdp, scv, snl =CoLM_SnowLayersCombineDivide.snowlayerscombine(lb, snl, z_icesno[lb:], dz_icesno[lb:], zi_icesno[lb:],
                          wliq_icesno[lb:], wice_icesno[lb:], t_icesno[lb:], scv, snowdp)

        # Divide thick snow elements
        if snl < 0:
            snl, wliq_icesno[lb:], wice_icesno[lb:], t_icesno[lb:], dz_icesno[lb:], z_icesno[lb:], zi_icesno[lb - 1:0] =CoLM_SnowLayersCombineDivide.snowlayersdivide(lb, snl, z_icesno[lb:], dz_icesno[lb:], zi_icesno[lb - 1:0],
                             wliq_icesno[lb:], wice_icesno[lb:], t_icesno[lb:])

    if snl > maxsnl:
        wice_icesno[maxsnl + 1:snl] = 0.0
        wliq_icesno[maxsnl + 1:snl] = 0.0
        t_icesno[maxsnl + 1:snl] = 0.0
        z_icesno[maxsnl + 1:snl] = 0.0
        dz_icesno[maxsnl + 1:snl] = 0.0

    if lb >= 1:
        wliq_icesno[0] = max(1e-8, wliq_icesno[0] + qsdew * deltim)
        wice_icesno[0] = max(1e-8, wice_icesno[0] + (qfros - qsubl) * deltim)

    return snl,  z_icesno, dz_icesno, zi_icesno, t_icesno, wice_icesno, wliq_icesno, scv, snowdp, gwat
    

def GLACIER_WATER_snicar(nl_ice,maxsnl,deltim,
                      z_icesno    ,dz_icesno   ,zi_icesno ,t_icesno,
                      wliq_icesno ,wice_icesno ,pg_rain   ,pg_snow ,
                      sm          ,scv         ,snowdp    ,imelt   ,
                      fiold       ,snl         ,qseva     ,qsdew   ,
                      qsubl       ,qfros       ,gwat      ,         
                      ssi         ,wimp        ,forc_us   ,forc_vs ,
                      forc_aer    ,
                      mss_bcpho   ,mss_bcphi   ,mss_ocpho,mss_ocphi,
                      mss_dst1    ,mss_dst2    ,mss_dst3  ,mss_dst4 ):
    lb = snl + 1
    if lb >= 1:
        gwat = pg_rain + sm - qseva
    else:
        gwat = CoLM_SoilSnowHydrology.snowwater_snicar(lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros,
                                dz_icesno[lb:], wice_icesno[lb:], wliq_icesno[lb:], forc_aer,
                                mss_bcpho[lb:], mss_bcphi[lb:], mss_ocpho[lb:], mss_ocphi[lb:],
                                mss_dst1[lb:], mss_dst2[lb:], mss_dst3[lb:], mss_dst4[lb:])

    # Surface runoff and infiltration
    if snl < 0:
        # Compaction rate for snow
        # Natural compaction and metamorphosis. The compaction rate is recalculated for every new timestep
        lb = snl + 1  # lower bound of array
        dz_icesno[lb:] = CoLM_SnowLayersCombineDivide.snowcompaction(lb, deltim, imelt[lb:], fiold[lb:], t_icesno[lb:],
                    wliq_icesno[lb:], wice_icesno[lb:], forc_us, forc_vs, dz_icesno[lb:])

        # Combine thin snow elements
        lb = maxsnl + 1
        wice_icesno[lb:], wliq_icesno[lb:], t_icesno[lb:],  dz_icesno[lb:], z_icesno[lb:], zi_icesno[lb:], snowdp, scv, snl = CoLM_SnowLayersCombineDivide.snowlayerscombine_snicar(lb,
                                                                                                snl, z_icesno[lb:], dz_icesno[lb:], zi_icesno[lb:],
                                                                                                wliq_icesno[lb:], wice_icesno[lb:], t_icesno[lb:], scv, snowdp,
                                                                                                mss_bcpho[lb:], mss_bcphi[lb:], mss_ocpho[lb:], mss_ocphi[lb:],
                                                                                                mss_dst1[lb:], mss_dst2[lb:], mss_dst3[lb:], mss_dst4[lb:])

        # Divide thick snow elements
        if snl < 0:
            CoLM_SnowLayersCombineDivide.snowlayersdivide_snicar(lb, snl, z_icesno[lb:], dz_icesno[lb:], zi_icesno[lb:],
                                    wliq_icesno[lb:], wice_icesno[lb:], t_icesno[lb:],
                                    mss_bcpho[lb:], mss_bcphi[lb:], mss_ocpho[lb:], mss_ocphi[lb:],
                                    mss_dst1[lb:], mss_dst2[lb:], mss_dst3[lb:], mss_dst4[lb:])

    if snl > maxsnl:
        wice_icesno[maxsnl:snl] = 0.
        wliq_icesno[maxsnl:snl] = 0.
        t_icesno[maxsnl:snl] = 0.
        z_icesno[maxsnl:snl] = 0.
        dz_icesno[maxsnl:snl] = 0.

    if lb >= 1:
        wliq_icesno[0] = max(1.e-8, wliq_icesno[0] + qsdew * deltim)
        wice_icesno[0] = max(1.e-8, wice_icesno[0] + (qfros - qsubl) * deltim)

    return snl,  z_icesno, dz_icesno, zi_icesno, t_icesno, wice_icesno, wliq_icesno, scv, snowdp, gwat, mss_bcpho   ,mss_bcphi   ,mss_ocpho,mss_ocphi, mss_dst1    ,mss_dst2    ,mss_dst3  ,mss_dst4