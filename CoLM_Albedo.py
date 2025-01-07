import numpy as np
from CoLM_SnowSnicar import CoLM_SnowSnicar
from CoLM_Aerosol import CoLM_Aerosol

def twostream(chil, rho, tau, green, lai, sai, coszen, albg, albv, tran, thermk, extkb, extkd, ssun, ssha, nl_colm):
    """
        calculation of canopy albedos via two stream approximation (direct
!     and diffuse ) and partition of incident solar
!    Original author: Yongjiu Dai, June 11, 2001
    """
    eup = np.zeros((2, 2))
    edown = np.zeros((2, 2))

    # projected area of phytoelements in direction of mu and
    # ! average inverse diffuse optical depth per unit leaf area
    phi1 = 0.5 - 0.633 * chil - 0.33 * chil * chil
    phi2 = 0.877 * (1. - 2. * phi1)

    proj = phi1 + phi2 * coszen
    extkb = (phi1 + phi2 * coszen) / coszen

    extkd = 0.719

    if abs(phi1) > 1.e-6 and abs(phi2) > 1.e-6:
        zmu = 1. / phi2 * (1. - phi1 / phi2 * np.log((phi1 + phi2) / phi1))
    elif abs(phi1) <= 1.e-6:
        zmu = 1. / 0.877
    elif abs(phi2) <= 1.e-6:
        zmu = 1. / (2. * phi1)

    zmu2 = zmu * zmu

    if nl_colm['LULC_USGS']:
        sai_ = 0.
    else:
        sai_ = sai

    lsai = lai + sai_
    power3 = (lai + sai_) / zmu
    power3 = min(50.0, power3)
    power3 = max(1.e-5, power3)
    thermk = np.exp(-power3)
    if lsai <= 1e-6:
        return

    # ----------------------------------------------------------------------
    # calculate average scattering coefficient, leaf projection and
    # !     other coefficients for two-stream model.
    # ----------------------------------------------------------------------
    # account for stem optical property effects
    for iw in range(0, 2):  # WAVE_BAND_LOOP
        # Account for stem optical property effects
        scat = lai / lsai * (tau[iw, 0] + rho[iw, 0]) + sai_ / lsai * (tau[iw, 1] + rho[iw, 1])
        as_ = scat / 2.0 * proj / (proj + coszen * phi2)
        as_ *= (1.0 - coszen * phi1 / (proj + coszen * phi2) * np.log(
            (proj + coszen * phi2 + coszen * phi1) / (coszen * phi1)))

        # Account for stem optical property effects
        upscat = lai / lsai * tau[iw, 0] + sai_ / lsai * tau[iw, 1]
        upscat = 0.5 * (scat + (scat - 2.0 * upscat) * ((1.0 + chil) / 2.0) ** 2)
        betao = (1.0 + zmu * extkb) / (scat * zmu * extkb) * as_

        # Intermediate variables identified in appendix of SE-85
        be = 1.0 - scat + upscat
        ce = upscat
        de = scat * zmu * extkb * betao
        fe = scat * zmu * extkb * (1.0 - betao)

        psi = np.sqrt(be ** 2 - ce ** 2) / zmu
        power1 = min(psi * lsai, 50.0)
        power2 = min(extkb * lsai, 50.0)
        s1 = np.exp(-power1)
        s2 = np.exp(-power2)

        # Calculation of direct albedos and canopy transmittances
        p1 = be + zmu * psi
        p2 = be - zmu * psi
        p3 = be + zmu * extkb
        p4 = be - zmu * extkb

        f1 = 1.0 - albg[iw, 1] * p1 / ce
        f2 = 1.0 - albg[iw, 1] * p2 / ce

        h1 = -(de * p4 + ce * fe)
        h4 = -(fe * p3 + ce * de)

        sigma = (zmu * extkb) ** 2 + (ce ** 2 - be ** 2)

        if abs(sigma) > 1e-10:
            hh1 = h1 / sigma
            hh4 = h4 / sigma

            m1 = f1 * s1
            m2 = f2 / s1
            m3 = (albg[iw, 0] - (hh1 - albg[iw, 1] * hh4)) * s2

            n1 = p1 / ce
            n2 = p2 / ce
            n3 = -hh4

            hh2 = (m3 * n2 - m2 * n3) / (m1 * n2 - m2 * n1)
            hh3 = (m3 * n1 - m1 * n3) / (m2 * n1 - m1 * n2)

            hh5 = hh2 * p1 / ce
            hh6 = hh3 * p2 / ce

            albv[iw, 0] = hh1 + hh2 + hh3
            tran[iw, 0] = hh4 * s2 + hh5 * s1 + hh6 / s1

            eup[iw, 0] = hh1 * (1.0 - s2 ** 2) / (2.0 * extkb) + hh2 * (1.0 - s1 * s2) / (extkb + psi) + hh3 * (
                    1.0 - s2 / s1) / (extkb - psi)
            edown[iw, 0] = hh4 * (1.0 - s2 ** 2) / (2.0 * extkb) + hh5 * (1.0 - s1 * s2) / (extkb + psi) + hh6 * (
                    1.0 - s2 / s1) / (extkb - psi)
        else:
            m1 = f1 * s1
            m2 = f2 / s1
            m3 = h1 / zmu ** 2 * (lsai + 1.0 / (2.0 * extkb)) * s2 + albg[iw, 1] / ce * (
                    -h1 / (2.0 * extkb * zmu ** 2) * (p3 * lsai + p4 / (2.0 * extkb)) - de) * s2 + albg[iw, 0] * s2

            n1 = p1 / ce
            n2 = p2 / ce
            n3 = 1.0 / ce * (h1 * p4 / (4.0 * extkb ** 2 * zmu) + de)

            hh2 = (m3 * n2 - m2 * n3) / (m1 * n2 - m2 * n1)
            hh3 = (m3 * n1 - m1 * n3) / (m2 * n1 - m1 * n2)

            hh5 = hh2 * p1 / ce
            hh6 = hh3 * p2 / ce

            albv[iw, 0] = -h1 / (2.0 * extkb * zmu ** 2) + hh2 + hh3
            tran[iw, 0] = 1.0 / ce * (
                    -h1 / (2.0 * extkb * zmu ** 2) * (p3 * lsai + p4 / (2.0 * extkb)) - de) * s2 + hh5 * s1 + hh6 / s1

            eup[iw, 0] = (hh2 - h1 / (2.0 * extkb * zmu ** 2)) * (1.0 - s2 ** 2) / (2.0 * extkb) + hh3 * (
                    lsai - 0.0) + h1 / (
                                 2.0 * extkb * zmu ** 2) * (lsai * s2 ** 2 - (1.0 - s2 ** 2) / (2.0 * extkb))

            edown[iw, 0] = (hh5 - (h1 * p4 / (4.0 * extkb ** 2 * zmu) + de) / ce) * (1.0 - s2 ** 2) / (
                    2.0 * extkb) + hh6 * (lsai - 0.0) + h1 * p3 / (ce * 4.0 * extkb ** 2 * zmu ** 2) * (
                                   lsai * s2 ** 2 - (1.0 - s2 ** 2) / (2.0 * extkb))

        ssun[iw, 0] = (1.0 - scat) * (1.0 - s2 + 1.0 / zmu * (eup[iw, 0] + edown[iw, 0]))
        ssha[iw, 0] = scat * (1.0 - s2) + (albg[iw, 1] * tran[iw, 0] + albg[iw, 0] * s2 - tran[iw, 0]) - albv[iw, 0] - (
                1.0 - scat) / zmu * (eup[iw, 0] + edown[iw, 0])

        # Calculation of diffuse albedos and canopy transmittances
        m1 = f1 * s1
        m2 = f2 / s1
        m3 = 0.0

        n1 = p1 / ce
        n2 = p2 / ce
        n3 = 1.0

        hh7 = -m2 / (m1 * n2 - m2 * n1)
        hh8 = -m1 / (m2 * n1 - m1 * n2)

        hh9 = hh7 * p1 / ce
        hh10 = hh8 * p2 / ce

        albv[iw, 1] = hh7 + hh8
        tran[iw, 1] = hh9 * s1 + hh10 / s1

        if abs(sigma) > 1e-10:
            eup[iw, 1] = hh7 * (1.0 - s1 * s2) / (extkb + psi) + hh8 * (1.0 - s2 / s1) / (extkb - psi)
            edown[iw, 1] = hh9 * (1.0 - s1 * s2) / (extkb + psi) + hh10 * (1.0 - s2 / s1) / (extkb - psi)
        else:
            eup[iw, 1] = hh7 * (1.0 - s1 * s2) / (extkb + psi) + hh8 * (lsai - 0.0)
            edown[iw, 1] = hh9 * (1.0 - s1 * s2) / (extkb + psi) + hh10 * (lsai - 0.0)

        ssun[iw, 1] = (1.0 - scat) / zmu * (eup[iw, 1] + edown[iw, 1])
        ssha[iw, 1] = tran[iw, 1] * (albg[iw, 1] - 1.0) - (albv[iw, 1] - 1.0) - (1.0 - scat)
    # ! WAVE_BAND_LOOP

    tran[:, 2] = s2

    return albv, tran, thermk, extkb, extkd, ssun, ssha

# def albland (vars_global.maxsnl,lai,sai,patchtype,nl_colm):
def albland(nl_colm, vars_global, gblock, tfrz, ipatch, patchtype, deltim,
            soil_s_v_alb, soil_d_v_alb, soil_s_n_alb, soil_d_n_alb,
            chil, rho, tau, fveg, green, lai, sai, coszen,
            wt, fsno, scv, scvold, sag, ssw, pg_snow, forc_t, t_grnd, t_soisno, dz_soisno,
            snl, wliq_soisno, wice_soisno, snw_rds, snofrz,
            mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi,
            mss_dst1, mss_dst2, mss_dst3, mss_dst4,
            alb, ssun, ssha, ssoi, ssno, ssno_lyr, thermk, extkb, extkd,mpi):
    """
    Determine net radiation.
    Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
    software         : Calculates fragmented albedos (direct and diffuse) in wavelength regions split at 0.7um.
                       (1) soil albedos: as in BATS formulations, which are the function of soil color and moisture in
                        the surface soil layer
                       (2) snow albedos: as in BATS formulations, which are inferred from the calculations of Wiscombe
                       and Warren (1980) and the snow model and data of Anderson(1976), and the function of snow age,
                       grain size,solar zenith angle, pollution, the amount of the fresh snow
                       (3) canopy albedo: two-stream approximation model
                       (4) glacier albedos: as in BATS, which are set to constants (0.8 for visible beam,
                        0.55 for near-infrared)
                       (5) lake and wetland albedos: as in BATS, which depend on cosine solar zenith angle, based on
                       data in Henderson-Sellers (1986). The frozen lake and wetland albedos are set to constants
                        (0.6 for visible beam, 0.4 for near-infrared)
                       (6) over the snow covered tile, the surface albedo is estimated by a linear combination of
                        albedos for snow, canopy and bare soil (or lake, wetland, glacier).

    Args:
        ipatch          (ndarray)  : patch index
        patchtype       (ndarray)  : land patch type (0=soil, 1=urban or built-up, 2=wetland, 3=land ice, 4=deep lake)
        snl             (ndarray)  : number of snow layers
        deltim          (float)    : seconds in a time step [second]
        soil_s_v_alb    (float)    : albedo of visible of the saturated soil
        soil_d_v_alb    (float)    : albedo of visible of the dry soil
        soil_s_n_alb    (float)    : albedo of near infrared of the saturated soil
        soil_d_n_alb    (float)    : albedo of near infrared of the dry soil
        chil            (float)    : albedo of near infrared of the dry soil
        rho
        tau
        fveg
        green
        lai
        sai
        coszen
        wt
        fsno
        ssw
        scv
        scvold
        pg_snow
        forc_t
        t_grnd
        wliq_soisno
        wice_soisno
        mss_ocpho
        mss_ocphi
        mss_dst1
        mss_dst2
        mss_dst3
        mss_dst4
        sag

    Returns:
         alb                (float)    : averaged albedo [-]
         ssun
         ssha
         thermk
         extkb
         extkd
         ssoi
         ssno
         ssno_lyr
    """
    # -------------------------- Local variables - ---------------------------
    age = 0.0  # factor to reduce visible snow alb due to snow age [-]
    albg0 = 0.0  # temporary varaiable [-]
    albsoi = np.zeros((2, 2))  # soil albedo[-]
    albsno = np.zeros((2, 2))  # snow albedo[-]
    albsno_pur = np.zeros((2, 2))  # snow albedo[-]
    albsno_bc = np.zeros((2, 2))  # snow albedo[-]
    albsno_oc = np.zeros((2, 2))  # snow albedo[-]
    albsno_dst = np.zeros((2, 2))  # snow albedo[-]

    albg = np.zeros((2, 2))  # albedo, ground
    albv = np.zeros((2, 2))  # albedo, vegetation[-]

    # beta0  # upscattering
    # cff  # snow
    # conn  # constant( = 0.5) for visible snow alb calculation[-]
    # cons  # constant( = 0.2) for nir snow albedo calculation[-]
    # czen  # cosine
    # czf  # solar
    # dfalbl  # snow
    # dfalbs  # snow
    # dralbl  # snow
    # dralbs  # snow
    # lsai  # leaf and stem
    # sl  # factor
    # snal0  # alb
    # snal1  # alb
    # upscat  # upward
    tran = np.zeros((2, 3))  # canopy

    snwcp_ice = 0.0  # !excess precipitation due to snow capping [kg m-2 s-1]
    mss_cnc_bcphi = np.zeros(
        0 - vars_global.maxsnl - 1)  # mass concentration of  hydrophilic BC = np.zeros(col, lyr)[kg / kg]
    mss_cnc_bcpho = np.zeros(
        0 - vars_global.maxsnl - 1)  # mass concentration of  hydrophilic BC = np.zeros(col, lyr)[kg / kg]
    mss_cnc_ocphi = np.zeros(
        0 - vars_global.maxsnl - 1)  # mass concentration of  hydrophilic OC = np.zeros(col, lyr)[kg / kg]
    mss_cnc_ocpho = np.zeros(
        0 - vars_global.maxsnl - 1)  # mass concentration of  hydrophilic OC = np.zeros(col, lyr)[kg / kg]
    mss_cnc_dst1 = np.zeros(0 - vars_global.maxsnl - 1)  # mass concentration of dust aerosol species 1(col, lyr)[kg / kg]
    mss_cnc_dst2 = np.zeros(0 - vars_global.maxsnl - 1)  # mass concentration of dust aerosol species 2(col, lyr)[kg / kg]
    mss_cnc_dst3 = np.zeros(0 - vars_global.maxsnl - 1)  # mass concentration of dust aerosol species 3(col, lyr)[kg / kg]
    mss_cnc_dst4 = np.zeros(0 - vars_global.maxsnl - 1)  # mass concentration of dust aerosol species 4(col, lyr)[kg / kg]

    # ----------------------------------------------------------------------
    # 1. Initial set
    # ----------------------------------------------------------------------

    # visible and near infrared band albedo for new snow
    snal0 = 0.85  # visible band
    snal1 = 0.65  # near infrared

    # ----------------------------------------------------------------------
    # set default soil and vegetation albedos and solar absorption
    # TODO: need double check
    # alb = np.zeros((2, 2))  # averaged albedo
    # ssun = np.zeros((2, 2))  # sunlit canopy absorption for solar radiation
    # ssha = np.zeros((2, 2))  # shaded canopy absorption for solar radiation
    # thermk = 0.0  # canopy gap fraction for TIR radiation
    # extkb = 0.0  # direct solar extinction coefficient
    # extkd = 0.0  # diffuse and scattered diffuse PAR extinction coefficient

    # ssoi = np.zeros((2, 2))  # ground soil absorption
    # ssno = np.zeros((2, 2))  # ground snow absorption
    # ssno_lyr = np.zeros((2, 2, 0 - vars_global.maxsnl - 1))  # ground snow layer absorption by SNICAR
    tran = np.zeros((2, 3))  # ! canopy transmittances for solar radiation
    alb[:] = 1  # averaged
    albg[:] = 1  # ground
    albv[:] = 1  # vegetation
    ssun[:] = 0  # sunlit leaf absorption
    ssha[:] = 0  # shaded leaf absorption
    tran[:, 0] = 0.  # incident direct radiation diffuse transmittance
    tran[:, 1] = 1.  # incident diffuse radiation diffuse transmittance
    tran[:, 2] = 1.  # incident direct radiation direct transmittance

    # 07/06/2023, yuan: use the values of the previous timestep.
    # for nighttime longwave calculations.
    # thermk = 1.e-3
    if (lai + sai) <= 1.e-6:
        thermk = 1.0
    # thermk = np.where((lai + sai) <= 1.e-6, 1., 1.e-3)
    extkb = 1.
    extkd = 0.718

    # soil and snow absorption
    ssoi[:,:] = 0  # set initial soil absorption
    ssno[:,:] = 0  # set initial snow absorption
    ssno_lyr[:,:,:] = 0  # set initial snow layer absorption

    if patchtype == 0:
        if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
            pass

    # ----------------------------------------------------------------------
    #    Calculate column-integrated aerosol masses, and
    # !  mass concentrations for radiative calculations and output
    # !  (based on new snow level state, after SnowFilter is rebuilt.
    # !  NEEDS TO BE AFTER SnowFiler is rebuilt, otherwise there
    # !  can be zero snow layers but an active column in filter)
    # ----------------------------------------------------------------------

    do_capsnow = False  # !true => DO snow capping
    aerosol = CoLM_Aerosol(nl_colm, mpi,gblock,vars_global.spval)

    snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, mss_cnc_bcphi, mss_cnc_bcpho, mss_cnc_ocphi, mss_cnc_ocpho, mss_cnc_dst1, mss_cnc_dst2, mss_cnc_dst3, mss_cnc_dst4 = aerosol.AerosolMasses(deltim, snl, do_capsnow, wice_soisno[:5], wliq_soisno[:5], snwcp_ice, snw_rds, mss_bcpho,
                       mss_bcphi,
                       mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, mss_cnc_bcphi, mss_cnc_bcpho,
                       mss_cnc_ocphi, mss_cnc_ocpho, mss_cnc_dst1, mss_cnc_dst2, mss_cnc_dst3, mss_cnc_dst4, vars_global.maxsnl)

    # ----------------------------------------------------------------------
    # ! Snow aging routine based on Flanner and Zender (2006), Linking snowpack
    # ! microphysics and albedo evolution, JGR, and Brun (1989), Investigation of
    # ! wet-snow metamorphism in respect of liquid-water content, Ann. Glaciol.
    # ----------------------------------------------------------------------
    SS = CoLM_SnowSnicar(nl_colm, mpi)
    SS.SnowAge_grain(deltim, snl, dz_soisno[:6], pg_snow, snwcp_ice, snofrz,
                     do_capsnow, fsno, scv, wliq_soisno[:5], wice_soisno[:5],
                     t_soisno[:6], t_grnd, forc_t, snw_rds,aerosol.fresh_snw_rds_max,vars_global.maxsnl)

    lsai = lai + sai
    if coszen <= 0.0:
        return  snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, sag, alb, ssun, ssha, thermk, extkb, extkd, ssoi, ssno, ssno_lyr# Only calculate albedo when coszen > 0
    czen = max(coszen, 0.001)

    # ----------------------------------------------------------------------
    # 2. get albedo over land
    # ----------------------------------------------------------------------
    # 2.1 soil albedos, depends on moisture
    if patchtype <= 2:  # soil, urban, and wetland
        alb_s_inc = max(0.11 - 0.40 * ssw, 0.0)
        albg[0, 0] = min(soil_s_v_alb + alb_s_inc, soil_d_v_alb)
        albg[1, 0] = min(soil_s_n_alb + alb_s_inc, soil_d_n_alb)
        albg[:, 1] = albg[:, 0]  # diffused albedos setting
    # 2.2 albedos for permanent ice sheet.
    elif patchtype == 3:  # permanent ice sheet
        albg[0, :] = 0.8
        albg[1, :] = 0.55

    # 2.3 albedo for inland water
    elif patchtype >= 4:
        albg0 = 0.05 / (czen + 0.15)
        albg[:, 0] = albg0
        albg[:, 1] = 0.1  # Subin (2012)

        if t_grnd < tfrz:  # frozen lake and wetland
            albg[:, 0] = 0.6
            albg[:, 1] = 0.4

    # SAVE soil ground albedo
    albsoi[:, :] = albg[:, :]

    # ----------------------------------------------------------------------
    #  3. albedo for snow cover.
    # !    - Scheme 1: snow albedo depends on snow-age, zenith angle, and thickness
    # !                of snow age gives reduction of visible radiation [CoLM2014].
    # !    - Scheme 2: SNICAR model
    # ----------------------------------------------------------------------
    if scv > 0.:
        if not nl_colm['DEF_USE_SNICAR']:
            cons = 0.2
            conn = 0.5
            sl = 2.0  # sl helps control albedo zenith dependence

            # Update the snow age
            if snl == 0:
                sag = 0.
            SS.snowage()

            # Correction for snow age
            age = 1. - 1. / (1. + sag)
            dfalbs = snal0 * (1. - cons * age)

            # czf corrects albedo of new snow for solar zenith
            cff = ((1. + 1. / sl) / (1. + czen * 2. * sl) - 1. / sl)
            cff = max(cff, 0.)
            czf = 0.4 * cff * (1. - dfalbs)
            dralbs = dfalbs + czf
            dfalbl = snal1 * (1. - conn * age)
            czf = 0.4 * cff * (1. - dfalbl)
            dralbl = dfalbl + czf

            albsno[0, 0] = dralbs
            albsno[1, 0] = dralbl
            albsno[0, 1] = dfalbs
            albsno[1, 1] = dfalbl
        else:
            pass
    # ! 3.1 correction due to snow cover
    alb = (1. - fsno) * albg + fsno * albsno
    # ----------------------------------------------------------------------
    # ! 4. canopy albedos: two stream approximation or 3D canopy radiation transfer
    # ----------------------------------------------------------------------
    if lai + sai > 1e-6:
        if patchtype == 0:
            if nl_colm['LULC_USGS'] or nl_colm['LULC_IGBP']:
                albv, tran, thermk, extkb, extkd, ssun, ssha = twostream(chil, rho, tau, green, lai, sai, coszen, albg, albv, tran, thermk, extkb, extkd, ssun, ssha, nl_colm)
        else:
            albv, tran, thermk, extkb, extkd, ssun, ssha = twostream(chil, rho, tau, green, lai, sai, coszen, albg, albv, tran, thermk, extkb, extkd, ssun, ssha, nl_colm)

    if patchtype == 0:
        if nl_colm['LULC_IGBP_PFT']:
            pass
        if nl_colm['LULC_IGBP_PC']:
            pass

    ssoi[0, 0] = tran[0, 0] * (1 - albsoi[0, 1]) + tran[0, 2] * (1 - albsoi[0, 0])
    ssoi[1, 0] = tran[1, 0] * (1 - albsoi[1, 1]) + tran[1, 2] * (1 - albsoi[1, 0])
    ssoi[0, 1] = tran[0, 1] * (1 - albsoi[0, 1])
    ssoi[1, 1] = tran[1, 1] * (1 - albsoi[1, 1])

    ssno[0, 0] = tran[0, 0] * (1 - albsno[0, 1]) + tran[0, 2] * (1 - albsno[0, 0])
    ssno[1, 0] = tran[1, 0] * (1 - albsno[1, 1]) + tran[1, 2] * (1 - albsno[1, 0])
    ssno[0, 1] = tran[0, 1] * (1 - albsno[0, 1])
    ssno[1, 1] = tran[1, 1] * (1 - albsno[1, 1])

    return snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, sag, alb, ssun, ssha, thermk, extkb, extkd, ssoi, ssno, ssno_lyr




def albocean(oro, scv, coszrs):
    # ----------------------------------------------------------------------
    #   Compute surface albedos
    # !
    # ! Computes surface albedos for direct/diffuse incident radiation for
    # ! two spectral intervals:
    # !   s = 0.2-0.7 micro-meters
    # !   l = 0.7-5.0 micro-meters
    # !
    # ! Albedos specified as follows:
    # !
    # ! Ocean           Uses solar zenith angle to compute albedo for direct
    # !                 radiation; diffuse radiation values constant; albedo
    # !                 independent of spectral interval and other physical
    # !                 factors such as ocean surface wind speed.
    # !
    # ! Ocean with      Surface albs specified; combined with overlying snow
    # !   sea ice
    # !
    # ! For more details , see Briegleb, Bruce P., 1992: Delta-Eddington
    # ! Approximation for Solar Radiation in the NCAR Community Climate Model,
    # ! Journal of Geophysical Research, Vol 97, D7, pp7603-7612).
    # !
    # ! yongjiu dai and xin-zhong liang (08/01/2001)
    # ----------------------------------------------------------------------
    alb = np.zeros((2, 2))
    asices = 0.70  # 海冰的反照率 (0.2-0.7 micro-meters)
    asicel = 0.50  # 海冰的反照率 (0.7-5.0 micro-meters)
    asnows = 0.95  # 雪的反照率 (0.2-0.7 micro-meters)
    asnowl = 0.70  # 雪的反照率 (0.7-5.0 micro-meters)

    # 海冰
    if int(np.round(oro)) == 2:
        # 设置海冰表面的反照率
        alb[0, 0] = asices
        alb[1, 0] = asicel
        alb[0, 1] = alb[0, 0]
        alb[1, 1] = alb[1, 0]

        sasdif = asnows
        saldif = asnowl

        # 计算雪的反射率
        if scv > 0:
            if coszrs < 0.5:
                # 光线直射
                sasdir = min(0.98, sasdif + (1. - sasdif) * 0.5 * (3. / (1. + 4. * coszrs) - 1.))
                saldir = min(0.98, saldif + (1. - saldif) * 0.5 * (3. / (1. + 4. * coszrs) - 1.))
            else:
                # 光线散射
                sasdir = asnows
                saldir = asnowl

            # 计算总反照率
            snwhgt = 20. * scv / 1000.
            rghsnw = 0.25
            frsnow = snwhgt / (rghsnw + snwhgt)
            alb[0, 0] = alb[0, 0] * (1. - frsnow) + sasdir * frsnow
            alb[1, 0] = alb[1, 0] * (1. - frsnow) + saldir * frsnow
            alb[0, 1] = alb[0, 1] * (1. - frsnow) + sasdif * frsnow
            alb[1, 1] = alb[1, 1] * (1. - frsnow) + saldif * frsnow

    # 无冰的海洋
    if int(np.round(oro)) == 0:
        # 根据太阳天顶角计算海洋表面的反射率
        alb[1, 0] = .026 / (coszrs ** 1.7 + .065) + .15 * (coszrs - 0.1) * (coszrs - 0.5) * (coszrs - 1.)
        alb[0, 0] = alb[1, 0]
        alb[0, 1] = 0.06
        alb[1, 1] = 0.06

    return alb