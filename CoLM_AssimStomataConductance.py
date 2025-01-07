# ----------------------------------------------------------------------
# DESCRIPTION:
# calculation of canopy photosynthetic rate using the integrated
# !     model relating assimilation and stomatal conductance.
# !     Original author: Yongjiu Dai, 08/11/2001
# !     Revision author: Xingjie Lu, 2021
# !     Reference: Dai et al., 2004: A two-big-leaf model for canopy temperature,
# !         photosynthesis and stomatal conductance. J. Climate, 17: 2281-2299.
# !     units are converted from mks to biological units in this routine.
# !                          units
# !                         -------
# !      pco2m, pco2a, pco2i, po2m                : pascals
# !      co2a, co2s, co2i, h2oa, h2os, h2oa       : mol mol-1
# !      vmax25, respcp, assim, gs, gb, ga        : mol m-2 s-1
# !      effcon                                   : mol co2 mol quanta-1
# !      1/rb, 1/ra, 1/rst                        : m s-1
# !                       conversions
# !                      -------------
# !      1 mol h2o           = 0.018 kg
# !      1 mol co2           = 0.044 kg
# !      h2o (mol mol-1)     = ea / psrf ( pa pa-1 )
# !      h2o (mol mol-1)     = q*mm/(q*mm + 1)
# !      gs  (co2)           = gs (h2o) * 1./1.6
# !      gs  (mol m-2 s-1 )  = gs (m s-1) * 44.6*tf/t*p/po
# !      par (mol m-2 s-1 )  = par(w m-2) * 4.6*1.e-6
# !      mm  (molair/molh2o) = 1.611
# ----------------------------------------------------------------------
import os
import numpy as np

def sortin(eyy, pco2y, range1, gammas, ic, iterationtotal):
    if ic < 4:
        eyy_a = 1.0
        if eyy[0] < 0.0:
            eyy_a = -1.0
        pco2y[0] = gammas + 0.5 * range1
        pco2y[1] = gammas + range1 * (0.5 - 0.3 * eyy_a)
        pco2y[2] = pco2y[0] - (pco2y[0] - pco2y[1]) / (eyy[0] - eyy[1] + 1.e-10) * eyy[0]

        pmin = min(pco2y[0], pco2y[1])
        emin = min(eyy[0], eyy[1])
        if emin > 0.0 and pco2y[2] > pmin:
            pco2y[2] = gammas
        return

    n = ic - 1
    for j in range(1, n):
        a = eyy[j]
        b = pco2y[j]
        for i in range(j - 1, 0, -1):
            if eyy[i] <= a:
                break
            eyy[i + 1] = eyy[i]
            pco2y[i + 1] = pco2y[i]
        i=0
        eyy[i] = a
        pco2y[i] = b

    pco2b = 0.0
    is_val = 1
    for ix in range(n):
        if eyy[ix] < 0.0:
            pco2b = pco2y[ix]
            is_val = ix + 1
    i1 = max(1, is_val - 1)
    i1 = min(n - 2, i1)
    i2 = i1 + 1
    i3 = i1 + 2
    isp = is_val + 1
    isp = min(isp, n)
    is_val = isp - 1

    pco2yl = pco2y[is_val-1] - (pco2y[is_val-1] - pco2y[isp]) / (eyy[is_val-1] - eyy[isp] + 1.e-10) * eyy[is_val-1]

    ac1 = eyy[i1] ** 2 - eyy[i2] ** 2
    ac2 = eyy[i2] ** 2 - eyy[i3] ** 2
    bc1 = eyy[i1] - eyy[i2]
    bc2 = eyy[i2] - eyy[i3]
    cc1 = pco2y[i1] - pco2y[i2]
    cc2 = pco2y[i2] - pco2y[i3]
    bterm = (cc1 * ac2 - cc2 * ac1) / (bc1 * ac2 - ac1 * bc2 + 1.e-10)
    aterm = (cc1 - bc1 * bterm) / (ac1 + 1.e-10)
    cterm = pco2y[i2] - aterm * eyy[i2] ** 2 - bterm * eyy[i2]
    pco2yq = cterm
    pco2yq = max(pco2yq, pco2b)
    pco2y[ic - 1] = (pco2yl + pco2yq) / 2.0

    pco2y[ic - 1] = max(pco2y[ic - 1], 0.01)

    return eyy, pco2y
def calc_photo_params(tlef, po2m, par, psrf, rstfac, rb, effcon, vmax25,
                      trop, slti, hlti, shti, hhti, trda, trdm, cint):

    # 定义 c3 和 c4
    c3 = 0.
    if effcon > 0.07:
        c3 = 1.
    c4 = 1. - c3

    # tlef 对 CO2 补偿点、Michaelis-Menten常数进行调整
    qt = 0.1 * (tlef - trop)
    kc = 30. * 2.1**qt
    ko = 30000. * 1.2**qt
    gammas = 0.5 * po2m / (2600. * 0.57**qt) * c3
    rrkk = kc * (1. + po2m / ko) * c3

    # 最大容量
    vm = vmax25 * 2.1**qt
    templ = 1. + np.exp(slti * (hlti - tlef))
    temph = 1. + np.exp(shti * (tlef - hhti))
    vm = vm / temph * rstfac * c3 + vm / (templ * temph) * rstfac * c4
    vm = vm * cint[0]

    rgas = 8.314467591
    jmax25 = 1.97 * vmax25
    jmax = jmax25 * np.exp(37.e3 * (tlef - trop) / (rgas * trop * tlef)) * \
           (1. + np.exp((710. * trop - 220.e3) / (rgas * trop))) / \
           (1. + np.exp((710. * tlef - 220.e3) / (rgas * tlef)))
    jmax = jmax * rstfac
    jmax = jmax * cint[1]

    epar = min(4.6e-6 * par * effcon, jmax)
    respcp = 0.015 * c3 + 0.025 * c4
    respc = respcp * vmax25 * 2.0**qt / (1. + np.exp(trda * (tlef - trdm))) * rstfac
    respc = respc * cint[0]

    omss = (vmax25 / 2.) * (1.8**qt) / templ * rstfac * c3 + \
           (vmax25 / 5.) * (1.8**qt) * rstfac * c4
    omss = omss * cint[0]

    tprcor = 44.6 * 273.16 * psrf / 1.013e5

    # 叶片单侧边界层水汽导度
    gbh2o = 1. / rb * tprcor / tlef

    return vm, epar, respc, omss, gbh2o, gammas, rrkk, c3, c4

def stomata  (co_lm, vmax25, effcon, slti, hlti, shti, hhti, trda, trdm, trop, g1, g0, gradm, binter, tm,
            psrf, po2m, pco2m, pco2a, ea, ei, tlef, par,o3coefv, o3coefg,
            rb, ra, rstfac, cint, assim, respc, rst):

    iterationtotal = 6
    eyy = np.zeros(iterationtotal)
    pco2y = np.zeros(iterationtotal)
    # ................................................................................................
    #   The soil color and reflectance is from the work:
    #   Peter J. Lawrence and Thomas N. Chase, 2007:
    #   Representing a MODIS consistent land surface in the Community Land Model (CLM 3.0):
    #   Part 1 generating MODIS consistent land surface parameters
    # ................................................................................................
    vm, epar, respc, omss, gbh2o, gammas, rrkk, c3, c4 = calc_photo_params(tlef, po2m, par, psrf, rstfac, rb, effcon, vmax25,
                                                                         trop, slti, hlti, shti, hhti, trda, trdm, cint)
    # 计算 bintc
    bintc = binter * max(0.1, rstfac)
    bintc = bintc * cint[2]  # 使用索引访问数组的第三个元素，Python 中索引从 0 开始

    # 计算 tprcor
    tprcor = 44.6 * 273.16 * psrf / 1.013e5

    # 计算 co2m 和 co2a
    co2m = pco2m / psrf  # mol mol-1
    co2a = pco2a / psrf  # mol mol-1

    # 计算 range
    range1 = pco2m * (1. - 1.6 / gradm) - gammas

    # for ic in range(iterationtotal):
    #     pco2y[ic] = 0.0
    #     eyy[ic] = 0.0

    for ic in range(iterationtotal):
        sortin(eyy, pco2y, range1, gammas, ic, iterationtotal)
        pco2i = pco2y[ic]
    # ................................................................................................
    #                           NET ASSIMILATION
    # !     the leaf assimilation (or gross photosynthesis) rate is described
    # !     as the minimum of three limiting rates:
    # !     omc: the efficiency of the photosynthetic enzyme system (Rubisco-limited);
    # !     ome: the amount of PAR captured by leaf chlorophyll;
    # !     oms: the capacity of the leaf to export or utilize the products of photosynthesis.
    # !     to aviod the abrupt transitions, two quadratic equations are used:
    # !             atheta*omp^2 - omp*(omc+ome) + omc*ome = 0
    # !         btheta*assim^2 - assim*(omp+oms) + omp*oms = 0
    # ................................................................................................
        atheta = 0.877
        btheta = 0.95

        omc = vm * (pco2i - gammas) / (pco2i + rrkk) * c3 + vm * c4
        ome = epar * (pco2i - gammas) / (pco2i + 2.0 * gammas) * c3 + epar * c4
        oms = omss * c3 + omss * pco2i * c4

        sqrtin_1 = max(0.0, (ome + omc) ** 2 - 4.0 * atheta * ome * omc)
        omp = ((ome + omc) - np.sqrt(sqrtin_1)) / (2.0 * atheta)
        sqrtin_2 = max(0.0, (omp + oms) ** 2 - 4.0 * btheta * omp * oms)
        assim = max(0.0, (oms + omp - np.sqrt(sqrtin_2)) / (2.0 * btheta))

        assimn = assim - respc  # mol m-2 s-1
    #!-----------------------------------------------------------------------
# !                      STOMATAL CONDUCTANCE
# !
# !  (1)   pathway for co2 flux
# !                                                  co2m
# !                                                   o
# !                                                   |
# !                                                   |
# !                                                   <  |
# !                                        1.37/gsh2o >  |  Ac-Rd-Rsoil
# !                                                   <  v
# !                                                   |
# !                                     <--- Ac-Rd    |
# !     o------/\/\/\/\/\------o------/\/\/\/\/\------o
# !    co2i     1.6/gsh2o     co2s    1.37/gbh2o     co2a
# !                                                   | ^
# !                                                   | | Rsoil
# !                                                   | |
# !
# !  (2)   pathway for water vapor flux
# !
# !                                                  em
# !                                                   o
# !                                                   |
# !                                                   |
# !                                                   <  ^
# !                                           1/gsh2o >  | Ea
# !                                                   <  |
# !                                                   |
# !                                     ---> Ec       !
# !     o------/\/\/\/\/\------o------/\/\/\/\/\------o
# !     ei       1/gsh2o      es       1/gbh2o       ea
# !                                                   | ^
# !                                                   | | Eg
# !                                                   | |
# !
# !  (3)   the relationship between net assimilation and tomatal conductance :
# !        gsh2o = m * An * [es/ei] / [pco2s/p] + b
# !        es = [gsh2o *ei + gbh2o * ea] / [gsh2o + gbh2o]
# !        ===>
# !        a*gsh2o^2 + b*gsh2o + c = 0
# !-----------------------------------------------------------------------
        co2s = co2a - 1.37 * assimn / gbh2o  # mol mol-1

        # Ensure that co2st is within a specific range
        co2st = min(co2s, co2a)
        co2st = max(co2st, 1.e-5)

        assmt = max(1.e-12, assimn)

        # Check if DEF_USE_MEDLYNST flag is defined
        if co_lm['DEF_USE_MEDLYNST']:
            # Calculate vapor pressure deficit
            vpd = max((ei - ea), 50.) * 1.e-3

            # Calculate quadratic coefficients for stomatal conductance
            acp = 1.6 * assmt / co2st  # in mol m-2 s-1
            aquad = 1.
            bquad = -2 * (g0 * 1.e-6 + acp) - (g1 * acp) ** 2 / (gbh2o * vpd)  # in mol m-2 s-1
            cquad = (g0 * 1.e-6) ** 2 + (2 * g0 * 1.e-6 + acp * (1 - g1 ** 2) / vpd) * acp  # in (mol m-2 s-1)**2

            # Solve quadratic equation to get stomatal conductance
            sqrtin = max(0., (bquad ** 2 - 4. * aquad * cquad))
            gsh2o = (-bquad + np.sqrt(sqrtin)) / (2. * aquad)
        else:
            # Calculate stomatal conductance using alternative method
            hcdma = ei * co2st / (gradm * assmt)

            # Calculate quadratic coefficients for stomatal conductance
            aquad = hcdma
            bquad = gbh2o * hcdma - ei - bintc * hcdma
            cquad = -gbh2o * (ea + hcdma * bintc)

            # Solve quadratic equation to get stomatal conductance
            sqrtin = max(0., (bquad ** 2 - 4. * aquad * cquad))
            gsh2o = (-bquad + np.sqrt(sqrtin)) / (2. * aquad)
            # print(bquad, sqrtin, aquad, hcdma, bintc, '------------=============')

            # Calculate saturation vapor pressure deficit
            es = (gsh2o - bintc) * hcdma  # pa
            es = min(es, ei)
            es = max(es, 1.e-2)

            # Calculate stomatal conductance
            gsh2o = es / hcdma + bintc  # mol m-2 s-1

        # Calculate input CO2 pressure
        pco2in = (co2s - 1.6 * assimn / gsh2o) * psrf  # pa

        # Calculate the difference in CO2 pressure
        eyy[ic] = pco2i - pco2in  # pa

        # Check if absolute value of eyy(ic) is less than 0.1 and exit the loop if true
        if abs(eyy[ic]) < 0.1:
            break

    # Convert stomatal conductance (mol m-2 s-1) to resistance rst (s m-1)
    # print(gsh2o, tlef, tprcor, '----========')
    rst = min(1.e6, 1. / (gsh2o * tlef / tprcor))  # s m-1

    return assim, respc, rst