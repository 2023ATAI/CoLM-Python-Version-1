from CoLM_Const_Physical import denh2o,roverg,hvap,hsub,rgas,cpair, stefnc,denice,tfrz,vonkar,grav,cpliq,cpice
import numpy as np

   
    # ! Energy and Mass Balance Model of LAND ICE (GLACIER / ICE SHEET)
    # !
    # ! Original author: Yongjiu Dai, /05/2014/
    # !
    # ! REVISIONS:
    # ! Hua Yuan, 01/2023: added GLACIER_WATER_snicar() to account for SNICAR
    # !                    model effects on snow water [see snowwater_snicar()],
    # !                    snow layers combine [see snowlayerscombine_snicar()],
    # !                    snow layers divide  [see snowlayersdivide_snicar()]
    # !
    # ! Hua Yuan, 01/2023: added snow layer absorption in GLACIER_TEMP()n sensible heat
    #                    with canopy and ground for PFT and Plant Community (PC)
    
def glacier_temp(patchtype, lb, nl_ice, deltim, 
                 zlnd, zsno, capr, cnfac, 
                 forc_hgt_u, forc_hgt_t, forc_hgt_q, 
                 forc_us, forc_vs, forc_t, forc_q, 
                 forc_hpbl, 
                 forc_rhoair, forc_psrf, coszen, sabg, 
                 forc_frl, fsno, dz_icesno, z_icesno, 
                 zi_icesno, t_icesno, wice_icesno, wliq_icesno, 
                 scv, snowdp, imelt, taux, 
                 tauy, fsena, fevpa, lfevpa, 
                 fseng, fevpg, olrg, fgrnd, 
                 qseva, qsdew, qsubl, qfros, 
                 sm, tref, qref, trad, 
                 errore, emis, z0m, zol, 
                 rib, ustar, qstar, tstar, 
                 fm, fh, fq, pg_rain, 
                 pg_snow, t_precip, snofrz, sabg_snow_lyr):

    #=======================================================================
    # This is the main function to execute the calculation
    # of thermal processes and surface fluxes of the land ice (glacier and ice sheet)
    #
    # Original author : Yongjiu Dai and Nan Wei, /05/2014/
    # Modified by Nan Wei, 07/2017/  interaction btw prec and land ice
    #=======================================================================

    # Initialize local variables
    cgrnd, cgrndl, cgrnds, degdT, dqgdT, eg, egsmax, egidif, emg, fact, htvp, qg, qsatg, qsatgdT, qred, thm, th, thv, t_grnd, t_icesno_bef, tinc, ur, xmf = [None] * 23

    # [1] Initial set and propositional variables
    t_grnd = t_icesno[lb]
    t_icesno_bef = np.copy(t_icesno)

    emg = 0.97

    htvp = hvap
    if wliq_icesno[lb] <= 0 and wice_icesno[lb] > 0:
        htvp = hsub

    thm = forc_t + 0.0098 * forc_hgt_t
    th = forc_t * (100000. / forc_psrf)**(rgas / cpair)
    thv = th * (1. + 0.61 * forc_q)
    ur = max(0.1, np.sqrt(forc_us**2 + forc_vs**2))

    # [2] Specific humidity and its derivative at ground surface
    qred = 1.
    qsadv(t_grnd, forc_psrf, eg, degdT, qsatg, qsatgdT)

    qg = qred * qsatg
    dqgdT = qred * qsatgdT

    # [3] Compute sensible and latent fluxes and their derivatives with respect to ground temperature using ground temperatures from previous time step
    groundfluxes_glacier(zlnd, zsno, forc_hgt_u, forc_hgt_t, forc_hgt_q, 
                         forc_us, forc_vs, forc_t, forc_q, forc_rhoair, forc_psrf, 
                         ur, thm, th, thv, t_grnd, qg, dqgdT, htvp, 
                         forc_hpbl, fsno, cgrnd, cgrndl, cgrnds, 
                         taux, tauy, fsena, fevpa, fseng, fevpg, tref, qref, 
                         z0m, zol, rib, ustar, qstar, tstar, fm, fh, fq)

    # [4] Ground temperature
    groundtem_glacier(patchtype, lb, nl_ice, deltim, 
                      capr, cnfac, dz_icesno, z_icesno, zi_icesno, 
                      t_icesno, wice_icesno, wliq_icesno, scv, snowdp, 
                      forc_frl, sabg, sabg_snow_lyr, fseng, fevpg, cgrnd, htvp, emg, 
                      imelt, snofrz, sm, xmf, fact, pg_rain, pg_snow, t_precip)

    # [5] Correct fluxes to present ice temperature
    t_grnd = t_icesno[lb]
    tinc = t_icesno[lb] - t_icesno_bef[lb]
    fseng += tinc * cgrnds
    fevpg += tinc * cgrndl

    egsmax = (wice_icesno[lb] + wliq_icesno[lb]) / deltim
    egidif = max(0., fevpg - egsmax)
    fevpg = min(fevpg, egsmax)
    fseng += htvp * egidif

    fsena = fseng
    fevpa = fevpg
    lfevpa = htvp * fevpg

    qseva, qsubl, qfros, qsdew = 0., 0., 0., 0.

    if fevpg >= 0:
        qseva = min(wliq_icesno[lb] / deltim, fevpg)
        qsubl = fevpg - qseva
    else:
        if t_grnd < tfrz:
            qfros = abs(fevpg)
        else:
            qsdew = abs(fevpg)

    fgrnd = sabg + emg * forc_frl - emg * stefnc * t_icesno_bef[lb]**3 * (t_icesno_bef[lb] + 4. * tinc) - (fseng + fevpg * htvp) + cpliq * pg_rain * (t_precip - t_icesno[lb]) + cpice * pg_snow * (t_precip - t_icesno[lb])
    olrg = (1. - emg) * forc_frl + emg * stefnc * t_icesno_bef[lb]**4 + 4. * emg * stefnc * t_icesno_bef[lb]**3 * tinc
    emis = emg
    trad = (olrg / stefnc)**0.25

    # [6] Energy balance error
    errore = sabg + forc_frl - olrg - fsena - lfevpa - xmf + cpliq * pg_rain * (t_precip - t_icesno[lb]) + cpice * pg_snow * (t_precip - t_icesno[lb])
    for j in range(lb, nl_ice + 1):
        errore -= (t_icesno[j] - t_icesno_bef[j]) / fact[j]

    if abs(errore) > 0.2:
        print("GLACIER_TEMP: energy balance violation")
        print(f"Error: {errore}, sabg: {sabg}, forc_frl: {forc_frl}, olrg: {olrg}, fsena: {fsena}, lfevpa: {lfevpa}, xmf: {xmf}, pg_rain: {pg_rain}, pg_snow: {pg_snow}, t_precip: {t_precip}, t_icesno[lb]: {t_icesno[lb]}")
        
def glacier_water(nl_ice, maxsnl, deltim, 
              z_icesno, dz_icesno, zi_icesno, t_icesno, 
              wliq_icesno, wice_icesno, pg_rain, pg_snow, 
              sm, scv, snowdp, imelt, 
              fiold, snl, qseva, qsdew, 
              qsubl, qfros, gwat, 
              ssi, wimp, forc_us, forc_vs):

    # Initialize local variables
    lb = snl + 1
    if lb >= 1:
        gwat = pg_rain + sm - qseva
    else:
        snowwater(lb, deltim, ssi, wimp, 
                  pg_rain, qseva, qsdew, qsubl, qfros, 
                  dz_icesno[lb:0], wice_icesno[lb:0], wliq_icesno[lb:0], gwat)

    # Surface runoff and infiltration
    if snl < 0:
        # Compaction rate for snow
        # Natural compaction and metamorphosis. The compaction rate
        # is recalculated for every new timestep
        lb = snl + 1  # lower bound of array
        snowcompaction(lb, deltim, 
                       imelt[lb:0], fiold[lb:0], t_icesno[lb:0], 
                       wliq_icesno[lb:0], wice_icesno[lb:0], forc_us, forc_vs, dz_icesno[lb:0])

        # Combine thin snow elements
        lb = maxsnl + 1
        snowlayerscombine(lb, snl, 
                          z_icesno[lb:1], dz_icesno[lb:1], zi_icesno[lb-1:1], 
                          wliq_icesno[lb:1], wice_icesno[lb:1], t_icesno[lb:1], scv, snowdp)

        # Divide thick snow elements
        if snl < 0:
            snowlayersdivide(lb, snl, 
                             z_icesno[lb:0], dz_icesno[lb:0], zi_icesno[lb-1:0], 
                             wliq_icesno[lb:0], wice_icesno[lb:0], t_icesno[lb:0])

    if snl > maxsnl:
        wice_icesno[maxsnl+1:snl] = 0.
        wliq_icesno[maxsnl+1:snl] = 0.
        t_icesno[maxsnl+1:snl] = 0.
        z_icesno[maxsnl+1:snl] = 0.
        dz_icesno[maxsnl+1:snl] = 0.

    if lb >= 1:
        wliq_icesno[1] = max(1.e-8, wliq_icesno[1] + qsdew * deltim)
        wice_icesno[1] = max(1.e-8, wice_icesno[1] + (qfros - qsubl) * deltim)    