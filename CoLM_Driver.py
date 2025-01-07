import numpy as np
import CoLMMAIN
# from CoLM_Const_Physical import CoLM_Const_Physical


def CoLMDRIVER(nl_colm,nl_colm_forcing,  const_physical, gblock, mpi, VTV, VTI, isgreenwich, landpft, cama_var, idate, deltim, dolai, doalb, dosst, vtv, vti,
               forcing1, varforcing, fluxe, const_lc, numpatch, forcing, var_global):
    # Parallel section
    maxsnl = var_global.maxsnl
    forcmask = forcing.forcmask
    # print('=================')
    for i in range(numpatch):

        # if i < 274:
        #     continue
        # print(i, vti.patchclass[i] - 1, numpatch, '---------')

        # if i == 275:
        #     return

        if nl_colm_forcing['DEF_forcing']['has_missing_value'] and not forcmask[i]:
            continue
        
        if not vti.patchmask[i]:
            continue
        
        m = vti.patchclass[i]-1
        steps_in_one_deltim = 1
        
        if m == var_global.WATERBODY and vtv.snowdp[i] > 0.0:
            steps_in_one_deltim = int(np.ceil(deltim / 1800.0))
        
        deltim_phy = deltim / steps_in_one_deltim


        if not nl_colm['DEF_URBAN_RUN'] or m != nl_colm['URBAN']:
            for k in range(steps_in_one_deltim):
                # print(i,k,numpatch,steps_in_one_deltim,'------driver------')

                fluxe.oroflag[i], vtv.z_sno[:, i], vtv.dz_sno[:, i], vtv.t_soisno[:, i], \
                vtv.wliq_soisno[:, i], vtv.wice_soisno[:, i], vtv.hk[:, i], vtv.smp[:, i],vtv.t_lake[:, i], vtv.lake_icefrac[:, i], vtv.savedtke1[i], \
                vtv.vegwp[:, i], vtv.gs0sun[i], vtv.gs0sha[i], vtv.lai_old[i], vtv.o3uptakesun[i],\
                vtv.o3uptakesha[i], varforcing.forc_ozone[i],vtv.t_grnd[i], vtv.tleaf[i], vtv.ldew[i], vtv.ldew_rain[i], vtv.ldew_snow[i],\
                vtv.sag[i], vtv.scv[i], vtv.snowdp[i], vtv.zwt[i], vtv.wdsrf[i], vtv.wa[i],vtv. wetwat[i], vtv.snw_rds[:, i], \
                vtv.mss_bcpho[:, i], vtv.mss_bcphi[:, i], vtv.mss_ocpho[:, i],\
                vtv.mss_ocphi[:, i], vtv.mss_dst1[:, i], vtv.mss_dst2[:, i], vtv.mss_dst3[:, i],\
                vtv.mss_dst4[:, i],vtv.ssno_lyr[:, :, :, i], vtv.fveg[i], vtv.fsno[i], vtv.sigf[i], vtv.green[i],\
                vtv.lai[i], vtv.sai[i], vtv.coszen[i], vtv.alb[:, :, i], vtv.ssun[:, :, i], vtv.ssha[:, :, i],\
                vtv.ssoi[:, :, i], vtv.ssno[:, :, i],vtv.thermk[i], vtv.extkb[i], vtv.extkd[i],\
                vtv.laisun[i], vtv.laisha[i], vtv.rstfacsun_out[i], vtv.rstfacsha_out[i], vtv.gssun_out[i], vtv.gssha_out[i],\
                vtv.wat[i],vtv.rss[i],vtv.rootr[:, i], vtv.rootflux[:, i],vtv.h2osoi[:, i],vtv.assimsun_out[i], vtv.etrsun_out[i], vtv.assimsha_out[i], vtv.etrsha_out[i], \
                fluxe.taux[i], fluxe.tauy[i], fluxe.fsena[i], fluxe.fevpa[i], fluxe.lfevpa[i],\
                fluxe.fsenl[i], fluxe.fevpl[i], fluxe.etr[i], fluxe.fseng[i], fluxe.fevpg[i], fluxe.olrg[i], \
                fluxe.fgrnd[i], fluxe.xerr[i], fluxe.zerr[i],vtv.tref[i], vtv.qref[i], vtv.trad[i], \
                fluxe.rsur[i], fluxe.rsur_se[i], fluxe.rsur_ie[i], fluxe.rnof[i], fluxe.qintr[i], fluxe.qinfl[i],\
                fluxe.qdrip[i],fluxe.qcharge[i],vtv.rst[i], fluxe.assim[i], fluxe.respc[i], fluxe.sabvsun[i], fluxe.sabvsha[i], fluxe.sabg[i],\
                fluxe.sr[i], fluxe.solvd[i], fluxe.solvi[i], fluxe.solnd[i], fluxe.solni[i], fluxe.srvd[i], fluxe.srvi[i],\
                fluxe.srnd[i], fluxe.srni[i], fluxe.solvdln[i], fluxe.solviln[i], fluxe.solndln[i], fluxe.solniln[i],\
                fluxe.srvdln[i], fluxe.srviln[i], fluxe.srndln[i], fluxe.srniln[i],varforcing.forc_rain[i], varforcing.forc_snow[i], \
                vtv.emis[i], vtv.z0m[i], vtv.zol[i], vtv.rib[i], vtv.ustar[i], vtv.qstar[i],\
                vtv.tstar[i], vtv.fm[i], vtv.fh[i], vtv.fq[i] = CoLMMAIN.colm_main(nl_colm, var_global, const_physical, const_lc, gblock, mpi, VTV, VTI, isgreenwich, landpft,
                    i, idate, vtv.coszen[i], deltim_phy,
                    vti.patchlonr[i], vti.patchlatr[i], vti.patchclass[i], vti.patchtype[i],
                    doalb, dolai, dosst, fluxe.oroflag[i],
                    # Soil information and lake depth (placeholders for actual data)
                    vti.soil_s_v_alb[i], vti.soil_d_v_alb[i], vti.soil_s_n_alb[i], vti.soil_d_n_alb[i],
                    vti.vf_quartz[:, i], vti.vf_gravels[:, i], vti.vf_om[:, i], vti.vf_sand[:, i],
                    vti.wf_gravels[:, i], vti.wf_sand[:, i], vti.porsl[:, i], vti.psi0[:, i],
                    vti.bsw[:, i],
                    vti.theta_r[:, i],   vti.alpha_vgm[:, i], vti.n_vgm[:, i],vti.L_vgm[:, i],
                    vti.sc_vgm [:, i],   vti.fc_vgm[:, i],
                    vti.hksati[:, i], vti.csol[:, i], vti.k_solids[:, i], vti.dksatu[:, i],
                    vti.dksatf[:, i], vti.dkdry[:, i], vti.BA_alpha[:, i], vti.BA_beta[:, i],
                    const_lc.rootfr[:, m], vti.lakedepth[i], vti.dz_lake[:, i],
                                            vti.topostd[i], vti.BVIC[0,i],
                                            # Flood variables (if defined)
                    # cama_var.flddepth_cama[i], cama_var.fldfrc_cama[i], cama_var.fevpg_fld[i], cama_var.finfg_fld[i],
                    # Vegetation information (placeholders for actual data)
                    vti.htop[i], vti.hbot[i], const_lc.sqrtdi[m], const_lc.effcon[m], const_lc.vmax25[m],
                    const_lc.kmax_sun[m], const_lc.kmax_sha[m], const_lc.kmax_xyl[m], const_lc.kmax_root[m],
                    const_lc.psi50_sun[m], const_lc.psi50_sha[m], const_lc.psi50_xyl[m], const_lc.psi50_root[m],
                    const_lc.ck[m], const_lc.slti[m], const_lc.hlti[m], const_lc.shti[m], const_lc.hhti[m], const_lc.trda[m], const_lc.trdm[m],
                    const_lc.trop[m], const_lc.g1[m], const_lc.g0[m], const_lc.gradm[m], const_lc.binter[m], const_lc.extkn[m], const_lc.chil[m],
                    const_lc.rho[:, :, m], const_lc.tau[:, :, m],
                    # Atmospheric forcing (placeholders for actual data)
                    varforcing.forc_pco2m[i], varforcing.forc_po2m[i], varforcing.forc_us[i], varforcing.forc_vs[i],
                    varforcing.forc_t[i], varforcing.forc_q[i], varforcing.forc_prc[i], varforcing.forc_prl[i],
                    varforcing.forc_rain[i], varforcing.forc_snow[i], varforcing.forc_psrf[i], varforcing.forc_pbot[i],
                    varforcing.forc_sols[i], varforcing.forc_soll[i], varforcing.forc_solsd[i], varforcing.forc_solld[i],
                    varforcing.forc_frl[i], varforcing.forc_hgt_u[i], varforcing.forc_hgt_t[i], varforcing.forc_hgt_q[i],
                    varforcing.forc_rhoair[i], varforcing.forc_hpbl[i], varforcing.forc_aerdep[:, i],
                    # Land surface variables required for restart (placeholders for actual data)
                    vtv.z_sno[:, i], vtv.dz_sno[:, i], vtv.t_soisno[:, i],
                    vtv.wliq_soisno[:, i], vtv.wice_soisno[:, i], vtv.smp[:, i],
                    vtv.hk[:, i], vtv.t_grnd[i], vtv.tleaf[i], vtv.ldew[i], vtv.ldew_rain[i], vtv.ldew_snow[i],
                    vtv.sag[i], vtv.scv[i], vtv.snowdp[i], vtv.fveg[i], vtv.fsno[i], vtv.sigf[i], vtv.green[i],
                    vtv.lai[i], vtv.sai[i], vtv.alb[:, :, i], vtv.ssun[:, :, i], vtv.ssha[:, :, i],
                    vtv.ssoi[:, :, i], vtv.ssno[:, :, i],vtv.thermk[i], vtv.extkb[i], vtv.extkd[i],
                    vtv.vegwp[:, i], vtv.gs0sun[i], vtv.gs0sha[i], vtv.lai_old[i], vtv.o3uptakesun[i],
                    vtv.o3uptakesha[i], varforcing.forc_ozone[i], vtv.zwt[i], vtv.wdsrf[i], vtv.wa[i],vtv. wetwat[i],
                    vtv.t_lake[:, i], vtv.lake_icefrac[:, i], vtv.savedtke1[i], vtv.snw_rds[:, i],
                    vtv.ssno_lyr[:, :, :, i], vtv.mss_bcpho[:, i], vtv.mss_bcphi[:, i], vtv.mss_ocpho[:, i],
                    vtv.mss_ocphi[:, i], vtv.mss_dst1[:, i], vtv.mss_dst2[:, i], vtv.mss_dst3[:, i],
                    vtv.mss_dst4[:, i], vtv.laisun[i], vtv.laisha[i], vtv.rootr[:, i], vtv.rootflux[:, i],
                    vtv.rss[i], vtv.rstfacsun_out[i], vtv.rstfacsha_out[i], vtv.gssun_out[i], vtv.gssha_out[i],
                    vtv.assimsun_out[i], vtv.etrsun_out[i], vtv.assimsha_out[i], vtv.etrsha_out[i],
                    vtv.h2osoi[:, i], vtv.wat[i], fluxe.taux[i], fluxe.tauy[i], fluxe.fsena[i], fluxe.fevpa[i], fluxe.lfevpa[i],
                    fluxe.fsenl[i], fluxe.fevpl[i], fluxe.etr[i], fluxe.fseng[i], fluxe.fevpg[i], fluxe.olrg[i], fluxe.fgrnd[i],
                    vtv.trad[i], vtv.tref[i], vtv.qref[i], fluxe.rsur[i], fluxe.rsur_se[i], fluxe.rsur_ie[i], fluxe.rnof[i], fluxe.qintr[i], fluxe.qinfl[i],
                    fluxe.qdrip[i], vtv.rst[i], fluxe.assim[i], fluxe.respc[i], fluxe.sabvsun[i], fluxe.sabvsha[i], fluxe.sabg[i],
                    fluxe.sr[i], fluxe.solvd[i], fluxe.solvi[i], fluxe.solnd[i], fluxe.solni[i], fluxe.srvd[i], fluxe.srvi[i],
                    fluxe.srnd[i], fluxe.srni[i], fluxe.solvdln[i], fluxe.solviln[i], fluxe.solndln[i], fluxe.solniln[i],
                    fluxe.srvdln[i], fluxe.srviln[i], fluxe.srndln[i], fluxe.srniln[i], fluxe.qcharge[i], fluxe.xerr[i], fluxe.zerr[i],
                    vti.zlnd, vti.zsno, vti.csoilc, vti.dewmx, vti.wtfact, vti.capr, vti.cnfac, vti.ssi, vti.wimp, vti.pondmx, vti.smpmax,
                    vti.smpmin, vti.trsmx0, vti.tcrit, vtv.emis[i], vtv.z0m[i], vtv.zol[i], vtv.rib[i], vtv.ustar[i], vtv.qstar[i],
                    vtv.tstar[i], vtv.fm[i], vtv.fh[i], vtv.fq[i], vti.vic_b_infilt, vti.vic_Dsmax, vti.vic_Ds, vti.vic_Ws, vti.vic_c
                )
    

       

