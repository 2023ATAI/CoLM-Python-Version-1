import CoLM_Utils
import CoLM_Albland
import numpy as np
from CoLM_Hydro_SoilFunction import soil_vliq_from_psi,soil_psi_from_vliq,soil_hk_from_psi
from CoLM_Hydro_SoilWater import Hydro_SoilWater
import CoLM_Albedo


class IniTimeVariable(object):
    def __init__(self, mpi, nl_colm, landpatch, var_global, VTV) -> None:
        self.mpi = mpi
        self.nl_colm = nl_colm
        self.var_global = var_global
        self.VTV = VTV
        self.landpatch = landpatch

    def snow_ini(self, patchtype, maxsnl, snowdp):
        """
        Snow spatial discretization initially
        """

        z_soisno = [0.0] * (maxsnl + 1)
        dz_soisno = [0.0] * (maxsnl + 1)
        snl = 0

        if patchtype <= 3:  # non water bodies
            if snowdp < 0.01:
                snl = 0
            else:
                if 0.01 <= snowdp <= 0.03:
                    snl = -1
                    dz_soisno[0] = snowdp
                elif 0.03 < snowdp <= 0.04:
                    snl = -2
                    dz_soisno[-1] = snowdp / 2.
                    dz_soisno[0] = dz_soisno[-1]
                elif 0.04 < snowdp <= 0.07:
                    snl = -2
                    dz_soisno[-1] = 0.02
                    dz_soisno[0] = snowdp - dz_soisno[-1]
                elif 0.07 < snowdp <= 0.12:
                    snl = -3
                    dz_soisno[-2] = 0.02
                    dz_soisno[-1] = (snowdp - 0.02) / 2.
                    dz_soisno[0] = dz_soisno[-1]
                elif 0.12 < snowdp <= 0.18:
                    snl = -3
                    dz_soisno[-2] = 0.02
                    dz_soisno[-1] = 0.05
                    dz_soisno[0] = snowdp - dz_soisno[-2] - dz_soisno[-1]
                elif 0.18 < snowdp <= 0.29:
                    snl = -4
                    dz_soisno[-3] = 0.02
                    dz_soisno[-2] = 0.05
                    dz_soisno[-1] = (snowdp - dz_soisno[-3] - dz_soisno[-2]) / 2.
                    dz_soisno[0] = dz_soisno[-1]
                elif 0.29 < snowdp <= 0.41:
                    snl = -4
                    dz_soisno[-3] = 0.02
                    dz_soisno[-2] = 0.05
                    dz_soisno[-1] = 0.11
                    dz_soisno[0] = snowdp - dz_soisno[-3] - dz_soisno[-2] - dz_soisno[-1]
                elif 0.41 < snowdp <= 0.64:
                    snl = -5
                    dz_soisno[-4] = 0.02
                    dz_soisno[-3] = 0.05
                    dz_soisno[-2] = 0.11
                    dz_soisno[-1] = (snowdp - dz_soisno[-4] - dz_soisno[-3] - dz_soisno[-2]) / 2.
                    dz_soisno[0] = dz_soisno[-1]
                elif snowdp > 0.64:
                    snl = -5
                    dz_soisno[-4] = 0.02
                    dz_soisno[-3] = 0.05
                    dz_soisno[-2] = 0.11
                    dz_soisno[-1] = 0.23
                    dz_soisno[0] = snowdp - dz_soisno[-4] - dz_soisno[-3] - dz_soisno[-2] - \
                                   dz_soisno[-1]

                zi = 0.
                for i in range(snl + 1):
                    z_soisno[i] = zi - dz_soisno[i] / 2.
                    zi = -zi - dz_soisno[i]

        return z_soisno, dz_soisno, snl

    def ini_time_Var(self, nl_soil, zi_soi, nvegwcs, tfrz, denh2o, denice, maxsnl, tlai, tsai, ipatch, patchtype
        , porsl, psi0, hksati, soil_s_v_alb, soil_d_v_alb, soil_s_n_alb, soil_d_n_alb
        , z0m, zlnd, htop, z0mr, chil, rho, tau, z_soisno, dz_soisno
        , t_soisno, wliq_soisno, wice_soisno, smp, hk, zwt, wa,vegwp,gs0sun,gs0sha
        , t_grnd, tleaf, ldew, ldew_rain, ldew_snow, sag, scv
        , snowdp, fveg, fsno, sigf, green, lai, sai, coszen
        , snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi
        , mss_dst1, mss_dst2, mss_dst3, mss_dst4
        , alb, ssun, ssha, ssoi, ssno, ssno_lyr, thermk, extkb, extkd
        , trad, tref, qref, rst, emis, zol, rib
        , ustar, qstar, tstar, fm, fh, fq
        , use_soilini, nl_soil_ini, soil_z, soil_t, soil_w, use_snowini, snow_d
        , use_wtd, zwtmm, zc_soimm, zi_soimm, vliq_r, nprms, prms):


        wet = np.zeros(nl_soil)
        zi_soi_a = np.zeros(nl_soil + 1)

        if patchtype <= 5:
            # ----------------------------------------------------------------------------------------------------------------------
            # (1) SOIL temperature, water and SNOW
            #    Variables: t_soisno, wliq_soisno, wice_soisno snowdp, sag, scv, fsno, snl, z_soisno, dz_soisno
            # ----------------------------------------------------------------------------------------------------------------------
            if use_soilini:
                zi_soi_a = [i for i in range(zi_soi)]
                for j in range(nl_soil):
                    t_soisno[j] = CoLM_Utils.polint(soil_z, soil_t, nl_soil_ini, z_soisno[j])

                if patchtype <= 1:
                    for j in range(nl_soil):
                        wet[j] = CoLM_Utils.polint(soil_z, soil_w, nl_soil_ini, z_soisno[j])
                        wet[j] = min(max(wet[j], 0), porsl[j])

                        if zwt <= zi_soi_a[j - 1]:
                            wet[j] = porsl[j]
                        elif zwt <= zi_soi_a[j]:
                            wet[j] = ((zi_soi_a[j] - zwt) * porsl[j] + (zwt - zi_soi_a[j - 1]) *
                                      wet[j]) / (
                                             zi_soi_a[j] - zi_soi_a[j - 1])
                        if t_soisno[j] >= tfrz:
                            wliq_soisno[j] = wet[j] * dz_soisno[j] * denh2o
                            wice_soisno[j] = 0.0
                        else:
                            wliq_soisno[j] = 0.0
                            wice_soisno[j] = wet[j] * dz_soisno[j] * denice

                    # ! get wa from zwt
                    if zwt > zi_soi_a[nl_soil - 1]:
                        psi = psi0[nl_soil] - (zwt * 1000. - zi_soi_a[nl_soil - 1] * 1000.) * 0.5
                        vliq = soil_vliq_from_psi(psi, porsl[nl_soil - 1],
                                                           vliq_r[nl_soil - 1], psi0[nl_soil - 1],
                                                           nprms,
                                                           prms[:, nl_soil - 1])
                        # Calculate the amount of water above the deepest soil layer
                        wa = -(zwt * 1000. - zi_soi_a[nl_soil] * 1000.) * (porsl[nl_soil] - vliq)
                    else:
                        # If the water table is below the deepest soil layer, there is no water above the soil layers
                        wa = 0.
                # ! (2) wetland or (4) lake
                elif patchtype == 2 or patchtype == 4:
                    for j in range(nl_soil):
                        if t_soisno[j] >= tfrz:
                            wliq_soisno[j] = porsl[j] * dz_soisno[j] * denh2o
                            wice_soisno[j] = 0.0
                        else:
                            wliq_soisno[j] = 0.0
                            wice_soisno[j] = porsl[j] * dz_soisno[j] * denice
                    wa = 0

                # ! land ice
                elif patchtype == 3:
                    for j in range(nl_soil):
                        wliq_soisno[j] = 0.0
                        wice_soisno[j] = dz_soisno[j] * denice

                    wa= 0
                if not self.nl_colm['DEF_USE_VariablySaturatedFlow']:
                    wa = wa + 5000.0
            # ! soil temperature, water content
            else:
                for j in range(nl_soil):
                    if patchtype == 3:  # land ice
                        t_soisno[j] = 253.0
                        wliq_soisno[j] = 0.0
                        wice_soisno[j ] = dz_soisno[j] * denice
                    else:
                        t_soisno[j ] = 283.0
                        wliq_soisno[j] = dz_soisno[j] * porsl[j] * denh2o
                        wice_soisno[j ] = 0.0

            z0m = htop * z0mr

            if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
                pass

            if use_snowini:
                pass
            else:
                snowdp = 0.0
                sag = 0.0
                scv = 0.0
                fsno = 0.0
                snl = 0

                # Initialize arrays with placeholder values
                t_soisno[maxsnl+1:0] = -999.0
                wice_soisno[maxsnl+1:0] = 0.0
                wliq_soisno[maxsnl+1:0] = 0.0
                z_soisno[maxsnl+1:0] = 0.0
                dz_soisno[maxsnl+1:0] = 0.0
            # ----------------------------------------------------------------------------------------------------------------------
            # ! (2) SOIL aquifer and water table
            #       ! Variables: wa, zwt
            # ----------------------------------------------------------------------------------------------------------------------
            if not use_wtd:
                if not use_soilini:
                    if self.nl_colm['DEF_USE_VariablySaturatedFlow']:
                        wa = 0.0
                        zwt = zi_soimm[nl_soil-1] / 1000.0
                    else:
                        # Water table depth (initially at 1.0 m below the model bottom; wa when zwt
                        # is below the model bottom zi(nl_soil))
                        wa = 4800.0  # assuming aquifer capacity is 5000 mm
                        zwt = (25.0 + z_soisno[nl_soil-1]) + dz_soisno[
                            nl_soil-1] / 2.0 - wa / 1000.0 / 0.2  # to result in zwt = zi(nl_soil) + 1.0 m
            else:
                if patchtype <= 1:
                    HSW = Hydro_SoilWater(self.nl_colm)
                    wliq_soisno[:nl_soil], smp, hk, wa = HSW.get_water_equilibrium_state(zwtmm, nl_soil,wliq_soisno[:nl_soil], smp, hk, wa,
                                                zc_soimm, zi_soimm, porsl, vliq_r, psi0, hksati, nprms, prms)
                else:
                    wa = 0.0
                    zwt = 0.0
            # ----------------------------------------------------------------------------------------------------------------------
            #! (3) soil matrix potential hydraulic conductivity
            # ! Variables: smp, hk
            # ----------------------------------------------------------------------------------------------------------------------
            for j in range(nl_soil):
                if patchtype == 3 or t_soisno[j] < tfrz:  # land ice or frozen soil
                    smp[j] = 1.e3 * 0.3336e6 / 9.80616 * (t_soisno[j] - tfrz) / t_soisno[j]
                    hk[j] = 0.0
                else:
                    vliq = wliq_soisno[j] / (zi_soimm[j] - zi_soimm[j - 1])
                    smp[j] = soil_psi_from_vliq(vliq, porsl[j], vliq_r[j], psi0[j], nprms, prms[:, j])
                    hk[j] = soil_hk_from_psi(smp[j], psi0[j], hksati[j], nprms, prms[:, j])
            # ----------------------------------------------------------------------------------------------------------------------
            #! ! (4) Vegetation water and temperature
            # ! Variables: ldew_rain, ldew_snow, ldew, t_leaf, vegwp, gs0sun, gs0sha
            # ----------------------------------------------------------------------------------------------------------------------
            ldew_rain = 0.0
            ldew_snow = 0.0
            ldew = 0.0
            tleaf = t_soisno[0]
            if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
                self.VTV.vegwp[:self.var_global.nvegwcs] = -2.5e4
                gs0sun = 1.0e4
                gs0sha = 1.0e4

            if patchtype == 0:
                if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
                    pass
            # ----------------------------------------------------------------------------------------------------------------------
            # (5) Ground
            # Variables: t_grnd, wdsrf
            # ----------------------------------------------------------------------------------------------------------------------

            t_grnd = t_soisno[0]
            wdsrf = 0.

            # ----------------------------------------------------------------------------------------------------------------------
            # (6) Leaf area
            # Variables: sigf, lai, sai
            # ----------------------------------------------------------------------------------------------------------------------

            if patchtype == 0:
                if self.nl_colm['LULC_USGS'] or self.nl_colm['LULC_IGBP']:
                    sigf = fveg
                    lai = tlai[ipatch]
                    sai = tsai[ipatch] * sigf
                if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
                    pass
                else:
                    sigf = fveg
                    lai = tlai[ipatch]
                    sai = tsai[ipatch] * sigf

            # ----------------------------------------------------------------------------------------------------------------------
            # (7) SNICAR
            # Variables: snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi,
            #                  mss_dst1, mss_dst2, mss_dst3, mss_dst4
            # ----------------------------------------------------------------------------------------------------------------------

            snw_rds= 54.526
            mss_bcpho= 0.
            mss_bcphi= 0.
            mss_ocpho= 0.
            mss_ocphi= 0.
            mss_dst1= 0.
            mss_dst2= 0.
            mss_dst3= 0.
            mss_dst4= 0.

            # ----------------------------------------------------------------------------------------------------------------------
            # (8) surface albedo
            # Variables: alb, ssun, ssha, ssno, thermk, extkb, extkd
            # ----------------------------------------------------------------------------------------------------------------------

            wt = 0.0
            pg_snow = 0.0
            snofrz = 0.0
            ssw = min(1.0, 1.0e-3 * wliq_soisno[0] / dz_soisno[0])
            snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, sag, alb, ssun, ssha, thermk, extkb, extkd, ssoi, ssno, ssno_lyr=CoLM_Albedo.albland(self.nl_colm, self.var_global, tfrz, ipatch,patchtype,1800.,soil_s_v_alb,soil_d_v_alb,soil_s_n_alb,soil_d_n_alb,
            chil,rho,tau,fveg,green,lai,sai,max(0.001,coszen),
            wt,fsno,scv,scv,sag,ssw,pg_snow,273.15,t_grnd,t_soisno[:1],dz_soisno[:1],
            snl,wliq_soisno,wice_soisno,snw_rds,snofrz,
            mss_bcpho,mss_bcphi,mss_ocpho,mss_ocphi,
            mss_dst1,mss_dst2,mss_dst3,mss_dst4,
            alb,ssun,ssha,ssoi,ssno,ssno_lyr,thermk,extkb,extkd, self.mpi)
        #!ocean grid
        else:
            t_soisno = 300.
            wice_soisno = 0.
            wliq_soisno = 1000.
            z_soisno[maxsnl + 1: 0] = 0.
            dz_soisno[maxsnl + 1: 0] = 0.
            sigf = 0.
            fsno = 0.
            ldew_rain = 0.
            ldew_snow = 0.
            ldew = 0.
            scv = 0.
            sag = 0.
            snowdp = 0.
            tleaf = 300.
            if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
                vegwp[1: nvegwcs+1] = -2.5e4
                gs0sun = 1.0e4
                gs0sha = 1.0e4
            t_grnd = 300.
            oro = 0
            alb = CoLM_Albedo.albocean(oro, scv, coszen)
            ssun = 0.0
            ssha = 0.0
            ssoi = 0.0
            ssno = 0.0
            ssno_lyr = 0.0
            thermk = 0.0
            extkb = 0.0
            extkd = 0.0
        trad = t_grnd
        tref = t_grnd
        qref = 0.3
        emis = 1.0
        zol = -1.0
        rib = -0.1
        ustar = 0.25
        qstar = 0.001
        tstar = -1.5
        fm = np.log(30.0)
        fh = np.log(30.0)
        fq = np.log(30.0)