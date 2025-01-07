# ----------------------------------------------------------------------------------------------------------------------
#    Read in soil parameters; make unit conversion for soil physical process modeling;
#    ! soil parameters 8 layers => 10 layers
#    !
#    ! Original author: Yongjiu Dai, 03/2014
#    !
#    ! Revisions:
#    ! Nan Wei, 01/2019: read more parameters from mksrfdata results
#    ! Shupeng Zhang and Nan Wei, 01/2022: porting codes to parallel version
#    ! -----------------------------------------------------------------------------------------------------------------

import os
from CoLM_SoilColorRefl import soil_color_refl
import CoLM_NetCDFVectorBlk
import numpy as np

class SoilParametersReadin(object):
    def __init__(self, nl_colm, landpatch, mpi, vars_global) -> None:
        self.nl_colm = nl_colm
        self.landpatch = landpatch
        self.mpi = mpi
        self.vars_global = vars_global

    def soil_parameters_readin(self, dir_landdata, lc_year, srfdata, VT, gblock):
        cyear = f"{lc_year:04d}"
        landdir = (dir_landdata.strip() + '/soil/' + cyear.strip()).strip()

        if self.mpi.p_is_worker:
            if self.landpatch.numpatch > 0:
                self.soil_vf_quartz_mineral_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_vf_gravels_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_vf_om_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_vf_sand_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_wf_gravels_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_wf_sand_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_OM_density_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_BD_all_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_theta_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_psi_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_lambda_l = np.zeros(self.landpatch.numpatch)

                if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                    self.soil_theta_r_l = np.zeros(self.landpatch.numpatch)
                    self.soil_alpha_vgm_l = np.zeros(self.landpatch.numpatch)
                    self.soil_L_vgm_l = np.zeros(self.landpatch.numpatch)
                    self.soil_n_vgm_l = np.zeros(self.landpatch.numpatch)

                self.soil_k_s_l = np.zeros(self.landpatch.numpatch)
                self.soil_csol_l = np.zeros(self.landpatch.numpatch)
                self.soil_k_solids_l = np.zeros(self.landpatch.numpatch)
                self.soil_tksatu_l = np.zeros(self.landpatch.numpatch)
                self.soil_tksatf_l = np.zeros(self.landpatch.numpatch)
                self.soil_tkdry_l = np.zeros(self.landpatch.numpatch)
                self.soil_BA_alpha_l = np.zeros(self.landpatch.numpatch)
                self.soil_BA_beta_l = np.zeros(self.landpatch.numpatch)

        if self.nl_colm['USEMPI']:
            pass

        for nsl in range(8):

            if self.nl_colm['SinglePoint']:
                self.soil_vf_quartz_mineral_s_l[:] = srfdata.SITE_soil_vf_quartz_mineral[nsl]
                self.soil_vf_gravels_s_l[:] = srfdata.SITE_soil_vf_gravels[nsl]
                self.soil_vf_om_s_l[:] = srfdata.SITE_soil_vf_om[nsl]
                self.soil_vf_sand_s_l[:] = srfdata.SITE_soil_vf_sand[nsl]
                self.soil_wf_gravels_s_l[:] = srfdata.SITE_soil_wf_gravels[nsl]
                self.soil_wf_sand_s_l[:] = srfdata.SITE_soil_wf_sand[nsl]
                self.soil_OM_density_s_l[:] = srfdata.SITE_soil_OM_density[nsl]
                self.soil_BD_all_s_l[:] = srfdata.SITE_soil_BD_all[nsl]
                self.soil_theta_s_l[:] = srfdata.SITE_soil_theta_s[nsl]
                self.soil_psi_s_l[:] = srfdata.SITE_soil_psi_s[nsl]
                self.soil_lambda_l[:] = srfdata.SITE_soil_lambda[nsl]

                if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                    self.soil_theta_r_l[:] = srfdata.SITE_soil_theta_r[nsl]
                    self.soil_alpha_vgm_l[:] = srfdata.SITE_soil_alpha_vgm[nsl]
                    self.soil_L_vgm_l[:] = srfdata.SITE_soil_L_vgm[nsl]
                    self. soil_n_vgm_l[:] = srfdata.SITE_soil_n_vgm[nsl]

                self.soil_k_s_l[:] = srfdata.SITE_soil_k_s[nsl]
                self.soil_csol_l[:] = srfdata.SITE_soil_csol[nsl]
                self.soil_k_solids_l[:] = srfdata.SITE_soil_k_solids[nsl]
                self.soil_tksatu_l[:] = srfdata.SITE_soil_tksatu[nsl]
                self.soil_tksatf_l[:] = srfdata.SITE_soil_tksatf[nsl]
                self.soil_tkdry_l[:] = srfdata.SITE_soil_tkdry[nsl]
                self.soil_BA_alpha_l[:] = srfdata.SITE_soil_BA_alpha[nsl]
                self.soil_BA_beta_l[:] = srfdata.SITE_soil_BA_beta[nsl]

            else:
                pass

            if self.mpi.p_is_worker:
                for ipatch in range(self.landpatch.numpatch):
                    m = self.landpatch.landpatch.settyp[ipatch]
                    if m == 0:  # ocean
                        VT.vf_quartz[nsl, ipatch] = -1.e36
                        VT.vf_gravels[nsl, ipatch] = -1.e36
                        VT.vf_om[nsl, ipatch] = -1.e36
                        VT.vf_sand[nsl, ipatch] = -1.e36
                        VT.wf_gravels[nsl, ipatch] = -1.e36
                        VT.wf_sand[nsl, ipatch] = -1.e36
                        VT.OM_density[nsl, ipatch] = -1.e36
                        VT.BD_all[nsl, ipatch] = -1.e36
                        VT.wfc[nsl, ipatch] = -1.e36
                        VT.porsl[nsl, ipatch] = -1.e36
                        VT.psi0[nsl, ipatch] = -1.e36
                        VT.bsw[nsl, ipatch] = -1.e36
                        if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                            VT.theta_r[nsl, ipatch] = -1.e36
                            VT.alpha_vgm[nsl, ipatch] = -1.e36
                            VT.L_vgm[nsl, ipatch] = -1.e36
                            VT.n_vgm[nsl, ipatch] = -1.e36
                        VT.hksati[nsl, ipatch] = -1.e36
                        VT.csol[nsl, ipatch] = -1.e36
                        VT.k_solids[nsl, ipatch] = -1.e36
                        VT.dksatu[nsl, ipatch] = -1.e36
                        VT.dksatf[nsl, ipatch] = -1.e36
                        VT.dkdry[nsl, ipatch] = -1.e36
                        VT.BA_alpha[nsl, ipatch] = -1.e36
                        VT.BA_beta[nsl, ipatch] = -1.e36
                    else:  # non ocean
                        VT.vf_quartz[nsl, ipatch] = self.soil_vf_quartz_mineral_s_l[ipatch]
                        VT.vf_gravels[nsl, ipatch] = self.soil_vf_gravels_s_l[ipatch]
                        VT.vf_om[nsl, ipatch] = self.soil_vf_om_s_l[ipatch]
                        VT.vf_sand[nsl, ipatch] = self.soil_vf_sand_s_l[ipatch]
                        VT.wf_gravels[nsl, ipatch] = self.soil_wf_gravels_s_l[ipatch]
                        VT.wf_sand[nsl, ipatch] = self.soil_wf_sand_s_l[ipatch]
                        VT.OM_density[nsl, ipatch] = self.soil_OM_density_s_l[ipatch]
                        VT.BD_all[nsl, ipatch] = self.soil_BD_all_s_l[ipatch]
                        VT.porsl[nsl, ipatch] = self.soil_theta_s_l[ipatch]
                        VT.psi0[nsl, ipatch] = self.soil_psi_s_l[ipatch]
                        VT.bsw[nsl, ipatch] = 1.0/self.soil_lambda_l[ipatch]
                        VT.wfc[nsl, ipatch] = ((-339.9/self.soil_psi_s_l[ipatch])**(-1.0*self.soil_lambda_l[ipatch])) * self.soil_theta_s_l[ipatch]

                        if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                            VT.psi0[nsl, ipatch] = -10
                            VT.theta_r[nsl, ipatch] = self.soil_theta_r_l[ipatch]
                            VT.alpha_vgm[nsl, ipatch] = self.soil_alpha_vgm_l[ipatch]
                            VT.L_vgm[nsl, ipatch] = self.soil_L_vgm_l[ipatch]
                            VT.n_vgm[nsl, ipatch] = self.soil_n_vgm_l[ipatch]
                            VT.wfc[nsl, ipatch] = self.soil_theta_r_l[ipatch] + (self.soil_theta_s_l[ipatch] - self.soil_theta_r_l[ipatch])

                        VT.hksati[nsl, ipatch] = self.soil_k_s_l[ipatch] * 10./86400.  # cm/day -> mm/s
                        VT.csol[nsl, ipatch] = self.soil_csol_l[ipatch]                 # J/(m2 K)
                        VT.k_solids[nsl, ipatch] = self.soil_k_solids_l[ipatch]         # W/(m K)
                        VT.dksatu[nsl, ipatch] = self.soil_tksatu_l[ipatch]             # W/(m K)
                        VT.dksatf[nsl, ipatch] = self.soil_tksatf_l[ipatch]             # W/(m K)
                        VT.dkdry[nsl, ipatch] = self.soil_tkdry_l[ipatch]               # W/(m K)
                        VT.BA_alpha[nsl, ipatch] = self.soil_BA_alpha_l[ipatch]
                        VT.BA_beta[nsl, ipatch] = self.soil_BA_beta_l[ipatch]

# ----------------------------------------------------------------------------------------------------------------------
#       ! The parameters of the top NINTH soil layers were given by datasets
#       ! [0-0.045 (LAYER 1-2), 0.045-0.091, 0.091-0.166, 0.166-0.289,
#       !  0.289-0.493, 0.493-0.829, 0.829-1.383 and 1.383-2.296 m].
#       ! The NINTH layer's soil parameters will assigned to the bottom soil layer (2.296 - 3.8019m).
#    ! -----------------------------------------------------------------------------------------------------------------
        if self.mpi.p_is_worker:
            for nsl in range(9, 1, -1):
                VT.vf_quartz[nsl-1, :] = VT.vf_quartz[nsl - 2, :]
                VT.vf_gravels[nsl-1, :] = VT.vf_gravels[nsl - 2, :]
                VT.vf_om[nsl-1, :] = VT.vf_om[nsl - 2, :]
                VT.vf_sand[nsl-1, :] = VT.vf_sand[nsl - 2, :]
                VT.wf_gravels[nsl-1, :] = VT.wf_gravels[nsl - 2, :]
                VT.wf_sand[nsl-1, :] = VT.wf_sand[nsl - 2, :]
                VT.OM_density[nsl-1, :] = VT.OM_density[nsl - 2, :]
                VT.BD_all[nsl-1, :] = VT.BD_all[nsl - 2, :]
                VT.wfc[nsl-1, :] = VT.wfc[nsl - 2, :]
                VT.porsl[nsl-1, :] = VT.porsl[nsl - 2, :]
                VT.psi0[nsl-1, :] = VT.psi0[nsl - 2, :]
                VT.bsw[nsl-1, :] = VT.bsw[nsl - 2, :]

                if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                    VT.theta_r[nsl - 1, :] = VT.theta_r[nsl - 2, :]
                    VT.alpha_vgm[nsl - 1, :] = VT.alpha_vgm[nsl - 2, :]
                    VT.L_vgm[nsl - 1, :] = VT.L_vgm[nsl - 2, :]
                    VT.n_vgm[nsl - 1, :] = VT.n_vgm[nsl - 2, :]

                VT.hksati[nsl - 1, :] = VT.hksati[nsl - 2, :]
                VT.csol[nsl - 1, :] = VT.csol[nsl - 2, :]
                VT.k_solids[nsl - 1, :] = VT.k_solids[nsl - 2, :]
                VT.dksatu[nsl - 1, :] = VT.dksatu[nsl - 2, :]
                VT.dksatf[nsl - 1, :] = VT.dksatf[nsl - 2, :]
                VT.dkdry[nsl - 1, :] = VT.dkdry[nsl - 2, :]
                VT.BA_alpha[nsl - 1, :] = VT.BA_alpha[nsl - 2, :]
                VT.BA_beta[nsl - 1, :] = VT.BA_beta[nsl - 2, :]

            for nsl in range(self.vars_global.nl_soil, 9, -1):
                VT.vf_quartz[nsl - 1, :] = VT.vf_quartz[8, :]
                VT.vf_gravels[nsl - 1, :] = VT.vf_gravels[8, :]
                VT.vf_om[nsl - 1, :] = VT.vf_om[8, :]
                VT.vf_sand[nsl - 1, :] = VT.vf_sand[8, :]
                VT.wf_gravels[nsl - 1, :] = VT.wf_gravels[8, :]
                VT.wf_sand[nsl - 1, :] = VT.wf_sand[8, :]
                VT.OM_density[nsl - 1, :] = VT.OM_density[8, :]
                VT.BD_all[nsl - 1, :] = VT.BD_all[8, :]
                VT.wfc[nsl - 1, :] = VT.wfc[8, :]
                VT.porsl[nsl - 1, :] = VT.porsl[8, :]
                VT.psi0[nsl - 1, :] = VT.psi0[8, :]
                VT.bsw[nsl - 1, :] = VT.bsw[8, :]

                if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                    VT.theta_r[nsl - 1, :] = VT.theta_r[8, :]
                    VT.alpha_vgm[nsl - 1, :] = VT.alpha_vgm[8, :]
                    VT.L_vgm[nsl - 1, :] = VT.L_vgm[8, :]
                    VT.n_vgm[nsl - 1, :] = VT.n_vgm[8, :]

                VT.hksati[nsl - 1, :] = VT.hksati[8, :]
                VT.csol[nsl - 1, :] = VT.csol[8, :]
                VT.k_solids[nsl - 1, :] = VT.k_solids[8, :]
                VT.dksatu[nsl - 1, :] = VT.dksatu[8, :]
                VT.dksatf[nsl - 1, :] = VT.dksatf[8, :]
                VT.dkdry[nsl - 1, :] = VT.dkdry[8, :]
                VT.BA_alpha[nsl - 1, :] = VT.BA_alpha[8, :]
                VT.BA_beta[nsl - 1, :] = VT.BA_beta[8, :]

        #   Soil reflectance of broadband of visible(_v) and near-infrared(_n) of the sarurated(_s) and dry(_d) soil
        #   SCHEME 1: Guessed soil color type according to land cover classes

        if self.nl_colm['DEF_SOIL_REFL_SCHEME'] == 1:
            if self.mpi.p_is_worker:
                for ipatch in range(self.landpatch.numpatch):
                    m = self.landpatch.landpatch.settyp[ipatch]
                    VT.soil_s_v_alb[ipatch],VT.soil_d_v_alb[ipatch],VT.soil_s_n_alb[ipatch],VT.soil_d_n_alb[ipatch]= soil_color_refl(m)

         #   SCHEME 2: Read a global soil color map from CLM
        if self.nl_colm['DEF_SOIL_REFL_SCHEME'] == 2:
            if self.nl_colm['SinglePoint']:
                VT.soil_s_v_alb = srfdata.SITE_soil_s_v_alb
                VT.soil_d_v_alb = srfdata.SITE_soil_d_v_alb
                VT.soil_s_n_alb = srfdata.SITE_soil_s_n_alb
                VT.soil_d_n_alb = srfdata.SITE_soil_d_n_alb
            else:
                #  (1) Read in the albedo of visible of the saturated soil
                lndname = os.path.join(landdir.strip(), 'soil_s_v_alb_patches.nc')
                VT.soil_s_v_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'soil_s_v_alb', self.landpatch.landpatch, VT.soil_s_v_alb,self.nl_colm['USEMPI'],self.mpi,gblock)

                #  (2) Read in the albedo of visible of the dry soil
                lndname = os.path.join(landdir.strip(), 'soil_d_v_alb_patches.nc')
                VT.soil_d_v_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'soil_d_v_alb', self.landpatch.landpatch, VT.soil_d_v_alb,self.nl_colm['USEMPI'],self.mpi,gblock)

                #  (3) Read in the albedo of near infrared of the saturated soil
                lndname = os.path.join(landdir.strip(), 'soil_s_n_alb_patches.nc')
                VT.soil_s_n_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'soil_s_n_alb', self.landpatch.landpatch, VT.soil_s_n_alb,self.nl_colm['USEMPI'],self.mpi,gblock)

                #  (4) Read in the albedo of near infrared of the dry soil
                lndname = os.path.join(landdir.strip(), 'soil_d_n_alb_patches.nc')
                VT.soil_d_n_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'soil_d_n_alb', self.landpatch.landpatch, VT.soil_d_n_alb,self.nl_colm['USEMPI'],self.mpi,gblock)
        return VT