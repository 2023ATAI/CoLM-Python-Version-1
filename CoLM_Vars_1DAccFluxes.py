import numpy as np
import CoLM_TurbulenceLEddy
import CoLM_FrictionVelocity

class CoLM_Vars_1DAccFluxes:
    def __init__(self, nl_colm, mpi, landpatch, spval):
        self.mpi = mpi
        self.landpatch = landpatch
        self.numpatch = landpatch.numpatch
        self.nl_colm = nl_colm
        self.spval = spval
        self.nac = 0.0  # number of accumulation
        self.nac_ln = None  # Initialize as empty, to be set later
        
        # 1D arrays
        self.a_us = None
        self.a_vs = None
        self.a_t = None
        self.a_q = None
        self.a_prc = None
        self.a_prl = None
        self.a_pbot = None
        self.a_frl = None
        self.a_solarin = None
        self.a_hpbl = None
        
        self.a_taux = None
        self.a_tauy = None
        self.a_fsena = None
        self.a_lfevpa = None
        self.a_fevpa = None
        self.a_fsenl = None
        self.a_fevpl = None
        self.a_etr = None
        self.a_fseng = None
        self.a_fevpg = None
        self.a_fgrnd = None
        self.a_sabvsun = None
        self.a_sabvsha = None
        self.a_sabg = None
        self.a_olrg = None
        self.a_rnet = None
        self.a_xerr = None
        self.a_zerr = None
        self.a_rsur = None
        self.a_rsur_se = None
        self.a_rsur_ie = None
        self.a_rsub = None
        self.a_rnof = None
        self.a_qintr = None
        self.a_qinfl = None
        self.a_qdrip = None
        self.a_rstfacsun = None
        self.a_rstfacsha = None
        self.a_gssun = None
        self.a_gssha = None
        self.a_rss = None
        self.a_wdsrf = None
        self.a_zwt = None
        self.a_wa = None
        self.a_wat = None
        self.a_wetwat = None
        self.a_assim = None
        self.a_respc = None
        self.a_assimsun = None
        self.a_assimsha = None
        self.a_etrsun = None
        self.a_etrsha = None
        
        self.a_qcharge = None
        
        self.a_t_grnd = None
        self.a_tleaf = None
        self.a_ldew = None
        self.a_ldew_rain = None
        self.a_ldew_snow = None
        self.a_scv = None
        self.a_snowdp = None
        self.a_fsno = None
        self.a_sigf = None
        self.a_green = None
        self.a_lai = None
        self.a_laisun = None
        self.a_laisha = None
        self.a_sai = None
        
        # 3D array
        self.a_alb = None
        
        self.a_emis = None
        self.a_z0m = None
        self.a_trad = None
        self.a_tref = None
        self.a_qref = None
        self.a_rain = None
        self.a_snow = None
        self.a_ndep_to_sminn = None
        self.a_abm = None
        self.a_gdp = None
        self.a_peatf = None
        self.a_hdm = None
        self.a_lnfm = None
        self.a_ozone = None
        
        self.a_t_soisno = None
        self.a_wliq_soisno = None
        self.a_wice_soisno = None
        self.a_h2osoi = None
        self.a_rootr = None
        self.a_BD_all = None
        self.a_wfc = None
        self.a_OM_density = None
        self.a_vegwp = None
        self.a_t_lake = None
        self.a_lake_icefrac = None
    
    def allocate_acc_fluxes(self, landelm):
        if self.mpi.p_is_worker and self.numpatch > 0:
            # 1D arrays
            self.a_us = np.zeros(self.numpatch)
            self.a_vs = np.zeros(self.numpatch)
            self.a_t = np.zeros(self.numpatch)
            self.a_q = np.zeros(self.numpatch)
            self.a_prc = np.zeros(self.numpatch)
            self.a_prl = np.zeros(self.numpatch)
            self.a_pbot = np.zeros(self.numpatch)
            self.a_frl = np.zeros(self.numpatch)
            self.a_solarin = np.zeros(self.numpatch)
            self.a_hpbl = np.zeros(self.numpatch)
            
            self.a_taux = np.zeros(self.numpatch)
            self.a_tauy = np.zeros(self.numpatch)
            self.a_fsena = np.zeros(self.numpatch)
            self.a_lfevpa = np.zeros(self.numpatch)
            self.a_fevpa = np.zeros(self.numpatch)
            self.a_fsenl = np.zeros(self.numpatch)
            self.a_fevpl = np.zeros(self.numpatch)
            self.a_etr = np.zeros(self.numpatch)
            self.a_fseng = np.zeros(self.numpatch)
            self.a_fevpg = np.zeros(self.numpatch)
            self.a_fgrnd = np.zeros(self.numpatch)
            self.a_sabvsun = np.zeros(self.numpatch)
            self.a_sabvsha = np.zeros(self.numpatch)
            self.a_sabg = np.zeros(self.numpatch)
            self.a_olrg = np.zeros(self.numpatch)
            self.a_rnet = np.zeros(self.numpatch)
            self.a_xerr = np.zeros(self.numpatch)
            self.a_zerr = np.zeros(self.numpatch)
            self.a_rsur = np.zeros(self.numpatch)
            self.a_rsur_se = np.zeros(self.numpatch)
            self.a_rsur_ie = np.zeros(self.numpatch)
            self.a_rsub = np.zeros(self.numpatch)
            self.a_rnof = np.zeros(self.numpatch)
            self.a_qintr = np.zeros(self.numpatch)
            self.a_qinfl = np.zeros(self.numpatch)
            self.a_qdrip = np.zeros(self.numpatch)
            self.a_rstfacsun = np.zeros(self.numpatch)
            self.a_rstfacsha = np.zeros(self.numpatch)
            self.a_gssun = np.zeros(self.numpatch)
            self.a_gssha = np.zeros(self.numpatch)
            self.a_rss = np.zeros(self.numpatch)
            self.a_wdsrf = np.zeros(self.numpatch)
            
            self.a_zwt = np.zeros(self.numpatch)
            self.a_wa = np.zeros(self.numpatch)
            self.a_wat = np.zeros(self.numpatch)
            self.a_wetwat = np.zeros(self.numpatch)
            self.a_assim = np.zeros(self.numpatch)
            self.a_respc = np.zeros(self.numpatch)

            self.a_assimsun = np.zeros(self.numpatch)
            self.a_assimsha = np.zeros(self.numpatch)
            self.a_etrsun = np.zeros(self.numpatch)
            self.a_etrsha = np.zeros(self.numpatch)
            
            self.a_qcharge = np.zeros(self.numpatch)
            
            self.a_t_grnd = np.zeros(self.numpatch)
            self.a_tleaf = np.zeros(self.numpatch)
            self.a_ldew_rain = np.zeros(self.numpatch)
            self.a_ldew_snow = np.zeros(self.numpatch)
            self.a_ldew = np.zeros(self.numpatch)
            self.a_scv = np.zeros(self.numpatch)
            self.a_snowdp = np.zeros(self.numpatch)
            self.a_fsno = np.zeros(self.numpatch)
            self.a_sigf = np.zeros(self.numpatch)
            self.a_green = np.zeros(self.numpatch)
            self.a_lai = np.zeros(self.numpatch)
            self.a_laisun = np.zeros(self.numpatch)
            self.a_laisha = np.zeros(self.numpatch)
            self.a_sai = np.zeros(self.numpatch)
            
            # 3D array
            self.a_alb = np.zeros((2, 2, self.numpatch))
            
            self.a_emis = np.zeros(self.numpatch)
            self.a_z0m = np.zeros(self.numpatch)
            self.a_trad = np.zeros(self.numpatch)
            self.a_tref = np.zeros(self.numpatch)
            self.a_qref = np.zeros(self.numpatch)
            self.a_rain = np.zeros(self.numpatch)
            self.a_snow = np.zeros(self.numpatch)
            self.a_ndep_to_sminn = np.zeros(self.numpatch)

            self.a_abm = np.zeros(self.numpatch)
            self.a_gdp = np.zeros(self.numpatch)
            self.a_peatf = np.zeros(self.numpatch)
            self.a_hdm = np.zeros(self.numpatch)
            self.a_lnfm = np.zeros(self.numpatch)
            
            self.a_ustar = np.zeros(self.numpatch)
            self.a_ustar2 = np.zeros(self.numpatch)
            self.a_tstar = np.zeros(self.numpatch)
            self.a_qstar = np.zeros(self.numpatch)
            self.a_zol = np.zeros(self.numpatch)
            self.a_rib = np.zeros(self.numpatch)
            self.a_fm = np.zeros(self.numpatch)
            self.a_fh = np.zeros(self.numpatch)
            self.a_fq = np.zeros(self.numpatch)
            
            self.a_us10m = np.zeros(self.numpatch)
            self.a_vs10m = np.zeros(self.numpatch)
            self.a_fm10m = np.zeros(self.numpatch)
            
            self.a_sr = np.zeros(self.numpatch)
            self.a_solvd = np.zeros(self.numpatch)
            self.a_solvi = np.zeros(self.numpatch)
            self.a_solnd = np.zeros(self.numpatch)
            self.a_solni = np.zeros(self.numpatch)
            self.a_srvd = np.zeros(self.numpatch)
            self.a_srvi = np.zeros(self.numpatch)
            self.a_srnd = np.zeros(self.numpatch)
            self.a_srni = np.zeros(self.numpatch)
            self.a_solvdln = np.zeros(self.numpatch)
            self.a_solviln = np.zeros(self.numpatch)
            self.a_solndln = np.zeros(self.numpatch)
            self.a_solniln = np.zeros(self.numpatch)
            self.a_srvdln = np.zeros(self.numpatch)
            self.a_srviln = np.zeros(self.numpatch)
            self.a_srndln = np.zeros(self.numpatch)
            self.a_srniln = np.zeros(self.numpatch)
            
            self.nac_ln = np.zeros(self.numpatch)

            if self.mpi.p_is_worker:
                if self.nl_colm['CROP']:
                    pass
                    # self.landpatch.elm_patch.build (landelm, landpatch, use_frac = True, sharedfrac = pctshrpch)
                else:
                    self.landpatch.elm_patch.build (landelm, self.landpatch.landpatch, use_frac = True)

    def flush_acc_fluxes(self, maxsnl, nl_soil, nvegwcs, nl_lake):
        if self.mpi.p_is_worker:
            self.nac = 0
            #
            if self.numpatch > 0:
                # flush the Fluxes for accumulation
                self.a_us[:] = self.spval
                self.a_vs[:] = self.spval
                self.a_t[:] = self.spval
                self.a_q[:] = self.spval
                self.a_prc[:] = self.spval
                self.a_prl[:] = self.spval
                self.a_pbot[:] = self.spval
                self.a_frl[:] = self.spval
                self.a_solarin[:] = self.spval
                self.a_hpbl[:] = self.spval

                self.a_taux[:] = self.spval
                self.a_tauy[:] = self.spval
                self.a_fsena[:] = self.spval
                self.a_lfevpa[:] = self.spval
                self.a_fevpa[:] = self.spval
                self.a_fsenl[:] = self.spval
                self.a_fevpl[:] = self.spval
                self.a_etr[:] = self.spval
                self.a_fseng[:] = self.spval
                self.a_fevpg[:] = self.spval
                self.a_fgrnd[:] = self.spval
                self.a_sabvsun[:] = self.spval
                self.a_sabvsha[:] = self.spval
                self.a_sabg[:] = self.spval
                self.a_olrg[:] = self.spval
                self.a_rnet[:] = self.spval
                self.a_xerr[:] = self.spval
                self.a_zerr[:] = self.spval
                self.a_rsur[:] = self.spval
                self.a_rsub[:] = self.spval
                self.a_rnof[:] = self.spval

                # Conditional compilation for CatchLateralFlow
                if self.nl_colm['CatchLateralFlow']:
                    self.a_xwsur[:] = self.spval
                    self.a_xwsub[:] = self.spval

                self.a_qintr[:] = self.spval
                self.a_qinfl[:] = self.spval
                self.a_qdrip[:] = self.spval
                self.a_rstfacsun[:] = self.spval
                self.a_rstfacsha[:] = self.spval
                self.a_gssun[:] = self.spval
                self.a_gssha[:] = self.spval
                self.a_rss[:] = self.spval

                self.a_wdsrf[:] = self.spval
                self.a_zwt[:] = self.spval
                self.a_wa[:] = self.spval
                self.a_wat[:] = self.spval
                self.a_wetwat[:] = self.spval
                self.a_assim[:] = self.spval
                self.a_respc[:] = self.spval
                self.a_assimsun[:] = self.spval
                self.a_assimsha[:] = self.spval
                self.a_etrsun[:] = self.spval
                self.a_etrsha[:] = self.spval

                self.a_qcharge[:] = self.spval

                self.a_t_grnd[:] = self.spval
                self.a_tleaf[:] = self.spval
                self.a_ldew_rain[:] = self.spval
                self.a_ldew_snow[:] = self.spval
                self.a_ldew[:] = self.spval
                self.a_scv[:] = self.spval
                self.a_snowdp[:] = self.spval
                self.a_fsno[:] = self.spval
                self.a_sigf[:] = self.spval
                self.a_green[:] = self.spval
                self.a_lai[:] = self.spval
                self.a_laisun[:] = self.spval
                self.a_laisha[:] = self.spval
                self.a_sai[:] = self.spval

                self.a_alb[:,:,:] = self.spval

                self.a_emis[:] = self.spval
                self.a_z0m[:] = self.spval
                self.a_trad[:] = self.spval
                self.a_tref[:] = self.spval
                self.a_qref[:] = self.spval
                self.a_rain[:] = self.spval
                self.a_snow[:] = self.spval

                # Conditional compilation for URBAN_MODEL
                # if self.nl_colm['URBAN_MODEL'] and numurban > 0:
                #     a_t_room[:] = self.spval
                #     a_tafu[:] = self.spval
                #     a_fhac[:] = self.spval
                #     a_fwst[:] = self.spval
                #     a_fach[:] = self.spval
                #     a_fahe[:] = self.spval
                #     a_fhah[:] = self.spval
                #     a_vehc[:] = self.spval
                #     a_meta[:] = self.spval
                #     a_senroof[:] = self.spval
                #     a_senwsun[:] = self.spval
                #     a_senwsha[:] = self.spval
                #     a_sengimp[:] = self.spval
                #     a_sengper[:] = self.spval
                #     a_senurbl[:] = self.spval
                #     a_lfevproof[:] = self.spval
                #     a_lfevpgimp[:] = self.spval
                #     a_lfevpgper[:] = self.spval
                #     a_lfevpurbl[:] = self.spval
                #     a_troof[:] = self.spval
                #     a_twall[:] = self.spval

                # Conditional compilation for BGC
                # if self.nl_colm['BGC']:
                #     a_leafc[:] = self.spval
                #     a_leafc_storage[:] = self.spval
                #     a_leafc_xfer[:] = self.spval
                #     a_frootc[:] = self.spval
                #     a_frootc_storage[:] = self.spval
                #     a_frootc_xfer[:] = self.spval
                #     a_livestemc[:] = self.spval
                #     a_livestemc_storage[:] = self.spval
                #     a_livestemc_xfer[:] = self.spval
                #     a_deadstemc[:] = self.spval
                #     a_deadstemc_storage[:] = self.spval
                #     a_deadstemc_xfer[:] = self.spval
                #     a_livecrootc[:] = self.spval
                #     a_livecrootc_storage[:] = self.spval
                #     a_livecrootc_xfer[:] = self.spval
                #     a_deadcrootc[:] = self.spval
                #     a_deadcrootc_storage[:] = self.spval
                #     a_deadcrootc_xfer[:] = self.spval
                #     a_grainc[:] = self.spval
                #     a_grainc_storage[:] = self.spval
                #     a_grainc_xfer[:] = self.spval
                #     a_leafn[:] = self.spval
                #     a_leafn_storage[:] = self.spval
                #     a_leafn_xfer[:] = self.spval
                #     a_frootn[:] = self.spval
                #     a_frootn_storage[:] = self.spval
                #     a_frootn_xfer[:] = self.spval
                #     a_livestemn[:] = self.spval
                #     a_livestemn_storage[:] = self.spval
                #     a_livestemn_xfer[:] = self.spval
                #     a_deadstemn[:] = self.spval
                #     a_deadstemn_storage[:] = self.spval
                #     a_deadstemn_xfer[:] = self.spval
                #     a_livecrootn[:] = self.spval
                #     a_livecrootn_storage[:] = self.spval
                #     a_livecrootn_xfer[:] = self.spval
                #     a_deadcrootn[:] = self.spval
                #     a_deadcrootn_storage[:] = self.spval
                #     a_deadcrootn_xfer[:] = self.spval
                #     a_grainn[:] = self.spval
                #     a_grainn_storage[:] = self.spval
                #     a_grainn_xfer[:] = self.spval
                #     a_retransn[:] = self.spval
                #     a_gpp[:] = self.spval
                #     a_downreg[:] = self.spval
                #     a_ar[:] = self.spval
                #     a_cwdprod[:] = self.spval
                #     a_cwddecomp[:] = self.spval
                #     a_hr[:] = self.spval
                #     a_fpg[:] = self.spval
                #     a_fpi[:] = self.spval
                #     a_gpp_enftemp[:] = self.spval
                #     a_gpp_enfboreal[:] = self.spval
                #     a_gpp_dnfboreal[:] = self.spval
                #     a_gpp_ebftrop[:] = self.spval
                #     a_gpp_ebftemp[:] = self.spval

                #     a_gpp_dbftrop = np.full(shape_1d, self.spval)
                #     a_gpp_dbftemp = np.full(shape_1d, self.spval)
                #     a_gpp_dbfboreal = np.full(shape_1d, self.spval)
                #     a_gpp_ebstemp = np.full(shape_1d, self.spval)
                #     a_gpp_dbstemp = np.full(shape_1d, self.spval)
                #     a_gpp_dbsboreal = np.full(shape_1d, self.spval)
                #     a_gpp_c3arcgrass = np.full(shape_1d, self.spval)
                #     a_gpp_c3grass = np.full(shape_1d, self.spval)
                #     a_gpp_c4grass = np.full(shape_1d, self.spval)
                #     a_leafc_enftemp = np.full(shape_1d, self.spval)
                #     a_leafc_enfboreal = np.full(shape_1d, self.spval)
                #     a_leafc_dnfboreal = np.full(shape_1d, self.spval)
                #     a_leafc_ebftrop = np.full(shape_1d, self.spval)
                #     a_leafc_ebftemp = np.full(shape_1d, self.spval)
                #     a_leafc_dbftrop = np.full(shape_1d, self.spval)
                #     a_leafc_dbftemp = np.full(shape_1d, self.spval)
                #     a_leafc_dbfboreal = np.full(shape_1d, self.spval)
                #     a_leafc_ebstemp = np.full(shape_1d, self.spval)
                #     a_leafc_dbstemp = np.full(shape_1d, self.spval)
                #     a_leafc_dbsboreal = np.full(shape_1d, self.spval)
                #     a_leafc_c3arcgrass = np.full(shape_1d, self.spval)
                #     a_leafc_c3grass = np.full(shape_1d, self.spval)
                #     a_leafc_c4grass = np.full(shape_1d, self.spval)

                #     a_O2_DECOMP_DEPTH_UNSAT = np.full(shape_2d, self.spval)
                #     a_CONC_O2_UNSAT = np.full(shape_2d, self.spval)

                #     # Conditional compilation for CROP
                #     if CROP:
                #         a_pdcorn = np.full(shape_1d, self.spval)
                #         a_pdswheat = np.full(shape_1d, self.spval)
                #         a_pdwwheat = np.full(shape_1d, self.spval)
                #         a_pdsoybean = np.full(shape_1d, self.spval)
                #         a_pdcotton = np.full(shape_1d, self.spval)
                #         a_pdrice1 = np.full(shape_1d, self.spval)
                #         a_pdrice2 = np.full(shape_1d, self.spval)
                #         a_pdsugarcane = np.full(shape_1d, self.spval)
                #         a_plantdate = np.full(shape_1d, self.spval)
                #         a_fertnitro_corn = np.full(shape_1d, self.spval)
                #         a_fertnitro_swheat = np.full(shape_1d, self.spval)
                #         a_fertnitro_wwheat = np.full(shape_1d, self.spval)
                #         a_fertnitro_soybean = np.full(shape_1d, self.spval)
                #         a_fertnitro_cotton = np.full(shape_1d, self.spval)
                #         a_fertnitro_rice1 = np.full(shape_1d, self.spval)
                #         a_fertnitro_rice2 = np.full(shape_1d, self.spval)
                #         a_fertnitro_sugarcane = np.full(shape_1d, self.spval)
                #         a_irrig_method_corn = np.full(shape_1d, self.spval)
                #         a_irrig_method_swheat = np.full(shape_1d, self.spval)
                #         a_irrig_method_wwheat = np.full(shape_1d, self.spval)
                #         a_irrig_method_soybean = np.full(shape_1d, self.spval)
                #         a_irrig_method_cotton = np.full(shape_1d, self.spval)
                #         a_irrig_method_rice1 = np.full(shape_1d, self.spval)
                #         a_irrig_method_rice2 = np.full(shape_1d, self.spval)
                #         a_irrig_method_sugarcane = np.full(shape_1d, self.spval)
                #         a_cphase = np.full(shape_1d, self.spval)
                #         a_vf = np.full(shape_1d, self.spval)
                #         a_gddmaturity = np.full(shape_1d, self.spval)
                #         a_gddplant = np.full(shape_1d, self.spval)
                #         a_hui = np.full(shape_1d, self.spval)
                #         a_cropprod1c = np.full(shape_1d, self.spval)
                #         a_cropprod1c_loss = np.full(shape_1d, self.spval)
                #         a_cropseedc_deficit = np.full(shape_1d, self.spval)
                #         a_grainc_to_cropprodc = np.full(shape_1d, self.spval)
                #         a_grainc_to_seed = np.full(shape_1d, self.spval)
                #         a_fert_to_sminn = np.full(shape_1d, self.spval)
                #         a_irrig_rate = np.full(shape_1d, self.spval)
                #         a_deficit_irrig = np.full(shape_1d, self.spval)
                #         a_sum_irrig = np.full(shape_1d, self.spval)
                #         a_sum_irrig_count = np.full(shape_1d, self.spval)

                #     a_ndep_to_sminn = np.full(shape_1d, self.spval)

                #     a_abm = np.full(shape_1d, self.spval)
                #     a_gdp = np.full(shape_1d, self.spval)
                #     a_peatf = np.full(shape_1d, self.spval)
                #     a_hdm = np.full(shape_1d, self.spval)
                #     a_lnfm = np.full(shape_1d, self.spval)
                self.a_ozone = np.full(self.numpatch, self.spval)

                self.a_t_soisno = np.full((nl_soil-maxsnl,self.numpatch), self.spval)
                self.a_wliq_soisno = np.full((nl_soil-maxsnl,self.numpatch), self.spval)
                self.a_wice_soisno = np.full((nl_soil-maxsnl,self.numpatch), self.spval)
                self.a_h2osoi = np.full((nl_soil,self.numpatch), self.spval)
                self.a_rootr = np.full((nl_soil,self.numpatch), self.spval)
                self.a_BD_all = np.full((nl_soil,self.numpatch), self.spval)
                self.a_wfc = np.full((nl_soil,self.numpatch), self.spval)
                self.a_OM_density = np.full((nl_soil,self.numpatch), self.spval)

                #     # Plant Hydraulic parameters
                self.a_vegwp = np.full((nvegwcs,self.numpatch), self.spval)

                self.a_t_lake = np.full((nl_lake,self.numpatch), self.spval)
                self.a_lake_icefrac = np.full((nl_lake,self.numpatch), self.spval)

                #     # Conditional compilation for BGC
                #     BGC = True  # Set this flag based on your conditions
                #     if BGC:
                #         a_litr1c_vr = np.full(shape_2d, self.spval)
                #         a_litr2c_vr = np.full(shape_2d, self.spval)
                #         a_litr3c_vr = np.full(shape_2d, self.spval)
                #         a_soil1c_vr = np.full(shape_2d, self.spval)
                #         a_soil2c_vr = np.full(shape_2d, self.spval)
                #         a_soil3c_vr = np.full(shape_2d, self.spval)
                #         a_cwdc_vr = np.full(shape_2d, self.spval)
                #         a_litr1n_vr = np.full(shape_2d, self.spval)
                #         a_litr2n_vr = np.full(shape_2d, self.spval)
                #         a_litr3n_vr = np.full(shape_2d, self.spval)
                #         a_soil1n_vr = np.full(shape_2d, self.spval)
                #         a_soil2n_vr = np.full(shape_2d, self.spval)
                #         a_soil3n_vr = np.full(shape_2d, self.spval)
                #         a_cwdn_vr = np.full(shape_2d, self.spval)
                #         a_sminn_vr = np.full(shape_2d, self.spval)

                a_ustar = np.full(self.numpatch, self.spval)
                a_ustar2 = np.full(self.numpatch, self.spval)
                a_tstar = np.full(self.numpatch, self.spval)
                a_qstar = np.full(self.numpatch, self.spval)
                a_zol = np.full(self.numpatch, self.spval)
                a_rib = np.full(self.numpatch, self.spval)
                a_fm = np.full(self.numpatch, self.spval)
                a_fh = np.full(self.numpatch, self.spval)
                a_fq = np.full(self.numpatch, self.spval)

                a_us10m = np.full(self.numpatch, self.spval)
                a_vs10m = np.full(self.numpatch, self.spval)
                a_fm10m = np.full(self.numpatch, self.spval)

                a_sr = np.full(self.numpatch, self.spval)
                a_solvd = np.full(self.numpatch, self.spval)
                a_solvi = np.full(self.numpatch, self.spval)
                a_solnd = np.full(self.numpatch, self.spval)
                a_solni = np.full(self.numpatch, self.spval)
                a_srvd = np.full(self.numpatch, self.spval)
                a_srvi = np.full(self.numpatch, self.spval)
                a_srnd = np.full(self.numpatch, self.spval)
                a_srni = np.full(self.numpatch, self.spval)
                a_solvdln = np.full(self.numpatch, self.spval)
                a_solviln = np.full(self.numpatch, self.spval)
                a_solndln = np.full(self.numpatch, self.spval)
                a_solniln = np.full(self.numpatch, self.spval)
                a_srvdln = np.full(self.numpatch, self.spval)
                a_srviln = np.full(self.numpatch, self.spval)
                a_srndln = np.full(self.numpatch, self.spval)
                a_srniln = np.full(self.numpatch, self.spval)

                nac_ln = np.zeros(self.numpatch)
                   
    def deallocate_acc_fluxes (self):
        if self.mpi.p_is_worker:
            if self.landpatch.numpatch > 0:

                del self.a_us     
                del self.a_vs     
                del self.a_t      
                del self.a_q      
                del self.a_prc    
                del self.a_prl    
                del self.a_pbot   
                del self.a_frl    
                del self.a_solarin
                del self.a_hpbl   

                del self.a_taux      
                del self.a_tauy      
                del self.a_fsena     
                del self.a_lfevpa    
                del self.a_fevpa     
                del self.a_fsenl     
                del self.a_fevpl     
                del self.a_etr       
                del self.a_fseng     
                del self.a_fevpg     
                del self.a_fgrnd     
                del self.a_sabvsun   
                del self.a_sabvsha   
                del self.a_sabg      
                del self.a_olrg      
                del self.a_rnet      
                del self.a_xerr      
                del self.a_zerr      
                del self.a_rsur      
                del self.a_rsub      
                del self.a_rnof      
    # if CatchLateralFlow
    #             del self.a_xwsur     
    #             del self.a_xwsub     
    # #
                del self.a_qintr     
                del self.a_qinfl     
                del self.a_qdrip     
                del self.a_rstfacsun 
                del self.a_rstfacsha 
                del self.a_gssun     
                del self.a_gssha     
                del self.a_rss       
                del self.a_wdsrf     

                del self.a_zwt       
                del self.a_wa        
                del self.a_wat       
                del self.a_wetwat    
                del self.a_assim     
                del self.a_respc     

                del self.a_assimsun   #1
                del self.a_assimsha   #1
                del self.a_etrsun     #1
                del self.a_etrsha     #1

                del self.a_qcharge   

                del self.a_t_grnd    
                del self.a_tleaf     
                del self.a_ldew_rain 
                del self.a_ldew_snow 
                del self.a_ldew      
                del self.a_scv       
                del self.a_snowdp    
                del self.a_fsno      
                del self.a_sigf      
                del self.a_green     
                del self.a_lai       
                del self.a_laisun    
                del self.a_laisha    
                del self.a_sai       

                del self.a_alb  

                del self.a_emis      
                del self.a_z0m       
                del self.a_trad      
                del self.a_tref      
                del self.a_qref      
                del self.a_rain      
                del self.a_snow      
    # if URBAN_MODEL
    #             if numurban > 0:
    #                del self.a_t_room    
    #                del self.a_tafu      
    #                del self.a_fhac      
    #                del self.a_fwst      
    #                del self.a_fach      
    #                del self.a_fahe      
    #                del self.a_fhah      
    #                del self.a_vehc      
    #                del self.a_meta      

    #                del self.a_senroof   
    #                del self.a_senwsun   
    #                del self.a_senwsha   
    #                del self.a_sengimp   
    #                del self.a_sengper   
    #                del self.a_senurbl   

    #                del self.a_lfevproof 
    #                del self.a_lfevpgimp 
    #                del self.a_lfevpgper 
    #                del self.a_lfevpurbl 

    #                del self.a_troof     
    #                del self.a_twall     
                
    #

            if self.nl_colm['BGC']:
                del self.a_leafc              
                del self.a_leafc_storage      
                del self.a_leafc_xfer         
                del self.a_frootc             
                del self.a_frootc_storage     
                del self.a_frootc_xfer        
                del self.a_livestemc          
                del self.a_livestemc_storage  
                del self.a_livestemc_xfer     
                del self.a_deadstemc          
                del self.a_deadstemc_storage  
                del self.a_deadstemc_xfer     
                del self.a_livecrootc         
                del self.a_livecrootc_storage 
                del self.a_livecrootc_xfer    
                del self.a_deadcrootc         
                del self.a_deadcrootc_storage 
                del self.a_deadcrootc_xfer    
                del self.a_grainc             
                del self.a_grainc_storage     
                del self.a_grainc_xfer        
                del self.a_leafn              
                del self.a_leafn_storage      
                del self.a_leafn_xfer         
                del self.a_frootn             
                del self.a_frootn_storage     
                del self.a_frootn_xfer        
                del self.a_livestemn          
                del self.a_livestemn_storage  
                del self.a_livestemn_xfer     
                del self.a_deadstemn          
                del self.a_deadstemn_storage  
                del self.a_deadstemn_xfer     
                del self.a_livecrootn         
                del self.a_livecrootn_storage 
                del self.a_livecrootn_xfer    
                del self.a_deadcrootn         
                del self.a_deadcrootn_storage 
                del self.a_deadcrootn_xfer    
                del self.a_grainn             
                del self.a_grainn_storage     
                del self.a_grainn_xfer        
                del self.a_retransn           
                del self.a_gpp                
                del self.a_downreg            
                del self.a_ar                 
                del self.a_cwdprod            
                del self.a_cwddecomp          
                del self.a_hr                 
                del self.a_fpg                
                del self.a_fpi                
                del self.a_gpp_enftemp         #1
                del self.a_gpp_enfboreal       #2
                del self.a_gpp_dnfboreal       #3
                del self.a_gpp_ebftrop         #4
                del self.a_gpp_ebftemp         #5
                del self.a_gpp_dbftrop         #6
                del self.a_gpp_dbftemp         #7
                del self.a_gpp_dbfboreal       #8
                del self.a_gpp_ebstemp         #9
                del self.a_gpp_dbstemp         #10
                del self.a_gpp_dbsboreal       #11
                del self.a_gpp_c3arcgrass      #12
                del self.a_gpp_c3grass         #13
                del self.a_gpp_c4grass         #14
                del self.a_leafc_enftemp       #1
                del self.a_leafc_enfboreal     #2
                del self.a_leafc_dnfboreal     #3
                del self.a_leafc_ebftrop       #4
                del self.a_leafc_ebftemp       #5
                del self.a_leafc_dbftrop       #6
                del self.a_leafc_dbftemp       #7
                del self.a_leafc_dbfboreal     #8
                del self.a_leafc_ebstemp       #9
                del self.a_leafc_dbstemp       #10
                del self.a_leafc_dbsboreal     #11
                del self.a_leafc_c3arcgrass    #12
                del self.a_leafc_c3grass       #13
                del self.a_leafc_c4grass       #14

                del self.a_O2_DECOMP_DEPTH_UNSAT 
                del self.a_CONC_O2_UNSAT         

    # if CROP
    #             del self.a_pdcorn             
    #             del self.a_pdswheat           
    #             del self.a_pdwwheat           
    #             del self.a_pdsoybean          
    #             del self.a_pdcotton           
    #             del self.a_pdrice1            
    #             del self.a_pdrice2            
    #             del self.a_pdsugarcane        
    #             del self.a_plantdate          
    #             del self.a_fertnitro_corn     
    #             del self.a_fertnitro_swheat   
    #             del self.a_fertnitro_wwheat   
    #             del self.a_fertnitro_soybean  
    #             del self.a_fertnitro_cotton   
    #             del self.a_fertnitro_rice1    
    #             del self.a_fertnitro_rice2    
    #             del self.a_fertnitro_sugarcane
    #             del self.a_irrig_method_corn     
    #             del self.a_irrig_method_swheat   
    #             del self.a_irrig_method_wwheat   
    #             del self.a_irrig_method_soybean  
    #             del self.a_irrig_method_cotton   
    #             del self.a_irrig_method_rice1    
    #             del self.a_irrig_method_rice2    
    #             del self.a_irrig_method_sugarcane
    #             del self.a_cphase             
    #             del self.a_hui                
    #             del self.a_vf                 
    #             del self.a_gddmaturity        
    #             del self.a_gddplant           
    #             del self.a_cropprod1c         
    #             del self.a_cropprod1c_loss    
    #             del self.a_cropseedc_deficit  
    #             del self.a_grainc_to_cropprodc
    #             del self.a_grainc_to_seed     
    #             del self.a_fert_to_sminn      

    #             del self.a_irrig_rate         
    #             del self.a_deficit_irrig      
    #             del self.a_sum_irrig          
    #             del self.a_sum_irrig_count    
    # #
    #             del self.a_ndep_to_sminn      

    #             del self.a_abm                
    #             del self.a_gdp                
    #             del self.a_peatf              
    #             del self.a_hdm                
    #             del self.a_lnfm               

    # #
    # # Ozone stress variables
    #             del self.a_ozone              
    # # END ozone stress variables

    #             del self.a_t_soisno    
    #             del self.a_wliq_soisno 
    #             del self.a_wice_soisno 
    #             del self.a_h2osoi      
    #             del self.a_rootr       
    #             del self.a_BD_all      
    #             del self.a_wfc         
    #             del self.a_OM_density  
    # #Plant Hydraulic parameters
    #             del self.a_vegwp       
    # #END plant hydraulic parameters
    #             del self.a_t_lake      
    #             del self.a_lake_icefrac
            if self.nl_colm['BGC']:
                del self.a_litr1c_vr   
                del self.a_litr2c_vr   
                del self.a_litr3c_vr   
                del self.a_soil1c_vr   
                del self.a_soil2c_vr   
                del self.a_soil3c_vr   
                del self.a_cwdc_vr     
                del self.a_litr1n_vr   
                del self.a_litr2n_vr   
                del self.a_litr3n_vr   
                del self.a_soil1n_vr   
                del self.a_soil2n_vr   
                del self.a_soil3n_vr   
                del self.a_cwdn_vr     
                del self.a_sminn_vr    
                del self.decomp_vr_tmp 
    #

                del self.a_ustar     
                del self.a_ustar2    
                del self.a_tstar     
                del self.a_qstar     
                del self.a_zol       
                del self.a_rib       
                del self.a_fm        
                del self.a_fh        
                del self.a_fq        

                del self.a_us10m     
                del self.a_vs10m     
                del self.a_fm10m     

                del self.a_sr        
                del self.a_solvd     
                del self.a_solvi     
                del self.a_solnd     
                del self.a_solni     
                del self.a_srvd      
                del self.a_srvi      
                del self.a_srnd      
                del self.a_srni      
                del self.a_solvdln   
                del self.a_solviln   
                del self.a_solndln   
                del self.a_solniln   
                del self.a_srvdln    
                del self.a_srviln    
                del self.a_srndln    
                del self.a_srniln    

                del self.nac_ln   

    def accumulate_fluxes(self, numelm, nl_colm_forcing, forcing, var_1forcing, var_1dfluxes, vtv, vti, const_physical):
        r_ustar2_e = 0.0
        fh2m = 0.0
        fq2m = 0.0
        r_fm10m_e = 0.0
        r_fm_e = 0.0
        r_fh_e = 0.0
        r_fq_e = 0.0

        if self.mpi.p_is_worker:
            if self.landpatch.numpatch > 0:
                self.nac += 1

                self.acc1d(var_1forcing.forc_us, self.a_us)
                self.acc1d(var_1forcing.forc_vs, self.a_vs)
                self.acc1d(var_1forcing.forc_t, self.a_t)
                self.acc1d(var_1forcing.forc_q, self.a_q)
                self.acc1d(var_1forcing.forc_prc, self.a_prc)
                self.acc1d(var_1forcing.forc_prl, self.a_prl)
                self.acc1d(var_1forcing.forc_pbot, self.a_pbot)
                self.acc1d(var_1forcing.forc_frl, self.a_frl)

                self.acc1d(var_1forcing.forc_sols, self.a_solarin)
                self.acc1d(var_1forcing.forc_soll, self.a_solarin)
                self.acc1d(var_1forcing.forc_solsd, self.a_solarin)
                self.acc1d(var_1forcing.forc_solld, self.a_solarin)
                if self.nl_colm['DEF_USE_CBL_HEIGHT']:
                    self.acc1d(var_1forcing.forc_hpbl, self.a_hpbl)

                self.acc1d(var_1dfluxes.taux, self.a_taux)
                self.acc1d(var_1dfluxes.tauy, self.a_tauy)
                self.acc1d(var_1dfluxes.fsena, self.a_fsena)
                self.acc1d(var_1dfluxes.lfevpa, self.a_lfevpa)
                self.acc1d(var_1dfluxes.fevpa, self.a_fevpa)
                self.acc1d(var_1dfluxes.fsenl, self.a_fsenl)
                self.acc1d(var_1dfluxes.fevpl, self.a_fevpl)
                self.acc1d(var_1dfluxes.etr, self.a_etr)
                self.acc1d(var_1dfluxes.fseng, self.a_fseng)
                self.acc1d(var_1dfluxes.fevpg, self.a_fevpg)
                self.acc1d(var_1dfluxes.fgrnd, self.a_fgrnd)
                self.acc1d(var_1dfluxes.sabvsun, self.a_sabvsun)
                self.acc1d(var_1dfluxes.sabvsha, self.a_sabvsha)
                self.acc1d(var_1dfluxes.sabg, self.a_sabg)
                self.acc1d(var_1dfluxes.olrg, self.a_olrg)

                if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                    var_1dfluxes.rnet = np.where(forcing.forcmask,
                                    var_1dfluxes.sabg +var_1dfluxes. sabvsun + var_1dfluxes.sabvsha - var_1dfluxes.olrg + var_1forcing.forc_frl,
                                    var_1dfluxes.rnet)
                else:
                    var_1dfluxes.rnet = np.where(vti.patchmask,
                                    var_1dfluxes.sabg + var_1dfluxes.sabvsun + var_1dfluxes.sabvsha - var_1dfluxes.olrg + var_1forcing.forc_frl,
                                    var_1dfluxes.rnet)

                self.acc1d(var_1dfluxes.rnet, self.a_rnet)
                self.acc1d(var_1dfluxes.xerr, self.a_xerr)
                self.acc1d(var_1dfluxes.zerr, self.a_zerr)
                self.acc1d(var_1dfluxes.rsur, self.a_rsur)

                self.acc1d(var_1dfluxes.rsub, self.a_rsub)
                self.acc1d(var_1dfluxes.rnof, self.a_rnof)

                self.acc1d(var_1dfluxes.qintr, self.a_qintr)
                self.acc1d(var_1dfluxes.qinfl, self.a_qinfl)
                self.acc1d(var_1dfluxes.qdrip, self.a_qdrip)

                self.acc1d(vtv.rstfacsun_out, self.a_rstfacsun)
                self.acc1d(vtv.rstfacsha_out, self.a_rstfacsha)

                self.acc1d(vtv.gssun_out, self.a_gssun)
                self.acc1d(vtv.gssha_out, self.a_gssha)

                self.acc1d(vtv.rss, self.a_rss)
                self.acc1d(vtv.wdsrf, self.a_wdsrf)
                self.acc1d(vtv.zwt, self.a_zwt)
                self.acc1d(vtv.wa, self.a_wa)
                self.acc1d(vtv.wat, self.a_wat)
                self.acc1d(vtv.wetwat, self.a_wetwat)
                self.acc1d(var_1dfluxes.assim, self.a_assim)
                self.acc1d(var_1dfluxes.respc, self.a_respc)
                self.acc1d(vtv.assimsun_out, self.a_assimsun)
                self.acc1d(vtv.assimsha_out, self.a_assimsha)
                self.acc1d(vtv.etrsun_out, self.a_etrsun)
                self.acc1d(vtv.etrsha_out, self.a_etrsha)

                self.acc1d(var_1dfluxes.qcharge, self.a_qcharge)

                self.acc1d(vtv.t_grnd, self.a_t_grnd)
                self.acc1d(vtv.tleaf, self.a_tleaf)
                self.acc1d(vtv.ldew_rain, self.a_ldew_rain)
                self.acc1d(vtv.ldew_snow, self.a_ldew_snow)
                self.acc1d(vtv.ldew, self.a_ldew)
                self.acc1d(vtv.scv, self.a_scv)
                self.acc1d(vtv.snowdp, self.a_snowdp)
                self.acc1d(vtv.fsno, self.a_fsno)
                self.acc1d(vtv.sigf, self.a_sigf)
                self.acc1d(vtv.green, self.a_green)
                self.acc1d(vtv.lai, self.a_lai)
                self.acc1d(vtv.laisun, self.a_laisun)
                self.acc1d(vtv.laisha, self.a_laisha)
                self.acc1d(vtv.sai, self.a_sai)

                self.acc3d(vtv.alb, self.a_alb,self.spval)

                self.acc1d(vtv.emis, self.a_emis)
                self.acc1d(vtv.z0m, self.a_z0m)

                r_trad = np.full(self.landpatch.numpatch, self.spval)
                for i in range(self.landpatch.numpatch):
                    if nl_colm_forcing['DEF_forcing']['has_missing_value'] and not forcing.forcmask[i]:
                        continue
                    if not vti.patchmask[i]:
                        continue
                    # print(var_1dfluxes.olrg[i],const_physical.stefnc,'----olrh-----')
                    r_trad[i] = (var_1dfluxes.olrg[i] / const_physical.stefnc) ** 0.25

                self.acc1d(r_trad, self.a_trad)

                self.acc1d(vtv.tref, self.a_tref)
                self.acc1d(vtv.qref, self.a_qref)

                self.acc1d(var_1forcing.forc_rain, self.a_rain)
                self.acc1d(var_1forcing.forc_snow, self.a_snow)

                if self.nl_colm['DEF_USE_OZONESTRESS']:
                    self.acc1d(var_1forcing.forc_ozone, var_1dfluxes.a_ozone)

                self.acc2d(vtv.t_soisno, self.a_t_soisno,self.spval)
                self.acc2d(vtv.wliq_soisno, self.a_wliq_soisno,self.spval)
                self.acc2d(vtv.wice_soisno, self.a_wice_soisno,self.spval)

                self.acc2d(vtv.h2osoi, self.a_h2osoi,self.spval)
                self.acc2d(vtv.rootr, self.a_rootr,self.spval)
                self.acc2d(vti.BD_all, self.a_BD_all,self.spval)
                self.acc2d(vti.wfc, self.a_wfc,self.spval)
                self.acc2d(vti.OM_density, self.a_OM_density,self.spval)
                if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
                    self.acc2d(vtv.vegwp, self.a_vegwp,self.spval)

                self.acc2d(vtv.t_lake, self.a_t_lake,self.spval)
                self.acc2d(vtv.lake_icefrac, self.a_lake_icefrac,self.spval)

                r_ustar = np.full(self.landpatch.numpatch, self.spval)
                r_ustar2 = np.full(self.landpatch.numpatch, self.spval)
                r_tstar = np.full(self.landpatch.numpatch, self.spval)
                r_qstar = np.full(self.landpatch.numpatch, self.spval)
                r_zol = np.full(self.landpatch.numpatch, self.spval)
                r_rib = np.full(self.landpatch.numpatch, self.spval)
                r_fm = np.full(self.landpatch.numpatch, self.spval)
                r_fh = np.full(self.landpatch.numpatch, self.spval)
                r_fq = np.full(self.landpatch.numpatch, self.spval)
                r_us10m = np.full(self.landpatch.numpatch, self.spval)
                r_vs10m = np.full(self.landpatch.numpatch, self.spval)
                r_fm10m = np.full(self.landpatch.numpatch, self.spval)

                for ielm in range(1, numelm + 1):
                    istt = int(self.landpatch.elm_patch.substt[ielm - 1])-1
                    iend = int(self.landpatch.elm_patch.subend[ielm - 1])

                    # filter = np.full(int(iend - istt + 1), True)

                    filter = vti.patchmask[istt:iend]

                    if nl_colm_forcing['DEF_forcing']['has_missing_value']:
                        filter &= forcing.forcmask[istt :iend]

                    if not np.any(filter):
                        continue

                    sumwt = np.sum(self.landpatch.elm_patch.subfrc[istt :iend][filter])

                    z0m_av  = sum(vtv.z0m        [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    hgt_u   = sum(var_1forcing.forc_hgt_u [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    hgt_t   = sum(var_1forcing.forc_hgt_t [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    hgt_q   = sum(var_1forcing.forc_hgt_q [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    us      = sum(var_1forcing.forc_us    [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    vs      = sum(var_1forcing.forc_vs    [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    tm      = sum(var_1forcing.forc_t     [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    qm      = sum(var_1forcing.forc_q     [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    psrf    = sum(var_1forcing.forc_psrf  [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    taux_e  = sum(var_1dfluxes.taux       [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    tauy_e  = sum(var_1dfluxes.tauy       [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    fsena_e = sum(var_1dfluxes.fsena      [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt
                    fevpa_e = sum(var_1dfluxes.fevpa      [istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend][filter]) / sumwt

                    if self.nl_colm['DEF_USE_CBL_HEIGHT']:
                        hpbl = np.sum(var_1forcing.forc_hpbl[istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend] * filter) / sumwt

                    z0h_av = z0m_av
                    z0q_av = z0m_av

                    displa_av = 2.0 / 3.0 * z0m_av / 0.07

                    hgt_u = max(hgt_u, 5.0 + displa_av)
                    hgt_t = max(hgt_t, 5.0 + displa_av)
                    hgt_q = max(hgt_q, 5.0 + displa_av)

                    zldis = hgt_u - displa_av

                    rhoair = (psrf - 0.378 * qm * psrf / (0.622 + 0.378 * qm)) / (const_physical.rgas * tm)

                    r_ustar_e = np.sqrt(max(1.e-6, np.sqrt(taux_e**2 + tauy_e**2) / rhoair))
                    r_tstar_e = -fsena_e / (rhoair * r_ustar_e) / const_physical.cpair
                    r_qstar_e = -fevpa_e / (rhoair * r_ustar_e)

                    thm = tm + 0.0098 * hgt_t
                    th = tm * (100000.0 / psrf) ** (const_physical.rgas / const_physical.cpair)
                    thv = th * (1.0 + 0.61 * qm)

                    r_zol_e = (zldis * const_physical.vonkar * const_physical.grav * 
                            (r_tstar_e * (1.0 + 0.61 * qm) + 0.61 * th * r_qstar_e) / 
                            (r_ustar_e ** 2 * thv))

                    if r_zol_e >= 0.0:  # stable
                        r_zol_e = min(2.0, max(r_zol_e, 1.e-6))
                    else:  # unstable
                        r_zol_e = max(-100.0, min(r_zol_e, -1.e-6))

                    beta = 1.0
                    zii = 1000.0

                    thvstar = r_tstar_e * (1.0 + 0.61 * qm) + 0.61 * th * r_qstar_e
                    ur = np.sqrt(us ** 2 + vs ** 2)
                    if r_zol_e >= 0.0:
                        um = max(ur, 0.1)
                    else:
                        if self.nl_colm['DEF_USE_CBL_HEIGHT']:
                            zii = max(5.0 * hgt_u, hpbl)
                        wc = (-const_physical.grav * r_ustar_e * thvstar * zii / thv) ** (1.0 / 3.0)
                        wc2 = beta ** 2 * (wc ** 2)
                        um = max(0.1, np.sqrt(ur ** 2 + wc2))

                    obu = zldis / r_zol_e

                    if self.nl_colm['DEF_USE_CBL_HEIGHT']:
                        r_ustar2_e, fh2m, fq2m, r_fm10m_e, r_fm_e, r_fh_e, r_fq_e = CoLM_TurbulenceLEddy.moninobuk_leddy(hgt_u, hgt_t, hgt_q, displa_av, z0m_av, z0h_av, z0q_av, obu, um, hpbl, const_physical.vonkar)
                    else:
                        r_ustar2_e, fh2m, fq2m, r_fm10m_e, r_fm_e, r_fh_e, r_fq_e = CoLM_FrictionVelocity.moninobuk(const_physical, hgt_u, hgt_t, hgt_q, displa_av, z0m_av, z0h_av, z0q_av, obu, um)

                    # bug found by chen qiying 2013/07/01
                    r_rib_e = r_zol_e / const_physical.vonkar * r_ustar2_e ** 2 / (const_physical.vonkar / r_fh_e * um ** 2)
                    r_rib_e = min(5.0, r_rib_e)

                    r_us10m_e = us / um * r_ustar2_e / const_physical.vonkar * r_fm10m_e
                    r_vs10m_e = vs / um * r_ustar2_e / const_physical.vonkar * r_fm10m_e

                    # Assign values from element (gridcell in latitude-longitude mesh) to patches.
                    r_ustar[istt:iend] = r_ustar_e
                    r_ustar2[istt:iend] = r_ustar2_e
                    r_tstar[istt:iend] = r_tstar_e
                    r_qstar[istt:iend] = r_qstar_e
                    r_zol[istt:iend] = r_zol_e
                    r_rib[istt:iend] = r_rib_e
                    r_fm[istt:iend] = r_fm_e
                    r_fh[istt:iend] = r_fh_e
                    r_fq[istt:iend] = r_fq_e
                    r_us10m[istt:iend] = r_us10m_e
                    r_vs10m[istt:iend] = r_vs10m_e
                    r_fm10m[istt:iend] = r_fm10m_e

                    del filter

                # Assuming self.acc1d is a function that processes these arrays, likely summing or averaging them
                self.acc1d(r_ustar, self.a_ustar)
                self.acc1d(r_ustar2, self.a_ustar2)
                self.acc1d(r_tstar, self.a_tstar)
                self.acc1d(r_qstar, self.a_qstar)
                self.acc1d(r_zol, self.a_zol)
                self.acc1d(r_rib, self.a_rib)
                self.acc1d(r_fm, self.a_fm)
                self.acc1d(r_fh, self.a_fh)
                self.acc1d(r_fq, self.a_fq)

                self.acc1d(r_us10m, self.a_us10m)
                self.acc1d(r_vs10m, self.a_vs10m)
                self.acc1d(r_fm10m, self.a_fm10m)

                del r_ustar, r_ustar2, r_tstar, r_qstar, r_zol, r_rib, r_fm, r_fh, r_fq
                del r_us10m, r_vs10m, r_fm10m


                self.acc1d(var_1dfluxes.sr, self.a_sr)
                self.acc1d(var_1dfluxes.solvd, self.a_solvd)
                self.acc1d(var_1dfluxes.solvi, self.a_solvi)
                self.acc1d(var_1dfluxes.solnd, self.a_solnd)
                self.acc1d(var_1dfluxes.solni, self.a_solni)
                self.acc1d(var_1dfluxes.srvd, self.a_srvd)
                self.acc1d(var_1dfluxes.srvi, self.a_srvi)
                self.acc1d(var_1dfluxes.srnd, self.a_srnd)
                self.acc1d(var_1dfluxes.srni, self.a_srni)
                self.acc1d(var_1dfluxes.solvdln, self.a_solvdln)
                self.acc1d(var_1dfluxes.solviln, self.a_solviln)
                self.acc1d(var_1dfluxes.solndln, self.a_solndln)
                self.acc1d(var_1dfluxes.solniln, self.a_solniln)
                self.acc1d(var_1dfluxes.srvdln, self.a_srvdln)
                self.acc1d(var_1dfluxes.srviln, self.a_srviln)
                self.acc1d(var_1dfluxes.srndln, self.a_srndln)
                self.acc1d(var_1dfluxes.srniln, self.a_srniln)

                for i in range(self.landpatch.numpatch):
                    if var_1dfluxes.solvdln[i] != self.spval:
                        self.nac_ln[i] += 1

    def acc1d(self, var, s):
        """
        Accumulates values from `var` into `s`, considering special values.

        Parameters:
        var (numpy.ndarray): Input array with values to accumulate.
        s (numpy.ndarray): In-place accumulation array.
        spval (float): Special value to be skipped during accumulation.
        """
        # Loop through the array
        for i in range(var.shape[0]):
            if var[i] != self.spval:
                if s[i] != self.spval:
                    s[i] += var[i]
                else:
                    s[i] = var[i]
        return s

    def acc2d(self, var, s, spval):
        """
        Python equivalent of the Fortran subroutine acc2d.
        
        Parameters:
        var : 2D numpy array (input)
        s : 2D numpy array (input/output)
        spval : float (special value used for comparison)
        """
        # Ensure s is modified in place
        for i2 in range(var.shape[1]):
            for i1 in range(var.shape[0]):
                if var[i1, i2] != spval:
                    if s[i1, i2] != spval:
                        s[i1, i2] += var[i1, i2]
                    else:
                        s[i1, i2] = var[i1, i2]
        return s

    def acc3d(self, var, s, spval):
        """
        Python equivalent of the Fortran subroutine acc3d.
        
        Parameters:
        var : 3D numpy array (input)
        s : 3D numpy array (input/output)
        spval : float (special value used for comparison)
        """
        # Ensure s is modified in place
        for i3 in range(var.shape[2]):
            for i2 in range(var.shape[1]):
                for i1 in range(var.shape[0]):
                    if var[i1, i2, i3] != spval:
                        if s[i1, i2, i3] != spval:
                            s[i1, i2, i3] += var[i1, i2, i3]
                        else:
                            s[i1, i2, i3] = var[i1, i2, i3]
        return s