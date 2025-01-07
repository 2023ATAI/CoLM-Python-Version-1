import numpy as np
import os
from CoLM_RangeCheck import check_vector_data
import CoLM_TimeManager
# from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
import CoLM_NetCDFVectorBlk


class Vars_PFTimeVariables(object):
    def __init__(self) -> None:
        pass

    def read_pftime_variables(self, landpft, file_restart):
        # Call the equivalent read functions
        self.tleaf_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tleaf_p', landpft, self.tleaf_p)  #
        self.ldew_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ldew_p', landpft, self.ldew_p)  #
        self.ldew_rain_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ldew_rain_p', landpft,
                                                                 self.ldew_rain_p)  # depth of rain on foliage [mm]
        self.ldew_snow_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ldew_snow_p', landpft,
                                                                 self.ldew_snow_p)  # depth of snow on foliage [mm]
        self.sigf_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sigf_p', landpft, self.sigf_p)  #
        self.tlai_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tlai_p', landpft, self.tlai_p)  #
        self.lai_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'lai_p', landpft, self.lai_p)  #
        self.tsai_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tsai_p', landpft, self.tsai_p)  #
        self.sai_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sai_p', landpft, self.sai_p)  #
        self.ssun_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssun_p', 2, 2, landpft, self.ssun_p)  #
        self.ssha_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssha_p', 2, 2, landpft, self.ssha_p)  #
        self.thermk_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'thermk_p', landpft, self.thermk_p)  #
        self.fshade_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fshade_p', landpft, self.fshade_p)  #
        self.extkb_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'extkb_p', landpft, self.extkb_p)  #
        self.extkd_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'extkd_p', landpft, self.extkd_p)  #
        self.tref_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tref_p', landpft, self.tref_p)  #
        self.qref_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'qref_p', landpft, self.qref_p)  #
        self.rst_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'rst_p', landpft, self.rst_p)  #
        self.z0m_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'z0m_p', landpft, self.z0m_p)  #
        if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            self.vegwp_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vegwp_p', self.var_global.nvegwcs,
                                                                 landpft, self.vegwp_p)  #
            self.gs0sun_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'gs0sun_p', landpft, self.gs0sun_p)  #
            self.gs0sha_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'gs0sha_p', landpft, self.gs0sha_p)  #

        if self.nl_colm['DEF_USE_OZONESTRESS']:
            self.lai_old_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'lai_old_p', landpft, self.lai_old_p,
                                                                   defval=0.0)
            self.o3uptakesun_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'o3uptakesun_p', landpft,
                                                                       self.o3uptakesun_p, defval=0.0)
            self.o3uptakesha_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'o3uptakesha_p', landpft,
                                                                       self.o3uptakesha_p, defval=0.0)

        if self.nl_colm['DEF_USE_IRRIGATION']:
            self.irrig_method_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_p', landpft,
                                                                        self.irrig_method_p, defval=1)

def convert_to_three_digits(number):
    return f"{number:03d}"

def convert_to_five_digits(number):
    return f"{number:05d}"

class Vars_TimeVariables(object):
    def __init__(self, nl_colm, mpi, landpatch, var_global, gblock) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.landpatch = landpatch
        self.var_global = var_global
        self.gblock = gblock

        if self.mpi.p_is_worker:
            if (self.landpatch.numpatch > 0):
                # print(self.var_global.maxsnl,'-+-+-+-+-+-+-++-+-+')
                self.z_sno = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.dz_sno = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.t_soisno = np.full((self.var_global.nl_soil - self.var_global.maxsnl, self.landpatch.numpatch),
                                        var_global.spval)
                self.wliq_soisno = np.full((self.var_global.nl_soil - self.var_global.maxsnl, self.landpatch.numpatch),
                                           var_global.spval)
                self.wice_soisno = np.full((self.var_global.nl_soil - self.var_global.maxsnl, self.landpatch.numpatch),
                                           var_global.spval)
                self.smp = np.full((self.var_global.nl_soil, self.landpatch.numpatch), var_global.spval)
                self.hk = np.full((self.var_global.nl_soil, self.landpatch.numpatch), var_global.spval)
                self.h2osoi = np.full((self.var_global.nl_soil, self.landpatch.numpatch), var_global.spval)
                self.rootr = np.full((self.var_global.nl_soil, self.landpatch.numpatch), var_global.spval)
                self.rootflux = np.full((self.var_global.nl_soil, self.landpatch.numpatch), var_global.spval)
                # !Plant Hydraulic variables
                self.vegwp = np.full((self.var_global.nvegwcs, self.landpatch.numpatch), var_global.spval)
                self.gs0sun = np.full((self.landpatch.numpatch), var_global.spval)
                self.gs0sha = np.full((self.landpatch.numpatch), var_global.spval)
                # !END plant hydraulic variables
                # !Ozone Stress variables
                self.o3coefv_sun = np.full((self.landpatch.numpatch), var_global.spval)
                self.o3coefv_sha = np.full((self.landpatch.numpatch), var_global.spval)
                self.o3coefg_sun = np.full((self.landpatch.numpatch), var_global.spval)
                self.o3coefg_sha = np.full((self.landpatch.numpatch), var_global.spval)
                self.lai_old = np.full((self.landpatch.numpatch), var_global.spval)
                self.o3uptakesun = np.full((self.landpatch.numpatch), var_global.spval)
                self.o3uptakesha = np.full((self.landpatch.numpatch), var_global.spval)
                # !End ozone stress variables

                self.rstfacsun_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.rstfacsha_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.gssun_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.gssha_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.assimsun_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.assimsha_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.etrsun_out = np.full((self.landpatch.numpatch), var_global.spval)
                self.etrsha_out = np.full((self.landpatch.numpatch), var_global.spval)

                self.t_grnd = np.full((self.landpatch.numpatch), var_global.spval)
                self.tleaf = np.full((self.landpatch.numpatch), var_global.spval)
                self.ldew = np.full((self.landpatch.numpatch), var_global.spval)
                self.ldew_rain = np.full((self.landpatch.numpatch), var_global.spval)
                self.ldew_snow = np.full((self.landpatch.numpatch), var_global.spval)
                self.sag = np.full((self.landpatch.numpatch), var_global.spval)
                self.scv = np.full((self.landpatch.numpatch), var_global.spval)
                self.snowdp = np.full((self.landpatch.numpatch), var_global.spval)
                self.fveg = np.full((self.landpatch.numpatch), var_global.spval)
                self.fsno = np.full((self.landpatch.numpatch), var_global.spval)
                self.sigf = np.full((self.landpatch.numpatch), var_global.spval)
                self.green = np.full((self.landpatch.numpatch), var_global.spval)
                self.tlai = np.full((self.landpatch.numpatch), var_global.spval)
                self.lai = np.full((self.landpatch.numpatch), var_global.spval)
                self.laisun = np.full((self.landpatch.numpatch), var_global.spval)
                self.laisha = np.full((self.landpatch.numpatch), var_global.spval)
                self.tsai = np.full((self.landpatch.numpatch), var_global.spval)
                self.sai = np.full((self.landpatch.numpatch), var_global.spval)
                self.coszen = np.full((self.landpatch.numpatch), var_global.spval)
                self.alb = np.full((2, 2, self.landpatch.numpatch), var_global.spval)
                self.ssun = np.full((2, 2, self.landpatch.numpatch), var_global.spval)
                self.ssha = np.full((2, 2, self.landpatch.numpatch), var_global.spval)
                self.ssoi = np.full((2, 2, self.landpatch.numpatch), var_global.spval)
                self.ssno = np.full((2, 2, self.landpatch.numpatch), var_global.spval)
                self.thermk = np.full((self.landpatch.numpatch), var_global.spval)
                self.extkb = np.full((self.landpatch.numpatch), var_global.spval)
                self.extkd = np.full((self.landpatch.numpatch), var_global.spval)
                self.zwt = np.full((self.landpatch.numpatch), var_global.spval)
                self.wa = np.full((self.landpatch.numpatch), var_global.spval)
                self.wetwat = np.full((self.landpatch.numpatch), var_global.spval)
                self.wat = np.full((self.landpatch.numpatch), var_global.spval)
                self.wdsrf = np.full((self.landpatch.numpatch), var_global.spval)
                self.rss = np.full((self.landpatch.numpatch), var_global.spval)
                self.t_lake = np.full((self.var_global.nl_lake, self.landpatch.numpatch), var_global.spval)
                self.lake_icefrac = np.full((self.var_global.nl_lake, self.landpatch.numpatch), var_global.spval)
                self.savedtke1 = np.full((self.landpatch.numpatch), var_global.spval)

                self.snw_rds = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_bcpho = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_bcphi = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_ocpho = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_ocphi = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_dst1 = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_dst2 = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_dst3 = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.mss_dst4 = np.full((0 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                self.ssno_lyr = np.full((2, 2, 1 - self.var_global.maxsnl, self.landpatch.numpatch), var_global.spval)
                # print(self.ssno_lyr.shape,'*-*-*-*-*-*-*-')
                self.trad = np.full((self.landpatch.numpatch), var_global.spval)
                self.tref = np.full((self.landpatch.numpatch), var_global.spval)
                self.qref = np.full((self.landpatch.numpatch), var_global.spval)
                self.rst = np.full((self.landpatch.numpatch), var_global.spval)
                self.emis = np.full((self.landpatch.numpatch), var_global.spval)
                self.z0m = np.full((self.landpatch.numpatch), var_global.spval)
                self.displa = np.full((self.landpatch.numpatch), var_global.spval)
                self.zol = np.full((self.landpatch.numpatch), var_global.spval)
                self.rib = np.full((self.landpatch.numpatch), var_global.spval)
                self.ustar = np.full((self.landpatch.numpatch), var_global.spval)
                self.qstar = np.full((self.landpatch.numpatch), var_global.spval)
                self.tstar = np.full((self.landpatch.numpatch), var_global.spval)
                self.fm = np.full((self.landpatch.numpatch), var_global.spval)
                self.fh = np.full((self.landpatch.numpatch), var_global.spval)
                self.fq = np.full((self.landpatch.numpatch), var_global.spval)

                self.irrig_rate = np.full((self.landpatch.numpatch), var_global.spval)
                self.deficit_irrig = np.full((self.landpatch.numpatch), var_global.spval)
                self.sum_irrig = np.full((self.landpatch.numpatch), var_global.spval)
                self.sum_irrig_count = np.full((self.landpatch.numpatch), var_global.spval)
                self.n_irrig_steps_left = np.full((self.landpatch.numpatch), var_global.spval)
                self.tairday = np.full((self.landpatch.numpatch), var_global.spval)
                self.usday = np.full((self.landpatch.numpatch), var_global.spval)
                self.vsday = np.full((self.landpatch.numpatch), var_global.spval)
                self.pairday = np.full((self.landpatch.numpatch), var_global.spval)
                self.rnetday = np.full((self.landpatch.numpatch), var_global.spval)
                self.fgrndday = np.full((self.landpatch.numpatch), var_global.spval)
                self.potential_evapotranspiration = np.full((self.landpatch.numpatch), var_global.spval)

                self.irrig_method_corn = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_swheat = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_wwheat = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_soybean = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_cotton = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_rice1 = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_rice2 = np.full((self.landpatch.numpatch), var_global.spval)
                self.irrig_method_sugarcane = np.full((self.landpatch.numpatch), var_global.spval)

        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
        if self.nl_colm['BGC']:
            pass
        if self.nl_colm['LATERAL_FLOW']:
            pass
        if self.nl_colm['URBAN_MODEL']:
            pass

    def check_TimeVariables(self):
        check_vector_data('t_grnd      [K]    ', self.t_grnd, self.mpi, self.nl_colm)
        check_vector_data('tleaf       [K]    ', self.tleaf, self.mpi, self.nl_colm)
        check_vector_data('ldew        [mm]   ', self.ldew, self.mpi, self.nl_colm)
        check_vector_data('ldew_rain   [mm]   ', self.ldew_rain, self.mpi, self.nl_colm)
        check_vector_data('ldew_snow   [mm]   ', self.ldew_snow, self.mpi, self.nl_colm)
        check_vector_data('sag         [-]    ', self.sag, self.mpi, self.nl_colm)
        check_vector_data('scv         [mm]   ', self.scv, self.mpi, self.nl_colm)
        check_vector_data('snowdp      [m]    ', self.snowdp, self.mpi, self.nl_colm)
        check_vector_data('fveg        [-]    ', self.fveg, self.mpi, self.nl_colm)
        check_vector_data('fsno        [-]    ', self.fsno, self.mpi, self.nl_colm)
        check_vector_data('sigf        [-]    ', self.sigf, self.mpi, self.nl_colm)
        check_vector_data('green       [-]    ', self.green, self.mpi, self.nl_colm)
        check_vector_data('lai         [-]    ', self.lai, self.mpi, self.nl_colm)
        check_vector_data('tlai        [-]    ', self.tlai, self.mpi, self.nl_colm)
        check_vector_data('sai         [-]    ', self.sai, self.mpi, self.nl_colm)
        check_vector_data('tsai        [-]    ', self.tsai, self.mpi, self.nl_colm)
        check_vector_data('coszen      [-]    ', self.coszen, self.mpi, self.nl_colm)
        check_vector_data('alb         [-]    ', self.alb, self.mpi, self.nl_colm)
        check_vector_data('ssun        [-]    ', self.ssun, self.mpi, self.nl_colm)
        check_vector_data('ssha        [-]    ', self.ssha, self.mpi, self.nl_colm)
        check_vector_data('ssoi        [-]    ', self.ssoi, self.mpi, self.nl_colm)
        check_vector_data('ssno        [-]    ', self.ssno, self.mpi, self.nl_colm)
        check_vector_data('thermk      [-]    ', self.thermk, self.mpi, self.nl_colm)
        check_vector_data('extkb       [-]    ', self.extkb, self.mpi, self.nl_colm)
        check_vector_data('extkd       [-]    ', self.extkd, self.mpi, self.nl_colm)
        check_vector_data('zwt         [m]    ', self.zwt, self.mpi, self.nl_colm)
        check_vector_data('wa          [mm]   ', self.wa, self.mpi, self.nl_colm)
        check_vector_data('wetwat      [mm]   ', self.wetwat, self.mpi, self.nl_colm)
        check_vector_data('wdsrf       [mm]   ', self.wdsrf, self.mpi, self.nl_colm)
        check_vector_data('rss         [s/m]  ', self.rss, self.mpi, self.nl_colm)
        check_vector_data('t_lake      [K]    ', self.t_lake, self.mpi, self.nl_colm)
        check_vector_data('lake_icefrc [-]    ', self.lake_icefrac, self.mpi, self.nl_colm)
        check_vector_data('savedtke1   [W/m K]', self.savedtke1, self.mpi, self.nl_colm)
        check_vector_data('z_sno       [m]    ', self.z_sno, self.mpi, self.nl_colm)
        check_vector_data('dz_sno      [m]    ', self.dz_sno, self.mpi, self.nl_colm)
        check_vector_data('t_soisno    [K]    ', self.t_soisno, self.mpi, self.nl_colm)
        check_vector_data('wliq_soisno [kg/m2]', self.wliq_soisno, self.mpi, self.nl_colm)
        check_vector_data('wice_soisno [kg/m2]', self.wice_soisno, self.mpi, self.nl_colm)
        check_vector_data('smp         [mm]   ', self.smp, self.mpi, self.nl_colm)
        check_vector_data('hk          [mm/s] ', self.hk, self.mpi, self.nl_colm)
        if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            check_vector_data('vegwp       [m]    ', self.vegwp, self.mpi, self.nl_colm)
            check_vector_data('gs0sun      []     ', self.gs0sun, self.mpi, self.nl_colm)
            check_vector_data('gs0sha      []     ', self.gs0sha, self.mpi, self.nl_colm)
        if self.nl_colm['DEF_USE_OZONESTRESS']:
            check_vector_data('o3coefv_sun        ', self.o3coefv_sun, self.mpi, self.nl_colm)
            check_vector_data('o3coefv_sha        ', self.o3coefv_sha, self.mpi, self.nl_colm)
            check_vector_data('o3coefg_sun        ', self.o3coefg_sun, self.mpi, self.nl_colm)
            check_vector_data('o3coefg_sha        ', self.o3coefg_sha, self.mpi, self.nl_colm)
            check_vector_data('lai_old            ', self.lai_old, self.mpi, self.nl_colm)
            check_vector_data('o3uptakesun        ', self.o3uptakesun, self.mpi, self.nl_colm)
            check_vector_data('o3uptakesha        ', self.o3uptakesha, self.mpi, self.nl_colm)
        if self.nl_colm['DEF_USE_SNICAR']:
            check_vector_data('snw_rds     [m-6]  ', self.snw_rds, self.mpi, self.nl_colm)
            check_vector_data('mss_bcpho   [Kg]   ', self.mss_bcpho, self.mpi, self.nl_colm)
            check_vector_data('mss_bcphi   [Kg]   ', self.mss_bcphi, self.mpi, self.nl_colm)
            check_vector_data('mss_ocpho   [Kg]   ', self.mss_ocpho, self.mpi, self.nl_colm)
            check_vector_data('mss_ocphi   [Kg]   ', self.mss_ocphi, self.mpi, self.nl_colm)
            check_vector_data('mss_dst1    [Kg]   ', self.mss_dst1, self.mpi, self.nl_colm)
            check_vector_data('mss_dst2    [Kg]   ', self.mss_dst2, self.mpi, self.nl_colm)
            check_vector_data('mss_dst3    [Kg]   ', self.mss_dst3, self.mpi, self.nl_colm)
            check_vector_data('mss_dst4    [Kg]   ', self.mss_dst4, self.mpi, self.nl_colm)
            check_vector_data('ssno_lyr    [-]    ', self.ssno_lyr, self.mpi, self.nl_colm)
        if self.nl_colm['DEF_USE_IRRIGATION']:
            check_vector_data('irrig_rate            ', self.irrig_rate, self.mpi, self.nl_colm)
            check_vector_data('deficit_irrig         ', self.deficit_irrig, self.mpi, self.nl_colm)
            check_vector_data('sum_irrig             ', self.sum_irrig, self.mpi, self.nl_colm)
            check_vector_data('sum_irrig_count       ', self.sum_irrig_count, self.mpi, self.nl_colm)
            check_vector_data('n_irrig_steps_left    ', self.n_irrig_steps_left, self.mpi, self.nl_colm)
            check_vector_data('tairday               ', self.tairday, self.mpi, self.nl_colm)
            check_vector_data('usday                 ', self.usday, self.mpi, self.nl_colm)
            check_vector_data('vsday                 ', self.vsday, self.mpi, self.nl_colm)
            check_vector_data('pairday               ', self.pairday, self.mpi, self.nl_colm)
            check_vector_data('rnetday               ', self.rnetday, self.mpi, self.nl_colm)
            check_vector_data('fgrndday              ', self.fgrndday, self.mpi, self.nl_colm)
            check_vector_data('potential_evapotranspiration', self.potential_evapotranspiration, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_corn     ', self.irrig_method_corn, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_swheat   ', self.irrig_method_swheat, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_wwheat   ', self.irrig_method_wwheat, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_soybean  ', self.irrig_method_soybean, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_cotton   ', self.irrig_method_cotton, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_rice1    ', self.irrig_method_rice1, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_rice2    ', self.irrig_method_rice2, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_sugarcane', self.irrig_method_sugarcane, self.mpi, self.nl_colm)
        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
        if self.nl_colm['BGC']:
            pass
        if self.nl_colm['USEMPI']:
            pass

    def WRITE_TimeVariables(self, idate, lc_year, site, dir_restart):
        compress = self.nl_colm['DEF_REST_CompressLevel']

        cyear = lc_year
        cdate = str(idate[0]) + '-' + str(idate[1]) + '-' + str(idate[2])

        if self.mpi.p_is_master:
            if not os.path.exists(dir_restart + '/' + cdate):
                os.makedirs(dir_restart + '/' + cdate)
        if self.nl_colm['USEMPI']:
            pass

        file_restart = os.path.join(dir_restart, cdate, site + '_restart_' + cdate + '_lc' + str(cyear) + '.nc')

        CoLM_NetCDFVectorBlk.ncio_create_file_vector(file_restart, self.landpatch.landpatch, self.mpi, self.gblock,
                                                     self.nl_colm['USEMPI'])

        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'patch', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'])

        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'snow', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'], -self.var_global.maxsnl)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'snowp1', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'],
                                                          -self.var_global.maxsnl + 1)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'soilsnow', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'],
                                                          self.var_global.nl_soil - self.var_global.maxsnl)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'soil', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'], self.var_global.nl_soil)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'lake', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'], self.var_global.nl_lake)

        if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'vegnodes',
                                                              self.mpi, self.gblock, self.nl_colm['USEMPI'],
                                                              self.var_global.nvegwcs)

        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'band', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'], 2)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, self.landpatch.landpatch, 'rtyp', self.mpi,
                                                          self.gblock, self.nl_colm['USEMPI'], 2)

        # ! Time-varying state variables which reaquired by restart run
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'z_sno', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.z_sno, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'dz_sno', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.dz_sno, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 't_soisno', 'soilsnow',
                                                 self.var_global.nl_soil - self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.t_soisno, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'wliq_soisno', 'soilsnow',
                                                 self.var_global.nl_soil - self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.wliq_soisno, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'wice_soisno', 'soilsnow',
                                                 self.var_global.nl_soil - self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.wice_soisno, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'smp', 'soil', self.var_global.nl_soil, 'patch',
                                                 self.landpatch.landpatch, self.smp, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'hk', 'soil', self.var_global.nl_soil, 'patch',
                                                 self.landpatch.landpatch, self.hk, self.mpi, self.gblock, compress)
        if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'vegwp', 'vegnodes', self.var_global.nvegwcs,
                                                     'patch', self.landpatch.landpatch, self.vegwp, self.mpi,
                                                     self.gblock, compress)
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'gs0sun', 'patch', self.landpatch.landpatch,
                                                   self.gs0sun, compress)
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'gs0sha', 'patch', self.landpatch.landpatch,
                                                   self.gs0sha, compress)
        if self.nl_colm['DEF_USE_OZONESTRESS']:
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'lai_old', 'patch', self.landpatch.landpatch,
                                                   self.lai_old, compress)
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'o3uptakesun', 'patch', self.landpatch.landpatch,
                                                   self.o3uptakesun, compress)
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'o3uptakesha', 'patch', self.landpatch.landpatch,
                                                   self.o3uptakesha, compress)

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 't_grnd', 'patch', self.landpatch.landpatch, self.t_grnd,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'tleaf', 'patch', self.landpatch.landpatch, self.tleaf,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'ldew', 'patch', self.landpatch.landpatch, self.ldew,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'ldew_rain', 'patch', self.landpatch.landpatch,
                                               self.ldew_rain, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'ldew_snow', 'patch', self.landpatch.landpatch,
                                               self.ldew_snow, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'sag', 'patch', self.landpatch.landpatch, self.sag,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'scv', 'patch', self.landpatch.landpatch, self.scv,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'snowdp', 'patch', self.landpatch.landpatch, self.snowdp,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'fveg', 'patch', self.landpatch.landpatch, self.fveg,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'fsno', 'patch', self.landpatch.landpatch, self.fsno,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'sigf', 'patch', self.landpatch.landpatch, self.sigf,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'green', 'patch', self.landpatch.landpatch, self.green,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'lai', 'patch', self.landpatch.landpatch, self.lai,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'tlai', 'patch', self.landpatch.landpatch, self.tlai,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'sai', 'patch', self.landpatch.landpatch, self.sai,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'tsai', 'patch', self.landpatch.landpatch, self.tsai,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'coszen', 'patch', self.landpatch.landpatch, self.coszen,
                                               self.mpi, self.gblock, compress)

        CoLM_NetCDFVectorBlk.ncio_write_vector12(file_restart, 'alb', 'band', 2, 'rtyp', 2, 'patch',
                                                 self.landpatch.landpatch, self.alb, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector12(file_restart, 'ssun', 'band', 2, 'rtyp', 2, 'patch',
                                                 self.landpatch.landpatch, self.ssun, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector12(file_restart, 'ssha', 'band', 2, 'rtyp', 2, 'patch',
                                                 self.landpatch.landpatch, self.ssha, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector12(file_restart, 'ssoi', 'band', 2, 'rtyp', 2, 'patch',
                                                 self.landpatch.landpatch, self.ssoi, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector12(file_restart, 'ssno', 'band', 2, 'rtyp', 2, 'patch',
                                                 self.landpatch.landpatch, self.ssno, self.mpi, self.gblock, compress)

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'thermk', 'patch', self.landpatch.landpatch, self.thermk,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'extkb   ', 'patch', self.landpatch.landpatch, self.extkb,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'extkd', 'patch', self.landpatch.landpatch, self.extkd,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'zwt', 'patch', self.landpatch.landpatch, self.zwt,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'wa', 'patch', self.landpatch.landpatch, self.wa, self.mpi,
                                               self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'wetwat', 'patch', self.landpatch.landpatch, self.wetwat,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'wdsrf', 'patch', self.landpatch.landpatch, self.wdsrf,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'rss', 'patch', self.landpatch.landpatch, self.rss,
                                               self.mpi, self.gblock, compress)

        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 't_lake', 'lake', self.var_global.nl_lake, 'patch',
                                                 self.landpatch.landpatch, self.t_lake, self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'lake_icefrc', 'lake', self.var_global.nl_lake, 'patch',
                                                 self.landpatch.landpatch, self.lake_icefrac, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'savedtke1', 'patch', self.landpatch.landpatch,
                                               self.savedtke1, self.mpi, self.gblock, compress)

        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'snw_rds', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.snw_rds, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_bcpho', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_bcpho, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_bcphi', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_bcphi, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_ocpho', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_ocpho, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_ocphi', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_ocphi, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_dst1', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_dst1, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_dst2', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_dst2, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_dst3', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_dst3, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'mss_dst4', 'snow', -self.var_global.maxsnl, 'patch',
                                                 self.landpatch.landpatch, self.mss_dst4, self.mpi, self.gblock,
                                                 compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector14(file_restart, 'ssno_lyr', 'band', 2, 'rtyp', 2, 'snowp1',
                                                 -self.var_global.maxsnl + 1, 'patch', self.landpatch.landpatch,
                                                 self.ssno_lyr, self.mpi, self.gblock, compress)

        # ! Additional va_vectorriables required by reginal model (such as WRF ) RSM)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'trad', 'patch', self.landpatch.landpatch, self.trad,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'tref', 'patch', self.landpatch.landpatch, self.tref,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'qref', 'patch', self.landpatch.landpatch, self.qref,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'rst', 'patch', self.landpatch.landpatch, self.rst,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'emis', 'patch', self.landpatch.landpatch, self.emis,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'z0m', 'patch', self.landpatch.landpatch, self.z0m,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'zol', 'patch', self.landpatch.landpatch, self.zol,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'rib', 'patch', self.landpatch.landpatch, self.rib,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'ustar', 'patch', self.landpatch.landpatch, self.ustar,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'qstar', 'patch', self.landpatch.landpatch, self.qstar,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'tstar', 'patch', self.landpatch.landpatch, self.tstar,
                                               self.mpi, self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'fm', 'patch', self.landpatch.landpatch, self.fm, self.mpi,
                                               self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'fh', 'patch', self.landpatch.landpatch, self.fh, self.mpi,
                                               self.gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'fq', 'patch', self.landpatch.landpatch, self.fq, self.mpi,
                                               self.gblock, compress)

        if self.nl_colm['DEF_USE_IRRIGATION']:
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_rate', 'patch',
                                                              self.landpatch.landpatch, self.irrig_rate, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'deficit_irrig', 'patch',
                                                              self.landpatch.landpatch, self.deficit_irrig, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'sum_irrig', 'patch',
                                                              self.landpatch.landpatch, self.sum_irrig, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'sum_irrig_count', 'patch',
                                                              self.landpatch.landpatch, self.sum_irrig_count, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'n_irrig_steps_left', 'patch',
                                                              self.landpatch.landpatch, self.n_irrig_steps_left,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'tairday', 'patch',
                                                              self.landpatch.landpatch, self.tairday, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'usday', 'patch', self.landpatch.landpatch,
                                                              self.usday, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'vsday ', 'patch', self.landpatch.landpatch,
                                                              self.vsday, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'pairday ', 'patch',
                                                              self.landpatch.landpatch, self.pairday, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'rnetday ', 'patch',
                                                              self.landpatch.landpatch, self.rnetday, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'fgrndday ', 'patch',
                                                              self.landpatch.landpatch, self.fgrndday, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'potential_evapotranspiration', 'patch',
                                                              self.landpatch.landpatch,
                                                              self.potential_evapotranspiration, compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_corn', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_corn,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_swheat', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_swheat,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_wwheat', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_wwheat,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_soybean', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_soybean,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_cotton ', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_cotton,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_rice1 ', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_rice1,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_rice2 ', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_rice2,
                                                              compress)
            CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, 'irrig_method_sugarcane ', 'patch',
                                                              self.landpatch.landpatch, self.irrig_method_sugarcane,
                                                              compress)

        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
        if self.nl_colm['BGC']:
            pass
        if self.nl_colm['CatchLateralFlow']:
            pass
        if self.nl_colm['URBAN_MODEL']:
            pass

    def deallocate_TimeVariables(self):
        # --------------------------------------------------
        # Deallocates memory for CoLM 1d[numpatch] variables
        # --------------------------------------------------
        if self.mpi.p_is_worker:
            if self.landpatch.numpatch > 0:
                del self.z_sno
                del self.dz_sno
                del self.t_soisno
                del self.wliq_soisno
                del self.wice_soisno
                del self.smp
                del self.hk
                del self.h2osoi
                del self.rootr
                del self.rootflux

                # Plant Hydraulic variables
                del self.vegwp
                del self.gs0sun
                del self.gs0sha
                # END plant hydraulic variables
                # Ozone stress variables
                del self.o3coefv_sun  # Ozone stress factor for photosynthesis on sunlit leaf
                del self.o3coefv_sha  # Ozone stress factor for photosynthesis on shaded leaf
                del self.o3coefg_sun  # Ozone stress factor for stomata on sunlit leaf
                del self.o3coefg_sha  # Ozone stress factor for stomata on shaded leaf
                del self.lai_old  # lai in last time step
                del self.o3uptakesun  # Ozone does, sunlit leaf(mmol O3 / m ^ 2
                del self.o3uptakesha  # Ozone does, shaded leaf(mmol O3 / m ^ 2
                # END Ozone stress variables
                del self.rstfacsun_out
                del self.rstfacsha_out
                del self.gssun_out
                del self.gssha_out
                del self.assimsun_out
                del self.assimsha_out
                del self.etrsun_out
                del self.etrsha_out

                del self.t_grnd
                del self.tleaf
                del self.ldew
                del self.ldew_rain
                del self.ldew_snow
                del self.sag
                del self.scv
                del self.snowdp
                del self.fveg
                del self.fsno
                del self.sigf
                del self.green
                del self.tlai
                del self.lai
                del self.laisun
                del self.laisha
                del self.tsai
                del self.sai
                del self.coszen
                del self.alb
                del self.ssun
                del self.ssha
                del self.ssoi
                del self.ssno
                del self.thermk
                del self.extkb
                del self.extkd
                del self.zwt
                del self.wa
                del self.wetwat
                del self.wat
                del self.wdsrf
                del self.rss

                del self.t_lake  # new lake scheme

                del self.lake_icefrac  # newlake scheme
                del self.savedtke1  # newlake scheme

                del self.snw_rds
                del self.mss_bcpho
                del self.mss_bcphi
                del self.mss_ocpho
                del self.mss_ocphi
                del self.mss_dst1
                del self.mss_dst2
                del self.mss_dst3
                del self.mss_dst4
                del self.ssno_lyr

                del self.trad
                del self.tref
                del self.qref
                del self.rst
                del self.emis
                del self.z0m
                del self.displa
                del self.zol
                del self.rib
                del self.ustar
                del self.qstar
                del self.tstar
                del self.fm
                del self.fh
                del self.fq

                del self.irrig_rate
                del self.deficit_irrig
                del self.sum_irrig
                del self.sum_irrig_count
                del self.n_irrig_steps_left

                del self.tairday
                del self.usday
                del self.vsday
                del self.pairday
                del self.rnetday
                del self.fgrndday
                del self.potential_evapotranspiration

                del self.irrig_method_corn
                del self.irrig_method_swheat
                del self.irrig_method_wwheat
                del self.irrig_method_soybean
                del self.irrig_method_cotton
                del self.irrig_method_rice1
                del self.irrig_method_rice2
                del self.irrig_method_sugarcane

        # # if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
        # deallocate_PFTimeVariables
        #
        # # if (defined BGC)
        # deallocate_BGCTimeVariables
        #
        # # ifdef CatchLateralFlow
        # deallocate_HydroTimeVariables
        #
        # # if (defined URBAN_MODEL)
        # deallocate_UrbanTimeVariables

    def read_time_variables(self, idate, lc_year, site, dir_restart):
        # Initialize
        if self.mpi.p_is_master == 0:
            print('Loading Time Variables ...')

        # Land cover type year
        cyear = lc_year
        cdate = str(idate.year) + '-' + convert_to_three_digits(idate.day) + '-' + convert_to_five_digits(idate.sec)
        file_restart = f"{dir_restart}/{cdate}/{site}_restart_{cdate}_lc{cyear}.nc"

        # Read time-varying state variables required by restart run
        self.z_sno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'z_sno', -self.var_global.maxsnl,
                                                           self.landpatch.landpatch, self.z_sno, self.nl_colm['USEMPI'],
                                                           self.mpi, self.gblock)  # node depth [m]
        self.dz_sno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'dz_sno', -self.var_global.maxsnl,
                                                            self.landpatch.landpatch, self.dz_sno,
                                                            self.nl_colm['USEMPI'], self.mpi,
                                                            self.gblock)  # interface depth [m]
        self.t_soisno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 't_soisno',
                                                              self.var_global.nl_soil - self.var_global.maxsnl,
                                                              self.landpatch.landpatch,
                                                              self.t_soisno, self.nl_colm['USEMPI'], self.mpi,
                                                              self.gblock)  # soil temperature [K]
        self.wliq_soisno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wliq_soisno',
                                                                 self.var_global.nl_soil - self.var_global.maxsnl,
                                                                 self.landpatch.landpatch,
                                                                 self.wliq_soisno, self.nl_colm['USEMPI'], self.mpi,
                                                                 self.gblock)  # liquid water in layers [kg/m2]
        self.wice_soisno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wice_soisno',
                                                                 self.var_global.nl_soil - self.var_global.maxsnl,
                                                                 self.landpatch.landpatch,
                                                                 self.wice_soisno, self.nl_colm['USEMPI'], self.mpi,
                                                                 self.gblock)  # ice lens in layers [kg/m2]
        self.smp = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'smp', self.var_global.nl_soil,
                                                         self.landpatch.landpatch,
                                                         self.smp, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # soil matrix potential [mm]
        self.hk = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'hk', self.var_global.nl_soil,
                                                        self.landpatch.landpatch,
                                                        self.hk, self.nl_colm['USEMPI'], self.mpi,
                                                        self.gblock)  # hydraulic conductivity [mm h2o/s]

        if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            self.vegwp = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vegwp', self.var_global.nvegwcs,
                                                               self.landpatch.landpatch,
                                                               self.vegwp, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  # vegetation water potential [mm]
            self.gs0sun = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'gs0sun', self.landpatch.landpatch,
                                                                self.gs0sun, self.nl_colm['USEMPI'], self.mpi,
                                                                self.gblock)  # working copy of sunlit stomata conductance
            self.gs0sha = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'gs0sha', self.landpatch.landpatch,
                                                                self.gs0sha, self.nl_colm['USEMPI'], self.mpi,
                                                                self.gblock)  # working copy of shalit stomata conductance

        if self.nl_colm['DEF_USE_OZONESTRESS']:
            self.lai_old = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'lai_old', self.landpatch.landpatch,
                                                                 self.lai_old, self.nl_colm['USEMPI'], self.mpi,
                                                                 self.gblock)
            self.o3uptakesun = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'o3uptakesun',
                                                                     self.landpatch.landpatch,
                                                                     self.o3uptakesun, self.nl_colm['USEMPI'], self.mpi,
                                                                     self.gblock)
            self.o3uptakesha = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'o3uptakesha',
                                                                     self.landpatch.landpatch,
                                                                     self.o3uptakesha, self.nl_colm['USEMPI'], self.mpi,
                                                                     self.gblock)

        self.t_grnd = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 't_grnd', self.landpatch.landpatch,
                                                            self.t_grnd, self.nl_colm['USEMPI'], self.mpi,
                                                            self.gblock)  # ground surface temperature [K]
        self.tleaf = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tleaf', self.landpatch.landpatch,
                                                           self.tleaf, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # leaf temperature [K]
        self.ldew = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ldew', self.landpatch.landpatch,
                                                          self.ldew, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # depth of water on foliage [mm]
        self.ldew_rain = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ldew_rain', self.landpatch.landpatch,
                                                               self.ldew_rain, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  # depth of rain on foliage [mm]
        self.ldew_snow = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ldew_snow', self.landpatch.landpatch,
                                                               self.ldew_snow, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  # depth of snow on foliage [mm]
        self.sag = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sag', self.landpatch.landpatch,
                                                         self.sag, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # non dimensional snow age [-]
        self.scv = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'scv', self.landpatch.landpatch,
                                                         self.scv, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # snow cover, water equivalent [mm]
        self.snowdp = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'snowdp', self.landpatch.landpatch,
                                                            self.snowdp, self.nl_colm['USEMPI'], self.mpi,
                                                            self.gblock)  # snow depth [meter]
        self.fveg = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fveg', self.landpatch.landpatch,
                                                          self.fveg, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # fraction of vegetation cover
        self.fsno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fsno', self.landpatch.landpatch,
                                                          self.fsno, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # fraction of snow cover on ground
        self.sigf = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sigf', self.landpatch.landpatch,
                                                          self.sigf, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # fraction of veg cover, excluding snow-covered veg [-]
        self.green = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'green', self.landpatch.landpatch,
                                                           self.green, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # leaf greenness
        self.lai = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'lai', self.landpatch.landpatch,
                                                         self.lai, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # leaf area index
        self.tlai = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tlai', self.landpatch.landpatch,
                                                          self.tlai, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # leaf area index
        self.sai = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sai', self.landpatch.landpatch,
                                                         self.sai, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # stem area index
        self.tsai = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tsai', self.landpatch.landpatch,
                                                          self.tsai, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # stem area index
        self.coszen = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'coszen', self.landpatch.landpatch,
                                                            self.coszen, self.nl_colm['USEMPI'], self.mpi,
                                                            self.gblock)  # cosine of solar zenith angle
        self.alb = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'alb', 2, 2, self.landpatch.landpatch,
                                                         self.alb, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # averaged albedo [-]
        self.ssun = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssun', 2, 2, self.landpatch.landpatch,
                                                          self.ssun, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # sunlit canopy absorption for solar radiation (0-1, self.nl_colm['USEMPI'], self.mpi, self.gblock)
        self.ssha = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssha', 2, 2, self.landpatch.landpatch,
                                                          self.ssha, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # shaded canopy absorption for solar radiation (0-1, self.nl_colm['USEMPI'], self.mpi, self.gblock)
        self.ssoi = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssoi', 2, 2, self.landpatch.landpatch,
                                                          self.ssoi, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # soil absorption for solar radiation (0-1, self.nl_colm['USEMPI'], self.mpi, self.gblock)
        self.ssno = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssno', 2, 2, self.landpatch.landpatch,
                                                          self.ssno, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # snow absorption for solar radiation (0-1, self.nl_colm['USEMPI'], self.mpi, self.gblock)
        self.thermk = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'thermk', self.landpatch.landpatch,
                                                            self.thermk, self.nl_colm['USEMPI'], self.mpi,
                                                            self.gblock)  # canopy gap fraction for tir radiation
        self.extkb = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'extkb', self.landpatch.landpatch,
                                                           self.extkb, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # (k, g(mu, self.nl_colm['USEMPI'], self.mpi, self.gblock)/mu, self.nl_colm['USEMPI'], self.mpi, self.gblock) direct solar extinction coefficient
        self.extkd = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'extkd', self.landpatch.landpatch,
                                                           self.extkd, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # diffuse and scattered diffuse PAR extinction coefficient
        self.zwt = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'zwt', self.landpatch.landpatch,
                                                         self.zwt, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # the depth to water table [m]
        self.wa = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wa', self.landpatch.landpatch,
                                                        self.wa, self.nl_colm['USEMPI'], self.mpi,
                                                        self.gblock)  # water storage in aquifer [mm]
        self.wetwat = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wetwat', self.landpatch.landpatch,
                                                            self.wetwat, self.nl_colm['USEMPI'], self.mpi,
                                                            self.gblock)  # water storage in wetland [mm]
        self.wdsrf = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wdsrf', self.landpatch.landpatch,
                                                           self.wdsrf, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # depth of surface water [mm]
        self.rss = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'rss', self.landpatch.landpatch,
                                                         self.rss, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # soil surface resistance [s/m]

        self.t_lake = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 't_lake', self.var_global.nl_lake,
                                                            self.landpatch.landpatch, self.t_lake,
                                                            self.nl_colm['USEMPI'], self.mpi, self.gblock)  #
        self.lake_icefrac = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'lake_icefrc',
                                                                  self.var_global.nl_lake, self.landpatch.landpatch,
                                                                  self.lake_icefrac, self.nl_colm['USEMPI'], self.mpi,
                                                                  self.gblock)  #
        self.savedtke1 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'savedtke1', self.landpatch.landpatch,
                                                               self.savedtke1, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  #

        self.snw_rds = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'snw_rds', -self.var_global.maxsnl,
                                                             self.landpatch.landpatch, self.snw_rds,
                                                             self.nl_colm['USEMPI'], self.mpi, self.gblock)  #
        self.mss_bcpho = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_bcpho',
                                                               -self.var_global.maxsnl, self.landpatch.landpatch,
                                                               self.mss_bcpho, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  #
        self.mss_bcphi = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_bcphi',
                                                               -self.var_global.maxsnl, self.landpatch.landpatch,
                                                               self.mss_bcphi, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  #
        self.mss_ocpho = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_ocpho',
                                                               -self.var_global.maxsnl, self.landpatch.landpatch,
                                                               self.mss_ocpho, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  #
        self.mss_ocphi = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_ocphi',
                                                               -self.var_global.maxsnl, self.landpatch.landpatch,
                                                               self.mss_ocphi, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)  #
        self.mss_dst1 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_dst1',
                                                              -self.var_global.maxsnl, self.landpatch.landpatch,
                                                              self.mss_dst1, self.nl_colm['USEMPI'], self.mpi,
                                                              self.gblock)  #
        self.mss_dst2 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_dst2',
                                                              -self.var_global.maxsnl, self.landpatch.landpatch,
                                                              self.mss_dst2, self.nl_colm['USEMPI'], self.mpi,
                                                              self.gblock)  #
        self.mss_dst3 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_dst3',
                                                              -self.var_global.maxsnl, self.landpatch.landpatch,
                                                              self.mss_dst3, self.nl_colm['USEMPI'], self.mpi,
                                                              self.gblock)  #
        self.mss_dst4 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'mss_dst4',
                                                              -self.var_global.maxsnl, self.landpatch.landpatch,
                                                              self.mss_dst4, self.nl_colm['USEMPI'], self.mpi,
                                                              self.gblock)  #
        self.ssno_lyr = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ssno_lyr', 2, 2,
                                                              -self.var_global.maxsnl + 1, self.landpatch.landpatch,
                                                              self.ssno_lyr, self.nl_colm['USEMPI'], self.mpi,
                                                              self.gblock)  #

        # Additional variables required by reginal model (such as WRF , self.nl_colm['USEMPI'], self.mpi, self.gblock) RSM, self.nl_colm['USEMPI'], self.mpi, self.gblock)
        self.trad = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'trad', self.landpatch.landpatch,
                                                          self.trad, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # radiative temperature of surface [K]
        self.tref = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tref', self.landpatch.landpatch,
                                                          self.tref, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # 2 m height air temperature [kelvin]
        self.qref = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'qref', self.landpatch.landpatch,
                                                          self.qref, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # 2 m height air specific humidity
        self.rst = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'rst', self.landpatch.landpatch,
                                                         self.rst, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # canopy stomatal resistance (s/m, self.nl_colm['USEMPI'], self.mpi, self.gblock)
        self.emis = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'emis', self.landpatch.landpatch,
                                                          self.emis, self.nl_colm['USEMPI'], self.mpi,
                                                          self.gblock)  # averaged bulk surface emissivity
        self.z0m = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'z0m', self.landpatch.landpatch,
                                                         self.z0m, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # effective roughness [m]
        self.zol = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'zol', self.landpatch.landpatch,
                                                         self.zol, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # dimensionless height (z/L, self.nl_colm['USEMPI'], self.mpi, self.gblock) used in Monin-Obukhov theory
        self.rib = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'rib', self.landpatch.landpatch,
                                                         self.rib, self.nl_colm['USEMPI'], self.mpi,
                                                         self.gblock)  # bulk Richardson number in surface layer
        self.ustar = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ustar', self.landpatch.landpatch,
                                                           self.ustar, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # u* in similarity theory [m/s]
        self.qstar = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'qstar', self.landpatch.landpatch,
                                                           self.qstar, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # q* in similarity theory [kg/kg]
        self.tstar = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tstar', self.landpatch.landpatch,
                                                           self.tstar, self.nl_colm['USEMPI'], self.mpi,
                                                           self.gblock)  # t* in similarity theory [K]
        self.fm = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fm', self.landpatch.landpatch,
                                                        self.fm, self.nl_colm['USEMPI'], self.mpi,
                                                        self.gblock)  # integral of profile FUNCTION for momentum
        self.fh = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fh', self.landpatch.landpatch,
                                                        self.fh, self.nl_colm['USEMPI'], self.mpi,
                                                        self.gblock)  # integral of profile FUNCTION for heat
        self.fq = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fq', self.landpatch.landpatch,
                                                        self.fq, self.nl_colm['USEMPI'], self.mpi,
                                                        self.gblock)  # integral of profile FUNCTION for moisture

        if self.nl_colm['DEF_USE_IRRIGATION']:
            self.irrig_rate = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_rate',
                                                                    self.landpatch.landpatch,
                                                                    self.irrig_rate, self.nl_colm['USEMPI'], self.mpi,
                                                                    self.gblock)
            self.deficit_irrig = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'deficit_irrig',
                                                                       self.landpatch.landpatch,
                                                                       self.deficit_irrig, self.nl_colm['USEMPI'],
                                                                       self.mpi, self.gblock)
            self.sum_irrig = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sum_irrig', self.landpatch.landpatch,
                                                                   self.sum_irrig, self.nl_colm['USEMPI'], self.mpi,
                                                                   self.gblock)
            self.sum_irrig_count = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sum_irrig_count',
                                                                         self.landpatch.landpatch, self.sum_irrig_count,
                                                                         self.nl_colm['USEMPI'], self.mpi, self.gblock)
            self.n_irrig_steps_left = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'n_irrig_steps_left',
                                                                            self.landpatch.landpatch,
                                                                            self.n_irrig_steps_left,
                                                                            self.nl_colm['USEMPI'], self.mpi,
                                                                            self.gblock)
            self.tairday = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'tairday', self.landpatch.landpatch,
                                                                 self.tairday, self.nl_colm['USEMPI'], self.mpi,
                                                                 self.gblock)
            self.usday = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'usday', self.landpatch.landpatch,
                                                               self.usday, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)
            self.vsday = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vsday', self.landpatch.landpatch,
                                                               self.vsday, self.nl_colm['USEMPI'], self.mpi,
                                                               self.gblock)
            self.pairday = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'pairday', self.landpatch.landpatch,
                                                                 self.pairday, self.nl_colm['USEMPI'], self.mpi,
                                                                 self.gblock)
            self.rnetday = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'rnetday', self.landpatch.landpatch,
                                                                 self.rnetday, self.nl_colm['USEMPI'], self.mpi,
                                                                 self.gblock)
            self.fgrndday = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fgrndday', self.landpatch.landpatch,
                                                                  self.fgrndday, self.nl_colm['USEMPI'], self.mpi,
                                                                  self.gblock)
            self.potential_evapotranspiration = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart,
                                                                                      'potential_evapotranspiration',
                                                                                      self.landpatch.landpatch,
                                                                                      self.potential_evapotranspiration,
                                                                                      self.nl_colm['USEMPI'], self.mpi,
                                                                                      self.gblock)
            self.irrig_method_corn = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_corn',
                                                                           self.landpatch.landpatch,
                                                                           self.irrig_method_corn,
                                                                           self.nl_colm['USEMPI'], self.mpi,
                                                                           self.gblock)
            self.irrig_method_swheat = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_swheat',
                                                                             self.landpatch.landpatch,
                                                                             self.irrig_method_swheat,
                                                                             self.nl_colm['USEMPI'], self.mpi,
                                                                             self.gblock)
            self.irrig_method_wwheat = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_wwheat',
                                                                             self.landpatch.landpatch,
                                                                             self.irrig_method_wwheat,
                                                                             self.nl_colm['USEMPI'], self.mpi,
                                                                             self.gblock)
            self.irrig_method_soybean = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_soybean',
                                                                              self.landpatch.landpatch,
                                                                              self.irrig_method_soybean,
                                                                              self.nl_colm['USEMPI'], self.mpi,
                                                                              self.gblock)
            self.irrig_method_cotton = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_cotton',
                                                                             self.landpatch.landpatch,
                                                                             self.irrig_method_cotton,
                                                                             self.nl_colm['USEMPI'], self.mpi,
                                                                             self.gblock)
            self.irrig_method_rice1 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_rice1',
                                                                            self.landpatch.landpatch,
                                                                            self.irrig_method_rice1,
                                                                            self.nl_colm['USEMPI'], self.mpi,
                                                                            self.gblock)
            self.irrig_method_rice2 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_rice2',
                                                                            self.landpatch.landpatch,
                                                                            self.irrig_method_rice2,
                                                                            self.nl_colm['USEMPI'], self.mpi,
                                                                            self.gblock)
            self.irrig_method_sugarcane = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'irrig_method_sugarcane',
                                                                                self.landpatch.landpatch,
                                                                                self.irrig_method_sugarcane,
                                                                                self.nl_colm['USEMPI'], self.mpi,
                                                                                self.gblock)

        # Read additional variables based on specific conditions
        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
            # file_restart = f"{dir_restart}/{cdate}/{site}_restart_pft_{cdate}_lc{cyear}.nc"
            # vars_PFTime.read_pftime_variables(landpft ,file_restart)

        # if self.nl_colm['BGC']:
        #     file_restart = f"{dir_restart}/{cdate}/{site}_restart_bgc_{cdate}_lc{cyear}.nc"
        #     read_bgctime_variables(file_restart)

        if self.nl_colm['CatchLateralFlow']:
            pass
            # file_restart = f"{dir_restart}/{cdate}/{site}_restart_basin_{cdate}_lc{cyear}.nc"
            # read_hydrotime_variables(file_restart)

        # if self.nl_colm['URBAN_MODEL']:
        #     file_restart = f"{dir_restart}/{cdate}/{site}_restart_urban_{cdate}_lc{cyear}.nc"
        #     read_urbantime_variables(file_restart)

        if self.nl_colm['RangeCheck']:
            self.check_TimeVariables()

        if self.mpi.p_is_master:
            print('Loading Time Variables done.')

    def check_TimeVariables(self):
        if self.mpi.p_is_master:
            print("\nChecking Time Variables ...")

        # Check vector data
        check_vector_data('t_grnd      [K]    ', self.t_grnd, self.mpi, self.nl_colm)  # ground surface temperature [K]
        check_vector_data('tleaf       [K]    ', self.tleaf, self.mpi, self.nl_colm)  # leaf temperature [K]
        check_vector_data('ldew        [mm]   ', self.ldew, self.mpi, self.nl_colm)  # depth of water on foliage [mm]
        check_vector_data('ldew_rain   [mm]   ', self.ldew_rain, self.mpi,
                          self.nl_colm)  # depth of rain on foliage [mm]
        check_vector_data('ldew_snow   [mm]   ', self.ldew_snow, self.mpi,
                          self.nl_colm)  # depth of snow on foliage [mm]
        check_vector_data('sag         [-]    ', self.sag, self.mpi, self.nl_colm)  # non dimensional snow age [-]
        check_vector_data('scv         [mm]   ', self.scv, self.mpi,
                          self.nl_colm)  # snow cover, self. water equivalent [mm]
        check_vector_data('snowdp      [m]    ', self.snowdp, self.mpi, self.nl_colm)  # snow depth [meter]
        check_vector_data('fveg        [-]    ', self.fveg, self.mpi, self.nl_colm)  # fraction of vegetation cover
        check_vector_data('fsno        [-]    ', self.fsno, self.mpi, self.nl_colm)  # fraction of snow cover on ground
        check_vector_data('sigf        [-]    ',
                          self.sigf, self.mpi,
                          self.nl_colm)  # fraction of veg cover, self. excluding snow-covered veg [-]
        check_vector_data('green       [-]    ', self.green, self.mpi, self.nl_colm)  # leaf greenness
        check_vector_data('lai         [-]    ', self.lai, self.mpi, self.nl_colm)  # leaf area index
        check_vector_data('tlai        [-]    ', self.tlai, self.mpi, self.nl_colm)  # leaf area index
        check_vector_data('sai         [-]    ', self.sai, self.mpi, self.nl_colm)  # stem area index
        check_vector_data('tsai        [-]    ', self.tsai, self.mpi, self.nl_colm)  # stem area index
        check_vector_data('coszen      [-]    ', self.coszen, self.mpi, self.nl_colm)  # cosine of solar zenith angle
        check_vector_data('alb         [-]    ', self.alb, self.mpi, self.nl_colm)  # averaged albedo [-]
        check_vector_data('ssun        [-]    ', self.ssun, self.mpi,
                          self.nl_colm)  # sunlit canopy absorption for solar radiation (0-1, self.mpi, self.nl_colm)
        check_vector_data('ssha        [-]    ', self.ssha, self.mpi,
                          self.nl_colm)  # shaded canopy absorption for solar radiation (0-1, self.mpi, self.nl_colm)
        check_vector_data('ssoi        [-]    ', self.ssoi, self.mpi,
                          self.nl_colm)  # soil absorption for solar radiation (0-1, self.mpi, self.nl_colm)
        check_vector_data('ssno        [-]    ', self.ssno, self.mpi,
                          self.nl_colm)  # snow absorption for solar radiation (0-1, self.mpi, self.nl_colm)
        check_vector_data('thermk      [-]    ', self.thermk, self.mpi,
                          self.nl_colm)  # canopy gap fraction for tir radiation
        check_vector_data('extkb       [-]    ', self.extkb, self.mpi,
                          self.nl_colm)  # (k, self. g(mu, self.mpi, self.nl_colm)/mu, self.mpi, self.nl_colm) direct solar extinction coefficient
        check_vector_data('extkd       [-]    ', self.extkd, self.mpi,
                          self.nl_colm)  # diffuse and scattered diffuse PAR extinction coefficient
        check_vector_data('zwt         [m]    ', self.zwt, self.mpi, self.nl_colm)  # the depth to water table [m]
        check_vector_data('wa          [mm]   ', self.wa, self.mpi, self.nl_colm)  # water storage in aquifer [mm]
        check_vector_data('wetwat      [mm]   ', self.wetwat, self.mpi, self.nl_colm)  # water storage in wetland [mm]
        check_vector_data('wdsrf       [mm]   ', self.wdsrf, self.mpi, self.nl_colm)  # depth of surface water [mm]
        check_vector_data('rss         [s/m]  ', self.rss, self.mpi, self.nl_colm)  # soil surface resistance [s/m]
        check_vector_data('t_lake      [K]    ', self.t_lake, self.mpi, self.nl_colm)  # lake temperature [K]
        check_vector_data('lake_icefrc [-]    ', self.lake_icefrac, self.mpi, self.nl_colm)  # lake ice fraction
        check_vector_data('savedtke1   [W/m K]', self.savedtke1, self.mpi,
                          self.nl_colm)  # saved turbulent kinetic energy
        check_vector_data('z_sno       [m]    ', self.z_sno, self.mpi, self.nl_colm)  # node depth [m]
        check_vector_data('dz_sno      [m]    ', self.dz_sno, self.mpi, self.nl_colm)  # interface depth [m]
        check_vector_data('t_soisno    [K]    ', self.t_soisno, self.mpi, self.nl_colm)  # soil temperature [K]
        check_vector_data('wliq_soisno [kg/m2]', self.wliq_soisno, self.mpi,
                          self.nl_colm)  # liquid water in layers [kg/m2]
        check_vector_data('wice_soisno [kg/m2]', self.wice_soisno, self.mpi, self.nl_colm)  # ice lens in layers [kg/m2]
        check_vector_data('smp         [mm]   ', self.smp, self.mpi, self.nl_colm)  # soil matrix potential [mm]
        check_vector_data('hk          [mm/s] ', self.hk, self.mpi, self.nl_colm)  # hydraulic conductivity [mm h2o/s]

        if self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            check_vector_data('vegwp       [m]    ', self.vegwp, self.mpi,
                              self.nl_colm)  # vegetation water potential [mm]
            check_vector_data('gs0sun      []     ', self.gs0sun, self.mpi,
                              self.nl_colm)  # working copy of sunlit stomata conductance
            check_vector_data('gs0sha      []     ', self.gs0sha, self.mpi,
                              self.nl_colm)  # working copy of shaded stomata conductance

        if self.nl_colm['DEF_USE_OZONESTRESS']:
            check_vector_data('o3coefv_sun        ', self.o3coefv_sun, self.mpi, self.nl_colm)
            check_vector_data('o3coefv_sha        ', self.o3coefv_sha, self.mpi, self.nl_colm)
            check_vector_data('o3coefg_sun        ', self.o3coefg_sun, self.mpi, self.nl_colm)
            check_vector_data('o3coefg_sha        ', self.o3coefg_sha, self.mpi, self.nl_colm)
            check_vector_data('lai_old            ', self.lai_old, self.mpi, self.nl_colm)
            check_vector_data('o3uptakesun        ', self.o3uptakesun, self.mpi, self.nl_colm)
            check_vector_data('o3uptakesha        ', self.o3uptakesha, self.mpi, self.nl_colm)

        if self.nl_colm['DEF_USE_SNICAR']:
            check_vector_data('snw_rds     [m-6]  ', self.snw_rds, self.mpi, self.nl_colm)
            check_vector_data('mss_bcpho   [Kg]   ', self.mss_bcpho, self.mpi, self.nl_colm)
            check_vector_data('mss_bcphi   [Kg]   ', self.mss_bcphi, self.mpi, self.nl_colm)
            check_vector_data('mss_ocpho   [Kg]   ', self.mss_ocpho, self.mpi, self.nl_colm)
            check_vector_data('mss_ocphi   [Kg]   ', self.mss_ocphi, self.mpi, self.nl_colm)
            check_vector_data('mss_dst1    [Kg]   ', self.mss_dst1, self.mpi, self.nl_colm)
            check_vector_data('mss_dst2    [Kg]   ', self.mss_dst2, self.mpi, self.nl_colm)
            check_vector_data('mss_dst3    [Kg]   ', self.mss_dst3, self.mpi, self.nl_colm)
            check_vector_data('mss_dst4    [Kg]   ', self.mss_dst4, self.mpi, self.nl_colm)
            check_vector_data('ssno_lyr    [-]    ', self.ssno_lyr, self.mpi, self.nl_colm)

        if self.nl_colm['DEF_USE_IRRIGATION']:
            check_vector_data('irrig_rate            ', self.irrig_rate, self.mpi, self.nl_colm)
            check_vector_data('deficit_irrig         ', self.deficit_irrig, self.mpi, self.nl_colm)
            check_vector_data('sum_irrig             ', self.sum_irrig, self.mpi, self.nl_colm)
            check_vector_data('sum_irrig_count       ', self.sum_irrig_count, self.mpi, self.nl_colm)
            check_vector_data('n_irrig_steps_left    ', self.n_irrig_steps_left, self.mpi, self.nl_colm)
            check_vector_data('tairday               ', self.tairday, self.mpi, self.nl_colm)
            check_vector_data('usday                 ', self.usday, self.mpi, self.nl_colm)
            check_vector_data('vsday                 ', self.vsday, self.mpi, self.nl_colm)
            check_vector_data('pairday               ', self.pairday, self.mpi, self.nl_colm)
            check_vector_data('rnetday               ', self.rnetday, self.mpi, self.nl_colm)
            check_vector_data('fgrndday              ', self.fgrndday, self.mpi, self.nl_colm)
            check_vector_data('potential_evapotranspiration', self.potential_evapotranspiration, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_corn     ', self.irrig_method_corn, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_swheat   ', self.irrig_method_swheat, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_wwheat   ', self.irrig_method_wwheat, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_soybean  ', self.irrig_method_soybean, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_cotton   ', self.irrig_method_cotton, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_rice1    ', self.irrig_method_rice1, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_rice2    ', self.irrig_method_rice2, self.mpi, self.nl_colm)
            check_vector_data('irrig_method_sugarcane', self.irrig_method_sugarcane, self.mpi, self.nl_colm)

    def save_to_restart(self, idate, deltim, itstamp, ptstamp):
        # Assuming DEF_WRST_FREQ is defined somewhere in the Python code
        # Assuming isendofhour, isendofday, isendofmonth, isendofyear functions are defined

        rwrite = False

        if self.nl_colm['DEF_WRST_FREQ'] == 'TIMESTEP':
            rwrite = True
        elif self.nl_colm['DEF_WRST_FREQ'] == 'HOURLY':
            rwrite = CoLM_TimeManager.isendofhour(idate, deltim)
        elif self.nl_colm['DEF_WRST_FREQ'] == 'DAILY':
            rwrite = CoLM_TimeManager.isendofday(idate, deltim)
        elif self.nl_colm['DEF_WRST_FREQ'] == 'MONTHLY':
            rwrite = CoLM_TimeManager.isendofmonth(idate, deltim)
        elif self.nl_colm['DEF_WRST_FREQ'] == 'YEARLY':
            rwrite = CoLM_TimeManager.isendofyear(idate, deltim)
        else:
            print('Warning: Please USE one of TIMESTEP/HOURLY/DAILY/MONTHLY/YEARLY for restart frequency.')

        if rwrite:
            rwrite = ptstamp < itstamp

        return rwrite
