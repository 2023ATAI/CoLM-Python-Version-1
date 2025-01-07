import numpy as np
import os
import CoLM_RangeCheck
import CoLM_NetCDFVectorBlk
from CoLM_NetCDFSerial import NetCDFFile

class Vars_PFTimeInvariants(object):
    def __init__(self) -> None:
        self.pftclass = None
        self.pftfrac = None
        self.htop_p = None
        self.hbot_p = None
    
    def read_pf_time_invariants(self, file_restart, landpatch, landpft):
        self.pftclass = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'pftclass', landpft, self.pftclass)
        self.pftfrac = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'pftfrac', landpft, self.pftfrac)
        self.htop_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'htop_p', landpft, self.htop_p)
        landpft, self.hbot_p = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'hbot_p', landpft, self.hbot_p)

        if self.nl_colm['CROP']:
            self.patchclass = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'pct_crops', landpatch, self.pctshrpch)


class Vars_TimeInvariants(object):
    def __init__(self, nl_colm, mpi, landpatch, var_global) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.landpatch = landpatch
        self.var_global = var_global
        if self.mpi.p_is_worker:
            if (self.landpatch.numpatch > 0):
                self.patchclass = np.zeros(self.landpatch.numpatch, dtype=int)
                self.patchtype = np.zeros(self.landpatch.numpatch, dtype=int)
                self.patchmask = np.full(self.landpatch.numpatch, False, dtype=bool)
                self.patchlonr = np.zeros(self.landpatch.numpatch)
                self.patchlatr = np.zeros(self.landpatch.numpatch)
                self.lakedepth = np.zeros(self.landpatch.numpatch)
                self.dz_lake = np.zeros((self.var_global.nl_lake, self.landpatch.numpatch))
                # self.dz_lake = [[None] * self.landpatch.numpatch for _ in range(self.var_global.nl_lake)]
                self.soil_s_v_alb = np.zeros(self.landpatch.numpatch)
                self.soil_d_v_alb = np.zeros(self.landpatch.numpatch)
                self.soil_s_n_alb = np.zeros(self.landpatch.numpatch)
                self.soil_d_n_alb = np.zeros(self.landpatch.numpatch)
                self.vf_quartz = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.vf_gravels = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.vf_om = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.vf_sand = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.wf_gravels = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.wf_sand = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.OM_density = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.BD_all = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.wfc = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.porsl = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.psi0 = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.bsw = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                    self.theta_r = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                    self.alpha_vgm = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                    self.L_vgm = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                    self.n_vgm = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                    self.sc_vgm = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                    self.fc_vgm = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.vic_b_infilt = np.zeros(self.landpatch.numpatch)
                self.vic_Dsmax = np.zeros(self.landpatch.numpatch)
                self.vic_Ds = np.zeros(self.landpatch.numpatch)
                self.vic_Ws = np.zeros(self.landpatch.numpatch)
                self.vic_c = np.zeros(self.landpatch.numpatch)

                self.hksati = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.csol = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.k_solids = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.dksatu = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.dksatf = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.dkdry = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.BA_alpha = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.BA_beta = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))
                self.htop = np.zeros(self.landpatch.numpatch)
                self.hbot = np.zeros(self.landpatch.numpatch)
                self.dbedrock = np.zeros(self.landpatch.numpatch)
                self.ibedrock = np.zeros(self.landpatch.numpatch, dtype=int)
                self.topoelv = np.zeros(self.landpatch.numpatch)
                self.topostd = np.zeros(self.landpatch.numpatch)
                self.BVIC = np.zeros((self.var_global.nl_soil, self.landpatch.numpatch))

                self.zlnd = 0.0
                self.zsno = 0.0
                self.csoilc = 0.0
                self.dewmx = 0.0
                self.wtfact = 0.0
                self.capr = 0.0
                self.cnfac = 0.0
                self.ssi = 0.0
                self.wimp = 0.0
                self.pondmx = 0.0
                self.smpmax = 0.0
                self.smpmin = 0.0
                self.trsmx0 = 0.0
                self.tcrit = 0.0
                self.wetwatmax = 0.0

        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
        if self.nl_colm['BGC']:
            pass
        if self.nl_colm['URBAN_MODEL']:
            pass

    def check_TimeInvariants(self):
        if self.mpi.p_is_master:
            print('Checking Time Invariants ...')
        if self.nl_colm['USEMPI']:
            pass

        CoLM_RangeCheck.check_vector_data('lakedepth    [m]     ', self.lakedepth, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('dz_lake      [m]     ', self.dz_lake, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('soil_s_v_alb [-]     ', self.soil_s_v_alb, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('soil_d_v_alb [-]     ', self.soil_d_v_alb, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('soil_s_n_alb [-]     ', self.soil_s_n_alb, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('soil_d_n_alb [-]     ', self.soil_d_n_alb, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('vf_quartz    [m3/m3] ', self.vf_quartz, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('vf_gravels   [m3/m3] ', self.vf_gravels, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('vf_om        [m3/m3] ', self.vf_om, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('vf_sand      [m3/m3] ', self.vf_sand, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('wf_gravels   [kg/kg] ', self.wf_gravels, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('wf_sand      [kg/kg] ', self.wf_sand, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('OM_density   [kg/m3] ', self.OM_density, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('BD_all       [kg/m3] ', self.BD_all, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('wfc          [m3/m3] ', self.wfc, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('porsl        [m3/m3] ', self.porsl, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('psi0         [mm]    ', self.psi0, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('bsw          [-]     ', self.bsw, self.mpi, self.nl_colm)
        if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            CoLM_RangeCheck.check_vector_data('theta_r      [m3/m3] ', self.theta_r, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('alpha_vgm    [-]     ', self.alpha_vgm, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('L_vgm        [-]     ', self.L_vgm, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('n_vgm        [-]     ', self.n_vgm, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('sc_vgm       [-]     ', self.sc_vgm, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('fc_vgm       [-]     ', self.fc_vgm, self.mpi, self.nl_colm)

        CoLM_RangeCheck.check_vector_data('hksati       [mm/s]  ', self.hksati, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('csol         [J/m3/K]', self.csol, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('k_solids     [W/m/K] ', self.k_solids, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('dksatu       [W/m/K] ', self.dksatu, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('dksatf       [W/m/K] ', self.dksatf, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('dkdry        [W/m/K] ', self.dkdry, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('BA_alpha     [-]     ', self.BA_alpha, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('BA_beta      [-]     ', self.BA_beta, self.mpi, self.nl_colm)

        if self.nl_colm['DEF_USE_BEDROCK']:
            CoLM_RangeCheck.check_vector_data('dbedrock     [m]     ', self.dbedrock, self.mpi, self.nl_colm)

        CoLM_RangeCheck.check_vector_data('topoelv      [-]     ', self.topoelv, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('topostd      [-]     ', self.topostd, self.mpi, self.nl_colm)
        CoLM_RangeCheck.check_vector_data('BVIC      [-]     ', self.BVIC, self.mpi, self.nl_colm)

        if self.nl_colm['USEMPI']:
            pass

        if self.mpi.p_is_master:
            print('Checking Constants ...')
            print('zlnd   [m]    ', self.zlnd)  # roughness length for soil [m]
            print('zsno   [m]    ', self.zsno)  # roughness length for snow [m]
            print('csoilc [-]    ', self.csoilc)  # drag coefficient for soil under canopy [-]
            print('dewmx  [mm]   ', self.dewmx)  # maximum dew
            print('wtfact [-]    ', self.wtfact)  # fraction of model area with high water table
            print('capr   [-]    ', self.capr)  # tuning factor to turn first layer T into surface T
            print('cnfac  [-]    ', self.cnfac)  # Crank Nicholson factor between 0 and 1
            print('ssi    [-]    ', self.ssi)  # irreducible water saturation of snow
            print('wimp   [m3/m3]', self.wimp)  # water impremeable IF porosity less than wimp
            print('pondmx [mm]   ', self.pondmx)  # ponding depth (mm)
            print('smpmax [mm]   ', self.smpmax)  # wilting point potential in mm
            print('smpmin [mm]   ', self.smpmin)  # restriction for min of soil poten. (mm)
            print('trsmx0 [mm/s] ', self.trsmx0)  # max transpiration for moist soil+100% veg.  [mm/s]
            print('tcrit  [K]    ', self.tcrit)  # critical temp. to determine rain or snow
            print('wetwatmax [mm]', self.wetwatmax)  # maximum wetland water (mm)

        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
        if self.nl_colm['BGC']:
            pass

    def WRITE_TimeInvariants(self, lc_year, casename, dir_restart, landpatch, gblock):
        compress = self.nl_colm['DEF_REST_CompressLevel']
        cyear = lc_year

        if self.mpi.p_is_master:
            os.makedirs(os.path.join(dir_restart, 'const'), exist_ok=True)

        # Synchronize processes if using MPI
        if self.nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb, p_err)

        file_restart = os.path.join(dir_restart, 'const', f"{casename}_restart_const_{cyear}.nc")

        CoLM_NetCDFVectorBlk.ncio_create_file_vector(file_restart, landpatch.landpatch, self.mpi, gblock,
                                                     self.nl_colm['USEMPI'])

        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'patch', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'])
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'soil', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], self.var_global.nl_soil)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'lake', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], self.var_global.nl_lake)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'band', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], 2)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'rtyp', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], 2)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'snow', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], -self.var_global.maxsnl)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'snowp1', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], -self.var_global.maxsnl + 1)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'soilsnow', self.mpi,
                                                          gblock, self.nl_colm['USEMPI'],
                                                          self.var_global.nl_soil - self.var_global.maxsnl)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'soil', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], self.var_global.nl_soil)
        CoLM_NetCDFVectorBlk.ncio_define_dimension_vector(file_restart, landpatch.landpatch, 'lake', self.mpi, gblock,
                                                          self.nl_colm['USEMPI'], self.var_global.nl_lake)

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'patchclass', 'patch', landpatch.landpatch,
                                               self.patchclass, self.mpi, gblock)  #
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'patchtype', 'patch', landpatch.landpatch, self.patchtype,
                                               self.mpi, gblock)  #
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'patchmask', 'patch', landpatch.landpatch, self.patchmask,
                                               self.mpi, gblock)  #

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'patchlonr', 'patch', landpatch.landpatch, self.patchlonr,
                                               self.mpi, gblock)  #
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'patchlatr', 'patch', landpatch.landpatch, self.patchlatr,
                                               self.mpi, gblock)  #

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'lakedepth', 'patch', landpatch.landpatch, self.lakedepth,
                                               self.mpi, gblock, compress)  #
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'dz_lake', 'lake', self.var_global.nl_lake, 'patch',
                                                 landpatch.landpatch, self.dz_lake, self.mpi, gblock, compress)  #

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'soil_s_v_alb', 'patch', landpatch.landpatch,
                                               self.soil_s_v_alb, self.mpi, gblock,
                                               compress)  # albedo of visible of the saturated soil
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'soil_d_v_alb', 'patch', landpatch.landpatch,
                                               self.soil_d_v_alb, self.mpi, gblock,
                                               compress)  # albedo of visible of the dry soil
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'soil_s_n_alb', 'patch', landpatch.landpatch,
                                               self.soil_s_n_alb, self.mpi, gblock,
                                               compress)  # albedo of near infrared of the saturated soil
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'soil_d_n_alb', 'patch', landpatch.landpatch,
                                               self.soil_d_n_alb, self.mpi, gblock,
                                               compress)  # albedo of near infrared of the dry soil

        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'vf_quartz', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.vf_quartz, self.mpi, gblock,
                                                 compress)  # volumetric fraction of quartz within mineral soil
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'vf_gravels', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.vf_gravels, self.mpi, gblock,
                                                 compress)  # volumetric fraction of gravels
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'vf_om', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.vf_om, self.mpi, gblock,
                                                 compress)  # volumetric fraction of organic matter
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'vf_sand', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.vf_sand, self.mpi, gblock,
                                                 compress)  # volumetric fraction of sand
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'wf_gravels', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.wf_gravels, self.mpi, gblock,
                                                 compress)  # gravimetric fraction of gravels
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'wf_sand', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.wf_sand, self.mpi, gblock,
                                                 compress)  # gravimetric fraction of sand
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'OM_density', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.OM_density, self.mpi, gblock,
                                                 compress)  # OM_density
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'BD_all', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.BD_all, self.mpi, gblock,
                                                 compress)  # bulk density of soil
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'wfc', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.wfc, self.mpi, gblock,
                                                 compress)  # field capacity
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'porsl', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.porsl, self.mpi, gblock,
                                                 compress)  # fraction of soil that is voids [-]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'psi0', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.psi0, self.mpi, gblock,
                                                 compress)  # minimum soil suction [mm] (NOTE: "-" valued, self. mpi, gblock, compress)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'bsw', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.bsw, self.mpi, gblock,
                                                 compress)  # clapp and hornbereger "b" parameter [-]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'theta_r', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.theta_r, self.mpi, gblock,
                                                 compress)  # residual moisture content [-]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'BVIC', 'soil', self.var_global.nl_soil, 'patch',
                                                 landpatch.landpatch, self.BVIC, self.mpi, gblock,
                                                 compress)  # b parameter in Fraction of saturated soil in a grid calculated by VIC

        if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'alpha_vgm', 'soil', self.var_global.nl_soil,
                                                     'patch', landpatch.landpatch, self.alpha_vgm, self.mpi, gblock,
                                                     compress)  # a parameter corresponding approximately to the inverse of the air-entry value
            CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'L_vgm', 'soil', self.var_global.nl_soil, 'patch',
                                                     landpatch.landpatch, self.L_vgm, self.mpi, gblock,
                                                     compress)  # pore-connectivity parameter [dimensionless]
            CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'n_vgm', 'soil', self.var_global.nl_soil, 'patch',
                                                     landpatch.landpatch, self.n_vgm, self.mpi, gblock,
                                                     compress)  # a shape parameter [dimensionless]
            CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'sc_vgm', 'soil', self.var_global.nl_soil, 'patch',
                                                     landpatch.landpatch, self.sc_vgm, self.mpi, gblock,
                                                     compress)  # saturation at the air entry value in the classical vanGenuchten model [-]
            CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'fc_vgm', 'soil', self.var_global.nl_soil, 'patch',
                                                     landpatch.landpatch, self.fc_vgm, self.mpi, gblock,
                                                     compress)  # a scaling factor by using air entry value in the Mualem model [-]

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'vic_b_infilt', 'patch', landpatch.landpatch,
                                               self.vic_b_infilt, self.mpi, gblock)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'vic_Dsmax', 'patch', landpatch.landpatch, self.vic_Dsmax,
                                               self.mpi, gblock)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'vic_Ds', 'patch', landpatch.landpatch, self.vic_Ds,
                                               self.mpi, gblock)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'vic_Ws', 'patch', landpatch.landpatch, self.vic_Ws,
                                               self.mpi, gblock)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'vic_c', 'patch', landpatch.landpatch, self.vic_c,
                                               self.mpi, gblock)
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'hksati', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.hksati, self.mpi,
                                               gblock, compress)  # hydraulic conductivity at saturation [mm h2o/s]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'csol', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.csol, self.mpi,
                                               gblock, compress)  # heat capacity of soil solids [J/(m3 K, self. mpi, gblock, compress)]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'k_solids', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.k_solids, self.mpi,
                                               gblock, compress)  # thermal conductivity of soil solids [W/m-K]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'dksatu', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.dksatu, self.mpi,
                                               gblock, compress)  # thermal conductivity of saturated soil [W/m-K]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'dksatf', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.dksatf, self.mpi,
                                               gblock, compress)  # thermal conductivity of saturated soil [W/m-K]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'dkdry', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.dkdry, self.mpi,
                                               gblock, compress)  # thermal conductivity for dry soil  [W/(m-K, self. mpi, gblock, compress)]
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'BA_alpha', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.BA_alpha, self.mpi,
                                               gblock, compress)  # alpha in Balland and Arp(2005, self. mpi, gblock, compress) thermal conductivity scheme
        CoLM_NetCDFVectorBlk.ncio_write_vector10(file_restart, 'BA_beta', 'soil', self.var_global.nl_soil, 'patch',
                                               landpatch.landpatch, self.BA_beta, self.mpi,
                                               gblock, compress)  # beta in Balland and Arp(2005, self. mpi, gblock, compress) thermal conductivity scheme

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'htop', 'patch', landpatch.landpatch, self.htop, self.mpi,
                                               gblock)  #
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'hbot', 'patch', landpatch.landpatch, self.hbot, self.mpi,
                                               gblock)

        if self.nl_colm['DEF_USE_BEDROCK']:
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'debdrock', 'patch', landpatch.landpatch,
                                                   self.dbedrock, self.mpi, gblock)
            CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'ibedrock', 'patch', landpatch.landpatch,
                                                   self.ibedrock, self.mpi, gblock)

        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'topoelv', 'patch', landpatch.landpatch, self.topoelv,
                                               self.mpi, gblock)
        CoLM_NetCDFVectorBlk.ncio_write_vector(file_restart, 'topostd', 'patch', landpatch.landpatch, self.topostd,
                                               self.mpi, gblock)

        # Synchronize processes if using MPI
        if self.nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb, p_err)

        if self.mpi.p_is_master:
            # Additional writing for specific configurations
            if (not (self.nl_colm['VectorInOneFileS']) and (not self.nl_colm['VectorInOneFileP'])):
                pass
            netfile = NetCDFFile(self.mpi.USEMPI)
            # Create the file again if necessary
            netfile.ncio_write_serial(file_restart, 'zlnd', self.zlnd)  # roughness length for soil [m]
            netfile.ncio_write_serial(file_restart, 'zsno', self.zsno)  # roughness length for snow [m]
            netfile.ncio_write_serial(file_restart, 'csoilc', self.csoilc)  # drag coefficient for soil under canopy [-]
            netfile.ncio_write_serial(file_restart, 'dewmx', self.dewmx)  # maximum dew
            netfile.ncio_write_serial(file_restart, 'wtfact',
                                      self.wtfact)  # fraction of model area with high water table
            netfile.ncio_write_serial(file_restart, 'capr',
                                      self.capr)  # tuning factor to turn first layer T into surface T
            netfile.ncio_write_serial(file_restart, 'cnfac', self.cnfac)  # Crank Nicholson factor between 0 and 1
            netfile.ncio_write_serial(file_restart, 'ssi', self.ssi)  # irreducible water saturation of snow
            netfile.ncio_write_serial(file_restart, 'wimp', self.wimp)  # water impremeable if porosity less than wimp
            netfile.ncio_write_serial(file_restart, 'pondmx', self.pondmx)  # ponding depth (mm)
            netfile.ncio_write_serial(file_restart, 'smpmax', self.smpmax)  # wilting point potential in mm
            netfile.ncio_write_serial(file_restart, 'smpmin', self.smpmin)  # restriction for min of soil poten. (mm)
            netfile.ncio_write_serial(file_restart, 'trsmx0',
                                      self.trsmx0)  # max transpiration for moist soil+100% veg.  [mm/s]
            netfile.ncio_write_serial(file_restart, 'tcrit', self.tcrit)  # critical temp. to determine rain or snow
            netfile.ncio_write_serial(file_restart, 'wetwatmax', self.wetwatmax)  # maximum wetland water (mm)

        # Additional calls for other configurations
        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            pass
            # file_restart = os.path.join(dir_restart, 'const', f"{casename}_restart_pft_const_{cyear}.nc")
            # write_pftime_invariants(file_restart)

        if self.nl_colm['BGC']:
            pass
            # file_restart = os.path.join(dir_restart, 'const', f"{casename}_restart_bgc_const_{cyear}.nc")
            # write_bgc_time_invariants(file_restart)

        if self.nl_colm['URBAN_MODEL']:
            pass
            # file_restart = os.path.join(dir_restart, 'const', f"{casename}_restart_urban_const_{cyear}.nc")
            # write_urbtime_invariants(file_restart)

    # Call the function with appropriate arguments
    # write_time_invariants(lc_year, casename, dir_restart)
    def deallocate_TimeInvariants(self):
        # --------------------------------------------------
        # Deallocates  memory for CoLM 1d[numpatch] variables
        # --------------------------------------------------
        if self.mpi.p_is_worker:
            if self.landpatch.numpatch > 0:
                del self.patchclass
                del self.patchtype
                del self.patchmask

                del self.patchlonr
                del self.patchlatr

                del self.lakedepth
                del self.dz_lake

                del self.soil_s_v_alb
                del self.soil_d_v_alb
                del self.soil_s_n_alb
                del self.soil_d_n_alb

                del self.vf_quartz
                del self.vf_gravels
                del self.vf_om
                del self.vf_sand
                del self.wf_gravels
                del self.wf_sand
                del self.OM_density
                del self.BD_all
                del self.wfc
                del self.porsl
                del self.psi0
                del self.bsw
                del self.theta_r
                del self.BVIC

                if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                    del self.alpha_vgm
                    del self.L_vgm
                    del self.n_vgm
                    del self.sc_vgm
                    del self.fc_vgm

                del self.vic_b_infilt
                del self.vic_Dsmax
                del self.vic_Ds
                del self.vic_Ws
                del self.vic_c

                del self.hksati
                del self.csol
                del self.k_solids
                del self.dksatu
                del self.dksatf
                del self.dkdry
                del self.BA_alpha
                del self.BA_beta

                del self.htop
                del self.hbot

                del self.dbedrock
                del self.ibedrock

                del self.topoelv
                del self.topostd

        # # if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
        # deallocate_PFTimeInvariants
        #
        # # ifdef BGC
        # deallocate_BGCTimeInvariants
        #
        # # ifdef URBAN_MODEL
        # deallocate_UrbanTimeInvariants

    def read_time_invariants(self, lc_year, casename, dir_restart, gblock):
        # Local variables
        cyear = lc_year
        file_restart = f"{dir_restart}/const/{casename}_restart_const_lc{lc_year:04d}.nc"
        
        self.patchclass = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'patchclass', self.landpatch.landpatch, self.patchclass, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.patchtype = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'patchtype', self.landpatch.landpatch, self.patchtype, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.patchmask = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'patchmask', self.landpatch.landpatch, self.patchmask, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.patchlonr = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'patchlonr', self.landpatch.landpatch, self.patchlonr, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.patchlatr = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'patchlatr', self.landpatch.landpatch, self.patchlatr, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.lakedepth = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'lakedepth', self.landpatch.landpatch, self.lakedepth, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.dz_lake = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'dz_lake', self.var_global.nl_lake, self.landpatch.landpatch, self.dz_lake, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.soil_s_v_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'soil_s_v_alb', self.landpatch.landpatch, self.soil_s_v_alb, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.soil_d_v_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'soil_d_v_alb', self.landpatch.landpatch, self.soil_d_v_alb, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.soil_s_n_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'soil_s_n_alb', self.landpatch.landpatch, self.soil_s_n_alb, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.soil_d_n_alb = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'soil_d_n_alb', self.landpatch.landpatch, self.soil_d_n_alb, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.vf_quartz = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vf_quartz', self.var_global.nl_soil, self.landpatch.landpatch, self.vf_quartz, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.vf_gravels = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vf_gravels', self.var_global.nl_soil, self.landpatch.landpatch, self.vf_gravels, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.vf_om = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vf_om', self.var_global.nl_soil, self.landpatch.landpatch, self.vf_om, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.vf_sand = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'vf_sand', self.var_global.nl_soil, self.landpatch.landpatch, self.vf_sand, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.wf_gravels = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wf_gravels', self.var_global.nl_soil, self.landpatch.landpatch, self.wf_gravels, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.wf_sand = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wf_sand', self.var_global.nl_soil, self.landpatch.landpatch, self.wf_sand, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.OM_density = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'OM_density', self.var_global.nl_soil, self.landpatch.landpatch, self.OM_density, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.BD_all = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'BD_all', self.var_global.nl_soil, self.landpatch.landpatch, self.BD_all, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.wfc = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'wfc', self.var_global.nl_soil, self.landpatch.landpatch, self.wfc, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.porsl = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'porsl', self.var_global.nl_soil, self.landpatch.landpatch, self.porsl, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.psi0 = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'psi0', self.var_global.nl_soil, self.landpatch.landpatch, self.psi0, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.bsw = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'bsw', self.var_global.nl_soil, self.landpatch.landpatch, self.bsw, self.nl_colm['USEMPI'], self.mpi, gblock)
        
        # Conditional compilation and additional vectors
        self.theta_r = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'theta_r', self.var_global.nl_soil, self.landpatch.landpatch, self.theta_r, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.BVIC = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'BVIC', self.var_global.nl_soil,
                                                             self.landpatch.landpatch, self.BVIC,
                                                             self.nl_colm['USEMPI'], self.mpi, gblock)

        self.alpha_vgm = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'alpha_vgm', self.var_global.nl_soil, self.landpatch.landpatch, self.alpha_vgm, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.L_vgm = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'L_vgm', self.var_global.nl_soil, self.landpatch.landpatch, self.L_vgm, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.n_vgm = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'n_vgm', self.var_global.nl_soil, self.landpatch.landpatch, self.n_vgm, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.sc_vgm = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'sc_vgm', self.var_global.nl_soil, self.landpatch.landpatch, self.sc_vgm, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.fc_vgm = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'fc_vgm', self.var_global.nl_soil, self.landpatch.landpatch, self.fc_vgm, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.hksati = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'hksati', self.var_global.nl_soil, self.landpatch.landpatch, self.hksati, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.csol = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'csol', self.var_global.nl_soil, self.landpatch.landpatch, self.csol, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.k_solids = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'k_solids', self.var_global.nl_soil, self.landpatch.landpatch, self.k_solids, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.dksatu = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'dksatu', self.var_global.nl_soil, self.landpatch.landpatch, self.dksatu, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.dksatf = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'dksatf', self.var_global.nl_soil, self.landpatch.landpatch, self.dksatf, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.dkdry = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'dkdry', self.var_global.nl_soil, self.landpatch.landpatch, self.dkdry, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.BA_alpha = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'BA_alpha', self.var_global.nl_soil, self.landpatch.landpatch, self.BA_alpha, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.BA_beta = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'BA_beta', self.var_global.nl_soil, self.landpatch.landpatch, self.BA_beta, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.htop = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'htop', self.landpatch.landpatch, self.htop, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.hbot = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'hbot', self.landpatch.landpatch, self.hbot, self.nl_colm['USEMPI'], self.mpi, gblock)

        # Read bedrock data if defined
        if self.nl_colm['DEF_USE_BEDROCK']:
            self.dbedrock = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'debdrock', self.landpatch.landpatch, self.dbedrock, self.nl_colm['USEMPI'], self.mpi, gblock)
            self.ibedrock = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'ibedrock', self.landpatch.landpatch, self.ibedrock, self.nl_colm['USEMPI'], self.mpi, gblock)

        netfile = NetCDFFile(self.nl_colm['USEMPI'])
        # Read and broadcast serial data
        self.topoelv = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'topoelv', self.landpatch.landpatch, self.topoelv, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.topostd = CoLM_NetCDFVectorBlk.ncio_read_vector(file_restart, 'topostd', self.landpatch.landpatch, self.topostd, self.nl_colm['USEMPI'], self.mpi, gblock)
        self.zlnd = netfile.ncio_read_bcast_serial(file_restart, 'zlnd')
        self.zsno = netfile.ncio_read_bcast_serial(file_restart, 'zsno')
        self.csoilc = netfile.ncio_read_bcast_serial(file_restart, 'csoilc')
        self.dewmx = netfile.ncio_read_bcast_serial(file_restart, 'dewmx')
        self.wtfact = netfile.ncio_read_bcast_serial(file_restart, 'wtfact')
        self.capr = netfile.ncio_read_bcast_serial(file_restart, 'capr')
        self.cnfac = netfile.ncio_read_bcast_serial(file_restart, 'cnfac')
        self.ssi = netfile.ncio_read_bcast_serial(file_restart, 'ssi')
        self.wimp = netfile.ncio_read_bcast_serial(file_restart, 'wimp')
        self.pondmx = netfile.ncio_read_bcast_serial(file_restart, 'pondmx')
        self.smpmax = netfile.ncio_read_bcast_serial(file_restart, 'smpmax')
        self.smpmin = netfile.ncio_read_bcast_serial(file_restart, 'smpmin')
        self.trsmx0 = netfile.ncio_read_bcast_serial(file_restart, 'trsmx0')
        self.tcrit = netfile.ncio_read_bcast_serial(file_restart, 'tcrit')
        self.wetwatmax = netfile.ncio_read_bcast_serial(file_restart, 'wetwatmax')

        # Read additional data if specific conditions are met
        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            file_restart = f"{dir_restart}/const/{casename}_restart_pft_const_lc{lc_year}.nc"
            pftimeInvariants = Vars_PFTimeInvariants()
            pftimeInvariants.read_pf_time_invariants(file_restart, self.landpatch)

        # if 'BGC' in globals():
        #     file_restart = f"{dir_restart}/const/{casename}_restart_bgc_const_lc{lc_year:04d}.nc"
        #     read_bgc_time_invariants(file_restart)

        # if 'URBAN_MODEL' in globals():
        #     file_restart = f"{dir_restart}/const/{casename}_restart_urb_const_lc{lc_year:04d}.nc"
        #     read_urban_time_invariants(file_restart)

        if self.nl_colm['RangeCheck']:
            self.check_TimeInvariants()

        # Print completion message if master process
        if self.mpi.p_is_master:
            print('Loading Time Invariants done.')