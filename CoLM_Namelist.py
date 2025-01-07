import os
import sys
import config


class CoLM_Namelist(object):
    def __init__(self, path_root, info_define, mpi) -> None:
        self.mpi = mpi

        # A group includes one "IO" process and several "worker" processes.
        # Its size determines number of IOs in a job.
        path_default = os.path.join(path_root, '__base__/default_fsrfdata.yml')
        path_forcing = os.path.join(path_root, '__base__/default_forcing.yml')
        path_history = os.path.join(path_root, '__base__/default_history.yml')
        path_simulation_time = os.path.join(path_root, '__base__/default_simulation_time.yml')

        plat_system = sys.platform
        if 'win' in plat_system:
            path_default = path_root + '\\' + '__base__\\default_fsrfdata.yml'
            path_forcing = path_root + '\\' + '__base__\\default_forcing.yml'
            path_history = path_root + '\\' + '__base__\\default_history.yml'
            path_simulation_time = path_root + '\\' + '__base__\\default_simulation_time.yml'
        dic_default = config.parse_from_yaml(path_default)
        dic_forcing = config.parse_from_yaml(path_forcing)
        dic_history = config.parse_from_yaml(path_history)
        dic_simulation_time = config.parse_from_yaml(path_simulation_time)

        # 如果是主线程则执行下面
        if self.mpi.p_is_master:
            nlfile = os.path.join(path_root, 'Guandong_2020.yml')
            assert os.path.exists(nlfile), \
                'Config path ({}) does not exist'.format(nlfile)
            assert nlfile.endswith('yml') or nlfile.endswith('yaml'), \
                'Config file ({}) should be yaml format'.format(nlfile)

            self.nl_colm = config.parse_from_yaml(nlfile)
            self.nl_colm = config.merge_config_dicts(self.nl_colm, info_define)
            self.nl_colm = config.merge_config_dicts(self.nl_colm, dic_default)
            self.nl_colm = config.merge_config_dicts(self.nl_colm, dic_simulation_time)
            self.nl_colm.update({'DEF_domain': Domain_type()})  # SinglePoint no userd

            if self.nl_colm['USEMPI']:
                # CALL mpi_abort (p_comm_glb, p_err)
                pass

            self.nl_colm_forcing = config.parse_from_yaml(str(os.path.join(path_root, self.nl_colm['DEF_forcing_namelist'])))
            self.nl_colm_forcing = config.merge_config_dicts(self.nl_colm_forcing, dic_forcing)

            if self.nl_colm['SinglePoint']:
                self.nl_colm_forcing['DEF_forcing']['has_missing_value'] = False

            plat_system = sys.platform
            if 'win' in plat_system:
                self.nl_colm['DEF_dir_landdata'] = self.nl_colm['DEF_dir_output'] + '\\' + self.nl_colm['DEF_CASE_NAME'] + '\\landdata'
                self.nl_colm['DEF_dir_restart'] = self.nl_colm['DEF_dir_output'] + '\\' + self.nl_colm['DEF_CASE_NAME'] + '\\restart'
                self.nl_colm['DEF_dir_history'] = self.nl_colm['DEF_dir_output'] + '\\' + self.nl_colm['DEF_CASE_NAME'] + '\\history'
            else:
                self.nl_colm['DEF_dir_landdata'] = os.path.join(self.nl_colm['DEF_dir_output'], self.nl_colm['DEF_CASE_NAME'],
                                                'landdata')
                self.nl_colm['DEF_dir_restart'] = os.path.join(self.nl_colm['DEF_dir_output'], self.nl_colm['DEF_CASE_NAME'], 'restart')
                self.nl_colm['DEF_dir_history'] = os.path.join(self.nl_colm['DEF_dir_output'], self.nl_colm['DEF_CASE_NAME'], 'history')

            self.create_directory(self.nl_colm['DEF_dir_output'])
            self.create_directory(self.nl_colm['DEF_dir_landdata'])
            self.create_directory(self.nl_colm['DEF_dir_restart'])
            self.create_directory(self.nl_colm['DEF_dir_history'])

            if self.nl_colm['SinglePoint']:
                self.nl_colm['DEF_nx_blocks'] = 360
                self.nl_colm['DEF_ny_blocks'] = 180
                DEF_HIST_mode = 'one'

            # ===============================================================
            # ----- Macros&Namelist conflicts and dependency management -----
            # ===============================================================
            # ----- SOIL model related ------ Macros&Namelist conflicts and dependency management
            if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                print('*********************************************************************** ')
                print('Note: DEF_USE_VARIABLY_SATURATED_FLOW is automaticlly set to .true.  ')
                print('when using vanGenuchten_Mualem_SOIL_MODEL. ')
                self.nl_colm['DEF_USE_VARIABLY_SATURATED_FLOW'] = True
            if self.nl_colm['LATERAL_FLOW']:
                print('*********************************************************************** ')
                print('Note: DEF_USE_VARIABLY_SATURATED_FLOW is automaticlly set to .true.  ')
                print('when defined LATERAL_FLOW. ')
                self.nl_colm['DEF_USE_VARIABLY_SATURATED_FLOW'] = True

            # ----- subgrid type related ------ Macros&Namelist conflicts and dependency management
            if self.nl_colm['LULC_USGS'] or self.nl_colm['LULC_IGBP']:
                self.nl_colm['DEF_USE_LCT'] = True
                self.nl_colm['DEF_USE_PFT'] = False
                self.nl_colm['DEF_USE_PC'] = False
                self.nl_colm['DEF_FAST_PC'] = False
            if self.nl_colm['LULC_IGBP_PFT']:
                self.nl_colm['DEF_USE_LCT'] = False
                self.nl_colm['DEF_USE_PFT'] = True
                self.nl_colm['DEF_USE_PC'] = False
                self.nl_colm['DEF_FAST_PC'] = False

            if self.nl_colm['LULC_IGBP_PC']:
                self.nl_colm['DEF_USE_LCT'] = False
                self.nl_colm['DEF_USE_PFT'] = False
                self.nl_colm['DEF_USE_PC'] = True
                self.nl_colm['DEF_FAST_PC'] = False

            if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
                if not self.nl_colm['DEF_LAI_MONTHLY']:
                    print('*********************************************************************** ')
                    print('Warning: 8-day LAI data is not supported for ')
                    print('LULC_IGBP_PFT and LULC_IGBP_PC.')
                    print('Changed to monthly data, set DEF_LAI_MONTHLY = .true.')
                    self.nl_colm['DEF_LAI_MONTHLY'] = True

            # ----- BGC and CROP model related ------ Macros&Namelist conflicts and dependency management
            if self.nl_colm['BGC']:
                if self.nl_colm['DEF_USE_LAIFEEDBACK']:
                    self.nl_colm['DEF_USE_LAIFEEDBACK'] = False
                    print('*********************************************************************** ')
                    print('Warning: LAI feedback is not supported for BGC off.')
                    print('DEF_USE_LAIFEEDBACK is set to false automatically when BGC is turned off.')

                if self.nl_colm['DEF_USE_SASU']:
                    self.nl_colm['DEF_USE_SASU'] = False
                    print('*********************************************************************** ')
                    print('Warning: Semi-Analytic Spin-up is on when BGC is off.')
                    print('DEF_USE_SASU is set to false automatically when BGC is turned off.')

                if self.nl_colm['DEF_USE_PN']:
                    self.nl_colm['DEF_USE_PN'] = False
                    print('*********************************************************************** ')
                    print('Warning: Punctuated nitrogen addition spin up is on when BGC is off.')
                    print('DEF_USE_PN is set to false automatically when BGC is turned off.')

                if self.nl_colm['DEF_USE_NITRIF']:
                    self.nl_colm['DEF_USE_NITRIF'] = False
                    print('*********************************************************************** ')
                    print('Warning: Nitrification-Denitrification is on when BGC is off.')
                    print('DEF_USE_NITRIF is set to false automatically when BGC is turned off.')

                if self.nl_colm['DEF_USE_FIRE']:
                    self.nl_colm['DEF_USE_FIRE'] = False
                    print('*********************************************************************** ')
                    print('Warning: Fire model is on when BGC is off.')
                    print('DEF_USE_FIRE is set to false automatically when BGC is turned off.')

            if self.nl_colm['CROP']:
                if self.nl_colm['DEF_USE_FERT']:
                    self.nl_colm['DEF_USE_FERT'] = False
                    print('*********************************************************************** ')
                    print('Warning: Fertilization is on when CROP is off.')
                    print('DEF_USE_FERT is set to false automatically when CROP is turned off.')

                if self.nl_colm['DEF_USE_CNSOYFIXN']:
                    self.nl_colm['DEF_USE_CNSOYFIXN'] = False
                    print('*********************************************************************** ')
                    print('Warning: Soy nitrogen fixation is on when CROP is off.')
                    print('DEF_USE_CNSOYFIXN is set to false automatically when CROP is turned off.')

                if self.nl_colm['DEF_USE_IRRIGATION']:
                    self.nl_colm['DEF_USE_IRRIGATION'] = False
                    print('*********************************************************************** ')
                    print('Warning: irrigation is on when CROP is off.')
                    print('DEF_USE_IRRIGATION is set to false automatically when CROP is turned off.')

            if not self.nl_colm['DEF_USE_OZONESTRESS']:
                if self.nl_colm['DEF_USE_OZONEDATA']:
                    self.nl_colm['DEF_USE_OZONEDATA'] = False
                print('*********************************************************************** ')
                print('Warning: DEF_USE_OZONEDATA is not supported for OZONESTRESS off.')
                print('DEF_USE_OZONEDATA is set to false automatically.')

            # ----- SNICAR model ------ Macros&Namelist conflicts and dependency management
            self.nl_colm['DEF_file_snowoptics'] = os.path.join(self.nl_colm['DEF_dir_runtime'],
                                               'snicar/snicar_optics_5bnd_mam_c211006.nc')
            self.nl_colm['DEF_file_snowaging'] = os.path.join(self.nl_colm['DEF_dir_runtime'],
                                              'snicar/snicar_drdt_bst_fit_60_c070416.nc')

            if 'win' in plat_system:
                self.nl_colm['DEF_file_snowoptics'] = self.nl_colm['DEF_dir_runtime'] + '\\snicar\\snicar_optics_5bnd_mam_c211006.nc'
                self.nl_colm['DEF_file_snowaging'] = self.nl_colm['DEF_dir_runtime'] + '\\snicar\\snicar_drdt_bst_fit_60_c070416.nc'

            if not self.nl_colm['DEF_USE_SNICAR']:
                if self.nl_colm['DEF_Aerosol_Readin']:
                    self.nl_colm['DEF_Aerosol_Readin'] = False
                print('*********************************************************************** ')
                print('Warning: DEF_Aerosol_Readin is not needed for DEF_USE_SNICAR off. ')
                print('DEF_Aerosol_Readin is set to false automatically.')
            if not self.nl_colm['URBAN_MODEL']:
                if self.nl_colm['DEF_URBAN_RUN']:
                    print('*********************************************************************** ')
                    print('Note: SNICAR is not applied for URBAN model, but for other land covers. ')
            else:
                if self.nl_colm['DEF_URBAN_RUN']:
                    print('*********************************************************************** ')
                    print('Note: The Urban model is not opened. IF you want to run Urban model ')
                    print('please #define URBAN_MODEL in define.h. otherwise DEF_URBAN_RUN will ')
                    print('be set to false automatically.')
                    self.nl_colm['DEF_URBAN_RUN'] = False

            if self.nl_colm['LULCC']:
                if self.nl_colm['LULC_USGS'] or self.nl_colm['BGC']:
                    print('*********************************************************************** ')
                    print('Fatal ERROR: LULCC is not supported for LULC_USGS/BGC at present. STOP! ')
                    self.mpi.MPI_stop()

                if not self.nl_colm['DEF_LAI_MONTHLY']:
                    print('*********************************************************************** ')
                    print('Note: When LULCC is opened, DEF_LAI_MONTHLY ')
                    print('will be set to true automatically.')
                    DEF_LAI_MONTHLY = True

                if not self.nl_colm['DEF_LAI_CHANGE_YEARLY']:
                    print('*********************************************************************** ')
                    print('Note: When LULCC is opened, DEF_LAI_CHANGE_YEARLY ')
                    print('will be set to true automatically.')
                    DEF_LAI_CHANGE_YEARLY = True

                if self.nl_colm['LULC_IGBP_PC'] or self.nl_colm['URBAN']:
                    print('*********************************************************************** ')
                    print('Fatal ERROR: LULCC is not supported for LULC_IGBP_PC/URBAN at present. STOP! ')
                    print('It is coming soon. ')
                    self.mpi.MPI_stop()
            # ----- [Complement IF needed] ----- Macros&Namelist conflicts and dependency management

            if self.nl_colm['SinglePoint'] and self.nl_colm['SrfdataDiag']:
                print('*********************************************************************** ')
                print('Surface data diagnose is closed in SinglePoint case.')
                self.nl_colm['SrfdataDiag'] = False

        # -----END Macros&Namelist conflicts and dependency management -----
        # ===============================================================

        if self.nl_colm['USEMPI']:
            pass
            # 并行处理

        if self.mpi.p_is_master:
            path = ''
            if 'win' in plat_system:
                path = path_root + '\\' + self.nl_colm['DEF_hist_vars_namelist']
            else:
                path = os.path.join(path_root, self.nl_colm['DEF_hist_vars_namelist'])

            if os.path.exists(path):
                self.nl_colm_history = config.parse_from_yaml(nlfile)
                self.nl_colm_history = config.merge_config_dicts(self.nl_colm_history, dic_history)
            else:
                self.nl_colm_history = dic_history
                print('History namelist file: ' + self.nl_colm['DEF_hist_vars_namelist'] + ' does not exist.')

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


class Domain_type(object):
    def __init__(self) -> None:
        self.edges = -90.0
        self.edgen = 90.0
        self.edgew = -180.0
        self.edgee = 180.0
