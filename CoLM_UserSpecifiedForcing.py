import numpy as np 
import CoLM_Qsadv
class UserSpecifiedForcing:
    def __init__(self, nl_colm, nl_colm_forcing, gblock):
        self.dataset = ""
        self.gblock = gblock
        self.nl_colm = nl_colm
        self.nl_colm_forcing = nl_colm_forcing

        self.solarin_all_band = False
        self.HEIGHT_V = 0.0
        self.HEIGHT_T = 0.0
        self.HEIGHT_Q = 0.0

        self.NVAR = 0
        self.startyr = 0
        self.startmo = 0
        self.endyr = 0
        self.endmo = 0

        self.dtime = []
        self.offset = []

        self.leapyear = False
        self.data2d = False
        self.hightdim = False
        self.dim2d = False

        self.latname = ""
        self.lonname = ""
        self.groupby = ""

        self.fprefix = []
        self.vname = []
        self.tintalgo = []

    def init_user_specified_forcing(self):
        self.NVAR = self.nl_colm_forcing['DEF_forcing']['NVAR']
        NVAR_default = self.NVAR
        if self.nl_colm['DEF_USE_CBL_HEIGHT']:
            self.NVAR += 1

        self.dtime = [0] * self.NVAR
        self.offset = [0] * self.NVAR

        self.fprefix = [""] * self.NVAR
        self.vname = [""] * self.NVAR
        self.tintalgo = [""] * self.NVAR

        self.solarin_all_band = self.nl_colm_forcing['DEF_forcing']['solarin_all_band']
        self.HEIGHT_V = self.nl_colm_forcing['DEF_forcing']['HEIGHT_V']
        self.HEIGHT_T = self.nl_colm_forcing['DEF_forcing']['HEIGHT_T']
        self.HEIGHT_Q = self.nl_colm_forcing['DEF_forcing']['HEIGHT_Q']

        self.startyr = self.nl_colm_forcing['DEF_forcing']['startyr']
        self.startmo = self.nl_colm_forcing['DEF_forcing']['startmo']
        self.endyr = self.nl_colm_forcing['DEF_forcing']['endyr']
        self.endmo = self.nl_colm_forcing['DEF_forcing']['endmo']

        self.dtime = self.nl_colm_forcing['DEF_forcing']['dtime']
        self.offset = self.nl_colm_forcing['DEF_forcing']['offset']

        self.leapyear = self.nl_colm_forcing['DEF_forcing']['leapyear']
        self.data2d = self.nl_colm_forcing['DEF_forcing']['data2d']
        self.hightdim = self.nl_colm_forcing['DEF_forcing']['hightdim']
        self.dim2d = self.nl_colm_forcing['DEF_forcing']['dim2d']

        self.latname = self.nl_colm_forcing['DEF_forcing']['latname']
        self.lonname = self.nl_colm_forcing['DEF_forcing']['lonname']
        self.groupby = self.nl_colm_forcing['DEF_forcing']['groupby']

        for ivar in range(NVAR_default):
            self.fprefix[ivar] = self.nl_colm_forcing['DEF_forcing']['fprefix'][ivar]
            self.vname[ivar] = self.nl_colm_forcing['DEF_forcing']['vname'][ivar]
            self.tintalgo[ivar] = self.nl_colm_forcing['DEF_forcing']['tintalgo'][ivar]

        if self.nl_colm['DEF_USE_CBL_HEIGHT']:
            self.fprefix[-1] = self.nl_colm_forcing['DEF_forcing']['CBL_fprefix']
            self.vname[-1] = self.nl_colm_forcing['DEF_forcing']['CBL_vname']
            self.tintalgo[-1] = self.nl_colm_forcing['DEF_forcing']['CBL_tintalgo']
            self.dtime[-1] = self.nl_colm_forcing['DEF_forcing']['CBL_dtime']
            self.offset[-1] = self.nl_colm_forcing['DEF_forcing']['CBL_offset']

    def metfilename(self, year, month, day, var_i):
        yearstr = year
        monthstr = month
        case = self.nl_colm_forcing['DEF_forcing']['dataset']
        metfilename = ""

        if month <10:
            monthstr = '0' + str(month)

        if case == 'PRINCETON':
            metfilename = '/'+ self.fprefix[var_i]+ str(yearstr) + '-' + str(yearstr) + '.nc'
        elif case == 'GSWP3':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + '-' + str(monthstr) + '.nc'
        elif case == 'QIAN':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + '-' + str(monthstr) + '.nc'
        elif case == 'CRUNCEPV4':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + '-' + str(monthstr) + '.nc'
        elif case == 'CRUNCEPV7':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + '-' + str(monthstr) + '.nc'
        elif case == 'ERA5LAND':
            metfilename = '/' + self.fprefix[var_i] + '_' + str(yearstr) + '_' + str(monthstr)

            if var_i==1:
                metfilename = metfilename + '_2m_temperature.nc'
            elif var_i==2:
                metfilename = metfilename + '_specific_humidity.nc'
            elif var_i == 3:
                metfilename = metfilename + '_surface_pressure.nc'
            elif var_i == 4:
                metfilename = metfilename + '_total_precipitation_m_hr.nc'
            elif var_i==5:
                metfilename = metfilename + '_10m_u_component_of_wind.nc'
            elif var_i==6:
                metfilename = metfilename + '_10m_v_component_of_wind.nc'
            elif var_i==7:
                metfilename = metfilename + '_surface_solar_radiation_downwards_w_m2.nc'
            elif var_i==8:
                metfilename = metfilename + '_surface_thermal_radiation_downwards_w_m2.nc'
        elif case == 'ERA5':
            metfilename = '/' + self.fprefix[var_i] + '_' + str(yearstr) + '_' + str(monthstr)
            if var_i==1:
                metfilename = metfilename + '_2m_temperature.nc4'
            elif var_i==2:
                metfilename = metfilename + '_q.nc4'
            elif var_i == 3:
                metfilename = metfilename + '_surface_pressure.nc4'
            elif var_i == 4:
                metfilename = metfilename + '_mean_total_precipitation_rate.nc4'
            elif var_i==5:
                metfilename = metfilename + '_10m_u_component_of_wind.nc4'
            elif var_i==6:
                metfilename = metfilename + '_10m_v_component_of_wind.nc4'
            elif var_i==7:
                metfilename = metfilename + '_mean_surface_downward_short_wave_radiation_flux.nc4'
            elif var_i==8:
                metfilename = metfilename + '_mean_surface_downward_long_wave_radiation_flux.nc4'
        elif case == 'MSWX':
            metfilename = '/' + self.fprefix[var_i] + '_' + str(yearstr) + '_' + str(monthstr) + '.nc'
        elif case == 'WFDE5':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + str(monthstr) + '_v2.0.nc'
        elif case == 'CRUJRA':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + '.365d.noc.nc'
        elif case == 'WFDEI':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + '-' + str(monthstr) + '.nc'
        elif case == 'JRA3Q':
            metfilename = '/' + self.fprefix[var_i] + '_' + str(yearstr) + '_' + str(monthstr) + '.nc'
        elif case == 'JRA55':
            metfilename = '/' + self.fprefix[var_i] + '_' + str(yearstr) + '.nc'
        elif case == 'GDAS':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + str(monthstr) + '.nc4'
        elif case == 'CLDAS':
            metfilename = '/' + self.fprefix[var_i] + '-' + str(yearstr) + str(monthstr) + '.nc'
        elif case == 'CMFD':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + str(monthstr) + '.nc4'
        elif case == 'CMIP6':
            metfilename = '/' + self.fprefix[var_i] + '_' + str(yearstr) + '.nc'
        elif case == 'CMIP6':
            metfilename = '/' + self.fprefix[var_i] + str(yearstr) + str(monthstr) + '.nc'
        elif case == 'POINT':
            metfilename = '/' + self.fprefix[1]

        if self.nl_colm['DEF_USE_CBL_HEIGHT']:
            if var_i == 9:
                metfilename = '/' + self.fprefix[9] + '_' + str(yearstr) + '_' + str(monthstr) + '_boundary_layer_height.nc4'
        return metfilename


    def metpreprocess(self, grid, forcn):
        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
            # Implement SinglePoint logic
            pass
        else:
            for iblkme in range(self.gblock.nblkme):
                ib = self.gblock.xblkme[iblkme]
                jb = self.gblock.yblkme[iblkme]

                for j in range(grid.ycnt[jb]):
                    for i in range(grid.xcnt[ib]):
                        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'PRINCETON':
                            # Implement PRINCETON logic
                            es, esdT, qsat_tmp, dqsat_tmpdT = CoLM_Qsadv.qsadv(forcn[0].val[i, j], forcn[2].val[i, j])
                            if qsat_tmp < forcn[1].val[i, j]:
                                forcn[1].val[i, j] = qsat_tmp
                        elif self.nl_colm_forcing['DEF_forcing']['dataset'] == 'GSWP2':
                            # Implement GSWP2 logic
                            es, esdT, qsat_tmp, dqsat_tmpdT = CoLM_Qsadv.qsadv(forcn[0].val[i, j], forcn[2].val[i, j])
                            if qsat_tmp < forcn[1].val[i, j]:
                                forcn[1].val[i, j] = qsat_tmp
                        elif self.nl_colm_forcing['DEF_forcing']['dataset'] == 'GSWP3':
                            # Implement GSWP3 logic
                            if forcn[0].blk[ib,jb].val[i,j] < 212.0:
                                forcn[0].blk[ib,jb].val[i,j] = 212.0
                            if forcn[3].blk[ib,jb].val[i,j] < 0.0:
                                forcn[3].blk[ib,jb].val[i,j] = 0.0
                            es, esdT, qsat_tmp, dqsat_tmpdT = CoLM_Qsadv.qsadv(forcn[0].blk[ib,jb].val[i,j], forcn[2].blk[ib,jb].val[i,j])
                            if qsat_tmp < forcn[1].blk[ib,jb].val[i,j]:
                                forcn[1].blk[ib,jb].val[i,j] = qsat_tmp
                        elif self.nl_colm_forcing['DEF_forcing']['dataset'] == 'QIAN':
                            # Implement QIAN logic
                            es, esdT, qsat_tmp, dqsat_tmpdT = CoLM_Qsadv.qsadv(forcn[0].val[i, j], forcn[2].val[i, j])
                            if qsat_tmp < forcn[1].val[i, j]:
                                forcn[1].val[i, j] = qsat_tmp
                            e = forcn[2].val[i, j] * forcn[1].val[i, j] / (0.622 + 0.378 * forcn[1].val[i, j])
                            ea = 0.70 + 5.95e-05 * 0.01 * e * np.exp(1500.0 / forcn[0].val[i, j])
                            forcn[7].val[i, j] = ea * 5.670373e-8 * forcn[0].val[i, j]**4
                        # Continue implementing other cases...

        return forcn