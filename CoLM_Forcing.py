import copy

import numpy as np
import os
import math
from CoLM_DataType import DataType, BlockData
import CoLM_TimeManager
import CoLM_OrbCoszen
import CoLM_RangeCheck
import CoLM_NetCDFBlock
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_Grid import Grid_type
from CoLM_MonthlyinSituCO2MaunaLoa import CoLM_MonthlyinSituCO2MaunaLoa
from CoLM_Mapping_Grid2Pset import MappingGrid2PSet
from CoLM_UserSpecifiedForcing import UserSpecifiedForcing
import CoLM_NetCDFVectorBlk
from CoLM_ForcingDownscaling import CoLM_ForcingDownscaling


class Forcing(object):
    def __init__(self, nl_colm, nl_colm_forcing, gblock, landpatch, mesh, pixel, mpi, var_global) -> None:
        self.gblock = gblock
        self.landpatch = landpatch
        self.nl_colm_forcing = nl_colm_forcing
        self.nl_colm = nl_colm
        self.mesh = mesh
        self.pixel = pixel
        self.mpi = mpi
        self.co2 = CoLM_MonthlyinSituCO2MaunaLoa()
        self.datatype = DataType(self.gblock)
        self.userForcing = UserSpecifiedForcing(self.nl_colm, self.nl_colm_forcing, self.gblock)
        self.downscaling = CoLM_ForcingDownscaling(nl_colm)
        self.netfile = NetCDFFile(mpi)
        self.gforc = Grid_type(nl_colm, gblock)  # grid_type
        self.mg2p_forc = MappingGrid2PSet(nl_colm, gblock, mpi, var_global.spval)  # spatial_mapping_type
        self.forcmask_pch = None

        self.topo_grid = None
        self.maxelv_grid = None
        self.sumarea_grid = None
        self.forc_mask = None

        self.forc_topo_grid = None
        self.forc_maxelv_grid = None
        self.forc_t_grid = None
        self.forc_th_grid = None
        self.forc_q_grid = None
        self.forc_pbot_grid = None
        self.forc_rho_grid = None
        self.forc_prc_grid = None
        self.forc_prl_grid = None
        self.forc_lwrad_grid = None
        self.forc_swrad_grid = None
        self.forc_hgt_grid = None
        self.forc_us_grid = None
        self.forc_vs_grid = None

        self.forc_t_part = None
        self.forc_th_part = None
        self.forc_q_part = None
        self.forc_pbot_part = None
        self.forc_rhoair_part = None
        self.forc_prc_part = None
        self.forc_prl_part = None
        self.forc_frl_part = None
        self.forc_swrad_part = None
        self.forc_us_part = None
        self.forc_vs_part = None

        self.glacierss = False

        self.deltim_int = None

        self.forctime = None
        self.iforctime = None
        self.forcing_read_ahead = False
        self.forc_disk = None

        self.tstamp_LB = None
        self.tstamp_UB = None

        self.avgcos = None
        self.metdata = None
        self.forcn = None
        self.forcn_LB = None
        self.forcn_UB = None

    def setstampLB(self, mtstamp, var_i):
        year = mtstamp.year  # mtstamp.year
        month = 0
        mday = 0
        time_i = 0
        day = mtstamp.day  # mtstamp.day
        sec = mtstamp.sec  # mtstamp.sec

        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
            ntime = len(self.forctime)
            time_i = 1

            if mtstamp < self.forctime or self.forctime[ntime - 1] < mtstamp:
                print('Error: Forcing does not cover simulation period!')
                print(f'Need {mtstamp}, Forc start {self.forctime}, Forc END {self.forctime[ntime - 1]}')
                # CoLM_stop()
            else:
                while not (mtstamp < self.forctime[time_i + 1]):
                    time_i += 1
                self.iforctime[var_i] = time_i
                self.tstamp_LB[var_i] = self.forctime[self.iforctime[var_i]]
            return mtstamp, var_i, year, month, mday, time_i
        self.tstamp_LB[var_i].year = year
        self.tstamp_LB[var_i].day = day

        # In the case of one year one file
        if self.nl_colm_forcing['DEF_forcing']['groupby'] == 'year':
            # Calculate the initial second
            sec = 86400 * (day - 1) + sec
            time_i = int((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                         self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1
            sec = (time_i - 1) * self.nl_colm_forcing['DEF_forcing']['dtime'][var_i] + \
                  self.nl_colm_forcing['DEF_forcing']['offset'][var_i] - 86400 * (day - 1)
            self.tstamp_LB[var_i].sec = sec

            # Set time stamp (ststamp_LB)
            if sec < 0:
                self.tstamp_LB[var_i].sec = 86400 + sec
                self.tstamp_LB[var_i].day = day - 1
                if self.tstamp_LB[var_i].day == 0:
                    self.tstamp_LB[var_i].year = year - 1
                    if CoLM_TimeManager.isleapyear(self.tstamp_LB[var_i].year):
                        self.tstamp_LB[var_i].day = 366
                    else:
                        self.tstamp_LB[var_i].day = 365

            # Set record info (year, time_i)
            if sec < 0 or (sec == 0 and self.nl_colm_forcing['DEF_forcing']['offset'][var_i] != 0):
                if year == self.nl_colm_forcing['DEF_forcing']['startyr'] and month == \
                        self.nl_colm_forcing['DEF_forcing']['startmo'] and day == 1:
                    sec = self.nl_colm_forcing['DEF_forcing']['offset'][var_i]
                else:
                    sec = 86400 + sec
                    day -= 1
                    if day == 0:
                        year -= 1
                        if CoLM_TimeManager.isleapyear(year) and self.nl_colm_forcing['DEF_forcing']['leapyear']:
                            day = 366
                        else:
                            day = 365

            # In case of leap year with a non-leap year calendar
            if not self.nl_colm_forcing['DEF_forcing'].leapyear and CoLM_TimeManager.isleapyear(year) and day > 59:
                day -= 1

            # Get record time index
            sec = 86400 * (day - 1) + sec
            time_i = int((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                         self.nl_colm_forcing['DEF_forcing'].dtime[var_i]) + 1

        # In the case of one month one file
        if self.nl_colm_forcing['DEF_forcing']['groupby'] == 'month':
            months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
            if CoLM_TimeManager.isleapyear(year):
                months = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

            # Calculate initial month and day values
            month, mday = CoLM_TimeManager.julian2monthday(year, day)

            # Calculate initial second value
            sec = 86400 * (mday - 1) + sec

            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) * 1.0 /
                                int(self.nl_colm_forcing['DEF_forcing']['dtime'][var_i])) + 1

            sec = (time_i - 1) * self.nl_colm_forcing['DEF_forcing']['dtime'][var_i] + \
                  self.nl_colm_forcing['DEF_forcing']['offset'][var_i] - 86400 * (mday - 1)
            self.tstamp_LB[var_i].sec = sec

            # Set time stamp (ststamp_LB)
            if sec < 0:
                self.tstamp_LB[var_i].sec = 86400 + sec
                self.tstamp_LB[var_i].day = day - 1
                if self.tstamp_LB[var_i].day == 0:
                    self.tstamp_LB[var_i].year = year - 1
                    if CoLM_TimeManager.isleapyear(self.tstamp_LB[var_i].year):
                        self.tstamp_LB[var_i].day = 366
                    else:
                        self.tstamp_LB[var_i].day = 365

            # Set record info (year, month, time_i)
            if sec < 0 or (sec == 0 and int(self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) != 0):
                if year == self.nl_colm_forcing['DEF_forcing']['startyr'] and month == \
                        self.nl_colm_forcing['DEF_forcing']['startmo'] and mday == 1:
                    sec = self.nl_colm_forcing['DEF_forcing']['offset'][var_i]
                else:
                    sec = 86400 + sec
                    mday -= 1
                    if mday == 0:
                        month -= 1
                        if month == 0:
                            month = 12
                            year -= 1
                            mday = 31
                        else:
                            mday = months[month] - months[month - 1]

            # In case of leap year with a non-leap year calendar
            if not self.nl_colm_forcing['DEF_forcing']['leapyear'] and CoLM_TimeManager.isleapyear(
                    year) and month == 2 and mday == 29:
                mday = 28

            # Get record time index
            sec = 86400 * (mday - 1) + sec
            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                                self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1

        # In the case of one day one file
        if self.nl_colm_forcing['DEF_forcing']['groupby'] == 'day':
            # Calculate initial month and day values
            month, mday = CoLM_TimeManager.julian2monthday(year, day)

            # Calculate initial second value
            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) * 1.0 /
                                self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1
            sec = (time_i - 1) * self.nl_colm_forcing['DEF_forcing']['dtime'][var_i] + \
                  self.nl_colm_forcing['DEF_forcing']['offset'][var_i]
            self.tstamp_LB[var_i].sec = sec

            # Set time stamp (ststamp_LB)
            if sec < 0:
                self.tstamp_LB[var_i].sec = 86400 + sec
                self.tstamp_LB[var_i].day = day - 1
                if self.tstamp_LB[var_i].day == 0:
                    self.tstamp_LB[var_i].year = year - 1
                    if CoLM_TimeManager.isleapyear(self.tstamp_LB[var_i].year):
                        self.tstamp_LB[var_i].day = 366
                    else:
                        self.tstamp_LB[var_i].day = 365

                if year == self.nl_colm_forcing['DEF_forcing']['startyr'] and month == \
                        self.nl_colm_forcing['DEF_forcing']['startmo'] and mday == 1:
                    sec = self.nl_colm_forcing['DEF_forcing']['offset'][var_i]
                else:
                    sec = 86400 + sec
                    year = self.tstamp_LB[var_i].year
                    month, mday = CoLM_TimeManager.julian2monthday(self.tstamp_LB[var_i].year, month)

            # In case of leap year with a non-leap year calendar
            if not self.nl_colm_forcing['DEF_forcing']['leapyear'] and CoLM_TimeManager.isleapyear(
                    year) and month == 2 and mday == 29:
                mday = 28

            # Get record time index
            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                                self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1

        if time_i <= 0:
            print("got the wrong time record of forcing! STOP!")

        return mtstamp, var_i, year, month, mday, time_i
        # CoLM_stop()

    def metread_latlon(self, dir_forcing: str, idate):
        """Reads latitude and longitude data from forcing files."""

        # Local variables
        filename = ""
        year, month, day, time_i = '1995', '01', '01', 0
        mtstamp = None  # Placeholder for timestamp handling
        latxy = None
        lonxy = None
        lat_in = None
        lon_in = None

        # Assume DEF_forcing and gforc are objects or dictionaries with attributes
        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT' or self.nl_colm_forcing['DEF_forcing'][
            'dataset'] == 'CPL7':
            self.gforc.define_by_ndims(360, 180)
        else:
            temp = CoLM_TimeManager.Timestamp(idate[0], idate[1], idate[2])
            mtstamp, _, year, month, day, time_i = self.setstampLB(temp, 0)
            filename = dir_forcing.strip() + self.userForcing.metfilename(year, month, day, 1)
            self.tstamp_LB[0] = CoLM_TimeManager.Timestamp(-1, -1, -1)

            if self.nl_colm_forcing['DEF_forcing']['dim2d']:
                latxy = self.netfile.ncio_read_bcast_serial(filename, self.nl_colm_forcing['DEF_forcing']['latname'])
                lonxy = self.netfile.ncio_read_bcast_serial(filename, self.nl_colm_forcing['DEF_forcing']['lonname'])

                lat_in = latxy[:, 0]
                lon_in = lonxy[0, :]

                del latxy
                del lonxy
            else:
                lat_in = self.netfile.ncio_read_bcast_serial(filename, self.nl_colm_forcing['DEF_forcing']['latname'])
                lon_in = self.netfile.ncio_read_bcast_serial(filename, self.nl_colm_forcing['DEF_forcing']['lonname'])

            if not self.nl_colm_forcing['DEF_forcing']['regional']:

                self.gforc.define_by_center(lat_in, lon_in)
            else:
                self.gforc.define_by_center(lat_in, lon_in,
                                            south=self.nl_colm_forcing['DEF_forcing']['regbnd'][0],
                                            north=self.nl_colm_forcing['DEF_forcing']['regbnd'][1],
                                            west=self.nl_colm_forcing['DEF_forcing']['regbnd'][2],
                                            east=self.nl_colm_forcing['DEF_forcing']['regbnd'][3])

            del lat_in
            del lon_in

        self.gforc.set_rlon()
        self.gforc.set_rlat()

    def forcing_init(self, dir_forcing, deltatime, ststamp, lc_year, landelm, varforcing1,
                     patchtype, numelm, pctshrpch=None, etstamp=None):  # pctshrpch landcrop;patchtype vit
        year = 0
        month = 0
        day = 0
        time_i = 0

        self.userForcing.init_user_specified_forcing()
        self.co2.init_monthly_co2_mlo(self.nl_colm['DEF_SSP'])

        self.deltim_int = int(deltatime)
        self.tstamp_LB = []
        self.tstamp_UB = []

        for i in range(int(self.nl_colm_forcing['DEF_forcing']['NVAR'])):
            self.tstamp_LB.append(CoLM_TimeManager.Timestamp(-1, -1, -1))

        for i in range(int(self.nl_colm_forcing['DEF_forcing']['NVAR'])):
            self.tstamp_UB.append(CoLM_TimeManager.Timestamp(-1, -1, -1))
        idate = [ststamp.year, ststamp.day, ststamp.sec]

        self.metread_latlon(dir_forcing, idate)

        if self.mpi.p_is_io:
            self.forcn = []
            self.forcn_LB = []
            self.forcn_UB = []

            for ivar in range(self.nl_colm_forcing['DEF_forcing']['NVAR']):
                forcn_tmep = self.datatype.allocate_block_data(self.gforc)
                self.forcn.append(forcn_tmep)
                forcn_LB_temp = self.datatype.allocate_block_data(self.gforc)
                self.forcn_LB.append(forcn_LB_temp)
                forcn_UB_temp = self.datatype.allocate_block_data(self.gforc)
                self.forcn_UB.append(forcn_UB_temp)

            self.metdata = self.datatype.allocate_block_data(self.gforc)
            self.avgcos = self.datatype.allocate_block_data(self.gforc)

            if self.nl_colm['URBAN_MODEL'] and self.nl_colm['SinglePoint']:
                pass

        if not self.nl_colm_forcing['DEF_forcing']['has_missing_value']:
            self.forcmask = self.mg2p_forc.build(self.gforc, self.landpatch.landpatch, self.mesh.mesh, self.pixel)
            if self.nl_colm['DEF_USE_Forcing_Downscaling']:
                self.forcmask_elm = self.mg2p_forc_elm.build(self.gforc, landelm)
        else:
            ststamp, _, year, month, day, time_i = self.setstampLB(ststamp, 0)
            filename = f"{dir_forcing}{self.userForcing.metfilename(year, month, day, 0)}"
            self.tstamp_LB[0] = CoLM_TimeManager.Timestamp(-1, -1, -1)

            if self.mpi.p_is_worker:
                if self.landpatch.numpatch > 0:
                    self.forcmask = np.full(self.landpatch.numpatch, True)
                if self.nl_colm['DEF_USE_Forcing_Downscaling']:
                    if numelm > 0:
                        forcmask_elm = np.full(numelm, True)
            missing_value = None
            if self.mpi.p_is_master:
                missing_value = self.netfile.ncio_get_attr(filename, self.vname,
                                                           self.nl_colm_forcing['DEF_forcing'][
                                                               'missing_value_name'], missing_value)

            self.metdata = CoLM_NetCDFBlock.ncio_read_block_time(filename, self.vname, self.gforc, time_i,
                                                                 self.metdata, self.mpi, self.gblock)

            self.mg2p_forc.build(self.gforc, self.landpatch, self.metdata, missing_value=missing_value, pfilter=self.forcmask)
            if self.nl_colm['DEF_USE_Forcing_Downscaling']:
                self.mg2p_forc_elm.build(self.gforc, landelm, self.metdata, missing_value=missing_value, pfilter=forcmask_elm)

        if self.nl_colm['DEF_USE_Forcing_Downscaling']:
            cyear = lc_year
            lndname = f"{self.nl_colm['DEF_dir_landdata']}/topography/{cyear}/topography_patches.nc"
            varforcing1.forc_topo = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'topography_patches', self.landpatch,
                                                                          varforcing1.forc_topo,
                                                                          self.nl_colm['USERMPI'], self.mpi,
                                                                          self.gblock)

            if self.mpi.p_is_worker:
                if self.nl_colm['CROP']:
                    self.landpatch.elm_patch.build(landelm, self.landpatch, use_frac=True, sharedfrac=pctshrpch)
                else:
                    self.landpatch.elm_patch.build(landelm, self.landpatch, use_frac=True)

                varforcing1.forc_topo_elm = [
                    sum(varforcing1.forc_topo[istt:iend] * self.landpatch.elm_patch.subfrc[istt:iend])
                    for istt, iend in zip(self.landpatch.elm_patch.substt, self.landpatch.elm_patch.subend)]

                if self.landpatch.numpatch > 0:
                    self.glacierss = [patchtype[i] == 3 for i in range(self.landpatch.numpatch)]

        self.forcing_read_ahead = False
        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
            if self.nl_colm['USE_SITE_ForcingReadAhead'] and etstamp is not None:
                self.forcing_read_ahead = True
                self.forc_disk = self.metread_time(dir_forcing, ststamp, etstamp, deltatime)
            else:
                self.forc_disk = self.metread_time(dir_forcing)

            self.iforctime = np.zeros(self.nl_colm_forcing['DEF_forcing']['NVAR'])

        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
            filename = f"{dir_forcing}{self.nl_colm_forcing['DEF_forcing']['fprefix']}"

            if not self.nl_colm['URBAN_MODEL']:
                if self.netfile.ncio_var_exist(filename, 'reference_height_v'):
                    self.nl_colm_forcing['DEF_forcing']['Height_V'] = self.netfile.ncio_read_serial(filename,
                                                                                                    'reference_height_v')

                if self.netfile.ncio_var_exist(filename, 'reference_height_t'):
                    self.nl_colm_forcing['DEF_forcing']['Height_T'] = self.netfile.ncio_read_serial(filename,
                                                                                                    'reference_height_t')

                if self.netfile.ncio_var_exist(filename, 'reference_height_q'):
                    self.nl_colm_forcing['DEF_forcing']['Height_Q'] = self.netfile.ncio_read_serial(filename,
                                                                                                    'reference_height_q')
            else:
                pass
                # if ncio_var_exist(filename, 'measurement_height_above_ground'):
                #     ncio_read_serial(filename, 'measurement_height_above_ground', Height_V)
                #     ncio_read_serial(filename, 'measurement_height_above_ground', Height_T)
                #     ncio_read_serial(filename, 'measurement_height_above_ground', Height_Q)

    def read_forcing(self, idate, dir_forcing, varforcing1, varforcing2, numelm, rgas,
                     isgreenwich):  # rgas const_physical
        # Constants and variables initialization
        # self.nl_colm_forcing['DEF_forcing']['NVAR'] = 8  # Assuming the number of variables, this needs to be adjusted based on actual data
        # self.gforc = ...  # Initialize this with appropriate structure
        # mtstamp = datetime(idate, 1, 1) + timedelta(days=idate[1] - 1 + idate[2] / 24.0)

        # Read lower and upper boundary forcing data
        if self.mpi.p_is_io:
            self.metreadLBUB(idate, dir_forcing, isgreenwich)
            id_date = copy.copy(idate)
            # id_date = CoLM_TimeManager.adj2end(id_date)
            mtstamp = id_date
            # print(mtstamp.year, mtstamp.day,mtstamp.sec,'---time1----')
            has_u = True
            has_v = True

            for ivar in range(self.nl_colm_forcing['DEF_forcing']['NVAR']):
                # print(ivar, mtstamp.year, mtstamp.day,mtstamp.sec,'---time----')
                if ivar == 5 and self.nl_colm_forcing['DEF_forcing']['vname'][ivar] == 'None':
                    has_u = False
                if ivar == 6 and self.nl_colm_forcing['DEF_forcing']['vname'][ivar] == 'None':
                    has_v = False
                if self.nl_colm_forcing['DEF_forcing']['vname'][ivar] == 'None' or \
                        self.nl_colm_forcing['DEF_forcing']['tintalgo'][ivar] == 'None':
                    continue

                if mtstamp < self.tstamp_LB[ivar] or self.tstamp_UB[ivar] < mtstamp:
                    print("the data required is out of range! STOP!")

                dtLB = mtstamp - self.tstamp_LB[ivar]
                dtUB = self.tstamp_UB[ivar] - mtstamp

                if self.nl_colm_forcing['DEF_forcing']['tintalgo'][ivar] == 'nearest':
                    if dtLB <= dtUB:
                        self.forcn[ivar] = self.datatype.block_data_copy(self.forcn_LB[ivar], self.forcn[ivar])
                    else:
                        self.forcn[ivar] = self.datatype.block_data_copy(self.forcn_UB[ivar], self.forcn[ivar])

                if self.nl_colm_forcing['DEF_forcing']['tintalgo'][ivar] == 'linear':
                    if (dtLB + dtUB) > 0:
                        self.forcn[ivar] = self.datatype.block_data_linear_interp(
                            self.forcn_LB[ivar], dtUB / (dtLB + dtUB),
                            self.forcn_UB[ivar], dtLB / (dtLB + dtUB),
                            self.forcn[ivar])
                    else:
                        self.forcn[ivar] = self.datatype.block_data_copy(self.forcn_LB[ivar], self.forcn[ivar])

                if self.nl_colm_forcing['DEF_forcing']['tintalgo'][ivar] == 'coszen':
                    for iblkme in range(self.gblock.nblkme):
                        ib = self.gblock.xblkme[iblkme]
                        jb = self.gblock.yblkme[iblkme]

                        for j in range(self.gforc.ycnt[jb]):
                            for i in range(self.gforc.xcnt[ib]):
                                ilat = self.gforc.ydsp[jb] + j + 1
                                ilon = self.gforc.xdsp[ib] + i + 1
                                # print(self.gforc.ycnt[jb], self.gforc.xcnt[ib],ilat,ilon,'-=-=-=-=-=-')
                                if ilon > self.gforc.nlon:
                                    ilon -= self.gforc.nlon
                                calday = CoLM_TimeManager.calendarday_date(mtstamp, isgreenwich)
                                cosz = max(0.001, CoLM_OrbCoszen.orb_coszen(calday, self.gforc.rlon[ilon],
                                                                            self.gforc.rlat[ilat]))
                                # self.forcn_LB why 10,8 to 8,10,need to deal
                                # print(cosz, self.avgcos.blk[ib, jb].val[i, j] ,self.forcn_LB[ivar].blk[ib, jb].val[i, j],'++++++++')
                                self.forcn[ivar].blk[ib, jb].val[i, j] = cosz / self.avgcos.blk[ib, jb].val[i, j] * self.forcn_LB[ivar].blk[ib, jb].val[i, j]

            # Preprocess for forcing data
            self.forcn = self.userForcing.metpreprocess(self.gforc, self.forcn)
            forc_xy_solarin = self.datatype.allocate_block_data(self.gforc)
            varforcing2.forc_xy_t = self.datatype.block_data_copy(self.forcn[0], varforcing2.forc_xy_t)
            varforcing2.forc_xy_q = self.datatype.block_data_copy(self.forcn[1], varforcing2.forc_xy_q)
            varforcing2.forc_xy_psrf = self.datatype.block_data_copy(self.forcn[2], varforcing2.forc_xy_psrf)
            varforcing2.forc_xy_pbot = self.datatype.block_data_copy(self.forcn[2], varforcing2.forc_xy_pbot)
            varforcing2.forc_xy_prl = self.datatype.block_data_copy(self.forcn[3], varforcing2.forc_xy_prl, sca=2 / 3.0)
            varforcing2.forc_xy_prc = self.datatype.block_data_copy(self.forcn[3], varforcing2.forc_xy_prc, sca=1 / 3.0)
            forc_xy_solarin = self.datatype.block_data_copy(self.forcn[6], forc_xy_solarin)
            varforcing2.forc_xy_frl = self.datatype.block_data_copy(self.forcn[7], varforcing2.forc_xy_frl)
            if self.nl_colm['DEF_USE_CBL_HEIGHT']:
                varforcing2.forc_xy_hpbl = self.datatype.block_data_copy(self.forcn[8], varforcing2.forc_xy_hpbl)

            if has_u and has_v:
                varforcing2.forc_xy_us = self.datatype.block_data_copy(self.forcn[4], varforcing2.forc_xy_us)
                varforcing2.forc_xy_vs = self.datatype.block_data_copy(self.forcn[5], varforcing2.forc_xy_vs)
            elif has_u:
                varforcing2.forc_xy_us = self.datatype.block_data_copy(self.forcn[4], varforcing2.forc_xy_us,
                                                                       sca=1 / math.sqrt(2.0))
                varforcing2.forc_xy_vs = self.datatype.block_data_copy(self.forcn[4], varforcing2.forc_xy_vs,
                                                                       sca=1 / math.sqrt(2.0))
            elif has_v:
                varforcing2.forc_xy_us = self.datatype.block_data_copy(self.forcn[5], varforcing2.forc_xy_us,
                                                                       sca=1 / math.sqrt(2.0))
                varforcing2.forc_xy_vs = self.datatype.block_data_copy(self.forcn[5], varforcing2.forc_xy_vs,
                                                                       sca=1 / math.sqrt(2.0))
            else:
                if self.nl_colm_forcing['DEF_forcing']['dataset'] != 'CPL7':
                    print("At least one of the wind components must be provided! STOP!")

            # Additional processing and mapping
            varforcing2.forc_xy_hgt_u = self.datatype.flush_block_data(varforcing2.forc_xy_hgt_u,
                                                                       self.nl_colm_forcing['DEF_forcing']['HEIGHT_V'])
            varforcing2.forc_xy_hgt_t = self.datatype.flush_block_data(varforcing2.forc_xy_hgt_t,
                                                                       self.nl_colm_forcing['DEF_forcing']['HEIGHT_T'])
            varforcing2.forc_xy_hgt_q = self.datatype.flush_block_data(varforcing2.forc_xy_hgt_q,
                                                                       self.nl_colm_forcing['DEF_forcing']['HEIGHT_Q'])

            if self.nl_colm_forcing['DEF_forcing']['solarin_all_band']:
                if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'QIAN':
                    for iblkme in range(self.gblock.nblkme):
                        ib = self.gblock.xblkme[iblkme]
                        jb = self.gblock.yblkme[iblkme]
                        for j in range(self.gforc.ycnt[jb]):
                            for i in range(self.gforc.xcnt[ib]):
                                hsolar = forc_xy_solarin.blk[ib, jb].val[i, j] * 0.5
                                ratio_rvrf = min(0.99,
                                                 max(0.29548 + 0.00504 * hsolar - 1.4957e-05 * hsolar ** 2 + 1.4881e-08 * hsolar ** 3,
                                                     0.01))
                                varforcing2.forc_xy_soll.blk[ib, jb].val[i, j] = ratio_rvrf * hsolar
                                varforcing2.forc_xy_solld.blk[ib, jb].val[i, j] = (1.0 - ratio_rvrf) * hsolar
                                ratio_rvrf = min(0.99,
                                                 max(0.17639 + 0.00380 * hsolar - 9.0039e-06 * hsolar ** 2 + 8.1351e-09 * hsolar ** 3,
                                                     0.01))
                                varforcing2.forc_xy_sols.blk[ib, jb].val[i, j] = ratio_rvrf * hsolar
                                varforcing2.forc_xy_solsd.blk[ib, jb].val[i, j] = (1.0 - ratio_rvrf) * hsolar
                else:
                    for iblkme in range(self.gblock.nblkme):
                        ib = self.gblock.xblkme[iblkme]
                        jb = self.gblock.yblkme[iblkme]
                        for j in range(self.gforc.ycnt[jb]):
                            for i in range(self.gforc.xcnt[ib]):
                                ilat = self.gforc.ydsp[jb] + j +1
                                ilon = self.gforc.xdsp[ib] + i +1
                                if ilon > self.gforc.nlon:
                                    ilon -= self.gforc.nlon
                                a = forc_xy_solarin.blk[ib, jb].val[i, j]
                                calday = CoLM_TimeManager.calendarday_date(idate, isgreenwich)
                                sunang = CoLM_OrbCoszen.orb_coszen(calday, self.gforc.rlon[ilon], self.gforc.rlat[ilat])
                                cloud = (1160.0 * sunang - a) / (963.0 * sunang)
                                cloud = max(cloud, 0.0)
                                cloud = max(0.58, min(cloud, 1.0))
                                difrat = 0.0604 / (sunang - 0.0223) + 0.0683
                                if difrat < 0:
                                    difrat =0.0
                                if difrat >1.0:
                                    difrat=1.0
                                difrat = difrat + (1.0 - difrat) * cloud
                                vnrat = (580.0 - cloud * 464.0) / ((580.0 - cloud * 499.0) + (580.0 - cloud * 464.0))
                                varforcing2.forc_xy_sols.blk[ib, jb].val[i, j] = a * (1.0 - difrat) * vnrat
                                # print(varforcing2.forc_xy_sols.blk[ib, jb].val[i, j] ,'---7---')
                                varforcing2.forc_xy_soll.blk[ib, jb].val[i, j] = a * (1.0 - difrat) * (1.0 - vnrat)
                                varforcing2.forc_xy_solsd.blk[ib, jb].val[i, j] = a * difrat * vnrat
                                varforcing2.forc_xy_solld.blk[ib, jb].val[i, j] = a * difrat * (1.0 - vnrat)

            # Get atmosphere CO2 concentration data
            year = idate.year
            month, mday = CoLM_TimeManager.julian2monthday(idate.year, idate.day)
            pco2m = self.co2.get_monthly_co2_mlo(year, month) * 1.e-6
            varforcing2.forc_xy_pco2m = self.datatype.block_data_copy(varforcing2.forc_xy_pbot,
                                                                      varforcing2.forc_xy_pco2m, sca=pco2m)
            varforcing2.forc_xy_po2m = self.datatype.block_data_copy(varforcing2.forc_xy_pbot, varforcing2.forc_xy_po2m,
                                                                     sca=0.209)

        varforcing1.forc_pco2m = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_pco2m, varforcing1.forc_pco2m)

        varforcing1.forc_po2m = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_po2m, varforcing1.forc_po2m)
        varforcing1.forc_us = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_us, varforcing1.forc_us)
        varforcing1.forc_vs = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_vs, varforcing1.forc_vs)
        varforcing1.forc_psrf = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_psrf, varforcing1.forc_psrf)
        varforcing1.forc_sols = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_sols, varforcing1.forc_sols)
        varforcing1.forc_soll = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_soll, varforcing1.forc_soll)
        varforcing1.forc_solsd = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_solsd, varforcing1.forc_solsd)
        varforcing1.forc_solld = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_solld, varforcing1.forc_solld)
        varforcing1.forc_hgt_t = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_hgt_t, varforcing1.forc_hgt_t)
        varforcing1.forc_hgt_u = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_hgt_u, varforcing1.forc_hgt_u)

        varforcing1.forc_hgt_q = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_hgt_q, varforcing1.forc_hgt_q)

        if self.nl_colm['DEF_USE_CBL_HEIGHT']:
            varforcing1.forc_hpbl = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_hpbl, varforcing1.forc_hpbl)

        if not self.nl_colm['DEF_USE_Forcing_Downscaling']:
            varforcing1.forc_t = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_t, varforcing1.forc_t)
            varforcing1.forc_q = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_q, varforcing1.forc_q)
            varforcing1.forc_prc = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_prc, varforcing1.forc_prc)
            varforcing1.forc_prl = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_prl, varforcing1.forc_prl)
            varforcing1.forc_pbot = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_pbot, varforcing1.forc_pbot)
            varforcing1.forc_frl = self.mg2p_forc.map_aweighted(varforcing2.forc_xy_frl, varforcing1.forc_frl)

            if self.mpi.p_is_worker:
                for np in range(self.landpatch.numpatch):
                    if self.nl_colm_forcing['DEF_forcing']['has_missing_value'] and not self.forcmask[np]:
                        continue

                    # Temperature adjustments
                    if varforcing1.forc_t[np] < 180.0:
                        varforcing1.forc_t[np] = 180.0
                    if varforcing1.forc_t[np] > 326.0:
                        varforcing1.forc_t[np] = 326.0

                    varforcing1.forc_rhoair[np] = (
                            (varforcing1.forc_pbot[np] - 0.378 * varforcing1.forc_q[np] * varforcing1.forc_pbot[np] / (
                                    0.622 + 0.378 * varforcing1.forc_q[np]))
                            / (rgas * varforcing1.forc_t[np])
                    )
        else:
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_t, varforcing1.forc_t_elm)
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_q, varforcing1.forc_q_elm)
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_prc, varforcing1.forc_prc_elm)
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_prl, varforcing1.forc_prl_elm)
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_pbot, varforcing1.forc_pbot_elm)
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_frl, varforcing1.forc_lwrad_elm)
            self.mg2p_forc.map_aweighted(varforcing2.forc_xy_hgt_t, varforcing1.forc_hgt_elm)

            if self.mpi.p_is_worker:
                for ne in range(numelm):
                    if self.nl_colm_forcing['DEF_forcing']['has_missing_value'] and not self.forcmask_elm[ne]:
                        continue

                    # Temperature adjustments
                    if varforcing1.forc_t_elm[ne] < 180.0:
                        varforcing1.forc_t_elm[ne] = 180.0
                    if varforcing1.forc_t_elm[ne] > 326.0:
                        varforcing1.forc_t_elm[ne] = 326.0

                    varforcing1.forc_rho_elm[ne] = (
                            (varforcing1.forc_pbot_elm[ne] - 0.378 * varforcing1.forc_q_elm[ne] *
                             varforcing1.forc_pbot_elm[ne] / (0.622 + 0.378 * varforcing1.forc_q_elm[ne]))
                            / (rgas * varforcing1.forc_t_elm[ne])
                    )
                    varforcing1.forc_th_elm[ne] = varforcing1.forc_t_elm[ne] * (
                            1.e5 / varforcing1.forc_pbot_elm[ne]) ** (
                                                          self.downscaling.rair / self.downscaling.cpair)

                forc_t_c, forc_th_c, forc_q_c, forc_pbot_c, forc_rho_c, forc_prc_c, forc_prl_c, forc_lwrad_c, forc_swrad_c, forc_us_c, forc_vs_c = self.downscaling.downscale_forcings(
                    numelm, self.landpatch.numpatch, self.landpatch.elm_patch.substt, self.landpatch.elm_patch.subend,
                    self.glacierss, self.landpatch.elm_patch.subfrc,
                    varforcing1.forc_topo_elm, varforcing1.forc_t_elm, varforcing1.forc_th_elm, varforcing1.forc_q_elm,
                    varforcing1.forc_pbot_elm,
                    varforcing1.forc_rho_elm, varforcing1.forc_prc_elm, varforcing1.forc_prl_elm,
                    varforcing1.forc_lwrad_elm, varforcing1.forc_hgt_elm,
                    varforcing1.forc_topo, varforcing1.forc_t, varforcing1.forc_th, varforcing1.forc_q,
                    varforcing1.forc_pbot,
                    varforcing1.forc_rhoair, varforcing1.forc_prc, varforcing1.forc_prl, varforcing1.forc_frl)

        # Range check (if enabled)
        if self.nl_colm['RangeCheck']:
            # if USEMPI:
            #     mpi_barrier(p_comm_glb, p_err)
            if self.mpi.p_is_master:
                print('Checking forcing ...')

            CoLM_RangeCheck.check_vector_data('Forcing us    [m/s]   ', varforcing1.forc_us, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing vs    [m/s]   ', varforcing1.forc_vs, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing t     [kelvin]', varforcing1.forc_t, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing q     [kg/kg] ', varforcing1.forc_q, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing prc   [mm/s]  ', varforcing1.forc_prc, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing psrf  [pa]    ', varforcing1.forc_psrf, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing prl   [mm/s]  ', varforcing1.forc_prl, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing sols  [W/m2]  ', varforcing1.forc_sols, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing soll  [W/m2]  ', varforcing1.forc_soll, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing solsd [W/m2]  ', varforcing1.forc_solsd, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing solld [W/m2]  ', varforcing1.forc_solld, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Forcing frl   [W/m2]  ', varforcing1.forc_frl, self.mpi, self.nl_colm)
            if self.nl_colm['DEF_USE_CBL_HEIGHT']:
                CoLM_RangeCheck.check_vector_data('Forcing hpbl  ', varforcing1.forc_hpbl, self.mpi, self.nl_colm)

            # if USEMPI:
            #     mpi_barrier(p_comm_glb, p_err)
        return

    def setstampUB(self, var_i):
        """
        Set the upper boundary timestamp and return year, month, day, and time index.

        Parameters:
        var_i (int): Variable index.
        tstamp_LB (dict): Lower boundary timestamps.
        tstamp_UB (dict): Upper boundary timestamps.
        dtime (dict): Time step values.
        self.iforctime (dict): Forcing time indices.
        offset (dict): Offset values.
        groupby (str): Time grouping ('year', 'month', 'day').
        leapyear (bool): Flag indicating if the calendar is leap year.
        isleapyear (bool): Flag indicating if the year is a leap year.
        DEF_forcing (object): Forcing definitions with 'dataset' and 'vname' attributes.

        Returns:
        tuple: (year, month, mday, time_i)
        """
        year = month = mday = time_i = 0

        # Point dataset handling
        if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
            if self.tstamp_UB[var_i] == 'NULL':
                self.tstamp_UB[var_i] = self.forctime[self.iforctime[var_i] + 1]
            else:
                self.iforctime[var_i] += 1
                self.tstamp_LB[var_i] = self.forctime[self.iforctime[var_i]]
                self.tstamp_UB[var_i] = self.forctime[self.iforctime[var_i] + 1]

            time_i = self.iforctime[var_i] + 1
            year = self.tstamp_UB[var_i]['year']
            return year, month, mday, time_i

        # Calculate timestamps

        if self.tstamp_UB[var_i].year < 0:
            # print('--------------------------')
            # print(self.tstamp_UB[var_i].year, self.tstamp_UB[var_i].day, self.tstamp_UB[var_i].sec, '--------ub1---')
            self.tstamp_UB[var_i] = self.tstamp_LB[var_i] + self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]
            # print(self.tstamp_LB[var_i].year, self.tstamp_LB[var_i].day, self.tstamp_LB[var_i].sec, var_i,
            #       self.nl_colm_forcing['DEF_forcing']['dtime'][var_i],'-*-*-*-*-*-*-NULL2*--*-*-*-*-*-')
            # print(self.tstamp_UB[var_i].year, self.tstamp_UB[var_i].day, self.tstamp_UB[var_i].sec, '--------ub2---')
        else:
            self.tstamp_LB[var_i] = self.tstamp_UB[var_i]
            # print(self.tstamp_UB[var_i].year, self.tstamp_UB[var_i].day, self.tstamp_UB[var_i].sec, '--------ub1---')
            self.tstamp_UB[var_i] = self.tstamp_UB[var_i] + self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]
            # print(self.tstamp_UB[var_i].year, self.tstamp_UB[var_i].day, self.tstamp_UB[var_i].sec, var_i,
            #       self.nl_colm_forcing['DEF_forcing']['dtime'][var_i], '-*-*-*-*-*-*-*not null2--*-*-*-*-*-')
            # print(self.tstamp_UB[var_i].year, self.tstamp_UB[var_i].day, self.tstamp_UB[var_i].sec, '--------ub2---')

        year = self.tstamp_UB[var_i].year
        day = self.tstamp_UB[var_i].day
        sec = self.tstamp_UB[var_i].sec
        # print(year, day, sec,var_i,'-*-*-*-*-*-*-*--*-*-*-*-*-')
        # Adjust based on groupby
        months = []
        if self.nl_colm_forcing['DEF_forcing']['groupby'] == 'year':
            if sec == 86400 and self.nl_colm_forcing['DEF_forcing']['offset'][var_i] == 0:
                sec = 0
                day += 1
                if CoLM_TimeManager.isleapyear(year) and day == 367:
                    year += 1
                    day = 1
                elif not CoLM_TimeManager.isleapyear(year) and day == 366:
                    year += 1
                    day = 1

            if not self.nl_colm_forcing['DEF_forcing']['leapyear'] and CoLM_TimeManager.isleapyear(year) and day > 59:
                day -= 1

            sec = 86400 * (day - 1) + sec
            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                                self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1

        elif self.nl_colm_forcing['DEF_forcing']['groupby'] == 'month':
            if CoLM_TimeManager.isleapyear(year):
                months = [0,31,60,91,121,152,182,213,244,274,305,335,366]
            else:
                months = [0,31,59,90,120,151,181,212,243,273,304,334,365]
            month, mday = CoLM_TimeManager.julian2monthday(year, day)

            if sec == 86400 and self.nl_colm_forcing['DEF_forcing']['offset'][var_i] == 0:
                sec = 0
                mday += 1
                if mday > (months[month - 1] - (months[month - 2] if month > 1 else 0)):
                    mday = 1
                    if month == 12:
                        month = 1
                        year += 1
                    else:
                        month += 1

            if not self.nl_colm_forcing['DEF_forcing']['leapyear'] and CoLM_TimeManager.isleapyear(
                    year) and month == 2 and mday == 29:
                mday = 28

            sec = 86400 * (mday - 1) + sec
            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                                self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1

        elif self.nl_colm_forcing['DEF_forcing']['groupby'] == 'day':
            if CoLM_TimeManager.isleapyear(year):
                months = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
            else:
                months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
            month, mday = CoLM_TimeManager.julian2monthday(year, day)

            if sec == 86400 and self.nl_colm_forcing['DEF_forcing']['offset'][var_i] == 0:
                sec = 0
                mday += 1
                if mday > (months[month - 1] - (months[month - 2] if month > 1 else 0)):
                    mday = 1
                    if month == 12:
                        month = 1
                        year += 1
                    else:
                        month += 1

            if not self.nl_colm_forcing['DEF_forcing']['leapyear'] and CoLM_TimeManager.isleapyear(
                    year) and month == 2 and mday == 29:
                mday = 28

            time_i = math.floor((sec - self.nl_colm_forcing['DEF_forcing']['offset'][var_i]) /
                                self.nl_colm_forcing['DEF_forcing']['dtime'][var_i]) + 1

        if time_i <= 0:
            raise ValueError("Got the wrong time record of forcing! STOP!")

        return year, month, mday, time_i

    def metreadLBUB(self, idate, dir_forcing, isgreenwich):
        # Assuming some global definitions or external variables
        mtstamp = copy.copy(idate)

        for ivar in range(self.nl_colm_forcing['DEF_forcing']['NVAR']):
            if self.nl_colm_forcing['DEF_forcing']['vname'][ivar] == 'None':
                continue  # No data, skip

            # Lower and upper boundary data already exist, skip
            if self.tstamp_LB[ivar].year > 0 and self.tstamp_UB[ivar].year > 0 and self.tstamp_LB[ivar] <= mtstamp and mtstamp < self.tstamp_UB[ivar]:
                continue

            # Set lower boundary time stamp and get data
            if self.tstamp_LB[ivar].year <= 0:
                mtstamp, var_i, year, month, day, time_i = self.setstampLB(mtstamp, ivar)
                # Read forcing data
                # print(ivar, year, month, day,time_i,'---time_i---')
                filename = dir_forcing + '/' + self.userForcing.metfilename(year, month, day, ivar)
                # print(filename,'===')
                if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
                    if not self.nl_colm['URBAN_MODEL']:
                        if self.forcing_read_ahead:
                            self.metdata['blk'][self.gblock['xblkme'], self.gblock['yblkme']]['val'] = self.forc_disk[
                                time_i, ivar]
                        else:
                            self.metdata = CoLM_NetCDFBlock.ncio_read_site_time(filename,
                                                                                self.nl_colm_forcing['DEF_forcing'][
                                                                                    'vname'][ivar], time_i,
                                                                                self.metdata, self.gblock, self.mpi)
                    # else:
                    # if vname[ivar] == 'Rainf':
                    #     rainf = ncio_read_site_time(filename, 'Rainf', time_i, None)
                    #     snowf = ncio_read_site_time(filename, 'Snowf', time_i, None)
                    #     for iblkme in range(gblock['nblkme']):
                    #         ib = gblock['xblkme'][iblkme]
                    #         jb = gblock['yblkme'][iblkme]
                    #         metdata['blk'][ib, jb]['val'][0, 0] = rainf['blk'][ib, jb]['val'][0, 0] + snowf['blk'][ib, jb]['val'][0, 0]
                    # else:
                    #     ncio_read_site_time(filename, vname[ivar], time_i, metdata)
                else:
                    # for i in range(self.metdata.blk.shape[0]):
                    #     for j in range(self.metdata.blk.shape[1]):
                    #         print(self.metdata.blk[i, j].val.shape, '******3*****')
                    self.metdata = CoLM_NetCDFBlock.ncio_read_block_time(filename,
                                                                         self.nl_colm_forcing['DEF_forcing']['vname'][
                                                                             ivar], self.gforc, time_i, self.metdata,
                                                                         self.mpi, self.gblock)

                self.forcn_LB[ivar] = self.datatype.block_data_copy(self.metdata, self.forcn_LB[ivar])

            # Set upper boundary time stamp and get data
            if self.tstamp_UB[ivar].year <= 0 or self.tstamp_UB[ivar] <= mtstamp:
                if self.tstamp_UB[ivar].year > 0:
                    self.forcn_LB[ivar] = self.datatype.block_data_copy(self.forcn_UB[ivar], self.forcn_LB[ivar])
                # print(ivar, year, month, day, time_i,'---time1----')
                year, month, day, time_i = self.setstampUB(ivar)
                # print(ivar, year, month, day, time_i,'---time2----')

                if year <= self.nl_colm_forcing['DEF_forcing']['endyr']:
                    # Read forcing data
                    filename = dir_forcing + '/' + self.userForcing.metfilename(year, month, day, ivar)
                    # print(filename,'-----')
                    if self.nl_colm_forcing['DEF_forcing']['dataset'] == 'POINT':
                        if not self.nl_colm['URBAN_MODEL']:
                            if self.forcing_read_ahead:
                                self.metdata['blk'][self.gblock['xblkme'], self.gblock['yblkme']]['val'] = \
                                    self.forc_disk[time_i, ivar]
                            else:
                                self.metdata = CoLM_NetCDFBlock.ncio_read_site_time(filename,
                                                                                    self.nl_colm_forcing['DEF_forcing'][
                                                                                        'vname'][ivar], time_i,
                                                                                    self.metdata, self.mpi)
                        # else:
                        # if vname[ivar] == 'Rainf':
                        #     rainf = ncio_read_site_time(filename, 'Rainf', time_i, None)
                        #     snowf = ncio_read_site_time(filename, 'Snowf', time_i, None)
                        #     for iblkme in range(gblock['nblkme']):
                        #         ib = gblock['xblkme'][iblkme]
                        #         jb = gblock['yblkme'][iblkme]
                        #         metdata['blk'][ib, jb]['val'][0, 0] = rainf['blk'][ib, jb]['val'][0, 0] + snowf['blk'][ib, jb]['val'][0, 0]
                        # else:
                        #     ncio_read_site_time(filename, vname[ivar], time_i, metdata)
                    else:
                        self.metdata = CoLM_NetCDFBlock.ncio_read_block_time(filename,
                                                                             self.nl_colm_forcing['DEF_forcing'][
                                                                                 'vname'][ivar], self.gforc, time_i,
                                                                             self.metdata, self.mpi, self.gblock)

                    self.forcn_UB[ivar] = self.datatype.block_data_copy(self.metdata, self.forcn_UB[ivar])
                else:
                    print(f"NOTE: reaching the END of forcing data, always reuse the last time step data!")
                    print(year, self.nl_colm_forcing['DEF_forcing']['endyr'])

                if ivar == 7:  # Calculate time average coszen, for shortwave radiation
                    self.calavgcos(idate, isgreenwich)

    def metread_time(self, dir_forcing, ststamp=None, etstamp=None, deltime=None):
        year = 0
        month = 0
        day = 0
        hour = 0
        minute = 0
        second = 0
        filename = dir_forcing + self.nl_colm_forcing['DEF_forcing']['fprefix']
        forctime_sec = self.netfile.ncio_read_serial(filename, 'time')
        timeunit = self.netfile.ncio_get_attr(filename, 'time', 'units')

        timestr = f"{timeunit[14:18]} {timeunit[19:21]} {timeunit[22:24]} {timeunit[25:27]} {timeunit[28:30]} {timeunit[31:33]}"
        # year, month, day, hour, minute, second = map(int, timestr.split())

        forctime = np.zeros(len(forctime_sec), dtype=int)
        forctime['year'] = year
        forctime['day'] = CoLM_TimeManager.get_calday(month * 100 + day, CoLM_TimeManager.isleapyear(year))
        forctime['sec'] = hour * 3600 + minute * 60 + second + forctime_sec

        id = [forctime['year'], forctime['day'], forctime['sec']]
        id = CoLM_TimeManager.adj2end(id)
        forctime = id

        ntime = len(forctime)

        for itime in range(1, ntime):
            id = [forctime[itime - 1]['year'], forctime[itime - 1]['day'], forctime[itime - 1]['sec']]
            id = CoLM_TimeManager.ticktime(forctime_sec[itime] - forctime_sec[itime - 1], id)
            forctime[itime] = id

        if self.forcing_read_ahead:
            id = CoLM_TimeManager.ticktime(deltime, id)
            etstamp_f = id

            if (ststamp < forctime) or (etstamp_f < etstamp):
                print('Error: Forcing does not cover simulation period!')
                print('Model start ', ststamp, ' -> Model END ', etstamp)
                print('Forc  start ', forctime, ' -> Forc END  ', etstamp_f)
                # CoLM_stop()
            else:
                its = 0
                while not (ststamp < forctime[its + 1]):
                    its += 1
                    if its >= ntime:
                        break

                ite = ntime - 1
                while etstamp < forctime[ite - 1]:
                    ite -= 1
                    if ite <= 1:
                        break

                ntime = ite - its + 1

                forctime_ = np.zeros(ntime)
                for it in range(ntime):
                    forctime_[it] = forctime[it + its - 1]

                forctime = np.zeros(ntime, dtype=int)
                for it in range(ntime):
                    forctime[it] = forctime_[it]

                del forctime_

            forc_disk = np.zeros((len(forctime), self.nl_colm_forcing['DEF_forcing']['NVAR']))

            filename = dir_forcing + self.userForcing.metfilename(-1, -1, -1, -1)
            for ivar in range(self.nl_colm_forcing['DEF_forcing']['NVAR']):
                if self.vname[ivar] != 'NULL':
                    metcache = self.netfile.ncio_read_period_serial(filename, self.vname[ivar], its, ite, metcache)
                    forc_disk[:, ivar] = metcache[0, 0, :]

            if metcache is not None:
                del metcache

        return forc_disk

    def calavgcos(self, idate, isgreenwich):
        """
        Calculate the average cosine of the solar zenith angle.

        Parameters:
        idate (tuple): A tuple representing the date (year, month, day).
        tstamp_LB (datetime): Lower boundary timestamp.
        tstamp_UB (datetime): Upper boundary timestamp.
        deltim_int (timedelta): Time increment.
        avgcos (object): Object holding average cosine values.
        gblock (object): Object holding block data.
        gforc (object): Object holding forcing data.
        """
        tstamp = copy.copy(idate)
        ntime = 0

        while tstamp < self.tstamp_UB[6]:
            ntime += 1
            tstamp += self.deltim_int

        tstamp = idate
        self.avgcos = self.datatype.flush_block_data(self.avgcos, 0.0)  # Initialize avgcos with zero

        while tstamp < self.tstamp_UB[6]:
            for iblkme in range(self.gblock.nblkme):
                ib = self.gblock.xblkme[iblkme]
                jb = self.gblock.yblkme[iblkme]

                for j in range(self.gforc.ycnt[jb]):
                    for i in range(self.gforc.xcnt[ib]):
                        ilat = self.gforc.ydsp[jb] + j +1
                        ilon = self.gforc.xdsp[ib] + i+1

                        if ilon > self.gforc.nlon:
                            ilon -= self.gforc.nlon

                        calday = CoLM_TimeManager.calendarday_date(tstamp, isgreenwich)
                        cosz = CoLM_OrbCoszen.orb_coszen(calday, self.gforc.rlon[ilon], self.gforc.rlat[ilat])
                        cosz = max(0.001, cosz)
                        self.avgcos.blk[ib, jb].val[i, j] += cosz / ntime

            tstamp += self.deltim_int

    def forcing_reset(self):
        self.tstamp_LB[:] = self.timestamp[-1, -1, -1]
        self.tstamp_UB[:] = self.timestamp[-1, -1, -1]

    def forcing_final(self):
        if self.forcmask: del self.forcmask
        # if self.forcmask_elm : del self.forcmask_elm
        if self.glacierss: del self.glacierss
        if self.forctime: del self.forctime
        if self.iforctime: del self.iforctime
        if self.forc_disk: del self.forc_disk
        if self.tstamp_LB: del self.tstamp_LB
        if self.tstamp_UB: del self.tstamp_UB
