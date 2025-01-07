# -----------------------------------------------------------------------
# DESCRIPTION:
#
#    Reading data in netCDF files by 5 degree blocks.
#
#    The file name gives the boundaries of the block.
#    For example, file "RG_65_75_60_80.URB2010.nc" stores data in region
#    from 65N to 60N and 75E to 80E.
#
#    Notice that:
#    1. Subroutines loop over all 5 degree blocks in simulation region.
#    2. Latitude in files is from north to south.
#    3. "read_5x5_data_pft" reads data with dimension "pft" and permute
#       dimension (lon,lat,pft) in files to (pft,lon,lat) in variables.
#    4. "read_5x5_data_time" reads data with dimension "time"
#       at given time.
#    5. "read_5x5_data_pft_time" reads data with dimension "pft" and "time"
#       at given time and permute dimension (lon,lat,pft) in files
#       to (pft,lon,lat) in variables.
#
# -----------------------------------------------------------------------


import os
import sys
from netCDF4 import Dataset


class CoLM_5x5DataReadin(object):
    def __init__(self, mpi, gblock) -> None:
        self.mpi = mpi
        self.gblock = gblock
        self.N_PFT_modis = 16

    def read_5x5_data(self, dir_5x5, sfx, grid, dataname, rdata):
        nxglb = grid.nlon
        nyglb = grid.nlat

        nxbox = nxglb // 360 * 5
        nybox = nyglb // 180 * 5

        if self.mpi.p_is_io:
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]

                if grid.xcnt[iblk] == 0:
                    continue
                if grid.ycnt[jblk] == 0:
                    continue

                rdata.blk[iblk, jblk].val[:, :] = 0

                inorth = grid.ydsp[jblk] + 1
                isouth = grid.ydsp[jblk] + grid.ycnt[jblk]

                iwest = grid.xdsp[iblk] + 1
                ieast = grid.xdsp[iblk] + grid.xcnt[iblk]
                if ieast > nxglb:
                    ieast = ieast - nxglb

                ibox = grid.xdsp[iblk] // nxbox + 1
                jbox = grid.ydsp[jblk] // nybox + 1
                ibox0 = ibox

                while True:
                    i0, i1, j0, j1, il0, il1, jl0, jl1, ibox, jbox, ibox0, file_5x5 = self.this_block_and_move_to_next(dir_5x5, sfx, nxbox,
                                                                                                    nybox, nxglb,
                                                                                                    isouth, inorth,
                                                                                                    iwest, ieast,
                                                                                                    ibox, jbox, ibox0)
                    if os.path.exists(file_5x5):
                        with Dataset(file_5x5, "r") as nc:
                            dcache = nc.variables[dataname][i0: i1 - i0 + 1, j0:j1 - j0 + 1]

                            rdata.blk[iblk, jblk].val[il0:il1, jl0:jl1] = dcache

                    if jbox == -1:
                        break
        return rdata

    def this_block_and_move_to_next(self, dir_5x5, sfx, nxbox, nybox, nxglb, isouth, inorth, iwest, ieast,
                                    ibox, jbox, ibox0):
        xdsp = (ibox - 1) * nxbox
        ydsp = (jbox - 1) * nybox

        j0 = max(inorth - ydsp, 0)
        j1 = min(isouth - ydsp, nybox) + 1
        jl0 = j0 + ydsp - inorth
        jl1 = j1 + ydsp - inorth + 1

        if ieast >= iwest:
            i0 = max(iwest - xdsp, 0)
            i1 = min(ieast - xdsp, nxbox) + 1
            il0 = i0 + xdsp - iwest
            il1 = i1 + xdsp - iwest + 1
        else:
            if iwest <= xdsp + nxbox:
                i0 = max(iwest - xdsp, 0)
                i1 = nxbox  + 1
                il0 = i0 + xdsp - iwest
                il1 = i1 + xdsp - iwest + 1
            else:
                i0 = 0
                i1 = min(ieast - xdsp, nxbox) + 1
                il0 = i0 + xdsp + nxglb - iwest
                il1 = i1 + xdsp + nxglb - iwest + 1

        file_5x5 = os.path.join(dir_5x5, 'RG_' + str((19 - jbox) * 5) + '_' + str((ibox - 37) * 5) + '_' + str(
            (18 - jbox) * 5) + '_' + str((ibox - 36) * 5) + '.' + sfx + '.nc')
        if 'win' in sys.platform:
            file_5x5 = dir_5x5 + '//RG_' + str((19 - jbox) * 5) + '_' + str((ibox - 37) * 5) + '_' + str(
                (18 - jbox) * 5) + '_' + str((ibox - 36) * 5) + '.' + sfx + '.nc'

        if xdsp + 1 <= ieast <= xdsp + nxbox:
            if isouth <= ydsp + nybox:
                jbox = -1
            else:
                ibox = ibox0
                jbox += 1
        else:
            ibox = (ibox % (nxglb // nxbox)) + 1

        return i0, i1, j0, j1, il0, il1, jl0, jl1, ibox, jbox, ibox0, file_5x5

    def read_5x5_data_pft(self, dir_5x5, sfx, grid, dataname, rdata):
        nxglb = grid.nlon
        nyglb = grid.nlat

        nxbox = nxglb // 360 * 5
        nybox = nyglb // 180 * 5

        if self.mpi.p_is_io:
            for iblkme in range(len(self.gblock.nblkme)):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]

                if grid.xcnt[iblk] == 0:
                    continue
                if grid.ycnt[jblk] == 0:
                    continue

                rdata.blk[iblk][jblk].val[:, :, :] = 0

                inorth = grid.ydsp[jblk] + 1
                isouth = grid.ydsp[jblk] + grid.ycnt[jblk]

                iwest = grid.xdsp[iblk] + 1
                ieast = grid.xdsp[iblk] + grid.xcnt[iblk]
                if ieast > nxglb:
                    ieast = ieast - nxglb

                ibox = grid.xdsp[iblk] // nxbox + 1
                jbox = grid.ydsp[jblk] // nybox + 1
                ibox0 = ibox

                while True:
                    i0, i1, j0, j1, il0, il1, jl0, jl1, ibox, jbox, ibox0, file_5x5 = self.this_block_and_move_to_next(dir_5x5, sfx, nxbox,
                                                                                                    nybox, nxglb,
                                                                                                    isouth, inorth,
                                                                                                    iwest, ieast, ibox,
                                                                                                    jbox, ibox0)

                    if os.path.isfile(file_5x5):
                        with Dataset(file_5x5, 'r') as nc:
                            dcache = nc.variables[dataname][i0:i1 - i0 + 1, j0:j1 - j0 + 1, 0:self.N_PFT_modis]
                            for ipft in range(self.N_PFT_modis):
                                rdata.blk[iblk][jblk].val[ipft, il0:il1, jl0:jl1] = dcache[:, :, ipft]

                    if jbox == -1:
                        break
        return rdata

    def read_5x5_data_time(self, dir_5x5, sfx, grid, dataname, time, rdata):
        nxglb = grid.nlon
        nyglb = grid.nlat

        nxbox = nxglb // 360 * 5
        nybox = nyglb // 180 * 5

        if self.mpi.p_is_io:
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]


                if grid.xcnt[iblk] == 0:
                    continue
                if grid.ycnt[jblk] == 0:
                    continue

                rdata.blk[iblk][jblk].val[:, :] = 0

                inorth = grid.ydsp[jblk] + 1
                isouth = grid.ydsp[jblk] + grid.ycnt[jblk]

                iwest = grid.xdsp[iblk] + 1
                ieast = grid.xdsp[iblk] + grid.xcnt[iblk]
                # print(grid.ydsp[jblk-1],inorth,grid.ydsp[jblk+1], isouth, grid.ycnt[jblk],grid.xdsp[iblk-1],iwest,grid.xdsp[iblk+1],ieast,grid.xcnt[iblk],'----------++++++++')
                if ieast > nxglb:
                    ieast -= nxglb

                ibox = (grid.xdsp[iblk] + 1) // nxbox + 1
                jbox = (grid.ydsp[jblk] + 1) // nybox + 1
                ibox0 = ibox
                # print(nxbox,nybox, nxglb,isouth, inorth,iwest, ieast,ibox, jbox, ibox0,'**************')

                while True:
                    i0, i1, j0, j1, il0, il1, jl0, jl1, ibox, jbox, ibox0, file_5x5 = self.this_block_and_move_to_next(dir_5x5, sfx, nxbox,
                                                                                                    nybox, nxglb,
                                                                                                    isouth, inorth,
                                                                                                    iwest, ieast,
                                                                                                    ibox, jbox, ibox0)

                    fexists = os.path.exists(file_5x5)
                    if fexists:
                        dcache = None
                        with Dataset(file_5x5, "r") as nc:
                            # print(nc.variables[dataname].shape,'**************-------------')

                            dcache = nc.variables[dataname][time:time+1,j0:j1 , i0:i1+1]
                        # print(dcache.shape,'********')
                        # print(i0,i1,j0,j1,il0,il1, jl0,jl1,'-----')
                        _, row, col = dcache.shape

                        rdata.blk[iblk][jblk].val[il0:il1, jl0:jl1] = dcache.reshape(row,col).transpose()

                    if jbox == -1:
                        break
        return rdata

    def read_5x5_data_pft_time(self, dir_5x5, sfx, grid, dataname, time, rdata):
        nxglb = grid.nlon
        nyglb = grid.nlat

        nxbox = nxglb // 360 * 5
        nybox = nyglb // 180 * 5

        if self.mpi.p_is_io:

            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]
                if grid.xcnt[iblk] == 0 or grid.ycnt[jblk] == 0:
                    continue

                rdata.blk[iblk, jblk].val[:, :, :] = 0

                inorth = grid.ydsp[jblk] + 1
                isouth = grid.ydsp[jblk] + grid.ycnt[jblk]

                iwest = grid.xdsp[iblk] + 1
                ieast = grid.xdsp[iblk] + grid.xcnt[iblk]
                if ieast > nxglb:
                    ieast = ieast - nxglb

                ibox = grid.xdsp[iblk] // nxbox + 1
                jbox = grid.ydsp[jblk] // nybox + 1
                ibox0 = ibox

                while True:
                    i0, i1, j0, j1, il0, il1, jl0, jl1, ibox, jbox, ibox0, file_5x5 = self.this_block_and_move_to_next(dir_5x5, sfx, nxbox,
                                                                                                    nybox, nxglb,
                                                                                                    isouth, inorth,
                                                                                                    iwest, ieast,
                                                                                                    ibox, jbox, ibox0)

                    if os.path.exists(file_5x5):
                        with Dataset(file_5x5, 'r') as nc:
                            dcache = nc.variables[dataname][i0:i1 - i0 + 1, j0:j1 - j0 + 1, 0:self.N_PFT_modis, time:0]
                            for ipft in range(self.N_PFT_modis):
                                rdata.blk[iblk][jblk].val[ipft, il0:il1, jl0:jl1] = dcache[:, :, ipft]

                    if jbox == -1:
                        break
        return rdata


class grid_type:
    def __init__(self, nlon, nlat, xcnt, ycnt, xdsp, ydsp):
        self.nlon = nlon
        self.nlat = nlat
        self.xcnt = xcnt
        self.ycnt = ycnt
        self.xdsp = xdsp
        self.ydsp = ydsp


class block_data_real8_3d:
    def __init__(self, blk):
        self.blk = blk


class block:
    def __init__(self, val):
        self.val = val

    # Define grid, gblock, N_PFT_modis according to your implementation
