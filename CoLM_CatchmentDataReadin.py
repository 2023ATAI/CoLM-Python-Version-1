# --------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Reading preprocessed MERIT Hydro data and generated catchment data in netcdf files.
#
#    1. If "in_one_file" is false, then the data is orgnized by 5 degree blocks.   
#    The file name gives the southwest corner of the block. 
#    For example, file "n60e075.nc" stores data in region from 65N to 60N and 75E to 80E, 
#    Subroutines loop over all 5 degree blocks in simulation region.
# 
#    2. Data is saved in variables with types of "block_data_xxxxx_xd".
# 
#    3. Latitude in files is from north to south. 
# 
# --------------------------------------------------------------------------------------
from CoLM_NetCDFSerial import NetCDFFile
import CoLM_Utils
import numpy as np
import os


def catchment_data_read(mpi, gblock, file_meshdata_in, dataname, grid, rdata, spv=None):
    nxhbox = 6000
    nyhbox = 6000
    nxhglb = 432000
    nyhglb = 216000
    i1next = 0

    il0 = 0
    il1 = 0
    if0 = 0
    if1 = 0
    i0next = 0

    if mpi.p_is_master:
        if grid.yinc == 1:
            print('Warning: latitude in catchment data should be from north to south.')
    nc_mesh = NetCDFFile(mpi)
    in_one_file = nc_mesh.ncio_var_exist(file_meshdata_in, dataname)

    if in_one_file:
        file_mesh = file_meshdata_in
        latitude = nc_mesh.ncio_read_bcast_serial(file_mesh, 'latitude')
        longitude = nc_mesh.ncio_read_bcast_serial(file_mesh, 'longitude')

        if mpi.p_is_io:
            nlat = len(latitude)
            nlon = len(longitude)

            isouth = CoLM_Utils.find_nearest_south(latitude[nlat], grid.nlat, grid.lat_s)
            inorth = CoLM_Utils.find_nearest_north(latitude[1], grid.nlat, grid.lat_n)

            for ilon in range(nlon):
                CoLM_Utils.normalize_longitude(longitude[ilon])

            iwest = CoLM_Utils.find_nearest_west(longitude(1), grid.nlon, grid.lon_w)
            ieast = CoLM_Utils.find_nearest_east(longitude(nlon), grid.nlon, grid.lon_e)

            for iblkme in range(gblock.nblkme):
                iblk = gblock.xblkme[iblkme]
                jblk = gblock.yblkme[iblkme]

                if spv:
                    rdata.blk[iblk, jblk].val[:, :] = spv
                else:
                    rdata.blk[iblk, jblk].val[:, :] = 0

                if (inorth > grid.ydsp[jblk] + nyhbox) or (isouth < grid.ydsp[jblk] + 1):
                    continue

                j0 = max(inorth, grid.ydsp(jblk) + 1)
                j1 = min(isouth, grid.ydsp(jblk) + grid.ycnt(jblk))

                jl0 = j0 - grid.ydsp(jblk)
                jl1 = j1 - grid.ydsp(jblk)
                jf0 = j0 - inorth + 1
                jf1 = j1 - inorth + 1

                i0min = grid.xdsp[iblk] + 1
                i1max = grid.xdsp[iblk] + grid.xcnt[iblk]
                if i1max > grid.nlon:
                    i1max = i1max - grid.nlon

                while i0min != i1max and not (
                CoLM_Utils.lon_between_floor(grid.lon_w[i0min], grid.lon_w[iwest], grid.lon_e[ieast])):
                    i0min = 1 if i0min > grid.nlon else i0min + 1

                if CoLM_Utils.lon_between_floor(grid.lon_w(i0min), grid.lon_w[iwest], grid.lon_e[ieast]):
                    i0 = i0min
                    i1 = i0
                    i1next = 1 if i1next > grid.nlon else i1 + 1

                    while i1 != i1max and \
                            CoLM_Utils.lon_between_floor(grid.lon_w[i1next], grid.lon_w[iwest], grid.lon_e[ieast]):
                        i1 = i1next
                        i1next = 1 if i1next > grid.nlon else i1 + 1

                    if0 = if0 + grid.nlon if if0 <= 0 else i0 - iwest + 1
                    if1 = if1 + grid.nlon if if1 <= 0 else i1 - iwest + 1
                    il0 = il0 + grid.nlon if il0 <= 0 else i0 - grid.xdsp[iblk]
                    il1 = il1 + grid.nlon if il1 <= 0 else i1 - grid.xdsp[iblk]

                    dcache = nc_mesh.ncio_read_part_serial(file_mesh, dataname, (jf0, if0), (jf1, if1))
                    dcache = np.transpose(dcache)
                    rdata.blk[iblk, jblk].val[il0:il1, jl0:jl1] = dcache

                if CoLM_Utils.lon_between_ceil(grid.lon_e[i1max], grid.lon_w[iwest], grid.lon_e[ieast]):
                    i1 = i1max
                    i0 = i1
                    i0next = grid.nlon if i0next == 0 else i0 - 1
                    while i0 != i0min and \
                            CoLM_Utils.lon_between_ceil(grid.lon_e(i0next), grid.lon_w[iwest], grid.lon_e[ieast]):
                        i0 = i0next
                        i0next = grid.nlon if i0next == 0 else i0 - 1

                    if i0 != i0min:
                        if0 = if0 + grid.nlon if if0 <= 0 else i0 - iwest + 1
                        if1 = if1 + grid.nlon if if1 <= 0 else i1 - iwest + 1
                        il0 = il0 + grid.nlon if il0 <= 0 else i0 - grid.xdsp[iblk]
                        il1 = il1 + grid.nlon if il1 <= 0 else i1 - grid.xdsp[iblk]

                        dcache = nc_mesh.ncio_read_part_serial(file_mesh, dataname, [jf0, if0], [jf1, if1])
                        dcache = np.transpose(dcache)
                        rdata.blk[iblk, jblk].val[il0:il1, jl0:jl1] = dcache

    else:
        if mpi.p_is_io:
            # remove suffix ".nc"
            path_mesh = file_meshdata_in[1:len(file_meshdata_in) - 3]

            for iblkme in range(gblock.nblkme):
                iblk = gblock.xblkme[iblkme]
                jblk = gblock.yblkme[iblkme]

                if spv is not None:
                    rdata.blk[iblk, jblk].val[:, :] = spv
                else:
                    rdata.blk[iblk, jblk].val[:, :] = 0

                inorth = grid.ydsp(jblk) + 1
                isouth = grid.ydsp(jblk) + grid.ycnt(jblk)

                iwest = grid.xdsp[iblk] + 1
                ieast = grid.xdsp[iblk] + grid.xcnt[iblk]
                if ieast > nxhglb:
                    ieast = ieast - nxhglb

                ibox = grid.xdsp[iblk] / nxhbox + 1
                jbox = grid.ydsp(jblk) / nyhbox + 1

                while True:
                    xdsp = (ibox - 1) * nxhbox
                    ydsp = (jbox - 1) * nyhbox

                    j0 = max(inorth - ydsp, 1)
                    j1 = min(isouth - ydsp, nyhbox)
                    jl0 = j0 + ydsp - inorth + 1
                    jl1 = j1 + ydsp - inorth + 1

                    if ieast >= iwest:
                        i0 = max(iwest - xdsp, 1)
                        i1 = min(ieast - xdsp, nxhbox)
                        il0 = i0 + xdsp - iwest + 1
                        il1 = i1 + xdsp - iwest + 1
                    else:
                        if iwest <= xdsp + nxhbox:
                            i0 = max(iwest - xdsp, 1)
                            i1 = nxhbox
                            il0 = i0 + xdsp - iwest + 1
                            il1 = i1 + xdsp - iwest + 1
                        else:
                            i0 = 1
                            i1 = min(ieast - xdsp, nxhbox)
                            il0 = i0 + xdsp + nxhglb - iwest + 1
                            il1 = i1 + xdsp + nxhglb - iwest + 1

                    pre1 = 0
                    pre2 = 0
                    if jbox <= 18:
                        pre1 = (18 - jbox) * 5
                    else:
                        pre1 = (jbox - 18) * 5

                    if ibox <= 36:
                        pre2 = (37 - ibox) * 5
                    else:
                        pre2 = (ibox - 37) * 5

                    file_mesh = path_mesh + '/' + str(pre1) + str(pre2) + ".nc"

                    if os.path.exists(file_mesh):
                        dcache = nc_mesh.ncio_read_part_serial(file_mesh, dataname, (j0, i0), (j1, i1))
                        dcache = np.transpose(dcache)
                        rdata.blk[iblk, jblk].val[il0:il1, jl0:jl1] = dcache

                    if (ieast >= xdsp + 1) and (ieast <= xdsp + nxhbox):
                        if isouth <= ydsp + nyhbox:
                            break
                        else:
                            ibox = grid.xdsp[iblk] / nxhbox + 1
                            jbox = jbox + 1
                    else:
                        ibox = ibox % (nxhglb / nxhbox) + 1
    return rdata
