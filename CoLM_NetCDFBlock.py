# 读取nc文件
# import xarray as xr
import os.path

import numpy as np

# from CoLM_NetCDFSerial import NetCDFFile
from netCDF4 import Dataset

def ncio_read_block(filename, dataname, mpi, gblock, grid, rdata):
    # try:
    nc_file = Dataset(filename, 'r')

    # var_data = nc_file.variables[dataname][:]

    if mpi.p_is_io:
        count2 = np.zeros(2, dtype='int')

        for iblkme in range(gblock.nblkme):
            iblk = gblock.xblkme[iblkme]
            jblk = gblock.yblkme[iblkme]

            ndims = [grid.xcnt[iblk], grid.ycnt[jblk]]
            if any(dim == 0 for dim in ndims):
                continue

            start2 = [grid.xdsp[iblk] + 1, grid.ydsp[jblk] + 1]
            count2[0] = min(grid.xcnt[iblk], grid.nlon - grid.xdsp[iblk])
            count2[1] = grid.ycnt[jblk]

            if count2[0] == grid.xcnt[iblk]:
                # start2要1在前，0在后，因为count2是正确顺序，而Python读取数据尺寸是43200，86400，而其他数据是86400，43200，需要换维度
                # print(type(start2[0]), type(start2[1]), start2[0], start2[1])
                # print(type(count2[0]), type(count2[1]), count2[0], count2[1])
                # rdata.blk[iblk, jblk].val = nc_file.variables[dataname][start2[0]:start2[0] + count2[0],start2[1]:start2[1] + count2[1]]
                rdata.blk[iblk, jblk].val = nc_file.variables[dataname][start2[1]:start2[1] + count2[1],
                                            start2[0]:start2[0] + count2[0]].transpose()
                pass
            else:
                # rdata.blk[iblk, jblk].val[0:count2[0], :] = nc_file.variables[dataname][start2[0]:start2[0] + count2[0],
                #                                             start2[1]:start2[1] + count2[1]]
                rdata.blk[iblk, jblk].val[0:count2[0], :] = nc_file.variables[dataname][start2[1]:start2[1] + count2[1],
                                                            start2[0]:start2[0] + count2[0]].transpose()
                start2[0] = 0
                start_mem = count2[0] + 1
                count2[0] = grid.xdsp[iblk] + grid.xcnt[iblk] - grid.nlon
                # rdata.blk[iblk, jblk].val[start_mem:ndims[0], :] = nc_file.variables[dataname][
                #                                                    start2[0]:start2[0] + count2[0],
                #                                                    start2[1]:start2[1] + count2[1]]
                rdata.blk[iblk, jblk].val[start_mem:ndims[0], :] = nc_file.variables[dataname][
                                                                   start2[1]:start2[1] + count2[0],
                                                                   start2[0]:start2[0] + count2[1]]
                pass
    nc_file.close()
    return rdata
    # except Exception as e:
    #     raise RuntimeError(f"{dataname} in file {filename} not found: {str(e)}")


def ncio_read_block_time(filename, dataname, grid, itime, rdata, mpi, gblock):
    # Local variables
    if mpi.p_is_io:  # Assuming p_is_io is defined elsewhere
        if os.path.exists(filename):

            nc_file = Dataset(filename, 'r')

            for iblkme in range(gblock.nblkme):
                iblk = gblock.xblkme[iblkme]
                jblk = gblock.yblkme[iblkme]

                ndims = [grid.xcnt[iblk], grid.ycnt[jblk]]
                if any(dim == 0 for dim in ndims):
                    continue

                start3 = [ grid.xdsp[iblk] + 1,grid.ydsp[jblk] + 1,itime-1]
                count3 = [min(grid.xcnt[iblk], grid.nlon - grid.xdsp[iblk]), grid.ycnt[jblk], 1]
                # print(start3, count3,'-------start--------')
                #
                # start3 = [itime, grid.ydsp[jblk] + 1, grid.xdsp[iblk] + 1]
                # count3 = [1, grid.ycnt[jblk], min(grid.xcnt[iblk], grid.nlon - grid.xdsp[iblk])]

                if len(nc_file.variables[dataname].shape) == 2:
                    if count3[2] == grid.xcnt[iblk]:
                        rdata.blk[iblk, jblk].val = nc_file.variables[dataname][start3[1]: start3[1] + count3[1],
                                                    start3[2]: start3[2] + count3[2]].transpose()
                    else:
                        rdata.blk[iblk, jblk].val[:count3[1], :] = nc_file.variables[dataname][
                                                                   start3[1]: start3[1] + count3[1],
                                                                   start3[2]: start3[2] + count3[2]].transpose()
                        start3[2] = 0
                        start_mem = count3[2] + 1
                        count3[2] = grid.xdsp[iblk] + grid.xcnt[iblk] - grid.nlon
                        rdata.blk[iblk, jblk].val[start_mem:ndims[0], :] = nc_file.variables[dataname][
                                                                           start3[1]: start3[1] + count3[1],
                                                                           start3[2]: start3[2] + count3[2]].transpose()
                else:
                    if count3[0] == grid.xcnt[iblk]:
                        temp = None
                        if dataname=='TBOT' or dataname=='QBOT' or dataname=='PSRF' or dataname=='WIND' or dataname=='FSDS' or dataname=='FLDS':
                            temp = nc_file.variables[dataname][start3[2]: start3[2]+1, start3[1]: start3[1]+ count3[1],start3[1]: start3[1]+ count3[0] ]
                        else:
                            temp = nc_file.variables[dataname][start3[2]: start3[2] + count3[2], start3[0]: start3[0] + count3[0], start3[1]: start3[1] + count3[1]]
                        if len(temp.shape)==3 and temp.shape[0]==1:
                            temp = temp[0,:,:]
                            if temp.shape[0]==count3[0]:
                                rdata.blk[iblk, jblk].val = temp
                            else:
                                rdata.blk[iblk, jblk].val = np.transpose(temp)

                        else:
                            rdata.blk[iblk, jblk].val = temp

                        # print(rdata.blk[iblk, jblk].val.shape,'======shape==1= ===')
                    else:
                        rdata.blk[iblk, jblk].val[:count3[0], :] = np.reshape(nc_file.variables[dataname][
                                                                     start3[0]: start3[0] + count3[0],
                                                                     start3[1]: start3[1] + count3[1],
                                                                     start3[2]: start3[2] + count3[2]], (count3[0], -1))
                        start3[0] = 0
                        start_mem = count3[0] + 1
                        count3[0] = grid.xdsp[iblk] + grid.xcnt[iblk] - grid.nlon
                        rdata.blk[iblk, jblk].val[start_mem:ndims[0], :] = np.reshape(nc_file.variables[dataname][
                                                                              start3[0]: start3[0] + count3[0],
                                                                              start3[1]: start3[1] + count3[1],
                                                                              start3[2]: start3[2] + count3[2]], count3[0], -1)

                        # print(rdata.blk[iblk, jblk].val.shape, '======shape==2= ===')

                    # if count3[2] == grid.xcnt[iblk]:
                    #     rdata.blk[iblk, jblk].val = np.reshape(nc_file.variables[dataname][start3[0]: start3[0] + count3[0],
                    #                                 start3[1]: start3[1] + count3[1],
                    #                                 start3[2]: start3[2] + count3[2]], (count3[1], -1))
                    # else:
                    #     rdata.blk[iblk, jblk].val[:count3[1], :] = np.reshape(nc_file.variables[dataname][
                    #                                                  start3[0]: start3[0] + count3[0],
                    #                                                  start3[1]: start3[1] + count3[1],
                    #                                                  start3[2]: start3[2] + count3[2]], (count3[1], -1))
                    #     start3[2] = 0
                    #     start_mem = count3[2] + 1
                    #     count3[2] = grid.xdsp[iblk] + grid.xcnt[iblk] - grid.nlon
                    #     rdata.blk[iblk, jblk].val[start_mem:ndims[0], :] = np.reshape(nc_file.variables[dataname][
                    #                                                           start3[0]: start3[0] + count3[0],
                    #                                                           start3[1]: start3[1] + count3[1],
                    #                                                           start3[2]: start3[2] + count3[2]], count3[1], -1)

            nc_file.close()
            return rdata
    return None


def ncio_read_block_time6(filename, dataname, grid, ndim1, itime, rdata, mpi, gblock):
    # Local variables
    if mpi.p_is_io:  # Assuming p_is_io is defined elsewhere
        if os.path.exists(filename):

            nc_file = Dataset(filename, 'r')
            # print(nc_file.variables[dataname].shape, '-------------------')
            # varid = nc_file.variables[dataname][:]

            for iblkme in range(gblock.nblkme):
                iblk = gblock.xblkme[iblkme]
                jblk = gblock.yblkme[iblkme]

                ndims = [ndim1, grid.xcnt[iblk], grid.ycnt[jblk]]
                if any(dim == 0 for dim in ndims):
                    continue

                start4 = [1, grid.xdsp[iblk] + 1, grid.ydsp[jblk] + 1, itime]

                count4 = [ndim1, min(grid.xcnt[iblk], grid.nlon - grid.xdsp[iblk]), grid.ycnt[jblk], 1]

                if count4[0] == grid.xcnt[iblk]:
                    rdata.blk[iblk, jblk].val = nc_file.variables[dataname][start4[0]: start4[0] + count4[0],
                                                start4[2]: start4[2] + count4[1],
                                                start4[1]: start4[1] + count4[2]]
                else:
                    rdata.blk[iblk, jblk].val[:, :start4[2], :] = nc_file.variables[dataname][
                                                                  start4[0]: start4[0] + count4[0],
                                                                  start4[2]: start4[2] + count4[1],
                                                                  start4[1]: start4[1] + count4[2]]
                    start4[1] = 0
                    start_mem = count4[1] + 1
                    count4[1] = grid.xdsp[iblk] + grid.xcnt[iblk] - grid.nlon
                    rdata.blk[iblk, jblk].val[start_mem:ndims[0], :] = nc_file.variables[dataname][
                                                                       start4[0]: start4[0] + count4[0],
                                                                       start4[2]: start4[2] + count4[1],
                                                                       start4[1]: start4[1] + count4[2]]

            nc_file.close()
            return rdata
    return None


def ncio_read_site_time(filename, dataname, itime, rdata, gblock, mpi):
    """
    Reads a 3D variable from a NetCDF file and updates rdata.

    Parameters:
    filename (str): Path to the NetCDF file.
    dataname (str): Name of the variable to read.
    itime (int): Time index to read from the variable.
    rdata (dict): Data structure to update with the read values.

    Returns:
    None
    """
    fid = False
    data = None

    # Check if the NetCDF file exists (handled by external check_ncfile_exist function)
    if mpi.p_is_io:
        if not os.path.exists(filename):
            print(f"NetCDF file {filename} does not exist.")

        if not fid:
            fid = True
            # Open the NetCDF file
            with Dataset(filename, 'r') as ds:
                # Get the dimension and variable IDs
                # time_dim = ds.dimensions['time'].size
                var = ds.variables[dataname]
                
                # Read the variable data
                start = [0, 0, itime]
                count = [1, 1, 1]
                data = var[start[0]:start[0]+count[0], start[1]:start[1]+count[1], start[2]:start[2]+count[2]]
                
                # Update the rdata structure (assuming rdata is a dict with keys for the block)
                xblkme = gblock['xblkme'][0]  # Example access; adjust based on actual structure
                yblkme = gblock['yblkme'][0]  # Example access; adjust based on actual structure
                rdata.blk[xblkme,yblkme].val = data
    return data