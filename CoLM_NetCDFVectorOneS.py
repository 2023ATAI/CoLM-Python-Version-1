# ----------------------------------------------------------------------------------
# DESCRIPTION:
#
#    High-level Subroutines to read and write variables in files with netCDF format.
#
#    CoLM read and write netCDF files mainly in three ways:
#    1. Serial: read and write data by a single process;
#    2. Vector: 1) read vector data by IO and scatter from IO to workers
#               2) gather from workers to IO and write vectors by IO
#    3. Block : read blocked data by IO
#               Notice: input file is a single file.
#    
#    This MODULE CONTAINS subroutines of "2. Vector".
#    
#    Two implementations can be used,
#    1) "MOD_NetCDFVectorBlk.F90": 
#       A vector is saved in separated files, each associated with a block. 
#       READ/WRITE are fast in this way and compression can be used.
#       However, there may be too many files, especially when blocks are small. 
#       CHOOSE this implementation by "#undef VectorInOneFile" in include/define.h
#    2) "MOD_NetCDFVectorOne.F90": 
#       A vector is saved in one file. 
#       READ/WRITE may be slow in this way.
#       CHOOSE this implementation by "#define VectorInOneFileS" in include/define.h
#
# ----------------------------------------------------------------------------------
from netCDF4 import Dataset
import numpy as np
import os
from datetime import datetime
from CoLM_DataType import Pointer


def get_time_now():
    # Get the current date and time
    now = datetime.utcnow()

    # Format date, time, and zone as per the Fortran code
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M%S")
    zone = now.strftime("%z")

    # Concatenate the formatted date, time, and zone with the desired format
    formatted_time = f"{date}-{time[:2]}:{time[2:4]}:{time[4:6]} UTC{zone[:3]}:{zone[3:5]}"

    return formatted_time


def get_file_data(filename, mode):
    try:
        data = Dataset(filename, mode)
        return data
    except:
        print('Warning:', filename, 'not found.')
        return None


class CoLM_NetCDFVector(object):
    def __init__(self, colm, mpi, gblock) -> None:
        self.colm = colm
        self.mpi = mpi
        self.gblock = gblock

    def ncio_open_vector(self, filename, dataname, exit_on_err):
        ncid = get_file_data(filename, 'r')
        if ncid is None:
            return None, None
        vecname = None
        if not dataname in ncid.groups:
            print('Warning: ' + dataname + ' in ' + filename + ' not found.')
        else:
            grpid = ncid.groups[dataname]
            vecname = grpid.getncattr('vector_name')
            if vecname is not None and exit_on_err:
                print('Netcdf error in reading', dataname, 'from', filename)
                # CALL CoLM_STOP()

        return ncid, vecname

    def ncio_create_file_vector(self, filename, pixelset):
        if '/media/zjl/7C24CC1724CBD276' in filename:
            names = filename.split('/')
            s = ''
            for n in names:
                s = '/' + n
            filename = '/home/zjl' + s
        folder_path = os.path.dirname(filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pixelset = pixelset
        if self.mpi.p_is_master:  # Assuming p_is_master is defined elsewhere
            with Dataset(filename, 'w', format='NETCDF4') as ncid:
                ncid.setncattr('create_time', get_time_now())

        if self.colm['USEMPI']:
            pass
            # CALL mpi_barrier (p_comm_glb, p_err)

    def ncio_define_dimension_vector(self, filename, pixelset, dimname, dimlen=None):
        if self.mpi.p_is_io:  # Assuming p_is_io is defined elsewhere
            if self.mpi.p_iam_io == 0:  # Assuming p_iam_io is defined elsewhere
                if '/media/zjl/7C24CC1724CBD276' in filename:
                    names = filename.split('/')
                    s = ''
                    for n in names:
                        s = '/' + n
                    filename = '/home/zjl' + s
                ncid = get_file_data(filename, 'a')
                if ncid is None:
                    return
                if dimlen is not None:
                    if dimname not in ncid.dimensions.keys():
                        dimid = ncid.createDimension(dimname, dimlen)
                else:
                    for iblkall in range(pixelset.nblkall):
                        iblk = pixelset.xblkall[iblkall]
                        jblk = pixelset.yblkall[iblkall]
                        blockname = self.gblock.get_blockname(iblk, jblk)

                        if dimname + '_' + blockname not in ncid.dimensions.keys():
                            dimid = ncid.createDimension(dimname + '_' + blockname, pixelset.vlenall[iblk, jblk])
                ncid.close()

    def ncio_write_vector(self, filename, dataname, vecname, pixelset, wdata, compress_level=None):
        if self.colm['USEMPI']:
            if self.mpi.p_is_master:
                pass
                # DO i = 0, p_np_io-1
                #     CALL mpi_recv (isrc, 1, MPI_INTEGER, MPI_ANY_SOURCE, mpi_tag_mesg, &
                #     p_comm_glb, p_stat, p_err)
                #     CALL mpi_send (lock, 1, MPI_INTEGER, isrc, mpi_tag_mesg, p_comm_glb, p_err)
                #     CALL mpi_recv (lock, 1, MPI_INTEGER, isrc, mpi_tag_mesg, p_comm_glb, p_stat, p_err)

        if self.mpi.p_is_io:
            if compress_level is not None:
                self.ncio_define_variable_vector(filename, pixelset, vecname, dataname, 'i4', compress=compress_level)
            else:
                self.ncio_define_variable_vector(filename, pixelset, vecname, dataname, 'i4')

            rbuff = [Pointer()] * pixelset.nblkgrp

            for iblkgrp in range(pixelset.nblkgrp):
                iblk = int(pixelset.xblkgrp[iblkgrp])
                jblk = int(pixelset.yblkgrp[iblkgrp])

                rbuff[iblkgrp].val = np.zeros(int(pixelset.vecgs.vlen[iblk, jblk]))

                if self.colm['USEMPI']:
                    pass
                    # mpi_gatherv(MPI_IN_PLACE, 0, MPI_INTEGER, rbuff, pixelset.vecgs.vcnt[:, iblk - 1, jblk - 1],
                    #             pixelset.vecgs.vdsp[:, iblk - 1, jblk - 1], MPI_INTEGER, p_root, p_comm_group, p_err)
                else:
                    istt = int(pixelset.vecgs.vstt[iblk, jblk])
                    iend = int(pixelset.vecgs.vend[iblk, jblk])
                    rbuff[iblkgrp].val = wdata[istt:iend]

            if self.colm['USEMPI']:
                pass
                # CALL mpi_send (p_iam_glb, 1, MPI_INTEGER, 0, mpi_tag_mesg, p_comm_glb, p_err)
                # CALL mpi_recv (lock, 1, MPI_INTEGER, 0, mpi_tag_mesg, p_comm_glb, p_stat, p_err)
            if '/media/zjl/7C24CC1724CBD276' in filename:
                names = filename.split('/')
                s = ''
                for n in names:
                    s = '/' + n
                filename = '/home/zjl' + s

            ncid = get_file_data(filename, 'a')
            if ncid is None:
                return None, None
            grpid = None
            if not dataname in ncid.groups:
                print('Warning: ' + dataname + ' in ' + filename + ' not found.')
            else:
                grpid = ncid.groups[dataname]

            for iblkgrp in range(pixelset.nblkgrp):
                iblk = int(pixelset.xblkgrp[iblkgrp])
                jblk = int(pixelset.yblkgrp[iblkgrp])
                blockname = self.gblock.get_blockname(iblk, jblk)

                var_name = vecname + '_' + blockname

                grpid.variables[var_name] = rbuff[iblkgrp].val
                # del rbuff[iblkgrp].val

            ncid.close()

            if self.colm['USEMPI']:
                pass
                # CALL mpi_send (lock, 1, MPI_INTEGER, 0, mpi_tag_mesg, p_comm_glb, p_err)
        if self.colm['USEMPI']:
            pass
            # if self.mpi.p_is_worker:
            #     for iblkgrp in range(pixelset.nblkgrp):
            #         iblk = pixelset.xblkgrp[iblkgrp]
            #         jblk = pixelset.yblkgrp[iblkgrp]

            #         if pixelset.vecgs.vlen[iblk, jblk] > 0:
            #             sbuff = wdata[pixelset.vecgs.vstt[iblk - 1, jblk - 1]:pixelset.vecgs.vend[iblk - 1, jblk - 1] + 1]
            #         else:
            #             sbuff = np.empty(1, dtype=np.int32)

            #         mpi_gatherv(sbuff, pixelset.vecgs.vlen[iblk - 1, jblk - 1], MPI_INTEGER,
            #                     MPI_INULL_P, MPI_INULL_P, MPI_INULL_P, MPI_INTEGER,  # insignificant on workers
            #                     p_root, p_comm_group, p_err)

            #         if sbuff.size > 0:
            #             sbuff = None
            # CALL mpi_barrier (p_comm_glb, p_err)

    def ncio_define_variable_vector(self, filename, pixelset, vecname, dataname, datatype,
                                    dim1name=None, dim2name=None, dim3name=None, compress=None):
        if self.mpi.p_iam_io == 0:
            ndims = 1
            if '/media/zjl/7C24CC1724CBD276' in filename:
                names = filename.split('/')
                s = ''
                for n in names:
                    s = '/' + n
                filename = '/home/zjl' + s
            ncid = get_file_data(filename, 'a')

            dimids = []

            if ncid is None:
                return
            if dim1name is not None:
                dimids.append(dim1name)
            if dim2name is not None:
                dimids.append(dim2name)
            if dim3name is not None:
                dimids.append(dim3name)

            dimids.append("")

            grpid = ncid.createGroup(dataname)
            grpid.setncattr('vector_name', vecname)

            for iblkall in range(pixelset.nblkall):
                iblk = int(pixelset.xblkall[iblkall])
                jblk = int(pixelset.yblkall[iblkall])

                blockname = self.gblock.get_blockname(iblk, jblk)
                varname = f"{vecname}_{blockname}"
                dimids[len(dimids)-1] = varname

                if compress is not None:
                    varid = grpid.createVariable(varname, datatype, dimensions=dimids, zlib=True, complevel=compress)
                else:
                    varid = grpid.createVariable(varname, datatype, dimensions=dimids)
            del dimids
            ncid.close()
        if self.colm['USEMPI']:
            pass
        # CALL mpi_barrier (p_comm_io, p_err)

