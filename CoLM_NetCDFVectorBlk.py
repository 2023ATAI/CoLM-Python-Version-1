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
import os.path
from CoLM_Utils import overload_decorator
import numpy as np
from CoLM_NetCDFSerial import NetCDFFile


def ncio_read_vector5(filename, dataname, pixelset, rdata, usermpi, mpi, gblock, defval=None):
    # Initialize rdata if not already allocated
    if mpi.p_is_worker:
        if pixelset.nset > 0 and rdata is None:
            rdata = np.zeros(pixelset.nset, dtype=int)

    any_file_exists = False

    if mpi.p_is_io:
        # Open the NetCDF file
        noerr = False
        sbuff = None
        ok = False
        net_cdf = NetCDFFile(usermpi)

        # ncid, grpid, vecname, noerr = self.ncio_open_vector(filename, dataname, defval is None)

        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            sbuff = np.zeros(pixelset.vecgs.vlen[iblk, jblk], dtype=int)

            fileblock = gblock.get_filename_block(filename, iblk, jblk)
            # print(filename,fileblock,'**************')
            this_file_exists = os.path.exists(fileblock)
            any_file_exists = any_file_exists or this_file_exists

            if net_cdf.ncio_var_exist(fileblock, dataname):
                sbuff = net_cdf.ncio_read_serial(fileblock, dataname)
            elif defval is not None:
                sbuff[:] = defval

            if usermpi:
                pass
            else:
                istt = pixelset.vecgs.vstt[iblk, jblk]
                iend = pixelset.vecgs.vend[iblk, jblk]
                # print(iblk,jblk,istt,iend,len(sbuff),'**************')

                rdata[istt:iend + 1] = sbuff
            del sbuff
        # ncid.close()

    return rdata


def ncio_read_vector6(filename, dataname, ndim1, pixelset, rdata, usermpi, mpi, gblock, defval=None):
    if mpi.p_is_worker:
        if pixelset.nset > 0 and rdata is None:
            rdata = np.zeros((ndim1, pixelset.nset))

    any_file_exists = False
    net_cdf = NetCDFFile(usermpi)

    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]

            sbuff = np.zeros((ndim1, pixelset.vecgs.vlen[iblk, jblk]))
            fileblock = gblock.get_filename_block(filename, iblk, jblk)

            this_file_exists = os.path.exists(fileblock)
            any_file_exists = any_file_exists or this_file_exists

            if net_cdf.ncio_var_exist(fileblock, dataname):
                sbuff = net_cdf.ncio_read_serial(fileblock, dataname)
            elif defval is not None:
                sbuff[:, :] = defval

            if usermpi:
                pass
            else:
                istt = pixelset.vecgs.vstt[iblk, jblk]
                iend = pixelset.vecgs.vend[iblk, jblk]
                # print(istt,iend,rdata.shape,sbuff.shape,'-----------')
                # rdata[:, istt:iend + 1] = sbuff.transpose(1, 0)
                if sbuff.shape[0] != ndim1:
                    sbuff = sbuff.transpose(1, 0)
                rdata[:, istt:iend + 1] = sbuff
            del sbuff

        if usermpi:
            pass

        if not any_file_exists:
            print(f'Warning: restart file {filename} not found.')

    if usermpi:
        pass
        # if mpi.p_is_worker:
        #     for iblkgrp in range(pixelset['nblkgrp']):
        #         iblk = pixelset['xblkgrp'][iblkgrp]
        #         jblk = pixelset['yblkgrp'][iblkgrp]

        #         if pixelset['vecgs']['vlen'][iblk, jblk] > 0:
        #             rbuff = np.empty((ndim1, pixelset['vecgs']['vlen'][iblk, jblk]))
        #         else:
        #             rbuff = np.empty((1, 1))

        #         rbuff = MPI.COMM_WORLD.scatterv([None, None, None, MPI.DOUBLE], root=0)

        #         if pixelset['vecgs']['vlen'][iblk, jblk] > 0:
        #             istt = pixelset['vecgs']['vstt'][iblk, jblk]
        #             iend = pixelset['vecgs']['vend'][iblk, jblk]
        #             rdata[:, istt:iend] = rbuff

    return rdata


def ncio_read_vector7(filename, dataname, ndim1, ndim2, pixelset, rdata, usermpi, mpi, gblock, defval=None):
    if mpi.p_is_worker:
        if pixelset.nset > 0 and rdata is None:
            rdata = np.zeros((ndim1, ndim2, pixelset.nset))

    any_file_exists = False
    net_cdf = NetCDFFile(usermpi)

    if mpi.p_is_io:

        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]

            sbuff = np.zeros((ndim1, ndim2, pixelset.vecgs.vlen[iblk, jblk]))
            fileblock = gblock.get_filename_block(filename, iblk, jblk)

            this_file_exists = os.path.exists(fileblock)
            any_file_exists = any_file_exists or this_file_exists

            if net_cdf.ncio_var_exist(fileblock, dataname):
                sbuff = net_cdf.ncio_read_serial(fileblock, dataname)
            elif defval is not None:
                sbuff[:, :, :] = defval

            if usermpi:
                pass
            else:
                istt = pixelset.vecgs.vstt[iblk, jblk]
                iend = pixelset.vecgs.vend[iblk, jblk]
                # print(istt,iend,rdata.shape,sbuff.shape,'-----------')
                rdata[:, :, istt:iend + 1] = sbuff.transpose(1,2,0)
                # rdata[:, :, istt:iend + 1] = sbuff
            del sbuff

        if usermpi:
            pass

        if not any_file_exists:
            print(f'Warning: restart file {filename} not found.')

    if usermpi:
        pass
        # if mpi.p_is_worker:
        #     for iblkgrp in range(pixelset['nblkgrp']):
        #         iblk = pixelset['xblkgrp'][iblkgrp]
        #         jblk = pixelset['yblkgrp'][iblkgrp]

        #         if pixelset['vecgs']['vlen'][iblk, jblk] > 0:
        #             rbuff = np.empty((ndim1, pixelset['vecgs']['vlen'][iblk, jblk]))
        #         else:
        #             rbuff = np.empty((1, 1))

        #         rbuff = MPI.COMM_WORLD.scatterv([None, None, None, MPI.DOUBLE], root=0)

        #         if pixelset['vecgs']['vlen'][iblk, jblk] > 0:
        #             istt = pixelset['vecgs']['vstt'][iblk, jblk]
        #             iend = pixelset['vecgs']['vend'][iblk, jblk]
        #             rdata[:, istt:iend] = rbuff

    return rdata


def ncio_read_vector8(filename, dataname, ndim1, ndim2, ndim3, pixelset, rdata, usermpi, mpi, gblock, defval=None):
    if mpi.p_is_worker:
        if pixelset.nset > 0 and rdata is None:
            rdata = np.zeros((ndim1, ndim2, ndim3, pixelset.nset))

    any_file_exists = False
    net_cdf = NetCDFFile(usermpi)

    if mpi.p_is_io:

        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]

            sbuff = np.zeros((ndim1, ndim2, ndim3, pixelset.vecgs.vlen[iblk, jblk]))
            fileblock = gblock.get_filename_block(filename, iblk, jblk)

            this_file_exists = os.path.exists(fileblock)
            any_file_exists = any_file_exists or this_file_exists

            if net_cdf.ncio_var_exist(fileblock, dataname):
                sbuff = net_cdf.ncio_read_serial(fileblock, dataname)
            elif defval is not None:
                sbuff[:, :, :, :] = defval

            if usermpi:
                pass
            else:
                istt = pixelset.vecgs.vstt[iblk, jblk]
                iend = pixelset.vecgs.vend[iblk, jblk]
                # print(istt,iend,rdata.shape,sbuff.shape,'-----------')
                rdata[:, :, :, istt:iend + 1] = sbuff.transpose(2, 3, 1, 0)
                # rdata[:, :, :, istt:iend + 1] = sbuff
            del sbuff

        if usermpi:
            pass

        if not any_file_exists:
            print(f'Warning: restart file {filename} not found.')

    if usermpi:
        pass
        # if mpi.p_is_worker:
        #     for iblkgrp in range(pixelset['nblkgrp']):
        #         iblk = pixelset['xblkgrp'][iblkgrp]
        #         jblk = pixelset['yblkgrp'][iblkgrp]

        #         if pixelset['vecgs']['vlen'][iblk, jblk] > 0:
        #             rbuff = np.empty((ndim1, pixelset['vecgs']['vlen'][iblk, jblk]))
        #         else:
        #             rbuff = np.empty((1, 1))

        #         rbuff = MPI.COMM_WORLD.scatterv([None, None, None, MPI.DOUBLE], root=0)

        #         if pixelset['vecgs']['vlen'][iblk, jblk] > 0:
        #             istt = pixelset['vecgs']['vstt'][iblk, jblk]
        #             iend = pixelset['vecgs']['vend'][iblk, jblk]
        #             rdata[:, istt:iend] = rbuff

    return rdata


@overload_decorator((ncio_read_vector5, 7),
                    (ncio_read_vector6, 8),
                    (ncio_read_vector7, 9),
                    (ncio_read_vector8, 10))
def ncio_read_vector(result):
    return result


def ncio_write_vector12(filename, dataname, dim1name, ndim1, dim2name, ndim2,
                        dim3name, pixelset, wdata, mpi, gblock, compress_level=None):
    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            istt = pixelset.vecgs.vstt[iblk, jblk]
            iend = pixelset.vecgs.vend[iblk, jblk]
            rbuff = wdata[:, :, istt:iend + 1]
            fileblock = gblock.get_filename_block(filename, jblk, iblk)
            netfile = NetCDFFile(mpi.USEMPI)
            if compress_level is not None:
                netfile.ncio_write_serial6(fileblock, dataname, rbuff, dim1name, dim2name, dim3name,
                                           compress=compress_level)
            else:
                netfile.ncio_write_serial6(fileblock, dataname, rbuff, dim1name, dim2name, dim3name)
            del rbuff


def ncio_write_vector14(filename, dataname, dim1name, ndim1, dim2name, ndim2,
                        dim3name, ndim3, dim4name, pixelset, wdata, mpi, gblock, compress_level=None):
    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            istt = pixelset.vecgs.vstt[iblk, jblk]
            iend = pixelset.vecgs.vend[iblk, jblk]
            rbuff = wdata[:, :, :, istt:iend + 1]
            fileblock = gblock.get_filename_block(filename, jblk, iblk)
            netfile = NetCDFFile(mpi.USEMPI)
            if compress_level is not None:
                netfile.ncio_write_serial8(fileblock, dataname, rbuff, dim1name, dim2name, dim3name, dim4name,
                                           compress=compress_level)
            else:
                netfile.ncio_write_serial8(fileblock, dataname, rbuff, dim1name, dim2name, dim3name, dim4name)
            del rbuff


def ncio_write_vector(filename, dataname, dimname, pixelset, wdata, mpi, gblock, compress_level=None):
    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            istt = pixelset.vecgs.vstt[iblk, jblk]
            iend = pixelset.vecgs.vend[iblk, jblk]
            rbuff = wdata[istt:iend + 1]
            fileblock = gblock.get_filename_block(filename, jblk, iblk)
            netfile = NetCDFFile(mpi.USEMPI)
            if compress_level is not None:
                netfile.ncio_write_serial5(fileblock, dataname, rbuff, dimname,
                                           compress=compress_level)
            else:
                netfile.ncio_write_serial4(fileblock, dataname, rbuff, dimname)
            del rbuff


def ncio_write_vector10(filename, dataname, dim1name, ndim1, dim2name, pixelset, wdata, mpi, gblock,
                        compress_level=None):
    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            istt = pixelset.vecgs.vstt[iblk, jblk]
            iend = pixelset.vecgs.vend[iblk, jblk]
            rbuff = wdata[:, istt:iend + 1]
            fileblock = gblock.get_filename_block(filename, jblk, iblk)
            netfile = NetCDFFile(mpi.USEMPI)
            if compress_level is not None:
                netfile.ncio_write_serial5(fileblock, dataname, rbuff, dim1name, dim2name,
                                           compress=compress_level)
            else:
                netfile.ncio_write_serial4(fileblock, dataname, rbuff, dim1name, dim2name)
            del rbuff


def ncio_create_file_vector(filename, pixelset, mpi, gblock, MPI):
    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            filename = gblock.get_filename_block(filename, jblk, iblk)
            netfile = NetCDFFile(MPI)

            netfile.ncio_create_file(filename)


def ncio_define_dimension_vector(filename, pixelset, dimname, mpi, gblock, MPI, dimlen=0):
    if mpi.p_is_io:
        for iblkgrp in range(pixelset.nblkgrp):
            iblk = pixelset.xblkgrp[iblkgrp]
            jblk = pixelset.yblkgrp[iblkgrp]
            fileblock = gblock.get_filename_block(filename, jblk, iblk)
            netfile = NetCDFFile(MPI)

            if os.path.exists(fileblock):

                if dimlen != 0:
                    netfile.ncio_define_dimension(fileblock, dimname, dimlen)
                else:
                    netfile.ncio_define_dimension(fileblock, dimname,
                                                  pixelset.vecgs.vlen[iblk, jblk])
