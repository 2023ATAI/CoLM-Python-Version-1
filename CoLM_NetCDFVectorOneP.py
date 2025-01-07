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


class CoLM_NetCDFVectorP(object):
    def __init__(self, colm, mpi, gblock) -> None:
        self.colm = colm
        self.mpi = mpi
        self.gblock = gblock

    def ncio_inquire_length_grp(self, filename, dataname, blkname):
        """
        Inquires the length of a specific variable within a group in a NetCDF file.

        Parameters:
        filename (str): The name of the NetCDF file.
        dataname (str): The name of the group within the NetCDF file.
        blkname (str): The name of the variable within the group.

        Returns:
        length (int): The length of the specified variable's dimension.
        """
        length = 0
        try:
            with Dataset(filename, 'r') as ncfile:
                # Get the group
                grp = ncfile.groups[dataname]
                # Check if the variable exists in the group
                if blkname in grp.variables:
                    var = grp.variables[blkname]
                    # Retrieve the dimensions of the variable
                    dims = var.shape
                    if dims:
                        # Length of the last dimension
                        length = dims[-1]
        except Exception as e:
            print(f"Error accessing file {filename}: {e}")
        return length
    
    def ncio_read_serial_grp_int64_1d(self,filename, dataname, blkname):
        varlen = self.ncio_inquire_length_grp(filename, dataname, blkname)
        if varlen > 0:
        # Open the NetCDF file
            with Dataset(filename, 'r') as ncfile:
                # Get the group
                group = ncfile.groups[dataname]
                
                # Get the variable
                variable = group.variables[blkname]
                
                # Get the data length
                varlen = len(variable)
                
                # Initialize the data array
                if varlen > 0:
                    rdata = variable[:]
                else:
                    rdata = []

        return rdata
    
    def ncio_open_vector(self, filename, dataname, exit_on_err):
        """
        Open a NetCDF file and retrieve group ID and vector name attribute.

        Parameters:
            filename (str): Path to the NetCDF file.
            dataname (str): Name of the group in the NetCDF file.
            exit_on_err (bool): Flag indicating whether to exit on error.

        Returns:
            tuple: (ncid, grpid, vecname, noerr)
                ncid (int): NetCDF file ID.
                grpid (int): Group ID within the NetCDF file.
                vecname (str): Vector name attribute.
                noerr (bool): Flag indicating success or failure.
        """
        noerr = True
        vecname = ""
        
        try:
            dataset = Dataset(filename, 'r')
            ncid = dataset.ncattrs()
        except FileNotFoundError:
            noerr = False
            print(f"Warning: {filename} not found.")
            if exit_on_err:
                raise RuntimeError(f"NetCDF error: {filename} not found.")
            return None, None, None, noerr

        if noerr:
            try:
                group = dataset.groups[dataname]
                grpid = group.ncattrs()
            except KeyError:
                noerr = False
                print(f"Warning: {dataname} in {filename} not found.")
                if exit_on_err:
                    raise RuntimeError(f"NetCDF error: {dataname} in {filename} not found.")
                return ncid, None, None, noerr

        if noerr:
            try:
                vecname = group.getncattr('vector_name')
            except AttributeError:
                noerr = False
                print(f"Warning: vector_name in {filename} not found.")
                if exit_on_err:
                    raise RuntimeError(f"NetCDF error: vector_name in {filename} not found.")
                return ncid, grpid, None, noerr
            
        if not noerr and exit_on_err:
            print('Netcdf error in reading ' + dataname + ' from ' + filename)
        
        return dataset, group, vecname, noerr


    def ncio_read_vector(self, filename, dataname, pixelset, rdata, defval=None):
        # Initialize rdata if not already allocated
        if self.mpi.p_is_worker:
            if pixelset.nset > 0 and rdata is None:
                rdata = np.zeros(pixelset.nset, dtype=int)

        if self.mpi.p_is_io:            
            # Open the NetCDF file
            noerr = False
            sbuff = None
            ok = False

            ncid, grpid, vecname, noerr = self.ncio_open_vector(filename, dataname, defval is None)
            
            for iblkgrp in range(pixelset.nblkgrp):
                iblk = pixelset.xblkgrp[iblkgrp]
                jblk = pixelset.yblkgrp[iblkgrp]

                if noerr:
                    blockname = self.gblock.get_blockname(jblk, iblk)
                    # blockname = self.gblock.get_blockname(iblk, jblk)互换位置
                    varname = f"{vecname}_{blockname}"
                    try:
                        variable = grpid.variables[varname]
                        sbuff = variable[:]
                        ok = True
                    except KeyError:
                        ok = False

                if not ok:
                    if defval is None:
                        print(f"NetCDF error reading {varname} from {filename}")
                    else:
                        sbuff[:] = defval[:]


                istt = pixelset.vecgs.vstt[iblk, jblk]
                iend = pixelset.vecgs.vend[iblk, jblk]
                print(iblk,jblk,istt,iend,len(sbuff),'**************')

                rdata[istt:iend] = sbuff
                del sbuff
            # ncid.close()

        return rdata
    


