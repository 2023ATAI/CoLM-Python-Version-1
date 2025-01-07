# 读取nc文件
# import xarray as xr
from netCDF4 import Dataset
import datetime
import numpy as np
import CoLM_TimeManager
from CoLM_Utils import overload_decorator

class NetCDFFile(object):
    def __init__(self, use_mpi) -> None:
        self.USEMPI = use_mpi

    def ncio_read_serial(self, filename, data_name):
        nc_file = self.ncio_read_file(filename, 'r')

        if nc_file is None:
            return None

        if data_name in nc_file.variables:
            # var = np.transpose(nc_file.variables[data_name][:])
            var = nc_file.variables[data_name][:]
            if var.size == 1:
                if var.dtype == 'float':
                    var = float(var)
            nc_file.close()
            return var
        else:
            print(f"Error: Variable '{data_name}' not found in the file.")
            nc_file.close()
            return None

    def ncio_inquire_varsize(self, filename, varname):
        nc_file = self.ncio_read_file(filename, 'r')

        if nc_file is None:
            return None

        if varname in nc_file.variables:
            var = nc_file.variables[varname][:]
            nc_file.close()
            return var.shape
        else:
            print(f"Error: Variable '{varname}' not found in the file.")
            nc_file.close()
            return None

    def ncio_read_file(self, filename, model):
        try:
            nc_file = Dataset(filename, model)
            return nc_file
        except FileNotFoundError:
            print("Error: File not found.")
            return None

    def ncio_create_file(self, filename):
        try:
            if filename is not None:
                nc_file = Dataset(filename, "w", format='NETCDF4')
                nc_file.setncattr('create_time', str(datetime.datetime.now()))
                # nc_file.enddef()
                nc_file.close()

        except Exception as e:
            print("An error occurred:", e)

    def ncio_define_dimension(self, filename, dimname, dimlen):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return

        if dimname not in ncfile.dimensions:
            # If dimension doesn't exist, define it
            if dimlen == 0:
                ncfile.createDimension(dimname, None)  # Unlimited dimension
            else:
                ncfile.createDimension(dimname, dimlen)
            dimid = dimname

            # Define variable based on dimension name
            varid = None
            if dimname == 'lon':
                varid = ncfile.createVariable('lon', 'f4', [dimid])
                varid.long_name = 'longitude'
                varid.units = 'degrees_east'
            elif dimname == 'lat':
                varid = ncfile.createVariable('lat', 'f4', [dimid])
                varid.long_name = 'latitude'
                varid.units = 'degrees_north'
            elif dimname == 'lat_cama':
                varid = ncfile.createVariable('lat_cama', 'f4', [dimid])
                varid.long_name = 'latitude'
                varid.units = 'degrees_north'
            elif dimname == 'lon_cama':
                varid = ncfile.createVariable('lon_cama', 'f4', [dimid])
                varid.long_name = 'longitude'
                varid.units = 'degrees_east'

        ncfile.close()

    def ncio_write_serial4(self, filename, dataname, wdata, dimname=None, compress=None):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return None

        if dataname not in ncfile.variables:
            if dimname is None:
                print('Warning: no dimension name for ' + dataname)
                return
            dimid = ncfile.dimensions[dimname].name

            if compress is not None:
                varid = ncfile.createVariable(dataname, 'i1', [dimid], zlib=True, complevel=compress)
            else:
                varid = ncfile.createVariable(dataname, 'i1', [dimid])

        # Write data
        ncfile.variables[dataname][:] = wdata
        ncfile.close()

    def ncio_write_serial(self, filename, dataname, wdata):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return None

        if dataname not in ncfile.variables:
            ncfile.createVariable(dataname, 'i4')

        # Write data
        ncfile.variables[dataname][:] = wdata
        ncfile.close()

    def ncio_write_serial5(self, filename, dataname, wdata, dim1name=None, dim2name=None, compress=None):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return None

        if dataname in ncfile.variables:
            pass
        else:
            # If variable doesn't exist, define it
            if dim1name is None or dim2name is None:
                print('Warning: no dimension name for', dataname)
                return

            dim1id = ncfile.dimensions[dim1name].name
            dim2id = ncfile.dimensions[dim2name].name

            if compress is not None:
                varid = ncfile.createVariable(dataname, 'i2', (dim1id, dim2id), complevel=compress)
            else:
                varid = ncfile.createVariable(dataname, 'i2', (dim1id, dim2id))

            # Write data
            # print(wdata,'---------------------')
            ncfile.variables[dataname] = wdata
        ncfile.close()

    # def ncio_write_serial3(self, filename, dataname, wdata, dimname):
    #     ncfile = self.ncio_read_file(filename)
    #
    #     if ncfile is None:
    #         return None
    #
    #     var = ncfile.createVariable(dataname, 'f8', dimname)  # f8表示64位浮点数
    #     var[:] = wdata
    #     ncfile.close()
    def ncio_read_period_serial_real8_2d(self, filename, dataname, timestt, timeend):
        # Check if the file exists
        # Allocate the output array
        rdata = None
        with Dataset(filename, 'r') as ncid:
            if ncid is None:
                return None

            varid = ncid.variables[dataname]
            data = varid[:, :, timestt:timeend + 1]
            rdata[:, :, :] = data

        return rdata

    def ncio_write_serial6(self, filename, dataname, wdata, dim1name=None, dim2name=None, dim3name=None, compress=None):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return

        # If variable doesn't exist, define it
        if dim1name is None or dim2name is None or dim3name is None:
            print('Warning: no dimension name for', dataname)
            return

        if dataname in ncfile.variables:
            ncfile.close()
            return

        dim1id = ncfile.dimensions[dim1name].name
        dim2id = ncfile.dimensions[dim2name].name
        dim3id = ncfile.dimensions[dim3name].name

        if compress is not None:
            varid = ncfile.createVariable(dataname, 'i4', (dim1id, dim2id, dim3id), zlib=True,
                                          complevel=compress)
        else:
            varid = ncfile.createVariable(dataname, 'i4', (dim1id, dim2id, dim3id))

        # Write data
        ncfile.variables[dataname][:] = wdata
        ncfile.close()

    def ncio_write_serial8(self, filename, dataname, wdata,
         dim1name, dim2name, dim3name, dim4name, compress):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return None

        # If variable doesn't exist, define it
        if dim1name is None or dim2name is None or dim3name is None or dim4name is None:
            print('Warning: no dimension name for', dataname)
            return

        dim1id = ncfile.dimensions[dim1name].name
        dim2id = ncfile.dimensions[dim2name].name
        dim3id = ncfile.dimensions[dim3name].name
        dim4id = ncfile.dimensions[dim4name].name

        if compress is not None:
            varid = ncfile.createVariable(dataname, 'i4', (dim1id, dim2id, dim3id,dim4id), zlib=True,
                                          complevel=compress)
        else:
            varid = ncfile.createVariable(dataname, 'i4', (dim1id, dim2id, dim3id,dim4id))

        # Write data
        ncfile.variables[dataname][:] = wdata
        ncfile.close()

    def ncio_put_attr(self, filename, varname, attrname, attrval):
        ncfile = self.ncio_read_file(filename, 'a')

        if ncfile is None:
            return None

        setattr(ncfile.variables[varname],attrname,attrval)

        ncfile.close()

    def ncio_read_bcast_serial(self, filename, dimname):
        rdata = self.ncio_read_serial(filename, dimname)
        if self.USEMPI:
            # CALL mpi_bcast (rdata, 1, MPI_INTEGER, p_root, p_comm_glb, p_err)
            pass
        return rdata

    def ncio_var_exist(self, filename, dataname):
        try:
            with Dataset(filename, 'r') as ncfile:
                is_exist = dataname in ncfile.variables.keys()
                return is_exist
        except FileNotFoundError:
            print("Error: File not found:",filename,dataname)
            return False

    def ncio_read_part_serial(self, filename, dataname, datastt, dataend):
        ncid, var = self.ncio_read_serial(filename, dataname)
        rdata = var[datastt[0]:dataend[0] + 1, datastt[1]:dataend[1] + 1]
        ncid.close()
        del var

        return rdata

    def ncio_get_attr(self, filename, varname, attrname):
        attrval = None
        ncid, var = self.ncio_read_serial(filename, varname)
        attrval = var[attrname]
        return attrval

    def ncio_inquire_length(self, filename, dataname):
        """
        Inquires the length of a specific variable in a NetCDF file.

        Parameters:
        filename (str): The name of the NetCDF file.
        dataname (str): The name of the variable.

        Returns:
        length (int): The length of the specified variable's dimension.
        """
        length = 0
        try:
            with Dataset(filename, 'r') as ncfile:
                # Check if the variable exists in the file
                if dataname in ncfile.variables:
                    var = ncfile.variables[dataname]
                    # Retrieve the dimensions of the variable
                    dims = var.shape
                    if dims:
                        # Length of the last dimension
                        length = dims[-1]
        except Exception as e:
            print(f"Error accessing file {filename}: {e}")
        return length
    
    def ncio_write_serial_time0d(self, filename, dataname, itime, wdata, dim1name=None):
        # Open the NetCDF file in write mode
        with Dataset(filename, 'r+') as dataset:
            try:
                # Try to get the variable ID
                var = dataset.variables[dataname]
            except KeyError:
                # If the variable does not exist
                if dim1name is None:
                    print(f'Warning: no dimension name for {dataname}')
                    return

                # Get the dimension ID
                dim = dataset.dimensions[dim1name]

                # Switch to define mode and define the variable
                dataset.setncattr('format', 'NETCDF4')  # Ensure compatibility with NETCDF4
                var = dataset.createVariable(dataname, np.float64, (dim1name,))
            
            # Write the data to the variable
            var[itime] = wdata

    def ncio_write_serial_time2d(self,filename, dataname, itime, wdata, dim1name, dim2name, dim3name, compress):
        try:
            with Dataset(filename, 'r+') as dataset:
                # 尝试获取变量ID
                try:
                    varid = dataset.variables[dataname]._varid
                except KeyError:
                    if not (dim1name and dim2name and dim3name):
                        print(f'Warning: no dimension name for {dataname}')
                        return

                    # 获取维度ID
                    dimid = []
                    for dimname in [dim1name, dim2name, dim3name]:
                        dimid.append(dataset.dimensions[dimname].name)

                    # 重新定义变量
                    dataset.set_auto_mask(False)
                    if compress is not None:
                        var = dataset.createVariable(dataname, np.float64, dimid, zlib=True, complevel=compress)
                    else:
                        var = dataset.createVariable(dataname, np.float64, dimid)
                    varid = var.varid

                # 将数据写入变量
                dataset.variables[dataname][0:wdata.shape[0], 0:wdata.shape[1], itime:itime+ 1] = wdata
        except Exception as e:
            print(f"An error occurred: {e}")

    def ncio_write_serial_time3d(self, filename, dataname, itime, wdata, dim1name, dim2name, dim3name, dim4name, compress):
        try:
            with Dataset(filename, 'r+') as dataset:
                # 尝试获取变量ID
                try:
                    varid = dataset.variables[dataname]._varid
                except KeyError:
                    if not (dim1name and dim2name and dim3name and dim4name):
                        print(f'Warning: no dimension name for {dataname}')
                        return

                    # 获取维度ID
                    dimid = []
                    for dimname in [dim1name, dim2name, dim3name, dim4name]:
                        dimid.append(dataset.dimensions[dimname].name)

                    # 重新定义变量
                    dataset.set_auto_mask(False)
                    if compress is not None:
                        var = dataset.createVariable(dataname, np.float64, dimid, zlib=True, complevel=compress)
                    else:
                        var = dataset.createVariable(dataname, np.float64, dimid)

                # 将数据写入变量
                dataset.variables[dataname][0:wdata.shape[0], 0:wdata.shape[1], 0:wdata.shape[2], itime:itime + 1] = wdata
        except Exception as e:
            print(f"An error occurred: {e}")

    def ncio_write_serial_time4d(self, filename, dataname, itime, wdata, dim1name, dim2name, dim3name, dim4name, dim5name, compress):
        try:
            with Dataset(filename, 'r+') as dataset:
                # 尝试获取变量ID
                try:
                    varid = dataset.variables[dataname]._varid
                except KeyError:
                    if not (dim1name and dim2name and dim3name and dim4name and dim5name):
                        print(f'Warning: no dimension name for {dataname}')
                        return

                    # 获取维度ID
                    dimid = []
                    for dimname in [dim1name, dim2name, dim3name, dim4name, dim5name]:
                        dimid.append(dataset.dimensions[dimname].name)

                    # 重新定义变量
                    dataset.set_auto_mask(False)
                    if compress is not None:
                        var = dataset.createVariable(dataname, np.float64, dimid, zlib=True, complevel=compress)
                    else:
                        var = dataset.createVariable(dataname, np.float64, dimid)

                # 将数据写入变量
                dataset.variables[dataname][0:wdata.shape[0], 0:wdata.shape[1], 0:wdata.shape[2], 0:wdata.shape[3], itime:itime + 1] = wdata
        except Exception as e:
            print(f"An error occurred: {e}")

    @overload_decorator((ncio_write_serial_time0d, 6),
                        (ncio_write_serial_time2d, 9),
                        (ncio_write_serial_time3d, 10),
                        (ncio_write_serial_time4d, 11))
    def ncio_write_serial_time(result):
        return result

    def ncio_write_time(self, filename, dataname, time_component, itime, adjust=None):
        """
        Python equivalent of the Fortran subroutine ncio_write_time.

        Parameters:
        filename (str): Path to the NetCDF file.
        dataname (str): Name of the data variable.
        time_component (list of int): Time components [year, month, day].
        itime (int): Time index (output).
        adjust (str, optional): Adjustment type ('HOURLY', 'DAILY', 'MONTHLY', 'YEARLY').
        """
        
        minutes = CoLM_TimeManager.minutes_since_1900(time_component.year, time_component.day, time_component.sec)
        
        if adjust is not None:
            adjust = adjust.strip()
            if adjust == 'HOURLY':
                minutes -= 30
            elif adjust == 'DAILY':
                minutes -= 720
            elif adjust == 'MONTHLY':
                minutes -= 21600
            elif adjust == 'YEARLY':
                minutes -= 262800
        
        with Dataset(filename, 'r+') as ds:
            if dataname in ds.variables:
                time_var = ds.variables[dataname]
                time_dim = ds.dimensions['time']
                timelen = len(time_dim)
                
                itime = 1
                if timelen > 0:
                    time_file = time_var[:]
                    for idx, time_val in enumerate(time_file):
                        if minutes == time_val:
                            itime = idx + 1
                            break
                    else:
                        itime = timelen + 1
            else:
                if 'time' not in ds.dimensions:
                    ds.createDimension('time', None)
                time_var = ds.createVariable(dataname, 'i4', ('time',))
                time_var.long_name = 'time'
                time_var.units = 'minutes since 1900-1-1 0:0:0'
                itime = 1
            
            time_var[itime-1] = minutes


