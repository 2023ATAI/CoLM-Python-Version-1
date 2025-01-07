import numpy as np
import os
from CoLM_Grid import Grid_type, GridConcatType
from CoLM_Mapping_Pset2Grid import MappingPset2Grid
from CoLM_DataType import DataType
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_Block import Block_type


class CoLM_HistGridded:
    def __init__(self, nl_colm, mpi, pixel, mesh, landpatch, gblock):
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.pixel = pixel
        self.mesh = mesh
        self.landpatch = landpatch
        self.gblock = gblock
        self.ghist = Grid_type(nl_colm, gblock)
        self.mp2g_hist = MappingPset2Grid(mpi, nl_colm, gblock)
        self.mp2g_hist_urb = MappingPset2Grid(mpi, nl_colm, gblock)
        self.hist_concat = GridConcatType()
        self.hist_data_id = 0
        self.netcdffile = NetCDFFile(nl_colm['USEMPI'])

    def hist_gridded_init(self, gforc, pctshrpch):
        if self.nl_colm['DEF_hist_grid_as_forcing']:
            # pass
            self.ghist.define_by_copy(gforc)
        else:
            self.ghist.define_by_res(self.nl_colm['DEF_hist_lon_res'], self.nl_colm['DEF_hist_lat_res'])

        if self.nl_colm['CROP']:
            self.mp2g_hist.build(self.landpatch, self.ghist, self.pixel, self.mesh.mesh)
        else:
            self.mp2g_hist.build(self.landpatch, self.ghist, self.pixel, self.mesh.mesh, pctshrpch)

        # if 'landurban' in globals():
        #     mp2g_hist_urb.build_arealweighted(ghist, landurban)
        self.hist_concat.set(self.ghist, self.gblock)

        # #ifdef SinglePoint
        #   hist_concat%ginfo%lat_c(:) = SITE_lat_location
        #   hist_concat%ginfo%lon_c(:) = SITE_lon_location

        if self.nl_colm['DEF_HIST_mode'] == 'one':
            self.hist_data_id = 1000

    def flux_map_and_write_2d(self, acc_vec, file_hist, varname, itime_in_file, sumarea, filter, longname, units, nac,
                              spval):
        """
        Python version of the Fortran subroutine flux_map_and_write_2d
        """

        # Normalize acc_vec
        if self.mpi.p_is_worker:
            acc_vec = np.where(acc_vec != spval, acc_vec / nac, acc_vec)

        # Allocate flux_xy_2d if IO process
        if self.mpi.p_is_io:
            datatype = DataType(self.gblock)
            flux_xy_2d = datatype.allocate_block_data(self.ghist)  # Assuming some allocation function

        # Map acc_vec to flux_xy_2d
        self.mp2g_hist.map(acc_vec, flux_xy_2d, spv=spval, msk=filter)

        # Normalize flux_xy_2d by sumarea if IO process
        if self.mpi.p_is_io:
            for iblkme in range(self.gblock.nblkme):
                xblk = self.gblock.xblkme[iblkme]
                yblk = self.gblock.yblkme[iblkme]

                for yloc in range(self.ghist.ycnt[yblk]):
                    for xloc in range(self.ghist.xcnt[xblk]):

                        if sumarea.blk[xblk, yblk].val[xloc, yloc] > 0.00001:
                            if flux_xy_2d.blk[xblk, yblk].val[xloc, yloc] != spval:
                                flux_xy_2d.blk[xblk, yblk].val[xloc, yloc] /= sumarea.blk[xblk, yblk].val[xloc, yloc]
                        else:
                            flux_xy_2d.blk[xblk, yblk].val[xloc, yloc] = spval

        # Write the results to the history file
        compress = self.nl_colm['DEF_HIST_CompressLevel']
        self.hist_write_var_real8_2d(file_hist, varname, self.ghist, itime_in_file, flux_xy_2d, compress, longname,
                                     units, spval)

    def hist_write_var_real8_2d(self, filename, dataname, grid, itime, wdata, compress, longname, units, spval):
        """
        Python version of the Fortran subroutine hist_write_var_real8_2d
        """
        if self.nl_colm['DEF_HIST_mode'] == 'one':
            if self.mpi.p_is_master:  # Placeholder for master process check
                vdata = np.full((self.hist_concat.ginfo.nlon, self.hist_concat.ginfo.nlat), spval)

                for iyseg in range(self.hist_concat.nyseg):
                    for ixseg in range(self.hist_concat.nxseg):
                        iblk = self.hist_concat.xsegs[ixseg].blk
                        jblk = self.hist_concat.ysegs[iyseg].blk
                        if self.gblock.pio[iblk, jblk] == self.mpi.p_iam_glb:
                            xbdsp = self.hist_concat.xsegs[ixseg].bdsp
                            ybdsp = self.hist_concat.ysegs[iyseg].bdsp
                            xgdsp = self.hist_concat.xsegs[ixseg].gdsp
                            ygdsp = self.hist_concat.ysegs[iyseg].gdsp
                            xcnt = self.hist_concat.xsegs[ixseg].cnt
                            ycnt = self.hist_concat.ysegs[iyseg].cnt

                            vdata[xgdsp:xgdsp + xcnt, ygdsp:ygdsp + ycnt] = \
                                wdata.blk[iblk, jblk].val[xbdsp:xbdsp + xcnt, ybdsp:ybdsp + ycnt]

                self.netcdffile.ncio_write_serial_time(filename, dataname, itime, vdata, 'lon', 'lat', 'time', compress)

                if itime == 1:
                    self.netcdffile.ncio_put_attr(filename, dataname, 'long_name', longname)
                    self.netcdffile.ncio_put_attr(filename, dataname, 'units', units)
                    self.netcdffile.ncio_put_attr(filename, dataname, 'missing_value', spval)

                del vdata
            self.hist_data_id += 1

        elif self.nl_colm['DEF_HIST_mode'] == 'block':
            if self.mpi.p_is_io:  # Placeholder for IO process check
                for iblkme in range(self.gblock.nblkme):
                    iblk = self.gblock.xblkme[iblkme]
                    jblk = self.gblock.yblkme[iblkme]

                    if grid.xcnt[iblk] == 0 or grid.ycnt[jblk] == 0:
                        continue

                    fileblock = self.gblock.get_filename_block(filename, iblk, jblk)

                    self.netcdffile.ncio_write_serial_time(fileblock, dataname, itime,
                                                           wdata.blk[iblk, jblk].val, 'lon', 'lat', 'time', compress)

    def flux_map_and_write_3d(self, acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, sumarea, filter,
                              longname, units, nac, spval):
        # Local variables
        datatype = DataType(self.gblock)
        flux_xy_3d = None
        # If worker process
        if self.mpi.p_is_worker:
            acc_vec[acc_vec != spval] /= nac

        # If IO process
        if self.mpi.p_is_io:
            flux_xy_3d = datatype.allocate_block_data3(self.ghist, ndim1, lb1)
            # Map the data
        self.mp2g_hist.map3d(acc_vec, flux_xy_3d, spv=spval, msk=filter)

        if self.mpi.p_is_io:
            # Loop through blocks and perform operations
            for iblkme in range(self.gblock.nblkme):
                xblk = self.gblock.xblkme[iblkme]
                yblk = self.gblock.yblkme[iblkme]

                for yloc in range(self.ghist.ycnt[yblk]):
                    for xloc in range(self.ghist.xcnt[xblk]):
                        if sumarea.blk[xblk, yblk].val[xloc, yloc] > 0.00001:
                            for i1 in range(flux_xy_3d.lb1 -1, flux_xy_3d.ub1):
                                if flux_xy_3d.blk[xblk, yblk].val[i1, xloc, yloc] != spval:
                                    flux_xy_3d.blk[xblk, yblk].val[i1, xloc, yloc] /= sumarea.blk[xblk, yblk].val[
                                        xloc, yloc]
                        else:
                            flux_xy_3d.blk[xblk, yblk].val[:, xloc, yloc] = spval

            # Write the variable
            compress = self.nl_colm['DEF_HIST_CompressLevel']
            self.hist_write_var3(file_hist, varname, dim1name, self.ghist, itime_in_file, flux_xy_3d, compress,
                                         longname, units, spval)

    def hist_write_var3(self, filename, dataname, dim1name, grid, itime, wdata, compress, longname, units, spval):
        # Determine mode
        if self.nl_colm['DEF_HIST_mode'] == 'one':
            if self.mpi.p_is_master:
                ndim1 = wdata.ub1 - wdata.lb1 + 1
                vdata = np.full((ndim1, self.hist_concat.ginfo.nlon, self.hist_concat.ginfo.nlat), spval)

                for iyseg in range(self.hist_concat.nyseg):
                    for ixseg in range(self.hist_concat.nxseg):
                        iblk = self.hist_concat.xsegs[ixseg].blk
                        jblk = self.hist_concat.ysegs[iyseg].blk
                        if self.gblock.pio[iblk, jblk] == self.mpi.p_iam_glb:
                            xbdsp = self.hist_concat.xsegs[ixseg].bdsp
                            ybdsp = self.hist_concat.ysegs[iyseg].bdsp
                            xgdsp = self.hist_concat.xsegs[ixseg].gdsp
                            ygdsp = self.hist_concat.ysegs[iyseg].gdsp
                            xcnt = self.hist_concat.xsegs[ixseg].cnt
                            ycnt = self.hist_concat.ysegs[iyseg].cnt

                            vdata[:, xgdsp+1:xgdsp + xcnt+1, ygdsp+1:ygdsp + ycnt+1] = \
                                wdata.blk[iblk, jblk].val[:, xbdsp+1:xbdsp + xcnt+1, ybdsp+1:ybdsp + ycnt+1]

                self.netcdffile.ncio_define_dimension(filename, dim1name, ndim1)
                self.netcdffile.ncio_write_serial_time(filename, dataname, itime, vdata, dim1name, 'lon', 'lat', 'time',
                                                       compress)

                if itime == 1:
                    self.netcdffile.ncio_put_attr(filename, dataname, 'long_name', longname)
                    self.netcdffile.ncio_put_attr(filename, dataname, 'units', units)
                    self.netcdffile.ncio_put_attr(filename, dataname, 'missing_value', spval)

                del vdata

            self.hist_data_id += (self.hist_data_id + 1)%1000

        elif self.nl_colm['DEF_HIST_mode'] == 'block':
            if self.mpi.p_is_io:
                for iblkme in range(self.gblock.nblkme):
                    iblk = self.gblock.xblkme[iblkme]
                    jblk = self.gblock.yblkme[iblkme]

                    if grid.xcnt[iblk] == 0 or grid.ycnt[jblk] == 0:
                        continue

                    fileblock = self.gblock.get_filename_block(filename, iblk, jblk)
                    self.netcdffile.ncio_define_dimension(fileblock, dim1name, wdata.ub1 - wdata.lb1 + 1)
                    self.netcdffile.ncio_write_serial_time(fileblock, dataname, itime, wdata.blk[iblk, jblk].val,
                                                           dim1name, 'lon', 'lat', 'time', compress)

    def hist_write_var4(self, filename, dataname, dim1name, dim2name, grid, itime, wdata, compress, longname, units, spval):
        # Determine mode
        if self.nl_colm['DEF_HIST_mode'] == 'one':
            if self.mpi.p_is_master:
                ndim1 = wdata.ub1 - wdata.lb1 + 1
                ndim2 = wdata.ub2 - wdata.lb2 + 1
                vdata = np.full((ndim1, ndim2, self.hist_concat.ginfo.nlon, self.hist_concat.ginfo.nlat), spval)
                for iyseg in range(self.hist_concat.nyseg):
                    for ixseg in range(self.hist_concat.nxseg):
                        iblk = self.hist_concat.xsegs[ixseg].blk
                        jblk = self.hist_concat.ysegs[iyseg].blk
                        if self.gblock.pio[iblk, jblk] == self.mpi.p_iam_glb:
                            xbdsp = self.hist_concat.xsegs[ixseg].bdsp
                            ybdsp = self.hist_concat.ysegs[iyseg].bdsp
                            xgdsp = self.hist_concat.xsegs[iyseg].gdsp
                            ygdsp = self.hist_concat.ysegs[iyseg].gdsp
                            xcnt = self.hist_concat.xsegs[iyseg].cnt
                            ycnt = self.hist_concat.ysegs[iyseg].cnt

                            vdata[:, :, xgdsp:xgdsp + xcnt, ygdsp:ygdsp + ycnt] = \
                                wdata.blk[iblk, jblk].val[:, :, xbdsp:xbdsp + xcnt, ybdsp:ybdsp + ycnt]

                self.netcdffile.ncio_define_dimension(filename, dim1name, ndim1)
                self.netcdffile.ncio_define_dimension(filename, dim2name, ndim2)

                self.netcdffile.ncio_write_serial_time(filename, dataname, itime, vdata,
                                                       dim1name, dim2name, 'lon', 'lat', 'time', compress)

                if itime == 1:
                    self.netcdffile.ncio_put_attr(filename, dataname, 'long_name', longname)
                    self.netcdffile.ncio_put_attr(filename, dataname, 'units', units)
                    self.netcdffile.ncio_put_attr(filename, dataname, 'missing_value', spval)

                del vdata

            self.hist_data_id += 1

        elif self.nl_colm['DEF_HIST_mode'] == 'block':
            if self.mpi.p_is_io:
                for iblkme in range(self.gblock.nblkme):
                    iblk = self.gblock.xblkme[iblkme]
                    jblk = self.gblock.yblkme[iblkme]

                    if grid.xcnt[iblk] == 0 or grid.ycnt[jblk] == 0:
                        continue

                    fileblock = self.gblock.get_filename_block(filename, iblk, jblk)

                    self.netcdffile.ncio_define_dimension(fileblock, dim1name, wdata.ub1 - wdata.lb1 + 1)
                    self.netcdffile.ncio_define_dimension(fileblock, dim2name, wdata.ub2 - wdata.lb2 + 1)

                    self.netcdffile.ncio_write_serial_time(fileblock, dataname, itime,
                                                           wdata.blk[iblk, jblk].val, dim1name, dim2name, 'lon', 'lat',
                                                           'time', compress)

    def flux_map_and_write_4d(self, acc_vec, file_hist, varname, itime_in_file, dim1name, lb1, ndim1, dim2name, lb2,
                              ndim2, sumarea, filter, longname, units, nac, spval):
        # Constants and placeholders
        flux_xy_4d = None
        # Assume acc_vec is a NumPy array
        if self.mpi.p_is_worker:
            acc_vec = np.where(acc_vec != spval, acc_vec / nac, acc_vec)
        if self.mpi.p_is_io:
            datatype = DataType(self.gblock)
            flux_xy_4d = datatype.allocate_block_data4(self.ghist, ndim1, ndim2, lb1 = lb1, lb2 = lb2)

        self.mp2g_hist.map4d(acc_vec, flux_xy_4d, spv=spval, msk=filter)

        if self.mpi.p_is_io:
            for iblkme in range(self.gblock.nblkme):
                xblk = self.gblock.xblkme[iblkme]
                yblk = self.gblock.yblkme[iblkme]

                for yloc in range(self.ghist.ycnt[yblk]):
                    for xloc in range(self.ghist.xcnt[xblk]):

                        if sumarea.blk[xblk, yblk].val[xloc, yloc] > 0.00001:
                            for i1 in range(flux_xy_4d.lb1 -1, flux_xy_4d.ub1):
                                for i2 in range(flux_xy_4d.lb2-1, flux_xy_4d.ub2):
                                    if flux_xy_4d.blk[xblk, yblk].val[i1, i2, xloc, yloc] != spval:
                                        flux_xy_4d.blk[xblk, yblk].val[i1, i2, xloc, yloc] /= \
                                        sumarea.blk[xblk, yblk].val[xloc, yloc]
                        else:
                            flux_xy_4d.blk[xblk, yblk].val[:, :, xloc, yloc] = np.full((flux_xy_4d.blk[xblk, yblk].val[:, :, xloc, yloc].shape[0],flux_xy_4d.blk[xblk, yblk].val[:, :, xloc, yloc].shape[1]),spval)

        compress = self.nl_colm['DEF_HIST_CompressLevel']
        self.hist_write_var4(file_hist, varname, dim1name, dim2name, self.ghist, itime_in_file, flux_xy_4d, compress,
                             longname, units, spval)

    def flux_map_and_write_ln(self, acc_vec, file_hist, varname, itime_in_file, sumarea, filter,
                              longname, units, spval, nac_ln):
        # Assuming p_is_worker and p_is_io are predefined
        if self.mpi.p_is_worker:
            for i in range(acc_vec.shape[0]):
                if acc_vec[i] != spval and nac_ln[i] > 0:
                    acc_vec[i] /= nac_ln[i]

        if self.mpi.p_is_io:
            datatype = DataType(self.gblock)
            flux_xy_2d = datatype.allocate_block_data(self.ghist)

        # Assuming self.mp2g_hist.map is a function that maps acc_vec to flux_xy_2d
        self.mp2g_hist.map(acc_vec, flux_xy_2d, spv=spval, msk=filter)

        if self.mpi.p_is_io:
            for iblkme in range(self.gblock.nblkme):
                xblk = self.gblock.xblkme[iblkme]
                yblk = self.gblock.yblkme[iblkme]

                for yloc in range(self.ghist.ycnt[yblk]):
                    for xloc in range(self.ghist.xcnt[xblk]):
                        if (sumarea.blk[xblk, yblk].val[xloc, yloc] > 0.00001 and
                                flux_xy_2d.blk[xblk, yblk].val[xloc, yloc] != spval):
                            flux_xy_2d.blk[xblk, yblk].val[xloc, yloc] /= sumarea.blk[xblk, yblk].val[xloc, yloc]
                        else:
                            flux_xy_2d.blk[xblk, yblk].val[xloc, yloc] = spval

        compress = self.nl_colm['DEF_HIST_CompressLevel']
        self.hist_write_var_real8_2d(file_hist, varname, self.ghist, itime_in_file, flux_xy_2d,
                                     compress, longname, units, spval)

    def hist_gridded_write_time(self, filename, dataname, time, itime):
        if self.nl_colm['DEF_HIST_mode'] == 'one':
            if self.mpi.p_is_master:
                netcdffile_tmep = NetCDFFile(self.nl_colm['USEMPI'])
                if not os.path.exists(filename):
                    netcdffile_tmep.ncio_create_file(filename)
                    netcdffile_tmep.ncio_define_dimension(filename, 'time', 0)
                    netcdffile_tmep.ncio_define_dimension(filename, 'lat', self.hist_concat.ginfo.nlat)
                    netcdffile_tmep.ncio_define_dimension(filename, 'lon', self.hist_concat.ginfo.nlon)

                    netcdffile_tmep.ncio_write_serial4(filename, 'lat', self.hist_concat.ginfo.lat_c, 'lat')
                    netcdffile_tmep.ncio_put_attr(filename, 'lat', 'long_name', 'latitude')
                    netcdffile_tmep.ncio_put_attr(filename, 'lat', 'units', 'degrees_north')

                    netcdffile_tmep.ncio_write_serial4(filename, 'lon', self.hist_concat.ginfo.lon_c, 'lon')
                    netcdffile_tmep.ncio_put_attr(filename, 'lon', 'long_name', 'longitude')
                    netcdffile_tmep.ncio_put_attr(filename, 'lon', 'units', 'degrees_east')

                netcdffile_tmep.ncio_write_time(filename, dataname, time, itime, self.nl_colm['DEF_HIST_FREQ'])

        elif self.nl_colm['DEF_HIST_mode'] == 'block':
            if self.mpi.p_is_io:
                for iblkme in range(self.gblock.nblkme):
                    iblk = self.gblock.xblkme[iblkme]
                    jblk = self.gblock.yblkme[iblkme]

                    if self.ghist.ycnt[jblk] <= 0 or self.ghist.xcnt[iblk] <= 0:
                        continue

                    fileblock = self.gblock.get_filename_block(filename, iblk, jblk)

                    if not os.path.exists(fileblock):
                        netcdffile_tmep = NetCDFFile(self.nl_colm['USEMPI'])
                        netcdffile_tmep.ncio_create_file(fileblock)
                        netcdffile_tmep.ncio_define_dimension(fileblock, 'time', 0)
                        self.hist_write_grid_info(fileblock, self.ghist, iblk, jblk)

                    netcdffile_tmep.ncio_write_time(fileblock, dataname, time, itime, self.nl_colm['DEF_HIST_FREQ'])

    def hist_write_grid_info(self, fileblock, grid, iblk, jblk):
        """
        Python equivalent of the Fortran subroutine hist_write_grid_info.

        Parameters:
        fileblock (str): Path to the NetCDF file block.
        grid (dict): Grid data structure containing latitude and longitude information.
        iblk (int): Block index in x direction.
        jblk (int): Block index in y direction.
        """

        # Allocate arrays
        lon_w = np.zeros(grid.xcnt[iblk])
        lon_e = np.zeros(grid.xcnt[iblk])
        lat_s = np.zeros(grid.ycnt[jblk])
        lat_n = np.zeros(grid.ycnt[jblk])

        yl = grid.ydsp[jblk] + 1
        yu = grid.ydsp[jblk] + grid.ycnt[jblk]

        lat_s[:] = grid.lat_s[yl - 1:yu]  # Adjusted for Python 0-indexing
        lat_n[:] = grid.lat_n[yl - 1:yu]

        if grid.xdsp[iblk] + grid.xcnt[iblk] > grid.nlon:
            xl = grid.xdsp[iblk] + 1
            xu = grid.nlon
            nx = grid.nlon - grid.xdsp[iblk]
            lon_w[:nx] = grid.lon_w[xl - 1:xu]  # Adjusted for Python 0-indexing
            lon_e[:nx] = grid.lon_e[xl - 1:xu]

            xl = 0  # Adjusted for Python 0-indexing
            xu = grid.xcnt[iblk] - nx
            lon_w[nx:] = grid.lon_w[xl:xu]
            lon_e[nx:] = grid.lon_e[xl:xu]
        else:
            xl = grid.xdsp[iblk] + 1
            xu = grid.xdsp[iblk] + grid.xcnt[iblk]
            lon_w[:] = grid.lon_w[xl - 1:xu]  # Adjusted for Python 0-indexing
            lon_e[:] = grid.lon_e[xl - 1:xu]

        netcdffile_tmep = NetCDFFile(self.nl_colm['USEMPI'])
        netcdffile_tmep.ncio_create_file(fileblock)

        # Define dimensions and write data to NetCDF file
        grid.ycnt[jblk] = netcdffile_tmep.ncio_define_dimension(fileblock, 'lat', grid.ycnt[jblk])
        grid.xcnt[iblk] = netcdffile_tmep.ncio_define_dimension(fileblock, 'lon', grid.xcnt[iblk])

        netcdffile_tmep.ncio_write_serial(fileblock, 'lat_s', lat_s, 'lat')
        netcdffile_tmep.ncio_write_serial(fileblock, 'lat_n', lat_n, 'lat')
        netcdffile_tmep.ncio_write_serial(fileblock, 'lon_w', lon_w, 'lon')
        netcdffile_tmep.ncio_write_serial(fileblock, 'lon_e', lon_e, 'lon')
