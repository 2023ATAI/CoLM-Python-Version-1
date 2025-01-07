# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Pixels are rasterized points defined by fine-resolution data.
#
#    CoLM use multiple grids to construct pixels. Grids are assimilated into pixel
#    coordinate one by one. One grid is assimilated by adding grid lines not present
#    in pixel coordinate. In other words, pixel coordinate is the union of all grids.
#
#    Pixels are used to carry out land surface tessellation. The grids used to
#    construct pixels are associated with surface data such as land cover types, soil
#    parameters, plant function types, leaf area index and forest height.
#    By using pixels, these variables are downscaled to fine resolution.
#
#    In pixel data type, region boundaries and each pixel boundaries are defined.
#    Subroutines to assimilate grid and map pixel to grid are defined as methods.
#
# ------------------------------------------------------------------------------------
import numpy as np
import CoLM_Utils
from CoLM_NetCDFSerial import NetCDFFile


class Pixel_type(object):
    def __init__(self, mpi, user_mpi) -> None:
        self.MPI = mpi
        self.lat_s = [None]
        self.lat_n = [None]
        self.lon_w = [None]
        self.lon_e = [None]
        self.netfile = NetCDFFile(user_mpi)
        self.nlon = 1
        self.nlat = 1
        self.edges = 0
        self.edgen = 0
        self.edgew = 0
        self.edgee = 0

    def set_edges(self, edges_in, edgen_in, edgew_in, edgee_in):

        self.nlon = 1
        self.nlat = 1

        self.edges = edges_in
        self.edgen = edgen_in
        self.edgew = edgew_in
        self.edgee = edgee_in

        self.edgew = CoLM_Utils.normalize_longitude(self.edgew)
        self.edgee = CoLM_Utils.normalize_longitude(self.edgee)

        self.lat_s[0] = self.edges
        self.lat_n[0] = self.edgen
        self.lon_w[0] = self.edgew
        self.lon_e[0] = self.edgee

        if self.MPI.p_is_master:
            print('Region information:')
            print(' (south,north,west,east) = (' + str(self.edges) + ',' + str(self.edgen) + ',' + str(
                self.edgew) + ',' + str(self.edgee) + ')')

    def assimilate_latlon(self, nlat, lat_s, lat_n, nlon, lon_w, lon_e):
        yinc = 1
        if lat_s[0] <= lat_s[nlat - 1]:
            south = lat_s[0]
            north = lat_n[nlat - 1]
        else:
            yinc = -1
            south = lat_s[nlat - 1]
            north = lat_n[0]
        ny = 0
        ytmp = np.zeros(self.nlat + nlat + 2, dtype='float')
        # print(self.nlat,len(self.lat_n),'+++++++++++++++')

        for iy1 in range(self.nlat):
            ytmp[ny] = self.lat_s[iy1]

            if (self.lat_s[iy1] < north) and (self.lat_n[iy1] > south):
                ys2 = CoLM_Utils.find_nearest_south(self.lat_s[iy1], nlat, lat_s)
                yn2 = CoLM_Utils.find_nearest_north(self.lat_n[iy1], nlat, lat_n)
                for iy2 in range(ys2, yn2 + yinc, yinc):
                    if lat_s[iy2] > self.lat_s[iy1]:
                        ny = ny + 1
                        ytmp[ny] = lat_s[iy2]

                if lat_n[yn2] < self.lat_n[iy1]:
                    ny = ny + 1
                    ytmp[ny] = lat_n[yn2]
            ny = ny + 1
        ytmp[ny] = self.lat_n[self.nlat - 1]

        del self.lat_s
        del self.lat_n

        self.nlat = ny
        # print(len(ytmp),ny, 'ytmp len --------------')
        self.lat_s = ytmp[0:ny]
        self.lat_n = ytmp[1:ny + 1]
        # print(self.nlat, len(self.lat_n),'------------------')

        del ytmp
        # print(len(self.lat_n),'lat_n +++++++++++++++')

        west = lon_w[0]
        east = lon_e[nlon - 1]

        xtmp = np.zeros(self.nlon + nlon + 2, dtype='float')

        nx = 0
        for ix1 in range(self.nlon):
            xtmp[nx] = self.lon_w[ix1]

            if (CoLM_Utils.lon_between_floor(self.lon_w[ix1], west, east) or
                    CoLM_Utils.lon_between_ceil(self.lon_e[ix1], west, east) or
                    CoLM_Utils.lon_between_floor(west, self.lon_w[ix1], self.lon_e[ix1]) or
                    CoLM_Utils.lon_between_ceil(east, self.lon_w[ix1], self.lon_e[ix1])):

                xw2 = CoLM_Utils.find_nearest_west(self.lon_w[ix1], nlon, lon_w)
                xe2 = CoLM_Utils.find_nearest_east(self.lon_e[ix1], nlon, lon_e)

                if not CoLM_Utils.lon_between_floor(self.lon_w[ix1], lon_w[xw2], lon_e[xw2]):
                    xw2 = (xw2 + 1) % nlon
                if not CoLM_Utils.lon_between_ceil(self.lon_e[ix1], lon_w[xe2], lon_e[xe2]):
                    xe2 -= 1
                    if xe2 == -1:
                        xe2 = nlon - 1

                if (CoLM_Utils.lon_between_floor(lon_w[xw2], self.lon_w[ix1], self.lon_e[ix1])) \
                        and (lon_w[xw2] != self.lon_w[ix1]):
                    nx = nx + 1
                    xtmp[nx] = lon_w[xw2]

                if xw2 != xe2:
                    ix2 = (xw2 + 1) % nlon
                    # print(ix2,xe2,ix2%nlon,'-------------------')
                    while True:
                        nx = nx + 1
                        # print(nx,len(xtmp),ix2,len(lon_w),xe2,nlon,'test---------')
                        xtmp[nx] = lon_w[ix2]

                        if ix2 == xe2:
                            break
                        ix2 = (ix2 + 1) % nlon

                if (CoLM_Utils.lon_between_ceil(lon_e[xe2], self.lon_w[ix1], self.lon_e[ix1])) \
                        and (lon_e[xe2] != self.lon_e[ix1]):
                    nx = nx + 1
                    xtmp[nx] = lon_e[xe2]
            nx = nx + 1
        # print(self.nlon-1,nx, 'nlon len ---------------')
        xtmp[nx] = self.lon_e[self.nlon - 1]

        # del self.lon_w
        # del self.lon_e

        self.nlon = nx
        # self.lon_w = np.zeros(self.nlon)
        # self.lon_e = np.zeros(self.nlon)

        self.lon_w = xtmp[0:nx]
        self.lon_e = xtmp[1:nx+1]
        # print(self.nlon, len(self.lon_w),'**************************')

        del xtmp

    def assimilate_gblock(self, gblock):
        self.assimilate_latlon(gblock.nyblk, gblock.lat_s, gblock.lat_n, gblock.nxblk, gblock.lon_w, gblock.lon_e)

    def assimilate_grid(self, grid):
        self.assimilate_latlon(grid.nlat, grid.lat_s, grid.lat_n, grid.nlon, grid.lon_w, grid.lon_e)

    def map_to_grid(self, grd):
        # if grd.xgrd is not None:
        #     del grd.xgrd
        # if grd.ygrd is not None:
        #     del grd.ygrd

        grd.ygrd = np.zeros(self.nlat, dtype=int)
        south = 0
        north = 0
        west = 0
        east = 0

        if grd.yinc == 1:
            south = grd.lat_s[0]
            north = grd.lat_n[grd.nlat - 1]
        else:
            south = grd.lat_s[grd.nlat - 1]
            north = grd.lat_n[0]

        iy1 = 0
        while True:
            if (self.lat_s[iy1] < north) and (self.lat_n[iy1] > south):
                iy2 = CoLM_Utils.find_nearest_south(self.lat_s[iy1], grd.nlat, grd.lat_s)
                while self.lat_n[iy1] <= grd.lat_n[iy2]:
                    grd.ygrd[iy1] = iy2
                    iy1 = iy1 + 1
                    if iy1 > self.nlat - 1:
                        break
            else:
                grd.ygrd[iy1] = -1
                iy1 = iy1 + 1
            if iy1 > self.nlat - 1:
                break
        grd.xgrd = np.zeros(self.nlon, dtype=int)

        west = grd.lon_w[0]
        east = grd.lon_e[grd.nlon - 1]

        ix1 = 0
        while True:
            if CoLM_Utils.lon_between_floor(self.lon_w[ix1], west, east) or \
                    CoLM_Utils.lon_between_ceil(self.lon_e[ix1], west, east):

                ix2 = CoLM_Utils.find_nearest_west(self.lon_w[ix1], grd.nlon, grd.lon_w)
                while CoLM_Utils.lon_between_ceil(self.lon_e[ix1], grd.lon_w[ix2], grd.lon_e[ix2]):
                    grd.xgrd[ix1] = ix2
                    ix1 = ix1 + 1
                    if ix1 > self.nlon - 1:
                        break
            else:
                grd.xgrd[ix1] = -1
                ix1 = ix1 + 1
            if ix1 > self.nlon - 1:
                break
        return grd

    def save_to_file(self, dir_landdata):
        if self.MPI.p_is_master:
            filename = dir_landdata + '/pixel.nc'

            self.netfile.ncio_create_file(filename=filename)

            self.netfile.ncio_write_serial(filename, 'edges', self.edges)
            self.netfile.ncio_write_serial(filename, 'edgen', self.edgen)
            self.netfile.ncio_write_serial(filename, 'edgew', self.edgew)
            self.netfile.ncio_write_serial(filename, 'edgee', self.edgee)

            self.netfile.ncio_define_dimension(filename, 'latitude', self.nlat)
            self.netfile.ncio_define_dimension(filename, 'longitude', self.nlon)

            self.netfile.ncio_write_serial4(filename, 'lat_s', self.lat_s, 'latitude')
            self.netfile.ncio_write_serial4(filename, 'lat_n', self.lat_n, 'latitude')
            self.netfile.ncio_write_serial4(filename, 'lon_w', self.lon_w, 'longitude')
            self.netfile.ncio_write_serial4(filename, 'lon_e', self.lon_e, 'longitude')

    def load_from_file(self, dir_landdata):
        filename = f"{dir_landdata}/pixel.nc"

        self.edges = self.netfile.ncio_read_bcast_serial(filename, 'edges')
        self.edgen = self.netfile.ncio_read_bcast_serial(filename, 'edgen')
        self.edgew = self.netfile.ncio_read_bcast_serial(filename, 'edgew')
        self.edgee = self.netfile.ncio_read_bcast_serial(filename, 'edgee')
        self.lat_s = self.netfile.ncio_read_bcast_serial(filename, 'lat_s')
        self.lat_n = self.netfile.ncio_read_bcast_serial(filename, 'lat_n')
        self.lon_w = self.netfile.ncio_read_bcast_serial(filename, 'lon_w')
        self.lon_e = self.netfile.ncio_read_bcast_serial(filename, 'lon_e')

        self.nlon = len(self.lon_w)
        self.nlat = len(self.lat_s)

    def pixel_free_mem(self):
        if self.lat_s is not None:
            del self.lat_s
        if self.lat_n is not None:
            del self.lat_n
        if self.lon_w is not None:
            del self.lon_w
        if self.lon_e is not None:
            del self.lon_e
