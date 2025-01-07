# -------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Definition of latitude-longitude grids and data types related to grids. 
#
#    Latitude-longitude grid can be defined by
#    1. "name"   : frequently used grids is predefined in this module;
#    2. "ndims"  : how many longitude and latitude grids are used globally;
#    3. "res"    : longitude and latitude resolutions in radian
#    4. "center" : longitude and latitude grid centers, and the border lines 
#                  are defined by center lines of grid centers; the region
#                  boundaries is optional.
#    5. "file"   : read grid informations from a file, the variables are
#                  'lat_s', 'lat_n', 'lon_w', 'lon_e'
#    6. "copy"   : copy grid informations from an existing grid
# 
#    Grid centers in radian can be calculated by using "set_rlon" and "set_rlat"
# 
#    Two additional data types are defined:
#    1. "grid_list_type"   : list of grid boxes;
#    2. "grid_concat_type" : used to concatenate grids distributed in blocks.
# 
# Created by Shupeng Zhang, May 2023
# -------------------------------------------------------------------------------
import numpy as np
import math
import CoLM_Utils
from CoLM_NetCDFSerial import NetCDFFile
from  CoLM_UserDefFun import findloc_ud


class GridInfoType:
    def __init__(self):
        self.nlat = 0
        self.nlon = 0
        self.lat_s = None
        self.lat_n = None
        self.lon_w = None
        self.lon_e = None
        self.lon_c = None  # grid center
        self.lat_c = None  # grid center

class GridList:
    def __init__(self, ng):
        self.ng = ng
        self.ilat = None
        self.ilon = None
class SegmentType(object):
    def __init__(self):
        self.blk = 0
        self.cnt = 0
        self.bdsp = 0
        self.gdsp = 0


class Grid_type(object):
    def newObject(self, o, len_newobject):
        if o is not None:
            o = np.zeros(len_newobject, dtype='int')
            return o

    def __init__(self, nl_colm, gblock) -> None:
        self.nl_colm = nl_colm
        self.gblock = gblock

        self.lat_s = None
        self.lat_n = None
        self.lon_w = None
        self.lon_e = None

        self.xgrd = None
        self.ygrd = None
        self.nlat = 0
        self.nlon = 0

        self.yinc = 1

        self.xdsp = None
        self.ydsp = None
        self.xcnt = None
        self.ycnt = None
        self.xblk = None
        self.yblk = None
        self.xloc = None
        self.yloc = None

        self.rlon = None
        self.rlat = None

    def reset(self, nlon, nlat) -> None:
        self.xgrd = None
        self.ygrd = None

        self.nlat = nlat
        self.nlon = nlon
        self.lat_s = np.zeros(nlat)
        self.lat_n = np.zeros(nlat)
        self.lon_w = np.zeros(nlon)
        self.lon_e = np.zeros(nlon)

    def define_by_name(self, gridname):
        if gridname == 'merit_90m':
            nlat = 180 * 60 * 20
            nlon = 360 * 60 * 20

            self.nlat = nlat
            self.nlon = nlon

            self.grid_init()

            del_lat = 180.0 / nlat
            for ilat in range(self.nlat):
                self.lat_s[ilat] = 90.0 - del_lat * (ilat + 1) - del_lat / 2.0
                self.lat_n[ilat] = 90.0 - del_lat * ilat - del_lat / 2.0

            del_lon = 360.0 / nlon
            for ilon in range(self.nlon):
                self.lon_w[ilon] = -180.0 + del_lon * ilon - del_lon / 2.0
                self.lon_e[ilon] = -180.0 + del_lon * (ilon + 1) - del_lon / 2.0

            self.normalize()
            self.set_blocks()

        if gridname == 'colm_5km':
            self.define_by_ndims(8640, 4320)

        if gridname == 'colm_1km':
            self.define_by_ndims(43200, 21600)

        if gridname == 'colm_500m':
            self.define_by_ndims(86400, 43200)

        if gridname == 'colm_100m':
            self.define_by_ndims(432000, 216000)

        if gridname == 'nitrif_2deg':
            self.define_by_ndims(144, 96)

    def grid_init(self):
        self.lat_s = np.zeros(self.nlat, dtype='float')
        self.lat_n = np.zeros(self.nlat, dtype='float')
        self.lon_w = np.zeros(self.nlon, dtype='float')
        self.lon_e = np.zeros(self.nlon, dtype='float')

    def define_from_file(self, filename, latname=None, lonname=None):
        netfile = NetCDFFile(self.nl_colm['USEMPI'])

        if not (latname is not None and lonname is not None):
            self.lat_s = netfile.ncio_read_bcast_serial(filename, 'lat_s')
            self.lat_n = netfile.ncio_read_bcast_serial(filename, 'lat_n')
            self.lon_w = netfile.ncio_read_bcast_serial(filename, 'lon_w')
            self.lon_e = netfile.ncio_read_bcast_serial(filename, 'lon_e')
            self.nlat = len(self.lat_s)
            self.nlon = len(self.lon_w)

            self.normalize()
            self.set_blocks()
        else:
            lat_in = netfile.ncio_read_bcast_serial(filename, latname)
            lon_in = netfile.ncio_read_bcast_serial(filename, lonname)
            self.define_by_center(lat_in, lon_in)

            del lat_in
            del lon_in

    def grid_define_by_copy(self, grid_in):
        self.init(grid_in.nlon, grid_in.nlat)

        self.lat_s = grid_in.lat_s
        self.lat_n = grid_in.lat_n
        self.lon_w = grid_in.lon_w
        self.lon_e = grid_in.lon_e

        self.normalize()
        self.set_blocks()

    def normalize(self):
        for ilon in range(self.nlon):
            self.lon_w[ilon] = CoLM_Utils.normalize_longitude(self.lon_w[ilon])
            self.lon_e[ilon] = CoLM_Utils.normalize_longitude(self.lon_e[ilon])

        for ilat in range(self.nlat):
            self.lat_s[ilat] = max(-90.0, min(90.0, self.lat_s[ilat]))
            self.lat_n[ilat] = max(-90.0, min(90.0, self.lat_n[ilat]))

        if self.lat_s[0] <= self.lat_s[self.nlat - 1]:
            self.yinc = 1
        else:
            self.yinc = -1

    def define_by_ndims(self, lon_points, lat_points):
        self.nlat = lat_points
        self.nlon = lon_points

        self.grid_init()

        del_lat = 180.0 / lat_points
        for ilat in range(self.nlat):
            self.lat_s[ilat] = 90.0 - del_lat * (ilat + 1)
            self.lat_n[ilat] = 90.0 - del_lat * ilat

        del_lon = 360.0 / lon_points
        for ilon in range(self.nlon):
            self.lon_w[ilon] = -180.0 + del_lon * ilon
            self.lon_e[ilon] = -180.0 + del_lon * (ilon + 1)

        self.lon_e[self.nlon - 1] = -180.0
        self.normalize()
        self.set_blocks()

    def set_blocks(self):
        self.xcnt = np.zeros(self.gblock.nxblk, dtype='int')
        self.xdsp = np.zeros(self.gblock.nxblk, dtype='int')
        self.ycnt = np.zeros(self.gblock.nyblk, dtype='int')
        self.ydsp = np.zeros(self.gblock.nyblk, dtype='int')

        self.xblk = np.zeros(self.nlon, dtype='int')
        self.yblk = np.zeros(self.nlat, dtype='int')

        self.xloc = np.zeros(self.nlon, dtype='int')
        self.yloc = np.zeros(self.nlat, dtype='int')

        edges = self.nl_colm['DEF_domain%edges']
        edgen = self.nl_colm['DEF_domain%edgen']
        edgew = self.nl_colm['DEF_domain%edgew']
        edgee = self.nl_colm['DEF_domain%edgee']

        edgew = CoLM_Utils.normalize_longitude(edgew)
        edgee = CoLM_Utils.normalize_longitude(edgee)

        jblk = 0
        ilat = 0

        if self.yinc == 1:
            self.ycnt[:] = 0
            self.yblk[:] = 0

            if edges < self.lat_s[0]:
                jblk = CoLM_Utils.find_nearest_south(self.lat_s[0], self.gblock.nyblk, self.gblock.lat_s)
            else:
                jblk = CoLM_Utils.find_nearest_south(edges, self.gblock.nyblk, self.gblock.lat_s)
                ilat = CoLM_Utils.find_nearest_south(edges, self.nlat, self.lat_s)

            self.ydsp[jblk] = ilat - 1

            while ilat <= self.nlat - 1:
                if self.lat_s[ilat] < edgen:
                    if self.lat_s[ilat] < self.gblock.lat_n[jblk]:

                        self.ycnt[jblk] = self.ycnt[jblk] + 1

                        self.yblk[ilat] = jblk
                        self.yloc[ilat] = self.ycnt[jblk] - 1

                        ilat = ilat + 1
                    else:
                        jblk = jblk + 1
                        if jblk <= self.gblock.nyblk:
                            self.ydsp[jblk] = ilat - 1
                        else:
                            break
                else:
                    break
        else:
            self.ycnt[:] = 0
            self.yblk[:] = 0

            if edgen > self.lat_n[0]:
                jblk = CoLM_Utils.find_nearest_north(self.lat_n[0], self.gblock.nyblk, self.gblock.lat_n)
            else:
                jblk = CoLM_Utils.find_nearest_north(edgen, self.gblock.nyblk, self.gblock.lat_n)
                ilat = CoLM_Utils.find_nearest_north(edgen, self.nlat, self.lat_n)

            self.ydsp[jblk] = ilat - 1

            while ilat <= self.nlat - 1:
                if self.lat_n[ilat] > edges:
                    if self.lat_n[ilat] > self.gblock.lat_s[jblk]:

                        self.ycnt[jblk] = self.ycnt[jblk] + 1

                        self.yblk[ilat] = jblk
                        self.yloc[ilat] = self.ycnt[jblk] - 1

                        ilat = ilat + 1
                    else:
                        jblk = jblk - 1
                        if jblk >= 0:
                            self.ydsp[jblk] = ilat - 1
                        else:
                            break
                else:
                    break

        self.xcnt[:] = 0
        self.xblk[:] = 0

        ilon = 0
        iblk = 0
        if (self.lon_w[0] != self.lon_e[self.nlon - 1]) and \
                (CoLM_Utils.lon_between_floor(edgew, self.lon_e[self.nlon - 1], self.lon_w[0])):
            iblk = CoLM_Utils.find_nearest_west(self.lon_w[0], self.gblock.nxblk, self.gblock.lon_w)
        else:
            iblk = CoLM_Utils.find_nearest_west(edgew, self.gblock.nxblk, self.gblock.lon_w)
            ilon = CoLM_Utils.find_nearest_west(edgew, self.nlon, self.lon_w)
        self.xdsp[iblk] = ilon - 1
        self.xcnt[iblk] = 1
        self.xblk[ilon] = iblk
        # self.xloc[ilon] = 0

        ilon_e = ilon
        if ilon_e == 0:
            ilon_e = self.nlon - 1
        ilon = (ilon + 1) % self.nlon
        while True:
            if CoLM_Utils.lon_between_floor(self.lon_w[ilon], edgew, edgee):
                if CoLM_Utils.lon_between_floor(self.lon_w[ilon], self.gblock.lon_w[iblk], self.gblock.lon_e[iblk]):

                    self.xcnt[iblk] = self.xcnt[iblk] + 1

                    self.xblk[ilon] = iblk
                    self.xloc[ilon] = self.xcnt[iblk] - 1

                    if ilon != ilon_e:
                        ilon = (ilon + 1) % self.nlon
                    else:
                        break
                else:
                    iblk = (iblk + 1) % self.gblock.nxblk
                    if self.xcnt[iblk] == 0:
                        self.xdsp[iblk] = ilon - 1
                    else:
                        ilon_e = self.xdsp[iblk] + self.xcnt[iblk]
                        if ilon_e > self.nlon:
                            ilon_e = ilon_e - self.nlon

                        self.xdsp[iblk] = ilon - 1
                        self.xcnt[iblk] = 0
                        while True:
                            self.xcnt[iblk] = self.xcnt[iblk] + 1
                            self.xblk[ilon] = iblk
                            self.xloc[ilon] = self.xcnt[iblk] - 1

                            if ilon != ilon_e:
                                ilon = (ilon+ 1) % self.nlon
                            else:
                                break
                        break
            else:
                break

    def define_by_center(self, lat_in, lon_in, south=None, north=None, west=None, east=None):
        self.nlat = len(lat_in)
        self.nlon = len(lon_in)
        self.lat_s = np.zeros(self.nlat)
        self.lat_n = np.zeros(self.nlat)
        self.lon_w = np.zeros(self.nlon)
        self.lon_e = np.zeros(self.nlon)

        self.reset(self.nlon, self.nlat)

        if lat_in[0] > lat_in[self.nlat - 1]:
            self.yinc = -1
        else:
            self.yinc = 1

        for ilat in range(self.nlat):
            if self.yinc == 1:
                if ilat < self.nlat - 1:
                    self.lat_n[ilat] = (lat_in[ilat] + lat_in[ilat + 1]) * 0.5
                else:
                    if north is not None:
                        self.lat_n[ilat] = north
                    else:
                        self.lat_n[ilat] = 90.0

                if ilat > 0:
                    self.lat_s[ilat] = (lat_in[ilat - 1] + lat_in[ilat]) * 0.5
                else:
                    if south is not None:
                        self.lat_s[ilat] = south
                    else:
                        self.lat_s[ilat] = -90.0
            elif self.yinc == -1:
                if ilat > 0:
                    self.lat_n[ilat] = (lat_in[ilat - 1] + lat_in[ilat]) * 0.5
                else:
                    if north is not None:
                        self.lat_n[ilat] = north
                    else:
                        self.lat_n[ilat] = 90.0

                if ilat < self.nlat - 1:
                    self.lat_s[ilat] = (lat_in[ilat] + lat_in[ilat + 1]) * 0.5
                else:
                    if south is not None:
                        self.lat_s[ilat] = south
                    else:
                        self.lat_s[ilat] = -90.0

        lon_in_n = lon_in
        for ilon in range(len(lon_in_n)):
            lon_in_n[ilon] = CoLM_Utils.normalize_longitude(lon_in_n[ilon])

        for ilon in range(self.nlon):
            ilone = (ilon+1) % self.nlon
            if lon_in_n[ilon] > lon_in_n[ilone]:
                self.lon_e[ilon] = (lon_in_n[ilon] + lon_in_n[ilone] + 360.0) * 0.5
            else:
                self.lon_e[ilon] = (lon_in_n[ilon] + lon_in_n[ilone]) * 0.5

            if ilon == self.nlon - 1 and east is not None:
                self.lon_e[self.nlon - 1] = east

            ilonw = ilon - 1
            if ilonw == -1:
                ilonw = self.nlon - 1
            if lon_in_n[ilonw] > lon_in_n[ilon]:
                self.lon_w[ilon] = (lon_in_n[ilonw] + lon_in_n[ilon] + 360.0) * 0.5
            else:
                self.lon_w[ilon] = (lon_in_n[ilonw] + lon_in_n[ilon]) * 0.5

            if ilon == 0 and west is not None:
                self.lon_w[0] = west

        del lon_in_n

        self.normalize()
        self.set_blocks()

    def define_by_res(self, lon_res, lat_res):
        lon_points = round(360.0 / lon_res)
        lat_points = round(180.0 / lat_res)
        self.define_by_ndims(lon_points, lat_points)

    def grid_free_men(self):
        if self.lat_s is not None:
            del self.lat_s
        if self.lat_n is not None:
            del self.lat_n
        if self.lon_w is not None:
            del self.lon_w
        if self.lon_e is not None:
            del self.lon_e

        if self.xdsp is not None:
            del self.xdsp
        if self.ydsp is not None:
            del self.ydsp

        if self.xcnt is not None:
            del self.xcnt
        if self.ycnt is not None:
            del self.ycnt

        if self.xblk is not None:
            del self.xblk
        if self.yblk is not None:
            del self.yblk

        if self.xloc is not None:
            del self.xloc
        if self.yloc is not None:
            del self.yloc

        # if self.xgrd is not None:
        #     del self.xgrd
        # if self.ygrd is not None:
        #     del self.ygrd

        # if self.rlon is not None:
        #     del self.rlon
        # if self.rlat is not None:
        #     del self.rlat

    def set_rlon(self):
        if self.rlon is None:
            self.rlon = np.zeros(self.nlon)

        lon = 0.0

        for ix in range(self.nlon):
            if self.lon_w[ix] <=self.lon_e[ix]:
                lon = (self.lon_w[ix]  + self.lon_e[ix]) * 0.5
            else:
                lon = (self.lon_e[ix] + self.lon_e[ix]) * 0.5 + 180.0

            lon = CoLM_Utils.normalize_longitude(lon)
            self.rlon[ix] = lon / 180.0 * math.pi

    def set_rlat(self):
        if self.rlat is None:
            self.rlat = np.zeros(self.nlat)

        for iy in range(self.nlat):
            self.rlat[iy] = (self.lat_s[iy] + self.lat_n[iy]) * 0.5 / 180.0 * math.pi
class GridConcatType:
    def __init__(self):
        self.ndatablk = 0
        self.nxseg = 0
        self.nyseg = 0
        self.xsegs = None
        self.ysegs = None
        self.ginfo = GridInfoType()

    def set(self, grid, gblock):
        ilat_l = findloc_ud(grid.yblk !=0)
        ilat_u = findloc_ud(grid.yblk !=0, True)

        self.ginfo.nlat = ilat_u - ilat_l + 1
        self.ginfo.lat_s = np.zeros(self.ginfo.nlat)
        self.ginfo.lat_n = np.zeros(self.ginfo.nlat)
        self.ginfo.lat_c = np.zeros(self.ginfo.nlat)

        self.nyseg = 0
        jblk = 0
        ilatloc = 0
        for ilat in range(ilat_l, ilat_u + 1):
            if grid.yblk[ilat] != jblk:
                self.nyseg += 1
                jblk = grid.yblk[ilat]

            self.ginfo.lat_c[ilatloc] = grid.lat_s[ilat]
            self.ginfo.lat_n[ilatloc] = grid.lat_n[ilat]
            self.ginfo.lat_c[ilatloc] = (grid.lat_s[ilat] + grid.lat_n[ilat]) * 0.5
            ilatloc += 1

        self.ysegs = []
        for i in range(self.nyseg):
            self.ysegs.append(SegmentType())

        iyseg = -1
        jblk = 0
        for ilat in range(ilat_l, ilat_u + 1):
            if grid.yblk[ilat] != jblk:
                # print(self.nyseg,len(self.ysegs),iyseg,ilat_l,ilat_u,'----not++++++++++++')
                iyseg += 1
                jblk = grid.yblk[ilat]
                self.ysegs[iyseg].blk = jblk
                self.ysegs[iyseg].bdsp = grid.yloc[ilat]
                self.ysegs[iyseg].gdsp = ilat - ilat_l
                self.ysegs[iyseg].cnt = 1
            else:
                # print(self.nyseg, ilat_l, ilat_u, '----yes++++++++++++')
                self.ysegs[iyseg].cnt += 1

        if np.all(grid.xblk > 0):
            ilon_w = 0
            ilon_e = grid.nlon - 1
        else:
            ilon_w = findloc_ud(grid.xblk != 0)
            while True:
                ilon = ilon_w - 1
                if ilon == -1:
                    ilon = grid.nlon -1

                if grid.xblk[ilon] != 0:
                    ilon_w = ilon
                else:
                    break

            ilon_e = ilon_w
            while True:
                ilon = ilon_e % grid.nlon + 1

                if grid.xblk[ilon] != 0:
                    ilon_e = ilon
                else:
                    break

        self.ginfo.nlon = ilon_e - ilon_w + 1
        if self.ginfo.nlon <= 0:
            self.ginfo.nlon += grid.nlon

        self.ginfo.lon_w = np.empty(self.ginfo.nlon)
        self.ginfo.lon_e = np.empty(self.ginfo.nlon)
        self.ginfo.lon_c = np.empty(self.ginfo.nlon)

        self.nxseg = 0
        ilon = ilon_w - 1
        iblk = 0
        ilonloc = -1
        while True:
            ilon = (ilon + 1) % grid.nlon
            if grid.xblk[ilon] != iblk:
                self.nxseg += 1
                iblk = grid.xblk[ilon]

            ilonloc += 1

            self.ginfo.lon_w[ilonloc] = grid.lon_w[ilon]
            self.ginfo.lon_e[ilonloc] = grid.lon_e[ilon]

            self.ginfo.lon_c[ilonloc] = (grid.lon_w[ilon] + grid.lon_e[ilon]) * 0.5
            if grid.lon_w[ilon] > grid.lon_e[ilon]:
                self.ginfo.lon_c[ilonloc] += 180.0
                # Assuming there's a function normalize_longitude defined
                self.ginfo.lon_c[ilonloc] = CoLM_Utils.normalize_longitude(
                    self.ginfo.lon_c[ilonloc])

            if ilon == ilon_e:
                break

        for ilon in range(1, self.ginfo.nlon):
            if self.ginfo.lon_c[ilon] < self.ginfo.lon_c[ilon - 1] and self.ginfo.lon_c[ilon] < 0:
                self.ginfo.lon_c[ilon] += 360.0

        if self.xsegs is not None:
            del self.xsegs

        self.xsegs = []
        for i in range(self.nyseg):
            self.xsegs.append(SegmentType())

        ixseg = -1
        iblk = 0
        ilon = ilon_w - 1
        ilonloc = -1
        while True:
            ilon = (ilon+1)%grid.nlon
            ilonloc += 1

            if grid.xblk[ilon] != iblk:
                ixseg += 1
                iblk = grid.xblk[ilon]
                self.xsegs[ixseg].blk = iblk
                self.xsegs[ixseg].bdsp = grid.xloc[ilon]
                self.xsegs[ixseg].gdsp = ilonloc
                self.xsegs[ixseg].cnt = 1
            else:
                self.xsegs[ixseg].cnt += 1

            if ilon == ilon_e:
                break

        self.ndatablk = 0

        for iyseg in range(self.nyseg):
            for ixseg in range(self.nxseg):
                iblk = self.xsegs[ixseg].blk
                jblk = self.ysegs[iyseg].blk
                if gblock.pio[iblk, jblk] >= 0:
                    self.ndatablk += 1

    def release(self, o):
        if o is not None:
            del o

    def grid_concat_free_mem(self):
        self.release(self.xsegs)
        self.release(self.ysegs)
        self.release(self.ginfo.lat_s)
        self.release(self.ginfo.lat_n)

        self.release(self.ginfo.lat_c)
        self.release(self.ginfo.lon_w)
        self.release(self.ginfo.lon_e)
        self.release(self.ginfo.lon_c)
