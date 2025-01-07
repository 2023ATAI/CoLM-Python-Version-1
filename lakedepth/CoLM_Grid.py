 #-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
import numpy as np
import math
import CoLM_Utils
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_Pixel import Pixel_type

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

class Grid_type(object):
    def newObject(o,len_newobject):
       if o is not None:
          del o 
          new_o = np.zeros(len_newobject)
          
          return new_o 
       
    def __init__(self, nl_colm, gblock, mpi) -> None:
        self.nl_colm = nl_colm
        self.gblock = gblock
        self.xgrd =[]
        self.ygrd = []
        self.pixel = Pixel_type(mpi)
        self.pixel.set_edges(nl_colm['DEF_domain'].edges, nl_colm['DEF_domain'].edgen, nl_colm['DEF_domain'].edgew, nl_colm['DEF_domain'].edgee)

    def define_by_name(self, gridname):
        if gridname == 'merit_90m':
            nlat = 180*60*20
            nlon = 360*60*20

            self.nlat = nlat
            self.nlon = nlon

            self.grid_init (self.nlon, self.nlat)

            del_lat = 180.0 / nlat
            for ilat in range(self.nlat):
                self.lat_s[ilat] = 90.0 - del_lat * (ilat+1) - del_lat/2.0
                self.lat_n[ilat] = 90.0 - del_lat * ilat - del_lat/2.0
            
            del_lon = 360.0 / nlon
            for ilon in range(self.nlon):
                self.lon_w[ilon] = -180.0 + del_lon * ilon - del_lon/2.0
                self.lon_e[ilon] = -180.0 + del_lon * (ilon+1) - del_lon/2.0

            self.normalize  ()
            self.set_blocks ()

        if gridname == 'colm_5km':
            self.define_by_ndims (8640,4320)

        if gridname == 'colm_1km':
            self.define_by_ndims (43200,21600)

        if gridname == 'colm_500m':
            self.define_by_ndims (86400,43200)

        if gridname == 'colm_100m':
            self.define_by_ndims (432000,216000)

        if gridname == 'nitrif_2deg':
            self.define_by_ndims (144,96)
    
    def grid_init(self, nlon, nlat):
        self.nlat = nlat
        self.nlon = nlon
        self.lat_s = self.newObject(nlat)
        self.lat_n = self.newObject(nlat)
        self.lon_w = self.newObject(nlon)
        self.lon_e = self.newObject(nlon)

    def define_from_file (self, filename, latname='', lonname=''):
        self.netfile = NetCDFFile(self.nl_colm['USEMPI'], filename)

        if not len(latname) == 0 and len(lonname) == 0:
            self.netfile.ncio_read_bcast_serial()
            self.lat_s = self.netfile.ncio_read_bcast_serial ( 'lat_s' )
            self.lat_n = self.netfile.ncio_read_bcast_serial ( 'lat_n' )
            self.lon_w = self.netfile.ncio_read_bcast_serial ( 'lon_w' )
            self.lon_e = self.netfile.ncio_read_bcast_serial ( 'lon_e' )
            self.nlat = len(self.lat_s)
            self.nlon = len(self.lon_w)

            self.normalize  ()
            self.set_blocks ()
        else:
            lat_in = self.netfile.ncio_read_bcast_serial (latname)
            lon_in = self.netfile.ncio_read_bcast_serial (lonname)
            self.define_by_center (lat_in, lon_in)
         
        del lat_in
        del lon_in

    def normalize (self):
        for ilon in range( self.nlon):
            CoLM_Utils.normalize_longitude (self.lon_w[ilon])
            CoLM_Utils.normalize_longitude (self.lon_e[ilon])

        for ilat in range(self.nlat):
            self.lat_s[ilat] = max(-90.0, min(90.0, self.lat_s[ilat]))
            self.lat_n[ilat] = max(-90.0, min(90.0, self.lat_n[ilat]))

        if self.lat_s[0] <= self.lat_s[self.nlat-1]:
            self.yinc = 1
        else:
            self.yinc = -1

    def define_by_ndims (self, lon_points, lat_points):
        self.nlat = lat_points
        self.nlon = lon_points

        self.grid_init (self.nlon, self.nlat)

        del_lat = 180.0 / lat_points
        for ilat in range( self.nlat):
            self.lat_s[ilat] = 90.0 - del_lat * (ilat+1)
            self.lat_n[ilat] = 90.0 - del_lat * ilat

        del_lon = 360.0 / lon_points
        for ilon in range( self.nlon):
            self.lon_w[ilon] = -180.0 + del_lon * ilon
            self.lon_e[ilon] = -180.0 + del_lon * (ilon+1)

        self.lon_e[self.nlon-1] = -180.0
        self.normalize  ()
        self.set_blocks ()

    def set_blocks (self):
        self.xcnt = self.newObject(self.gblock.nxblk)
        self.xdsp = self.newObject(self.gblock.nxblk)
        self.ycnt = self.newObject(self.gblock.nyblk)
        self.ydsp = self.newObject(self.gblock.nyblk)

        self.xblk = self.newObject(self.nlon)
        self.yblk = self.newObject(self.nlat)

        self.xloc = self.newObject(self.nlon)
        self.yloc = self.newObject(self.nlat)

        edges = self.nl_colm['DEF_domain'].edges
        edgen = self.nl_colm['DEF_domain'].edgen
        edgew = self.nl_colm['DEF_domain'].edgew
        edgee = self.nl_colm['DEF_domain'].edgee

        edgew = CoLM_Utils.normalize_longitude (edgew)
        edgee = CoLM_Utils.normalize_longitude (edgee)

        jblk = 0
        ilat = 1

        if self.yinc == 1:

            self.ycnt[:] = 0
            self.yblk[:] = 0

            if edges < self.lat_s[0]:
                jblk = CoLM_Utils.find_nearest_south (self.lat_s[0], self.gblock.nyblk, self.gblock.lat_s)
                # ilat = 1
            else:
                jblk = CoLM_Utils.find_nearest_south (edges, self.gblock.nyblk, self.gblock.lat_s)
                ilat = CoLM_Utils.find_nearest_south (edges, self.nlat, self.lat_s)

            self.ydsp[jblk-1] = ilat - 1

            while ilat <= self.nlat:
                if self.lat_s[ilat-1] < edgen:
                    if self.lat_s[ilat-1] < self.gblock.lat_n[jblk-1]:

                        self.ycnt[jblk-1] = self.ycnt[jblk-1] + 1

                        self.yblk[ilat-1] = jblk
                        self.yloc[ilat-1] = self.ycnt[jblk-1]

                        ilat = ilat + 1
                    else:
                        jblk = jblk + 1
                        if jblk <= self.gblock.nyblk:
                            self.ydsp[jblk-1] = ilat - 1
                        else:
                            break
                else:
                    break
        else:
            self.ycnt[:] = 0
            self.yblk[:] = 0

            if edgen > self.lat_n[0]:
                jblk = CoLM_Utils.find_nearest_north (self.lat_n[0], self.gblock.nyblk, self.gblock.lat_n)
                # ilat = 1
            else:
                jblk = CoLM_Utils.find_nearest_north (edgen, self.gblock.nyblk, self.gblock.lat_n)
                ilat = CoLM_Utils.find_nearest_north (edgen, self.nlat, self.lat_n)

            self.ydsp[jblk-1] = ilat - 1

            while ilat <= self.nlat:
                if self.lat_n[ilat-1] > edges:
                    if self.lat_n[ilat-1] > self.gblock.lat_s[jblk-1]:

                        self.ycnt[jblk-1] = self.ycnt[jblk-1] + 1

                        self.yblk[ilat-1] = jblk
                        self.yloc[ilat-1] = self.ycnt[jblk-1]

                        ilat = ilat + 1
                    else:
                        jblk = jblk - 1
                        if jblk >= 1:
                            self.ydsp[jblk-1] = ilat - 1
                        else:
                            break
                else:
                    break

        self.xcnt[:] = 0
        self.xblk[:] = 0

        if (self.lon_w[0] != self.lon_e[self.nlon-1]) and \
            (CoLM_Utils.lon_between_floor(edgew, self.lon_e[self.nlon-1], self.lon_w[0])):
            iblk = CoLM_Utils.find_nearest_west (self.lon_w[0], self.gblock.nxblk, self.gblock.lon_w)
            ilon = 1
        else:
            iblk = CoLM_Utils.find_nearest_west (edgew, self.gblock.nxblk, self.gblock.lon_w)
            ilon = CoLM_Utils.find_nearest_west (edgew, self.nlon, self.lon_w)

        self.xdsp[iblk-1] = ilon - 1
        self.xcnt[iblk-1] = 1
        self.xblk[ilon-1] = iblk
        self.xloc[ilon-1] = 1

        ilon_e = ilon - 1
        if ilon_e == 0:
            ilon_e = self.nlon
        ilon = ilon % self.nlon +1
        while True:
            if CoLM_Utils.lon_between_floor(self.lon_w[ilon-1], edgew, edgee):
                if CoLM_Utils.lon_between_floor(self.lon_w[ilon-1], self.gblock.lon_w[iblk-1], self.gblock.lon_e[iblk-1]):

                    self.xcnt[iblk-1] = self.xcnt[iblk-1] + 1

                    self.xblk[ilon-1] = iblk
                    self.xloc[ilon-1] = self.xcnt[iblk-1]

                    if ilon != ilon_e:
                        ilon = ilon % self.nlon + 1
                    else:
                        break
                else:
                    iblk = iblk % self.gblock.nxblk + 1
                    if self.xcnt[iblk-1] == 0:
                        self.xdsp[iblk-1] = ilon - 1
                    else:
                        ilon_e = self.xdsp[iblk-1] + self.xcnt[iblk-1]
                        if ilon_e > self.nlon:
                            ilon_e = ilon_e - self.nlon

                        self.xdsp[iblk-1] = ilon - 1
                        self.xcnt[iblk-1] = 0
                        while True:
                            self.xcnt[iblk-1] = self.xcnt[iblk-1] + 1
                            self.xblk[ilon-1] = iblk
                            self.xloc[ilon-1] = self.xcnt[iblk-1]

                            if ilon != ilon_e:
                                ilon = ilon % self.nlon + 1
                            else:
                                break
                        break
            else:
                break

    def define_by_center (self, lat_in, lon_in, south, north, west, east):
        self.nlat = len(lat_in)
        self.nlon = len(lon_in)

        self.init (self.nlon, self.nlat)

        if lat_in[0] > lat_in[self.nlat-1]:
            self.yinc = -1
        else:
            self.yinc = 1

        for ilat in range( self.nlat):
            if self.yinc == 1:
                if ilat < self.nlat-1:
                    self.lat_n[ilat] = (lat_in[ilat] + lat_in[ilat+1]) * 0.5
                else:
                    if north is not None:
                        self.lat_n[ilat] = north
                    else:
                        self.lat_n[ilat] = 90.0

                if ilat > 0:
                    self.lat_s[ilat] = (lat_in[ilat-1] + lat_in[ilat]) * 0.5
                else:
                    if south is not None:
                        self.lat_s[ilat] = south
                    else:
                        self.lat_s[ilat] = -90.0
            elif self.yinc == -1:
                if ilat > 0:
                    self.lat_n[ilat] = (lat_in[ilat-1] + lat_in[ilat]) * 0.5
                else:
                    if north is not None:
                        self.lat_n[ilat] = north
                    else:
                        self.lat_n[ilat] = 90.0

                if ilat < self.nlat-1:
                    self.lat_s[ilat] = (lat_in[ilat] + lat_in[ilat+1]) * 0.5
                else:
                    if south is not None:
                        self.lat_s[ilat] = south
                    else:
                        self.lat_s[ilat] = -90.0

        lon_in_n = np.zeros(len(lon_in))

        lon_in_n = lon_in
        for ilon in range(len(lon_in_n)):
            CoLM_Utils.normalize_longitude (lon_in_n[ilon])

        for ilon in range(self.nlon):
            ilone = (ilon+1) % self.nlon + 1
            if lon_in_n[ilon] > lon_in_n[ilone-1]:
                self.lon_e[ilon] = (lon_in_n[ilon] + lon_in_n[ilone-1] + 360.0) * 0.5
            else:
                self.lon_e[ilon] = (lon_in_n[ilon] + lon_in_n[ilone-1]) * 0.5

            if ilon == self.nlon-1 and east is not None:
                self.lon_e[self.nlon-1] = east

            ilonw = ilon - 1
            if ilonw == -1:
                ilonw = self.nlon-1
            if lon_in_n[ilonw] > lon_in_n[ilon]:
                self.lon_w[ilon] = (lon_in_n[ilonw] + lon_in_n[ilon] + 360.0) * 0.5
            else:
                self.lon_w[ilon] = (lon_in_n[ilonw] + lon_in_n[ilon]) * 0.5

            if ilon == 0 and west is not None:
                self.lon_w[0] = west

        del lon_in_n

        self.normalize  ()
        self.set_blocks ()

    def define_by_res(self, lon_res, lat_res):
        lon_points = round(360.0 / lon_res)
        lat_points = round(180.0 / lat_res)
        self.define_by_ndims(lon_points, lat_points)

    def assimilate_grid(self):
        self.pixel.assimilate_grid(self.nlat, self.lat_s, self.lat_n, self.nlon, self.lon_w, self.lon_e)

    def map_to_grid(self):
        self.xgrd, self.ygrd = self.pixel.map_to_grid(self.yinc, self.lat_s, self.lat_n, self.nlat, self.lon_w, self.lon_e, self.nlon)


class GridConcatType:
    def __init__(self):
        self.ndatablk = 0
        self.nxseg = 0
        self.nyseg = 0
        self.xsegs = None
        self.ysegs = None
        self.ginfo = GridInfoType()

    def set(self, grid, gblock):
        ilat_l = np.argmax(grid.yblk != 0)
        ilat_u = len(grid.yblk) - np.argmax(grid.yblk[::-1] != 0)
        
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
            ilatloc += 1
            self.ginfo.lat_c[ilatloc - 1] = grid.lat_s[ilat]
            self.ginfo.lat_n[ilatloc - 1] = grid.lat_n[ilat]
            self.ginfo.lat_c[ilatloc - 1] = (grid.lat_s[ilat] + grid.lat_n[ilat]) * 0.5

        self.ysegs = np.zeros(self.nyseg)

        iyseg = 0
        jblk = 0
        for ilat in range(ilat_l, ilat_u + 1):
            if grid.yblk[ilat] != jblk:
                iyseg += 1
                jblk = grid.yblk[ilat]
                self.ysegs[iyseg].blk  = jblk
                self.ysegs[iyseg].bdsp = grid.yloc[ilat] - 1
                self.ysegs[iyseg].gdsp = ilat - ilat_l
                self.ysegs[iyseg].cnt  = 1
            else:
                self.ysegs[iyseg].cnt += 1

        if np.all(grid.xblk > 0):
            ilon_w = 1
            ilon_e = grid.nlon
        else:
            ilon_w = np.argmax(grid.xblk != 0) + 1##########
            while True:
                ilon = ilon_w - 1
                if ilon == 0:
                    ilon = grid.nlon

                if grid.xblk[ilon - 1] != 0:
                    ilon_w = ilon
                else:
                    break

            ilon_e = ilon_w
            while True:
                ilon = ilon_e % grid.nlon + 1

                if grid.xblk[ilon - 1] != 0:
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
        ilonloc = 0
        while True:
            ilon = ilon % grid.nlon + 1
            if grid.xblk[ilon - 1] != iblk:
                self.nxseg += 1
                iblk = grid.xblk[ilon - 1]

            ilonloc += 1
            self.ginfo.lon_w[ilonloc - 1] = grid.lon_w[ilon - 1]
            self.ginfo.lon_e[ilonloc - 1] = grid.lon_e[ilon - 1]

            self.ginfo.lon_c[ilonloc - 1] = (grid.lon_w[ilon - 1] + grid.lon_e[ilon - 1]) * 0.5
            if grid.lon_w[ilon - 1] > grid.lon_e[ilon - 1]:
                self.ginfo.lon_c[ilonloc - 1] += 180.0
                # Assuming there's a function normalize_longitude defined
                CoLM_Utils.normalize_longitude(self.ginfo.lon_c[ilonloc - 1])###############

            if ilon == ilon_e:
                break

        for ilon in range(2, self.ginfo.nlon+1):
            if self.ginfo.lon_c[ilon-1] < self.ginfo.lon_c[ilon - 2] and self.ginfo.lon_c[ilon-1] < 0:
                self.ginfo.lon_c[ilon-1] += 360.0

        if hasattr(self, 'xsegs'):
            del self.xsegs
        self.xsegs = np.empty(self.nxseg, dtype=object)

        ixseg = 0
        iblk = 0
        ilon = ilon_w - 1
        ilonloc = 0
        while True:
            ilon = (ilon . grid.nlon) + 1
            ilonloc += 1
            if grid.xblk[ilon - 1] != iblk:
                ixseg += 1
                iblk = grid.xblk[ilon - 1]
                self.xsegs[ixseg].blk  = iblk
                self.xsegs[ixseg].bdsp = grid.xloc[ilon] - 1
                self.xsegs[ixseg].gdsp = ilonloc - 1
                self.xsegs[ixseg].cnt = 1
            else:
                self.xsegs[ixseg - 1].cnt += 1

            if ilon == ilon_e:
                break

        self.ndatablk = 0

        for iyseg in range(1, self.nyseg + 1):
            for ixseg in range(1, self.nxseg + 1):
                iblk = self.xsegs[ixseg - 1]['blk']
                jblk = self.ysegs[iyseg - 1]['blk']
                if gblock.pio[iblk - 1, jblk - 1] >= 0:
                    self.ndatablk += 1

    def release(self,o):
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
        