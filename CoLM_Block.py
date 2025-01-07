# -------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    To deal with high-resolution data, the globe is divided into blocks.
#
#     (180W,90N)                           (180E,90N)
#        .-----------------------------------.
#        |         |         |        |      |
#        |         |         |        |      |
#        |         |         |        |      |
#        .-----------------------------------.
#        |         |         |        |      |
#        |         |         |        |      |
#        |         |         |        |      |
#        .-----------------------------------.
#        |         |         |        |      |
#        |         |         |        |      |
#        |         |         |        |      |
#        .-----------------------------------.
#     (180W,90S)                           (180E,90S)
#   
#    1.
#    Boundaries for block (i,j) is saved in 
#    "gblock%lat_s(j), gblock%lat_n(j), gblock%lon_w(i), gblock%lon_e(i)" 
#    for south, north, west and east boundaries respectively.
#
#    2.
#    The (i,j) element of 2D array gblock%pio saves the global communication
#    number of process which is in charge of Input/Output of block (i,j).
#
#    3.
#    For Input/Output processes, "gblock%nblkme, gblock%xblkme(:), gblock%yblkme(:)"
#    save the locations of blocks which are handled by themselves.
#
#    4.
#    Division of blocks can be generated by number of blocks globally (by set_by_size), 
#    or set by predefined boundaries in files (by set_by_file).
# 
# -------------------------------------------------------------------------------------
import CoLM_Utils
from CoLM_NetCDFSerial import NetCDFFile
import numpy as np
import os
import math
import sys


class Block_type(object):
    def __init__(self, nl_colm, mpi, site_lon_location, site_lat_location) -> None:
        self.MPI = mpi
        self.nl_colm = nl_colm

        self.site_lon_location = site_lon_location
        self.site_lat_location = site_lat_location
        self.xblkme = None
        self.yblkme = None
        self.pio = None
        self.nblkme = 0

        if os.path.exists(nl_colm['DEF_BlockInfoFile']):
            self.netfile = NetCDFFile(nl_colm['USEMPI'])
            self.lat_s = self.netfile.ncio_read_bcast_serial(nl_colm['DEF_BlockInfoFile'], 'lat_s')
            self.lat_n = self.netfile.ncio_read_bcast_serial(nl_colm['DEF_BlockInfoFile'], 'lat_n')
            self.lon_w = self.netfile.ncio_read_bcast_serial(nl_colm['DEF_BlockInfoFile'], 'lon_w')
            self.lon_e = self.netfile.ncio_read_bcast_serial(nl_colm['DEF_BlockInfoFile'], 'lon_e')

            self.nyblk = len(self.lat_s)
            self.nxblk = len(self.lon_w)

            # blocks should be from south to north
            if self.lat_s[0] > self.lat_s[self.nyblk - 1]:
                self.lat_s = self.lat_s[self.nyblk - 1:0:-1]
                self.lat_n = self.lat_n[self.nyblk - 1:0:-1]
        else:
            if nl_colm['DEF_AverageElementSize'] > 0:
                self.nxblk = math.floor(360. / (nl_colm['DEF_AverageElementSize'] / 120. * 50))
                self.nxblk = min(self.nxblk, 360)
                while self.nxblk < 360 and 360 % self.nxblk != 0:
                    self.nxblk = self.nxblk + 1

                self.nyblk = math.floor(180. / (nl_colm['DEF_AverageElementSize'] / 120. * 50))
                self.nyblk = min(self.nxblk, 180)
                while (self.nyblk < 180) and 180 % self.nyblk != 0:
                    self.nyblk = self.nyblk + 1
            else:
                self.nxblk = nl_colm['DEF_nx_blocks']
                self.nyblk = nl_colm['DEF_ny_blocks']

            if 360 % nl_colm['DEF_nx_blocks'] != 0 or 180 % nl_colm['DEF_ny_blocks'] != 0:
                if mpi.p_is_master:
                    print('Number of blocks in longitude should be a factor of 360 ')
                    print(' and Number of blocks in latitude should be a factor of 180.')
                    mpi.CoLM_stop()

            self.lon_w = np.zeros(self.nxblk)
            self.lon_e = np.zeros(self.nxblk)

            for iblk in range(self.nxblk):
                self.lon_w[iblk] = -180.0 + 360.0 / self.nxblk * iblk
                self.lon_e[iblk] = -180.0 + 360.0 / self.nxblk * (iblk + 1)

                self.lon_w[iblk] = CoLM_Utils.normalize_longitude(self.lon_w[iblk])
                self.lon_e[iblk] = CoLM_Utils.normalize_longitude(self.lon_e[iblk])

            self.lat_s = np.zeros(self.nyblk)
            self.lat_n = np.zeros(self.nyblk)

            for jblk in range(self.nyblk):
                self.lat_s[jblk] = -90.0 + 180.0 / self.nyblk * jblk
                self.lat_n[jblk] = -90.0 + 180.0 / self.nyblk * jblk + 1

            if self.MPI.p_is_master:
                print('Block information:')
                print(str(self.nxblk) + ' blocks in longitude,' + str(self.nyblk) + ' blocks in latitude.')
            self.nl_colm = nl_colm
            self.init_pio()

    def clip(self, numblocks):
        edges = self.nl_colm['DEF_domain%edges']
        edgen = self.nl_colm['DEF_domain%edgen']
        edgew = self.nl_colm['DEF_domain%edgew']
        edgee = self.nl_colm['DEF_domain%edgee']
        # print(edges,edgen,edgew,edgee,'*********')

        iblk_south = CoLM_Utils.find_nearest_south(edges, self.nyblk, self.lat_s)
        iblk_north = CoLM_Utils.find_nearest_north(edgen, self.nyblk, self.lat_n)
        # print(iblk_south,iblk_north,'iblk_south_north----')

        edgew = CoLM_Utils.normalize_longitude(edgew)
        edgee = CoLM_Utils.normalize_longitude(edgee)
        # print(edgew,edgee,'edge-----')

        if edgew == edgee:
            iblk_west = 0
            iblk_east = self.nxblk - 1
        else:
            iblk_west = CoLM_Utils.find_nearest_west(edgew, self.nxblk, self.lon_w)
            iblk_east = CoLM_Utils.find_nearest_east(edgee, self.nxblk, self.lon_e)

            if iblk_west == iblk_east:
                if CoLM_Utils.lon_between_floor(edgee, self.lon_w[iblk_west], edgew):
                    iblk_west = 0
                    iblk_east = self.nxblk - 1
        # print(iblk_west,iblk_east,'iblk_west_east----2')

        if numblocks is not None:
            numblocks_y = iblk_north - iblk_south + 1

            if iblk_east >= iblk_west:
                numblocks_x = iblk_east - iblk_west + 1
            else:
                numblocks_x = self.nxblk - iblk_west + 1 + iblk_east

            numblocks = numblocks_x * numblocks_y
        return iblk_south, iblk_north, iblk_west, iblk_east, numblocks

    def init_pio(self):
        iblk_south = 0
        iblk_north = 0
        iblk_west = 0
        iblk_east = 0
        numblocks = 0
        if self.MPI.p_is_master:
            iblk_south, iblk_north, iblk_west, iblk_east, numblocks = self.clip(numblocks)

        if self.nl_colm['USEMPI']:
            pass
            # NetCDFFile.mpi_bcast (numblocks, 1, MPI_INTEGER, p_root, p_comm_glb, p_err)
            # self.MPI.divide_processes_into_groups(numblocks, self.DEF_PIO_groupsize)

        self.pio = np.full((self.nxblk, self.nyblk),-1)

        if self.MPI.p_is_master:
            iproc = -1
            for jblk in range(iblk_south, iblk_north + 1):
                iblk = iblk_west
                # print(iblk_south, iblk_north,iblk_west,iblk,jblk,'-------proot-------')

                while True:
                    if self.nl_colm['USEMPI']:
                        iproc = (iproc + 1) % self.MPI.p_np_io
                        self.pio[iblk, jblk] = self.MPI.p_address_io[iproc]
                    else:
                        self.pio[iblk, jblk] = self.MPI.p_root

                    if iblk != iblk_east:
                        iblk = iblk % self.nxblk + 1
                    else:
                        break
        if self.nl_colm['USEMPI']:
            # NetCDFFile.mpi_bcast (self.pio, self.nxblk * self.nyblk, MPI_INTEGER, p_root, p_comm_glb, p_err)
            pass

        if not self.nl_colm['SinglePoint']:
            self.nblkme = 0
            if self.MPI.p_is_io:
                self.nblkme = len(np.where(self.pio == self.MPI.p_iam_glb)[0])
                if self.nblkme > 0:
                    iblkme = 0
                    self.xblkme = np.zeros(self.nblkme, dtype='int')
                    self.yblkme = np.zeros(self.nblkme, dtype='int')
                    for iblk in range(self.nxblk):
                        for jblk in range(self.nyblk):
                            if self.MPI.p_iam_glb == self.pio[iblk, jblk]:
                                self.xblkme[iblkme] = iblk
                                self.yblkme[iblkme] = jblk
                                iblkme = iblkme + 1
        else:
            self.nblkme = 1
            self.xblkme = np.zeros(1, dtype='int')
            self.yblkme = np.zeros(1, dtype='int')
            self.site_lon_location = CoLM_Utils.normalize_longitude(self.site_lon_location)
            self.xblkme[0] = CoLM_Utils.find_nearest_west(self.site_lon_location, self.nxblk, self.lon_w)
            self.yblkme[0] = CoLM_Utils.find_nearest_south(self.site_lat_location, self.nyblk, self.lat_s)

    def save_to_file(self, dir_landdata):
        if self.MPI.p_is_master:
            filename = os.path.join(dir_landdata, 'block.nc')

            plat_system = sys.platform
            if 'win' in plat_system:
                filename = dir_landdata + '\\' + 'block.nc'

            netfile = NetCDFFile(self.MPI)

            netfile.ncio_create_file(filename)

            netfile.ncio_define_dimension(filename, 'longitude', self.nxblk)
            netfile.ncio_define_dimension(filename, 'latitude', self.nyblk)

            netfile.ncio_write_serial4(filename,'lat_s', self.lat_s, 'latitude')
            netfile.ncio_write_serial4(filename,'lat_n', self.lat_n, 'latitude')
            netfile.ncio_write_serial4(filename,'lon_w', self.lon_w, 'longitude')
            netfile.ncio_write_serial4(filename,'lon_e', self.lon_e, 'longitude')

    def get_blockname(self, iblk, jblk):
        if self.lat_s[jblk] < 0:
            cy = 's' + str(-int(self.lat_s[jblk]))
        else:
            cy = 'n' + str(int(self.lat_s[jblk]))

        if self.lon_w[iblk] < 0:
            cx = 'w' + str(-int(self.lon_w[iblk]))
        else:
            cx = 'e' + str(int(self.lon_w[iblk]))

        blockname = cx + '_' + cy

        return blockname

    def get_filename_block(self, filename, iblk, jblk):
        blockname = self.get_blockname(iblk, jblk)
        i = len(filename.strip())
        while i > 0:
            if filename[i - 2: i - 1] == '.':
                break
            i -= 1

        if i > 0:
            fileblock = filename[:i - 2] + '_' + blockname + '.nc'
        else:
            fileblock = filename + '_' + blockname + '.nc'

        return fileblock

    def block_free_mem(self):
        if self.lat_s is not None:
            del self.lat_s
        if self.lat_n is not None:
            del self.lat_n
        if self.lon_w is not None:
            del self.lon_w
        if self.lon_e is not None:
            del self.lon_e
        if self.pio is not None:
            del self.pio
        if self.xblkme is not None:
            del self.xblkme
        if self.yblkme is not None:
            del self.yblkme

    def load_from_file(self, dir_landdata, is_master=True):
        filename = f"{dir_landdata}/block.nc"
        netfile = NetCDFFile(self.MPI)

        self.lat_s = netfile.ncio_read_bcast_serial(filename, 'lat_s')
        self.lat_n = netfile.ncio_read_bcast_serial(filename, 'lat_n')
        self.lon_w = netfile.ncio_read_bcast_serial(filename, 'lon_w')
        self.lon_e = netfile.ncio_read_bcast_serial(filename, 'lon_e')

        self.nyblk = len(self.lat_s)
        self.nxblk = len(self.lon_w)

        if self.MPI.p_is_master:
            print('Block information:')
            print(f'{self.nxblk} blocks in longitude, {self.nyblk} blocks in latitude.\n')

        self.block_read_pio(dir_landdata)

    def block_read_pio(self, dir_landdata):
        iblk_south = 0
        iblk_north = 0
        iblk_west = 0
        iblk_east = 0
        if self.MPI.p_is_master:
            filename = f"{dir_landdata}/mesh/{self.nl_colm['DEF_LC_YEAR']}/mesh.nc"
            netfile = NetCDFFile(self.MPI)
            nelmblk = netfile.ncio_read_serial(filename, 'nelm_blk')
            numblocks = sum(nelmblk[nelmblk > 0])
            iblk_south, iblk_north, iblk_west, iblk_east,numblocks = self.clip(numblocks)

        # ifdef USEMPI
        #     mpi_bcast(numblocks, 1, MPI_INTEGER, p_root, p_comm_glb, p_err)
        #     divide_processes_into_groups(numblocks, DEF_PIO_groupsize)

        if self.MPI.p_is_master:
            self.pio = np.full((self.nxblk, self.nyblk), -1, dtype=int)

            # ifdef USEMPI
            #     allocate(nelm_io(0: p_np_io - 1))
            #     nelm_io(:) = 0
            #     jproc = -1

            for jblk in range(iblk_south, iblk_north + 1):
                iblk = iblk_west
                while True:
                    if self.nl_colm['USEMPI']:
                        pass
                    else:
                        self.pio[iblk,jblk] = self.MPI.p_root

                    if iblk != iblk_east:
                        iblk = (iblk % self.nxblk) + 1
                    else:
                        break

            if self.nl_colm['USEMPI']:
                pass
        if self.nl_colm['USEMPI']:
            pass

        if not self.nl_colm['SinglePoint']:
            self.nblkme = 0
            if self.MPI.p_is_io:
                self.nblkme = len(np.where(self.pio == self.MPI.p_iam_glb)[0])
                if self.nblkme > 0:
                    iblkme = 0
                    self.xblkme = np.zeros(self.nblkme, dtype=int)
                    self.yblkme = np.zeros(self.nblkme, dtype=int)
                    for iblk in range(self.nxblk):
                        for jblk in range(self.nyblk):
                            if self.MPI.p_iam_glb == self.pio[iblk, jblk]:
                                self.xblkme[iblkme] = iblk
                                self.yblkme[iblkme] = jblk
                                iblkme += 1
        else:
            self.nblkme = 1
            self.xblkme = np.zeros(1, dtype=int)
            self.yblkme = np.zeros(1, dtype=int)

            for iblk in range(self.nxblk):
                for jblk in range(self.nyblk):
                    if nelmblk[iblk, jblk] > 0:
                        self.xblkme[0] = iblk
                        self.yblkme[0] = jblk

        if nelmblk is not None:
            del nelmblk