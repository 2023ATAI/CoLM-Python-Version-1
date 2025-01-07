# -----------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    This module includes subroutines for checking the results of making surface data.
#
#    The surface data in vector form is mapped to gridded data with last
#    three dimensions of [type,longitude,latitude], which can be viewed by other softwares.
#
#    In GRIDBASED, the grid of gridded data is just the grid of the mesh.
#    In UNSTRUCTURED or CATCHMENT, the grid is user defined and the mapping uses area
#    weighted scheme.
#
# Created by Shupeng Zhang, May 2023
#
# Revisions:
# TODO
# -----------------------------------------------------------------------------------------
import os
import sys
import numpy as np
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_DataType import DataType
from CoLM_Grid import Grid_type, GridConcatType
from CoLM_Mapping_Pset2Grid import MappingPset2Grid


class CoLm_SrfdataDiag(object):
    def __init__(self, nl_colm, mpi, gblock) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.gblock = gblock
        self.netfile = NetCDFFile(mpi)
        self.dt = DataType(gblock)
        self.gdiag = Grid_type(nl_colm, gblock)
        self.srf_concat = GridConcatType()
        self.srf_concat.set(self.gdiag, gblock)
        self.m_elm2diag = MappingPset2Grid(mpi, nl_colm, gblock)
        self.m_patch2diag = MappingPset2Grid(mpi, nl_colm, gblock)
        self.m_pft2diag = MappingPset2Grid(mpi, nl_colm, gblock)
        self.m_urb2diag = MappingPset2Grid(mpi, nl_colm, gblock)

    def srfdata_diag_init(self, dir_landdata, landpatch, landelm, elm_patch, landpft, landurban, hru_patch, pctshrpch,
                          N_land_classification, pixel, mesh):
        landdir = os.path.join(dir_landdata, 'diag')
        elmid_r8 = None

        if 'win' in sys.platform:
            landdir = dir_landdata + '\\' + 'diag'

        if self.mpi.p_is_master:
            os.makedirs(landdir.strip())

        self.srf_concat.set(self.gdiag, self.gblock)

        self.m_elm2diag.build(landelm, self.gdiag, pixel, mesh)

        if not self.nl_colm['CROP']:
            self.m_patch2diag.build(landpatch, self.gdiag, pixel, mesh)
        else:
            self.m_patch2diag.build(landpatch, self.gdiag, pctshrpch, pixel, mesh)

        if self.nl_colm['LULC_IGBP_PFT'] or self.nl_colm['LULC_IGBP_PC']:
            self.m_pft2diag.build(landpft, self.gdiag, pixel, mesh)

        if self.nl_colm['URBAN_MODEL']:
            self.m_urb2diag.build(landurban, self.gdiag, pixel, mesh)

        srf_data_id = 666

        if self.mpi.p_is_worker:
            elmid_r8 = np.array(landelm.eindex, dtype=np.float64)

        landname = dir_landdata.strip() + '/diag/element.nc'
        self.srfdata_map_and_write(elmid_r8, landelm.settyp, [0], self.m_elm2diag,
                                   -1.0e36, landname, 'element', compress=1, write_mode='one')

        if self.mpi.p_is_worker:
            del elmid_r8

        typindex = [ityp for ityp in range(N_land_classification + 1)]

        landname = dir_landdata.strip() + '/diag/patchfrac_elm.nc'
        self.srfdata_map_and_write(elm_patch.subfrc, landpatch.settyp, typindex, self.m_patch2diag,
                                   -1.0e36, landname, 'patchfrac_elm', compress=1, write_mode='one')

        if self.nl_colm['CATCHMENT']:
            landname = dir_landdata.strip() + '/diag/patchfrac_hru.nc'
            self.srfdata_map_and_write(hru_patch.subfrc, landpatch.settyp, typindex, self.m_patch2diag,
                                       -1.0e36, landname, 'patchfrac_hru', compress=1, write_mode='one')

    def srfdata_map_and_write(self, vsrfdata, settyp, typindex, m_srf, spv, filename, dataname, compress,
                              write_mode=None, lastdimname=None, lastdimvalue=None):
        wmode = 'one'
        if write_mode is not None:
            wmode = write_mode.strip()

        ntyps = len(typindex)

        sumwt = None
        wdata = None

        if self.mpi.p_is_io:
            sumwt = self.dt.allocate_block_data2d(self.gdiag, ntyps)
            wdata = self.dt.allocate_block_data2d(self.gdiag, ntyps)

        if self.mpi.p_is_worker:
            if len(vsrfdata) > 0:
                vecone = np.ones_like(vsrfdata)

        m_srf.map_split(vecone, settyp, typindex, sumwt, spv)
        m_srf.map_split(vsrfdata, settyp, typindex, wdata, spv)

        if self.mpi.p_is_io:
            for iblkme in range(self.gblock.nblkme):
                ib = self.gblock.xblkme[iblkme]
                jb = self.gblock.yblkme[iblkme]

                mask = (sumwt.blk[ib, jb].val > 0.) and (wdata.blk[ib, jb].val != spv)
                wdata.blk[ib, jb].val[mask] /= sumwt.blk[ib, jb].val[mask]
                wdata.blk[ib, jb].val[~mask] = spv

        if wmode == 'one':
            if self.mpi.p_is_master:
                vdata = np.full((ntyps, self.srf_concat.ginfo.nlon, self.srf_concat.ginfo.nlat), spv)

                if self.nl_colm['USEMPI']:
                    pass
                else:
                    for iyseg in range(self.srf_concat.nyseg):
                        for ixseg in range(self.srf_concat.nxseg):
                            iblk = self.srf_concat.xsegs[ixseg].blk
                            jblk = self.srf_concat.ysegs[iyseg].blk
                            xbdsp = self.srf_concat.xsegs[ixseg].bdsp
                            ybdsp = self.srf_concat.ysegs[iyseg].bdsp
                            xgdsp = self.srf_concat.xsegs[ixseg].gdsp
                            ygdsp = self.srf_concat.ysegs[iyseg].gdsp
                            xcnt = self.srf_concat.xsegs[ixseg].cnt
                            ycnt = self.srf_concat.ysegs[iyseg].cnt

                            vdata[:, xgdsp + 1:xgdsp + xcnt, ygdsp + 1:ygdsp + ycnt] = \
                                wdata.blk[iblk, jblk].val[:, xbdsp + 1:xbdsp + xcnt, ybdsp + 1:ybdsp + ycnt]

                print(f'Please check gridded data < {dataname.strip()} > in {filename.strip()}')

                if not os.path.exists(filename.strip()):
                    self.netfilencio_create_file(filename.strip())

                    self.netfile.ncio_define_dimension(filename.strip(), 'TypeIndex', ntyps)
                    self.netfile.ncio_define_dimension(filename.strip(), 'lon', self.srf_concat.ginfo.nlon)
                    self.netfile.ncio_define_dimension(filename.strip(), 'lat', self.srf_concat.ginfo.nlat)

                    self.netfile.ncio_write_serial(filename.strip(), 'lat', self.srf_concat.ginfo.lat_c, 'lat')
                    self.netfile.ncio_put_attr(filename.strip(), 'lat', 'long_name', 'latitude')
                    self.netfile.ncio_put_attr(filename.strip(), 'lat', 'units', 'degrees_north')

                    self.netfile.ncio_write_serial(filename.strip(), 'lon', self.srf_concat.ginfo.lon_c, 'lon')
                    self.netfile.ncio_put_attr(filename.strip(), 'lon', 'long_name', 'longitude')
                    self.netfile.ncio_put_attr(filename.strip(), 'lon', 'units', 'degrees_east')

                    self.netfile.ncio_write_serial(filename.strip(), 'lat_s', self.srf_concat.ginfo.lat_s, 'lat')
                    self.netfile.ncio_put_attr(filename.strip(), 'lat_s', 'long_name', 'southern latitude boundary')
                    self.netfile.ncio_put_attr(filename.strip(), 'lat_s', 'units', 'degrees_north')

                    self.netfile.ncio_write_serial(filename.strip(), 'lat_n', self.srf_concat.ginfo.lat_n, 'lat')
                    self.netfile.ncio_put_attr(filename.strip(), 'lat_n', 'long_name', 'northern latitude boundary')
                    self.netfile.ncio_put_attr(filename.strip(), 'lat_n', 'units', 'degrees_north')

                    self.netfile.ncio_write_serial(filename.strip(), 'lon_w', self.srf_concat.ginfo.lon_w, 'lon')
                    self.netfile.ncio_put_attr(filename.strip(), 'lon_w', 'long_name', 'western longitude boundary')
                    self.netfile.ncio_put_attr(filename.strip(), 'lon_w', 'units', 'degrees_east')

                    self.netfile.ncio_write_serial(filename.strip(), 'lon_e', self.srf_concat.ginfo.lon_e, 'lon')
                    self.netfile.ncio_put_attr(filename.strip(), 'lon_e', 'long_name', 'eastern longitude boundary')
                    self.netfile.ncio_put_attr(filename.strip(), 'lon_e', 'units', 'degrees_east')

                    self.netfilencio_write_serial(filename.strip(), 'TypeIndex', typindex, 'TypeIndex')

                if lastdimname is not None and lastdimvalue is not None:
                    ilastdim = self.netfilencio_write_lastdim(filename.strip(), lastdimname, lastdimvalue)
                    self.netfilencio_write_serial_time(filename.strip(), dataname.strip(), ilastdim, vdata, 'TypeIndex',
                                                       'lon', 'lat', lastdimname.strip(), compress)
                else:
                    self.netfilencio_write_serial(filename.strip(), dataname.strip(), vdata, 'TypeIndex', 'lon', 'lat',
                                                  compress)

                self.netfilencio_put_attr(filename.strip(), dataname.strip(), 'missing_value', spv)

                del vdata

            if self.nl_colm['USEMPI']:
                pass

            srf_data_id += 1

        elif wmode == 'block':

            if self.mpi.p_is_io:
                for iblkme in range(self.gblock.nblkme):
                    iblk = self.gblock.xblkme[iblkme]
                    jblk = self.gblock.yblkme[iblkme]

                    if self.gdiag.xcnt[iblk - 1] == 0 or self.gdiag.ycnt[jblk - 1] == 0:
                        continue

                    fileblock = self.gblock.get_filename_block(filename, iblk, jblk)

                    if not os.path.exists(fileblock):
                        self.netfilencio_create_file(fileblock)
                        self.netfilencio_define_dimension(fileblock, 'TypeIndex', ntyps)
                        self.srf_write_grid_info(fileblock, self.gdiag, iblk, jblk)

                    if lastdimname is not None and lastdimvalue is not None:
                        ilastdim = self.netfilencio_write_lastdim(fileblock, lastdimname, lastdimvalue)
                        self.netfilencio_write_serial_time(fileblock, dataname, ilastdim, wdata.blk[iblk, jblk].val,
                                                           'TypeIndex', 'lon', 'lat', lastdimname, compress)
                    else:
                        self.netfilencio_write_serial(fileblock, dataname, wdata.blk[iblk, jblk].val, 'TypeIndex',
                                                      'lon', 'lat', compress)

                    self.netfilencio_put_attr(fileblock, dataname, 'missing_value', spv)

        if vecone is not None:
            del vecone

    def srf_write_grid_info(self, fileblock, grid, iblk, jblk):
        yl = grid.ydsp[jblk] + 1
        yu = grid.ydsp[jblk] + grid.ycnt[jblk]
        lat_s = grid.lat_s[yl:yu]
        lat_n = grid.lat_n[yl:yu]

        if grid.xdsp[iblk] + grid.xcnt[iblk] > grid.nlon:
            xl = grid.xdsp[iblk] + 1
            xu = grid.nlon
            nx = grid.nlon - grid.xdsp[iblk]
            lon_w = grid.lon_w[xl:xu]
            lon_e = grid.lon_e[xl:xu]

            xl = 0
            xu = grid.xcnt[iblk] - nx

            lon_w[nx:grid.xcnt[iblk]] = grid.lon_w[xl:xu - 1]
            lon_e[nx:grid.xcnt[iblk]] = grid.lon_e[xl:xu - 1]
            lon_w.extend(grid.lon_w[xl:xu - 1])
            lon_e.extend(grid.lon_e[xl:xu - 1])
        else:
            xl = grid.xdsp[iblk]
            xu = grid.xdsp[iblk] + grid.xcnt[iblk]
            lon_w = grid.lon_w[xl:xu - 1]
            lon_e = grid.lon_e[xl:xu - 1]

        self.netFile.ncio_define_dimension(fileblock, 'lat', grid.ycnt[jblk])
        self.netFile.ncio_define_dimension(fileblock, 'lon', grid.xcnt[iblk])
        self.netFile.ncio_write_serial(fileblock, 'lat_s', lat_s, 'lat')
        self.netFile.ncio_write_serial(fileblock, 'lat_n', lat_n, 'lat')
        self.netFile.ncio_write_serial(fileblock, 'lon_w', lon_w, 'lon')
        self.netFile.ncio_write_serial(fileblock, 'lon_e', lon_e, 'lon')
