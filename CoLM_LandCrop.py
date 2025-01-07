import numpy as np
from CoLM_5x5DataReadin import CoLM_5x5DataReadin
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
import CoLM_PixelsetShared


class CoLM_LandCrop(object):
    def __init__(self, mpi, nl_colm, gblock, landpatch, var_global, srfdata, mesh) -> None:
        self.mpi = mpi
        self.co_lm = nl_colm
        self.mesh = mesh
        self.var_global = var_global
        self.mpi = mpi
        self.gblock = gblock
        self.landpatch = landpatch
        self.srfdata = srfdata
        self.readin = CoLM_5x5DataReadin(mpi, gblock)
        self.pctshrpch = None
        self.cropclass = None

    def landcrop_build(self, lc_year, landelm, landhru, gpatch, gcrop):
        cyear = str(lc_year)
        pctshared_xy = None
        classshared = None
        pctshared = None
        cropdata = None

        if self.mpi.p_is_master:
            print("Making patches (crop shared) :")

        if self.co_lm['SinglePoint'] and self.co_lm['CROP']:
            if self.srfdata.SITE_landtype == self.var_global.CROPLAND and self.co_lm['USE_SITE_pctcrop']:
                numpatch = sum(self.srfdata.SITE_pctcrop > 0.)
                self.pctshrpch = self.srfdata.SITE_pctcrop[self.srfdata.SITE_pctcrop > 0.] / sum(
                    self.srfdata.SITE_pctcrop > 0.)
                self.cropclass = self.srfdata.SITE_croptyp[self.srfdata.SITE_pctcrop > 0.]

                self.landpatch.eindex = np.ones(numpatch)
                self.landpatch.ipxstt = np.ones(numpatch)
                self.landpatch.ipxend = np.ones(numpatch)
                self.landpatch.settyp = np.ones(numpatch) * self.var_global.CROPLAND
                self.landpatch.ielm = np.ones(numpatch)

                self.landpatch.nset = numpatch
                self.landpatch.set_vecgs()
                return

        if self.co_lm['USEMPI']:
            pass
            # CALL mpi_barrier (p_comm_glb, p_err)

        if self.mpi.p_is_io:
            dir_5x5 = self.co_lm['DEF_dir_rawdata'] + '/plant_15s'
            suffix = 'MOD' + cyear
            dt = DataType(self.gblock)
            pctcrop_xy = dt.allocate_block_data(gpatch)
            pctcrop_xy = self.readin.read_5x5_data(dir_5x5, suffix, gpatch, 'PCT_CROP', pctcrop_xy)
            dtt = DataType(self.gblock)
            pctshared_xy = dtt.allocate_block_data2d(gpatch, 2)
            for iblkme in range(self.gblock.nblkme):
                ib = self.gblock.xblkme[iblkme]
                jb = self.gblock.yblkme[iblkme]
                pctshared_xy.blk[ib - 1, jb - 1].val[0][:][:] = 1. - pctcrop_xy.blk[ib - 1][jb - 1].val / 100.
                pctshared_xy.blk[ib - 1, jb - 1].val[1][:][:] = pctcrop_xy.blk[ib - 1][jb - 1].val / 100.

        sharedfilter = [1]

        pctshared = CoLM_PixelsetShared.pixelsetshared_build(self.co_lm, self.mpi, self.mesh, self.gblock,
                                                             self.landpatch, gpatch, pctshared_xy, 2, sharedfilter,
                                                             classshared)

        if self.mpi.p_is_worker:
            if self.landpatch.nset > 0:
                self.landpatch.settyp[classshared == 2] = self.var_global.CROPLAND

        if self.mpi.p_is_io:
            file_patch = self.co_lm['DEF_dir_rawdata'] + '/global_CFT_surface_data.nc'
            dttt = DataType(self.gblock)
            cropdata = dttt.allocate_block_data2d(gcrop, self.var_global.N_CFT)
            CoLM_NetCDFBlock.ncio_read_block(file_patch, 'PCT_CFT', self.mpi, gcrop, self.var_global.N_CFT, cropdata)

        cropfilter = [self.var_global.CROPLAND]

        self.pctshrpch = CoLM_PixelsetShared.pixelsetshared_build(self.co_lm, self.mpi, self.mesh, self.gblock,
                                                                  self.landpatch, gcrop, cropdata,
                                                                  self.var_global.N_CFT, cropfilter,
                                                                  self.cropclass, fracin=pctshared)

        numpatch = self.landpatch.nset

        if pctshared is not None:
            del pctshared
        if classshared is not None:
            del classshared

        if self.co_lm['USEMPI']:
            pass
            # if self.mpi.p_is_worker:
            #     npatch_glb = mpi_reduce(numpatch, MPI_INTEGER, MPI_SUM, p_root, p_comm_worker, p_err)
            #     if p_iam_worker == 0:
            #         print('Total: ', npatch_glb, ' patches (with crop).')

            # mpi_barrier(p_comm_glb, p_err)
        else:
            print('Total: ', numpatch, ' patches.')

        self.landpatch.elm_patch.build(landelm, self.landpatch, use_frac=True, sharedfrac=self.pctshrpch)

        if self.co_lm['CATCHMENT']:
            self.landpatch.hru_patch.build(landhru, self.landpatch, use_frac=True, sharedfrac=self.pctshrpch)

        self.landpatch.write_patchfrac(self.co_lm['DEF_dir_landdata'], lc_year)
