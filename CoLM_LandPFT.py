# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Build pixelset "landpft" (Plant Function Type).
#
#    In CoLM, the global/regional area is divided into a hierarchical structure:
#    1. If GRIDBASED or UNSTRUCTURED is defined, it is
#       ELEMENT >>> PATCH
#    2. If CATCHMENT is defined, it is
#       ELEMENT >>> HRU >>> PATCH
#    If Plant Function Type classification is used, PATCH is further divided into PFT.
#    If Plant Community classification is used,     PATCH is further divided into PC.
#
#    "landpft" refers to pixelset PFT.
#
# Created by Shupeng Zhang, May 2023
#    porting codes from Hua Yuan's OpenMP version to MPI parallel version.
# ------------------------------------------------------------------------------------
import numpy as np
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
from CoLM_DataType import DataType
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_5x5DataReadin import CoLM_5x5DataReadin


class CoLM_LandPFT(object):
    def __init__(self, nl_colm, mpi, gblock, mesh, const_lc, var_global) -> None:
        self.nl_colm = nl_colm
        self.const_lc = const_lc
        self.mpi = mpi
        self.gblock = gblock
        self.var_global = var_global
        self.mesh = mesh
        self.vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
        self.readin = CoLM_5x5DataReadin(mpi, gblock)
        self.numpft = 0
        self.patch_pft_s = None
        self.patch_pft_e = None
        self.pft2patch = None

    def landpft_build(self, lc_year, SITE_pctpfts, SITE_pfttyp, landpatch, numpatch, cropclass, landpft, gpatch):
        # Local Variables
        dir_5x5 = self.nl_colm['DEF_dir_rawdata'] + '/plant_15s'
        cyear = str(lc_year)
        suffix = 'MOD' + cyear
        pctpft_patch = None
        pctpft_one = None
        area_one = None
        patchmask = None
        
        pft2patch = None
        npft_glb = None
        pctpft =None

        if self.mpi.p_is_master:
            print('Making land plant function type tiles:')

        if self.nl_colm['SinglePoint']:
            if self.nl_colm['USE_SITE_pctpfts']:
                if self.nl_colm['CROP']:
                    if self.const_lc.patchtypes[landpatch.settyp[0]] == 0:
                        self.numpft = len(np.where(SITE_pctpfts > 0.)[0])
                    elif landpatch.settyp(0) == self.var_global.CROPLAND:
                        self.numpft = numpatch
                    else:
                        self.numpft = 0
                else:
                    if self.const_lc.patchtypes[landpatch.settyp[0]] == 0 and landpatch.settyp[0] != self.var_global.CROPLAND:
                        self.numpft = len(np.where(SITE_pctpfts > 0.)[0])
                    else:
                        self.numpft = 0

                self.patch_pft_s = np.empty(numpatch)
                self.patch_pft_e = np.empty(numpatch)

                if self.numpft > 0:
                    landpft.eindex = np.ones(self.numpft)
                    landpft.settyp = np.zeros(self.numpft)
                    landpft.ipxstt = np.ones(self.numpft)
                    landpft.ipxend = np.ones(self.numpft)
                    landpft.ielm = np.ones(self.numpft)

                    pft2patch = np.empty(self.numpft)

                    if self.nl_colm['CROP']:
                        if self.const_lc.patchtypes[landpatch.settyp[0]] == 0:
                            landpft.settyp = SITE_pctpfts[SITE_pctpfts> 0]
                            pft2patch = np.ones(self.numpft)
                            self.patch_pft_s[:] = 1
                            self.patch_pft_e[:] = self.numpft
                        elif landpatch.settyp[0] == self.var_global.CROPLAND:
                            for ipft in range(self.numpft):
                                landpft.settyp[ipft] = cropclass[ipft] + self.var_global.N_PFT - 1
                                pft2patch[ipft] = ipft
                                self.patch_pft_s[ipft] = ipft
                                self.patch_pft_e[ipft] = ipft
                    else:
                        if self.const_lc.patchtypes[landpatch.settyp[0]] == 0 and landpatch.settyp[
                            0] != self.var_global.CROPLAND:
                            landpft.settyp = SITE_pfttyp[SITE_pctpfts > 0]
                            pft2patch = np.ones(self.numpft)
                            self.patch_pft_s[:] = 1
                            self.patch_pft_e[:] = self.numpft
                else:
                    print('Warning: land type', landpatch.settyp[0], 'for LULC_IGBP_PFT|LULC_IGBP_PC')
                    self.patch_pft_s[:] = -1
                    self.patch_pft_e[:] = -1

                landpft_nset = self.numpft
                # Assuming landpft_set_vecgs is a method to set the vector global sum
                landpft.set_vecgs()
                return

        # MPI Barrier
        if self.nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb, p_err)

        # IO section
        if self.mpi.p_is_io:
            dt = DataType(self.gblock)
            pctpft = dt.allocate_block_data2d(gpatch, self.readin.N_PFT_modis, lb1=0)
            pctpft = dt.flush_block_data(1.0)
            dir_5x5 = self.nl_colm['DEF_dir_rawdata'] + '/plant_15s'
            cyear = lc_year
            suffix = 'MOD' + cyear
            self.readin.read_5x5_data_pft(dir_5x5, suffix, gpatch, 'PCT_PFT', pctpft)
            if self.nl_colm['USEMPI']:
                pass
                # aggregation_data_daemon(gpatch, data_r8_3d_in1=pctpft, n1_r8_3d_in1=N_PFT_modis)

        # Worker section
        if self.mpi.p_is_worker:
            if numpatch > 0:
                pctpft_patch = np.zeros((self.var_global.N_PFT, numpatch))
                patchmask = np.empty(numpatch, dtype=bool)
                patchmask[:] = True

            for ipatch in range(numpatch):
                if self.nl_colm['CROP']:
                    if self.const_lc.patchtypes[landpatch.settyp[ipatch]] == 0:
                        ard = AggregationRequestData(self.nl_colm['USEMPI'], self.mpi, self.mesh.mesh,
                                                     self.gblock.pixel)
                        pctpft_one = ard.aggregation_request_data(landpatch, ipatch, gpatch, zip=False, area=area_one,
                                                                  data_r8_3d_in1=pctpft,
                                                                  n1_r8_3d_in1=self.readin.N_PFT_modis, lb1_r8_3d_in1=0)
                        sum_temp = np.sum(pctpft_one[0:self.var_global.N_PFT, :], axis=1)
                        sumarea = np.sum(area_one * sum_temp)

                        if sumarea <= 0.0:
                            patchmask[ipatch] = False
                        else:
                            for ipft in range(self.var_global.N_PFT):
                                pctpft_patch[ipft, ipatch] = np.sum(pctpft_one[ipft, :] * area_one) / sumarea
                else:
                    if self.const_lc.patchtypes[landpatch.settyp[ipatch]] == 0 \
                            and landpatch.settyp[ipatch - 1] != self.var_global.CROPLAND:
                        ard = AggregationRequestData(self.nl_colm['USEMPI'], self.mpi, self.mesh.mesh,
                                                     self.gblock.pixel)
                        pctpft_one = ard.aggregation_request_data(landpatch, ipatch, gpatch, zip=False, area=area_one,
                                                                  data_r8_3d_in1=pctpft,
                                                                  n1_r8_3d_in1=self.readin.N_PFT_modis,
                                                                  lb1_r8_3d_in1=0)
                        sumarea = np.sum(area_one * np.sum(pctpft_one[0:self.var_global.N_PFT, :], axis=1))

                        if sumarea <= 0.0:
                            patchmask[ipatch] = False
                        else:
                            for ipft in range(self.var_global.N_PFT):
                                pctpft_patch[ipft, ipatch] = np.sum(pctpft_one[ipft, :] * area_one) / sumarea

            if self.nl_colm['USEMPI']:
                pass
                # aggregation_worker_done()

            if numpatch > 0:
                npatch = len(patchmask)
                self.numpft = len(np.where(pctpft_patch > 0.)[0])
                if self.nl_colm['CROP']:
                    self.numpft = self.numpft + len(np.where(landpatch.settyp == self.var_global.CROPLAND)[0])
                if npatch > 0:
                    self.patch_pft_s = np.zeros(npatch)
                    self.patch_pft_e = np.zeros(npatch)
            else:
                self.numpft = 0

            if self.numpft > 0:
                pft2patch = np.zeros(self.numpft)

                landpft.eindex = np.zeros(self.numpft)
                landpft.settyp = np.zeros(self.numpft)
                landpft.ipxstt = np.zeros(self.numpft)
                landpft.ipxend = np.zeros(self.numpft)
                landpft.ielm = np.ones(self.numpft)

                npft = 0
                npatch = 0
                for ipatch in range(numpatch):
                    if patchmask[ipatch]:
                        npatch += 1
                        if self.nl_colm['CROP']:
                            if self.const_lc.patchtypes[landpatch.settyp[ipatch]] == 0:
                                self.patch_pft_s[npatch] = npft + 1
                                self.patch_pft_e[npatch] = npft + len(np.where(pctpft_patch[:, ipatch] > 0)[0])
                                for ipft in range(self.var_global.N_PFT):
                                    if pctpft_patch[ipft, ipatch] > 0:
                                        npft += 1
                                        landpft.ielm[npft] = landpatch.ielm[ipatch]
                                        landpft.eindex[npft] = landpatch.eindex[ipatch]
                                        landpft.ipxstt[npft] = landpatch.ipxstt[ipatch]
                                        landpft.ipxend[npft] = landpatch.ipxend[ipatch]
                                        landpft.settyp[npft] = ipft
                                        pft2patch[npft] = npatch
                            elif landpatch.settyp[ipatch] == self.var_global.CROPLAND:
                                npft += 1
                                self.patch_pft_s[npatch] = npft
                                self.patch_pft_e[npatch] = npft
                                landpft.ielm[npft] = landpatch.ielm[ipatch]
                                landpft.eindex[npft] = landpatch.eindex[ipatch]
                                landpft.ipxstt[npft] = landpatch.ipxstt[ipatch]
                                landpft.ipxend[npft] = landpatch.ipxend[ipatch]
                                landpft.settyp[npft] = cropclass[ipatch] + self.var_global.N_PFT
                                pft2patch[npft] = npatch
                            else:
                                self.patch_pft_s[npatch] = -1
                                self.patch_pft_e[npatch] = -1
                        else:
                            if (self.const_lc.patchtypes[landpatch.settyp[ipatch]] == 0 and landpatch.settyp[
                                ipatch] != self.var_global.CROPLAND):
                                self.patch_pft_s[npatch] = npft + 1
                                self.patch_pft_e[npatch] = npft + np.count_nonzero(pctpft_patch[:, ipatch] > 0)
                                for ipft in range(self.var_global.N_PFT):
                                    if pctpft_patch[ipft, ipatch] > 0:
                                        npft += 1
                                        landpft.ielm[npft] = landpatch.ielm[ipatch]
                                        landpft.eindex[npft] = landpatch.eindex[ipatch]
                                        landpft.ipxstt[npft] = landpatch.ipxstt[ipatch]
                                        landpft.ipxend[npft] = landpatch.ipxend[ipatch]
                                        landpft.settyp[npft] = ipft
                                        pft2patch[npft] = npatch
                            else:
                                self.patch_pft_s[npatch] = -1
                                self.patch_pft_e[npatch] = -1
        landpatch.pset_pack(patchmask, numpatch)

        landpft.nset = self.numpft
        landpft.set_vecgs()

        if self.nl_colm['USEMPI']:
            if self.mpi.p_is_worker:
                # npft_glb = mpi_reduce(self.numpft, MPI_SUM)
                if self.mpi.p_iam_worker == 0:
                    print('Total:', npft_glb, 'plant function type tiles.')
            # mpi_barrier(p_comm_glb, p_err)
        else:
            print('Total:', self.numpft, 'plant function type tiles.')

        if self.nl_colm['SinglePoint']:
            SITE_pfttyp[:] = landpft.settyp

        # Deallocate memory
        if pctpft_patch is not None:
            del pctpft_patch
        if pctpft_one is not None:
            del pctpft_one
        if area_one is not None:
            del area_one
        if patchmask is not None:
            del patchmask

    def map_patch_to_pft(self, numpatch, numpft, landpatch, landpft):
        if not self.mpi.p_is_worker:
            if numpatch <= 0 or numpft <= 0:
                return

            self.patch_pft_s = np.empty(numpatch, dtype=int)
            self.patch_pft_e = np.empty(numpatch, dtype=int)
            self.pft2patch = np.empty(numpft, dtype=int)

            ipft = 0
            for ipatch in range(numpatch):
                settyp = landpatch['settyp'][ipatch]
                if self.nl_colm['CROP']:
                    if self.const_lc.patchtypes[settyp] == 0:
                        self.patch_pft_s[ipatch] = ipft
                        while ipft <= numpft:
                            if (landpft.eindex[ipft] == landpatch.eindex[ipatch] and
                                    landpft.ipxstt[ipft] == landpatch.ipxstt[ipatch] and
                                    landpft.settyp[ipft] < self.var_global.N_PFT):
                                self.pft2patch[ipft] = ipatch
                                self.patch_pft_e[ipatch] = ipft
                                ipft += 1
                            else:
                                break
                    elif settyp == self.var_global.CROPLAND:
                        self.patch_pft_s[ipatch] = ipft
                        self.patch_pft_e[ipatch] = ipft
                        self.pft2patch[ipft] = ipatch
                        ipft += 1
                    else:
                        self.patch_pft_s[ipatch] = -1
                        self.patch_pft_e[ipatch] = -1
                else:
                    if self.const_lc.patchtypes[settyp] == 0 and settyp != self.var_global.CROPLAND:
                        self.patch_pft_s[ipatch] = ipft
                        while ipft <= numpft:
                            if (landpft.eindex[ipft] == landpatch.eindex[ipatch] and
                                    landpft.ipxstt[ipft] == landpatch.ipxstt[ipatch] and
                                    landpft.settyp[ipft] < self.var_global.N_PFT):
                                self.pft2patch[ipft] = ipatch
                                self.patch_pft_e[ipatch] = ipft
                                ipft += 1
                            else:
                                break
                    else:
                        self.patch_pft_s[ipatch] = -1
                        self.patch_pft_e[ipatch] = -1

        # return patch_pft_s, patch_pft_e, pft2patch
