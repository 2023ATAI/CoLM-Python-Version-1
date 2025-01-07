#------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
import numpy as np
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector

class MOD_LandPFT(object):
    def __init__(self, nl_colm, mpi, gblock, mesh, const_lc) -> None:
        self.nl_colm = nl_colm
        self.const_lc = const_lc
        self.mpi = mpi
        self.gblock = gblock
        self.mesh = mesh
        self.vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)

    def landpft_build(self, lc_year, SITE_pctpfts, var_global, landpatch, numpatch):
        # Local Variables
        dir_5x5 = self.nl_colm['DEF_dir_rawdata'] + '/plant_15s'
        cyear = str(lc_year)
        suffix = 'MOD' + cyear
        pctpft_patch = None
        pctpft_one = None
        area_one = None
        patchmask = None
        patch_pft_s = None
        patch_pft_e = None
        pft2patch = None
        npft_glb = None
        
        if self.mpi.p_is_master:
            print('Making land plant function type tiles:')
        
        #ifdef SinglePoint
        if self.nl_colm['USE_SITE_pctpfts']:
            if self.nl_colm['CROP']:
                if self.const_lc.patchtypes[landpatch.settyp[1]] == 0:
                    numpft = len(np.where(SITE_pctpfts > 0.)[0])
                elif landpatch.settyp(1) == var_global.CROPLAND:
                    numpft = numpatch
                else:
                    numpft = 0
            else:
                if self.const_lc.patchtypes[landpatch.settyp[1]] == 0 and landpatch.settyp[1]!=var_global.CROPLAND:
                    numpft = len(np.where(SITE_pctpfts > 0.)[0])
                elif landpatch.settyp(1) == var_global.CROPLAND:
                    numpft = numpatch
                else:
                    numpft = 0

       
            patch_pft_s = np.empty(numpatch)
            patch_pft_e = np.empty(numpatch)

            if numpft > 0:
                landpft_eindex = np.empty(numpft)
                landpft_settyp = np.empty(numpft)
                landpft_ipxstt = np.empty(numpft)
                landpft_ipxend = np.empty(numpft)
                landpft_ielm = np.ones(numpft)

                pft2patch = np.empty(numpft)

                if self.const_lc.patchtypes[landpatch.settyp[0] - 1] == 0:
                    landpft_settyp = np.pack(SITE_pfttyp, SITE_pctpfts > 0)
                    pft2patch = np.ones(numpft)
                    patch_pft_s[:] = 1
                    patch_pft_e[:] = numpft
                elif self.const_lc.patchtypes[landpatch.settyp[0] - 1] == 0 and landpatch.settyp[0] != var_global.CROPLAND:
                    landpft_settyp = np.empty(numpft)
                    pft2patch = np.empty(numpft)
                    patch_pft_s = np.empty(numpft)
                    patch_pft_e = np.empty(numpft)
                    for ipft in range(1, numpft + 1):
                        landpft_settyp[ipft - 1] = cropclass[ipft - 1] + var_global.N_PFT - 1
                        pft2patch[ipft - 1] = ipft
                        patch_pft_s[ipft - 1] = ipft
                        patch_pft_e[ipft - 1] = ipft
                else:
                    print('Warning: land type', landpatch.settyp[0], 'for LULC_IGBP_PFT|LULC_IGBP_PC')
                    patch_pft_s[:] = -1
                    patch_pft_e[:] = -1

            landpft_nset = numpft
            # Assuming landpft_set_vecgs is a method to set the vector global sum
            landpft_set_vecgs()

        # MPI Barrier
        if  self.nl_colm['USEMPI']:
            mpi_barrier(p_comm_glb, p_err)

        # IO section
        if self.mpi.p_is_io:
            gpatch = allocate_block_data(gpatch, pctpft, N_PFT_modis, lb1=0)
            flush_block_data(pctpft, 1.0)
            dir_5x5 = DEF_dir_rawdata.strip() + '/plant_15s'
            cyear = str(lc_year).zfill(4)
            suffix = 'MOD' + cyear
            read_5x5_data_pft(dir_5x5, suffix, gpatch, 'PCT_PFT', pctpft)
            if  self.nl_colm['USEMPI']:
                aggregation_data_daemon(gpatch, data_r8_3d_in1=pctpft, n1_r8_3d_in1=N_PFT_modis)

        # Worker section
        if self.mpi.p_is_worker:
            if numpatch > 0:
                pctpft_patch = np.zeros((N_PFT, numpatch))
                patchmask = np.ones(numpatch, dtype=bool)

            for ipatch in range(1, numpatch + 1):
                if self.const_lc.patchtypes[landpatch.settyp[ipatch - 1] - 1] == 0 or \
                (self.const_lc.patchtypes[landpatch.settyp[ipatch - 1] - 1] == 0 and landpatch.settyp[ipatch - 1] != var_global.CROPLAND):
                    aggregation_request_data(landpatch, ipatch, gpatch, zip=False, area=area_one, 
                                            data_r8_3d_in1=pctpft, data_r8_3d_out1=pctpft_one, n1_r8_3d_in1=N_PFT_modis,
                                            lb1_r8_3d_in1=0)
                    sumarea = np.sum(area_one * np.sum(pctpft_one[0:N_PFT, :], axis=1))

                    if sumarea <= 0.0:
                        patchmask[ipatch - 1] = False
                    else:
                        for ipft in range(0, N_PFT):
                            pctpft_patch[ipft, ipatch - 1] = np.sum(pctpft_one[ipft, :] * area_one) / sumarea

            if  self.nl_colm['USEMPI']:
                aggregation_worker_done()

            if numpatch > 0:
                npatch = np.count_nonzero(patchmask)
                numpft = np.count_nonzero(pctpft_patch > 0.)
                if npatch > 0:
                    patch_pft_s = np.zeros(npatch)
                    patch_pft_e = np.zeros(npatch)

            if numpft > 0:
                pft2patch = np.zeros(numpft)

                landpft_eindex = np.zeros(numpft)
                landpft_settyp = np.zeros(numpft)
                landpft_ipxstt = np.zeros(numpft)
                landpft_ipxend = np.zeros(numpft)
                landpft_ielm = np.ones(numpft)

                npft = 0
                npatch = 0
                for ipatch in range(1, numpatch + 1):
                    if patchmask[ipatch - 1]:
                        npatch += 1
                        if self.const_lc.patchtypes[landpatch.settyp[ipatch - 1] - 1] == 0 or \
                        (self.const_lc.patchtypes[landpatch.settyp[ipatch - 1] - 1] == 0 and landpatch.settyp[ipatch - 1] != var_global.CROPLAND):
                            patch_pft_s[npatch - 1] = npft + 1
                            patch_pft_e[npatch - 1] = npft + np.count_nonzero(pctpft_patch[:, ipatch - 1] > 0)
                            for ipft in range(0, N_PFT):
                                if pctpft_patch[ipft, ipatch - 1] > 0:
                                    npft += 1
                                    landpft_ielm[npft - 1] = landpatch.ielm[ipatch - 1]
                                    landpft_eindex[npft - 1] = landpatch.eindex[ipatch - 1]
                                    landpft_ipxstt[npft - 1] = landpatch.ipxstt[ipatch - 1]
                                    landpft_ipxend[npft - 1] = landpatch.ipxend[ipatch - 1]
                                    landpft_settyp[npft - 1] = ipft
                                    pft2patch[npft - 1] = npatch
                        elif landpatch.settyp[ipatch - 1] == var_global.CROPLAND:
                            npft += 1
                            patch_pft_s[npatch - 1] = npft
                            patch_pft_e[npatch - 1] = npft
                            landpft_ielm[npft - 1] = landpatch.ielm[ipatch - 1]
                            landpft_eindex[npft - 1] = landpatch.eindex[ipatch - 1]
                            landpft_ipxstt[npft - 1] = landpatch.ipxstt[ipatch - 1]
                            landpft_ipxend[npft - 1] = landpatch.ipxend[ipatch - 1]
                            landpft_settyp[npft - 1] = cropclass[ipatch - 1] + N_PFT - 1
                            pft2patch[npft - 1] = npatch
                        else:
                            patch_pft_s[npatch - 1] = -1
                            patch_pft_e[npatch - 1] = -1

        if  self.nl_colm['USEMPI']:
            if self.mpi.p_is_worker:
                npft_glb = mpi_reduce(numpft, MPI_SUM)
                if p_iam_worker == 0:
                    print('Total:', npft_glb, 'plant function type tiles.')
            mpi_barrier(p_comm_glb, p_err)
        else:
            print('Total:', numpft, 'plant function type tiles.')

        if self.nl_colm['SinglePoint']:
            SITE_pfttyp = np.zeros(numpft)
            SITE_pfttyp[:] = landpft_settyp

        # Deallocate memory
        if pctpft_patch is not None:
            del pctpft_patch
        if pctpft_one is not None:
            del pctpft_one
        if area_one is not None:
            del area_one
        if patchmask is not None:
            del patchmask

    # Define necessary functions and variables
