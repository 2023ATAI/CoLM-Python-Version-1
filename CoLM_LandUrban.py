# --------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Build pixelset "landurban".
#
# REVISIONS:
# --------------------------------------------------------------------------------------
import numpy as np
from CoLM_DataType import DataType, BlockData
import CoLM_Utils
from CoLM_5x5DataReadin import CoLM_5x5DataReadin
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_Pixelset import Pixelset_type


class CoLM_LandUrban(object):
    def __init__(self, mpi, nl_colm, gblock, landpatch, var_global, mesh) -> None:
        self.mesh = mesh
        self.var_global = var_global
        self.mpi = mpi
        self.gblock = gblock
        self.nl_colm = nl_colm
        self.patch2urban = None
        self.urban2patch = None
        self.readin = CoLM_5x5DataReadin(mpi, gblock)
        self.landpatch = landpatch
        self.landurban = Pixelset_type(nl_colm['USEMPI'], mpi, gblock, self.mesh)
        self.numpatch = 0
        self.numurban = 0

    def landurban_build(self, lc_year, gurban, landelm, landhru):
        # Local Variables

        dir_urban = ""
        data_urb_class = BlockData(self.gblock.nxblk, self.gblock.nyblk)

        # Local arrays
        ibuff = None
        types = None
        order = None

        # Index variables
        ipatch = 0
        jpatch = 0
        iurban = 0
        ie = 0
        ipxstt = 0
        ipxend = 0
        npxl = 0
        ipxl = 0
        nurb_glb = 0
        npatch_glb = 0

        # Local vars for landpath and landurban
        numpatch_ = 0
        eindex_ = None
        ipxstt_ = None
        ipxend_ = None
        settyp_ = None
        ielm_ = None

        numurban_ = 0
        urbclass = None

        suffix = ""
        cyear = ""

        if self.mpi.p_is_master:
            print("Making self.URBAN type tiles :")

        if self.nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb)

        # Allocate and read the grided LCZ/NCAR self.URBAN type
        if self.mpi.p_is_io:
            dir_urban = self.nl_colm['DEF_dir_rawdata'].strip() + '/urban_type'

            # Allocate and read data_urb_class
            dt = DataType(self.gblock)
            data_urb_class = dt.allocate_block_data(gurban)
            data_urb_class = dt.flush_block_data(0)

            suffix = 'URBTYP'
            if self.nl_colm['DEF_URBAN_type_scheme'] == 1:
                data_urb_class = self.readin.read_5x5_data(dir_urban, suffix, gurban, 'URBAN_DENSITY_CLASS',
                                                           data_urb_class)
            elif self.nl_colm['DEF_URBAN_type_scheme'] == 2:
                data_urb_class = self.readin.read_5x5_data(dir_urban, suffix, gurban, 'LCZ_DOM', data_urb_class)

            # Execute aggregation_data_daemon if using MPI
            if self.nl_colm['USEMPI']:
                pass
                # aggregation_data_daemon(gurban, data_i4_2d_in1=data_urb_class)

        if self.mpi.p_is_worker:
            if self.numpatch > 0:
                # a temporary self.numpatch with max self.URBAN patch
                numpatch_ = self.numpatch + np.count_nonzero(self.landpatch.settyp == self.var_global.URBAN) * (
                            self.var_global.N_URB - 1)

                eindex_ = np.zeros(numpatch_, dtype=np.int64)
                ipxstt_ = np.zeros(numpatch_, dtype=np.int32)
                ipxend_ = np.zeros(numpatch_, dtype=np.int32)
                settyp_ = np.zeros(numpatch_, dtype=np.int32)
                ielm_ = np.zeros(numpatch_, dtype=np.int32)

                # max self.URBAN patch number
                numurban_ = np.count_nonzero(self.landpatch.settyp == self.var_global.URBAN) * self.var_global.N_URB
                if numurban_ > 0:
                    urbclass = np.zeros(numurban_, dtype=np.int32)

            jpatch = 0
            iurban = 0

            # loop for temporary self.numpatch to filter duplicate self.URBAN patch
            for ipatch in range(self.numpatch):
                if self.landpatch.settyp[ipatch] == self.var_global.URBAN:
                    ie = self.landpatch.ielm[ipatch]
                    ipxstt = self.landpatch.ipxstt[ipatch]
                    ipxend = self.landpatch.ipxend[ipatch]

                    ard = AggregationRequestData(self.nl_colm['USEMPI'], self.mpi, self.mesh.mesh, self.gblock.pixel)
                    ibuff = ard.aggregation_request_data(self.landpatch, ipatch + 1, gurban, zip=False,
                                                         data_i4_2d_in1=data_urb_class)
                    if self.nl_colm['DEF_URBAN_type_scheme'] == 1:
                        ibuff = np.where(np.logical_or(ibuff < 1, ibuff > 3), 3, ibuff)
                    elif self.nl_colm['DEF_URBAN_type_scheme'] == 2:
                        ibuff = np.where(np.logical_or(ibuff > 10, ibuff == 0), 9, ibuff)

                    npxl = ipxend - ipxstt + 1
                    types = ibuff
                    order = np.arange(ipxstt, ipxend + 1)

                    # change order vars, types->regid
                    # add region information, because self.URBAN type may be same,
                    # but from different region in this self.URBAN patch
                    # relative code is changed
                    # CoLM_Utils.quicksort(npxl, types, order)
                    keys = np.lexsort((order, types))
                    order = [order[i] for i in keys]
                    types = [types[i] for i in keys]

                    self.mesh.mesh[ie].ilon[ipxstt - 1:ipxend] = self.mesh.mesh[ie].ilon[order]
                    self.mesh.mesh[ie].ilat[ipxstt - 1:ipxend] = self.mesh.mesh[ie].ilat[order]

                    for ipxl in range(ipxstt, ipxend + 1):
                        if ipxl != ipxstt:
                            if types[ipxl - 1] != types[ipxl - 2]:
                                ipxend_[jpatch] = ipxl - 1
                            else:
                                continue

                        jpatch += 1
                        eindex_[jpatch] = self.mesh.mesh[ie].indx
                        settyp_[jpatch] = self.var_global.URBAN
                        ipxstt_[jpatch] = ipxl
                        ielm_[jpatch] = ie

                        iurban += 1
                        urbclass[iurban] = types[ipxl - 1]

                    ipxend_[jpatch] = ipxend
                    types = None
                    order = None
                else:
                    jpatch = jpatch + 1
                    eindex_[jpatch] = self.landpatch.eindex[ipatch]
                    ipxstt_[jpatch] = self.landpatch.ipxstt[ipatch]
                    ipxend_[jpatch] = self.landpatch.ipxend[ipatch]
                    settyp_[jpatch] = self.landpatch.settyp[ipatch]
                    ielm_[jpatch] = self.landpatch.ielm[ipatch]

            if self.nl_colm['USEMPI']:
                pass
                # CALL aggregation_worker_done ()

            # Update self.numpatch
            self.numpatch = jpatch

            if self.numpatch > 0:
                # Update self.landpatch with new patch number
                self.landpatch.eindex = eindex_[:jpatch]
                self.landpatch.ipxstt = ipxstt_[:jpatch]
                self.landpatch.ipxend = ipxend_[:jpatch]
                self.landpatch.settyp = settyp_[:jpatch]
                self.landpatch.ielm = ielm_[:jpatch]

            if self.numpatch > 0:
                # Update self.URBAN patch number
                self.numurban = np.count_nonzero(self.landpatch.settyp == self.var_global.URBAN)
            else:
                self.numurban = 0

            if self.numurban > 0:
                # Allocate memory for landurban attributes
                self.landurban.eindex = np.array([self.landpatch.eindex[i] for i in range(len(self.landpatch.settyp)) if
                                                  self.landpatch.settyp[i] == self.var_global.URBAN])
                self.landurban.ipxstt = np.array([self.landpatch.ipxstt[i] for i in range(len(self.landpatch.settyp)) if
                                                  self.landpatch.settyp[i] == self.var_global.URBAN])
                self.landurban.ipxend = np.array([self.landpatch.ipxend[i] for i in range(len(self.landpatch.settyp)) if
                                                  self.landpatch.settyp[i] == self.var_global.URBAN])
                self.landurban.ielm = np.array([self.landpatch.ielm[i] for i in range(len(self.landpatch.settyp)) if
                                                self.landpatch.settyp[i] == self.var_global.URBAN])

                self.landurban.settyp = urbclass[:self.numurban]

            # Update land patch with self.URBAN type patch
            self.landurban.nset = self.numurban
            self.landpatch.nset = self.numpatch

            # Further code execution...

        self.landpatch.set_vecgs()
        self.landurban.set_vecgs()

        self.map_patch_to_urban()

        if self.nl_colm['USEMPI']:
            pass
            # if self.mpi.p_is_worker:
            #     nurb_glb = np.sum(self.numurban)
            #     if self.mpi.p_iam_worker == 0:
            #         print('Total: {} self.URBAN tiles.'.format(nurb_glb))
            # MPI barrier
        else:
            print('Total: {} self.URBAN tiles.'.format(self.numurban))

        if self.nl_colm['SinglePoint']:
            SITE_urbtyp = np.array(self.numurban)
            SITE_lucyid = np.empty(self.numurban)
            if not self.nl_colm['USE_SITE_urban_paras']:
                SITE_fveg_urb = np.empty(self.numurban)
                SITE_htop_urb = np.empty(self.numurban)
                SITE_flake_urb = np.empty(self.numurban)
                SITE_popden = np.empty(self.numurban)
                SITE_froof = np.empty(self.numurban)
                SITE_hroof = np.empty(self.numurban)
                SITE_hwr = np.empty(self.numurban)
                SITE_fgper = np.empty(self.numurban)
                SITE_fgimp = np.empty(self.numurban)
            SITE_em_roof = np.empty(self.numurban)
            SITE_em_wall = np.empty(self.numurban)
            SITE_em_gimp = np.empty(self.numurban)
            SITE_em_gper = np.empty(self.numurban)
            SITE_t_roommax = np.empty(self.numurban)
            SITE_t_roommin = np.empty(self.numurban)
            SITE_thickroof = np.empty(self.numurban)
            SITE_thickwall = np.empty(self.numurban)
            SITE_cv_roof = np.empty(self.var_global.nl_roof)
            SITE_cv_wall = np.empty(self.var_global.nl_wall)
            SITE_cv_gimp = np.empty(self.var_global.nl_soil)
            SITE_tk_roof = np.empty(self.var_global.nl_roof)
            SITE_tk_wall = np.empty(self.var_global.nl_wall)
            SITE_tk_gimp = np.empty(self.var_global.nl_soil)
            SITE_alb_roof = np.empty((2, 2))
            SITE_alb_wall = np.empty((2, 2))
            SITE_alb_gimp = np.empty((2, 2))
            SITE_alb_gper = np.empty((2, 2))
            SITE_urbtyp[:] = self.landurban.settyp

        # Assuming self.numpatch, npatch_glb, p_is_worker, p_iam_worker, p_root, p_comm_worker, p_err, p_comm_glb, USEMPI, DEF_dir_landdata, lc_year, landelm, self.landpatch, use_frac, CATCHMENT, hru_patch, elm_patch, write_patchfrac, ibuff, types, order, eindex_, ipxstt_, ipxend_, settyp_, ielm_, urbclass, and allocated() are defined

        if not self.nl_colm['CROP']:
            if self.nl_colm['USEMPI']:
                if self.mpi.p_is_worker:
                    # CALL mpi_reduce (self.numpatch, npatch_glb, 1, MPI_INTEGER, MPI_SUM, p_root, p_comm_worker, p_err)
                    if self.mpi.p_iam_worker == 0:
                        print('Total: {} patches.'.format(npatch_glb))
                # CALL mpi_barrier (p_comm_glb, p_err)
            else:
                print('Total: {} patches.'.format(self.numpatch))

            self.landpatch.elm_patch.build(landelm, self.landpatch, use_frac=True)
            if self.nl_colm['CATCHMENT']:
                self.landpatch.hru_patch.build(landhru, self.landpatch, use_frac=True)

            self.landpatch.write_patchfrac(self.nl_colm['DEF_dir_landdata'], lc_year)

        if 'ibuff' in locals() and ibuff is not None:
            del ibuff
        if 'types' in locals() and types is not None:
            del types
        if 'order' in locals() and order is not None:
            del order
        if 'eindex_' in locals() and eindex_ is not None:
            del eindex_
        if 'ipxstt_' in locals() and ipxstt_ is not None:
            del ipxstt_
        if 'ipxend_' in locals() and ipxend_ is not None:
            del ipxend_
        if 'settyp_' in locals() and settyp_ is not None:
            del settyp_
        if 'ielm_' in locals() and ielm_ is not None:
            del ielm_
        if 'urbclass' in locals() and urbclass is not None:
            del urbclass

    def map_patch_to_urban(self):
        if self.mpi.p_is_worker:
            if self.numpatch <= 0 or self.numurban <= 0:
                return

            if self.patch2urban is not None:
                del self.patch2urban
            if self.urban2patch is not None:
                del self.urban2patch

            iurban = 0
            for ipatch in range(self.numpatch):
                if self.landpatch.settyp[ipatch] == self.var_global.URBAN:
                    iurban += 1
                    self.patch2urban[ipatch] = iurban
                    self.urban2patch[iurban - 1] = ipatch
                else:
                    self.patch2urban[ipatch] = -1
