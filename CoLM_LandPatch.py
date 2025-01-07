# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Build pixelset "landpatch".
#
#    In CoLM, the global/regional area is divided into a hierarchical structure:
#    1. If GRIDBASED or UNSTRUCTURED is defined, it is
#       ELEMENT >>> PATCH
#    2. If CATCHMENT is defined, it is
#       ELEMENT >>> HRU >>> PATCH
#    If Plant Function Type classification is used, PATCH is further divided into PFT.
#    If Plant Community classification is used,     PATCH is further divided into PC.
#
#    "landpatch" refers to pixelset PATCH.
#
# Created by Shupeng Zhang, May 2023
#    porting codes from Hua Yuan's OpenMP version to MPI parallel version.
# ------------------------------------------------------------------------------------
from CoLM_Pixelset import Pixelset_type, SubsetType
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector

import CoLM_Utils
import numpy as np
import os


class CoLM_LandPatch(object):
    def __init__(self, nl_colm, mpi, gblock, pixel,  mesh, const_lc) -> None:
        self.nl_colm = nl_colm
        self.const_lc = const_lc
        self.mpi = mpi
        self.gblock = gblock
        self.pixel = pixel
        self.mesh = mesh
        self.landpatch = Pixelset_type(nl_colm['USEMPI'], mpi, gblock, self.mesh)
        self.vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
        self.numpatch = -1
        self.N_land_classification = 24  # GLCC USGS number of land cover category
        self.elm_patch = SubsetType(self.mesh, self.pixel)

    def landpatch_build(self, lc_year, SITE_landtype, var_global, landelm, landhru, gpatch, pixel):
        # Local Variables
        file_patch = ""
        cyear = str(lc_year)
        patchdata = None
        iloc, npxl, ipxl, numset = 0, 0, 0, 0
        ie, iset, ipxstt, ipxend = 0, 0, 0, 0
        types, order, ibuff = None, None, None
        eindex_tmp, settyp_tmp, ipxstt_tmp, ipxend_tmp, ielm_tmp = None, None, None, None, None
        msk = None
        npatch_glb = 0
        dominant_type = 0
        npxl_types = None

        cyear = str(lc_year)
        if self.mpi.p_is_master:
            print('Making land patches:')

        # Define SinglePoint
        if self.nl_colm['SinglePoint']:
            if self.nl_colm['CROP']:
                if (SITE_landtype == var_global.CROPLAND) and self.nl_colm['USE_SITE_pctcrop']:
                    return

            self.numpatch = 1

            self.landpatch.eindex = np.zeros(self.numpatch, dtype='int')
            self.landpatch.ipxstt = np.zeros(self.numpatch, dtype='int')
            self.landpatch.ipxend = np.zeros(self.numpatch, dtype='int')
            self.landpatch.settyp = np.zeros(self.numpatch)
            self.landpatch.ielm = np.zeros(self.numpatch, dtype='int')

            self.landpatch.settyp[:] = SITE_landtype

            self.landpatch.nset = self.numpatch
            self.landpatch.set_vecgs()

            return

        if self.nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb, p_err)

        # Define SinglePoint
        if not self.nl_colm['SinglePoint']:
            if self.mpi.p_is_io:
                dt = DataType(self.gblock)
                patchdata = dt.allocate_block_data(gpatch)

                if not self.nl_colm['LULC_USGS']:
                    # add parameter input for time year
                    file_patch = self.nl_colm['DEF_dir_rawdata'] + 'landtypes/landtype-igbp-modis-' + cyear + '.nc'
                else:
                    # TODO: need usgs land cover type data
                    file_patch = self.nl_colm['DEF_dir_rawdata'] + 'landtypes/landtype-usgs-update.nc'

                patchdata = CoLM_NetCDFBlock.ncio_read_block(file_patch, 'landtype', self.mpi, self.gblock, gpatch,
                                                             patchdata)

                if self.nl_colm['USEMPI']:
                    pass
                    # aggregation_data_daemon(gpatch, data_i4_2d_in1=patchdata)

        if self.mpi.p_is_worker:
            numset = landhru.numhru if self.nl_colm['CATCHMENT'] else self.mesh.numelm

            if numset > 0:
                if self.nl_colm['LULC_IGBP']:
                    self.N_land_classification = 17  # MODIS IGBP number of land cover category
                eindex_tmp = np.zeros(numset * self.N_land_classification,dtype=int)
                settyp_tmp = np.zeros(numset * self.N_land_classification,dtype=int)
                ipxstt_tmp = np.zeros(numset * self.N_land_classification,dtype=int)
                ipxend_tmp = np.zeros(numset * self.N_land_classification,dtype=int)
                ielm_tmp = np.zeros(numset * self.N_land_classification,dtype=int)

            self.numpatch = -1

            # for iset in range(numset):
            #     print(landelm.ielm[iset], landelm.ipxstt[iset], landelm.ipxend[iset], '-------landpatch---------')

            for iset in range(numset):
                if self.nl_colm['CATCHMENT']:
                    ie = landhru.ielm[iset]
                    ipxstt = landhru.ipxstt[iset]
                    ipxend = landhru.ipxend[iset]
                else:
                    ie = landelm.ielm[iset]
                    ipxstt = landelm.ipxstt[iset]
                    ipxend = landelm.ipxend[iset]

                npxl = ipxend - ipxstt + 1
                types = np.zeros(ipxend - ipxstt,dtype=int)
                ard = AggregationRequestData(self.nl_colm['USEMPI'], self.mpi, self.mesh.mesh, pixel)

                if not self.nl_colm['SinglePoint']:
                    if self.nl_colm['CATCHMENT']:
                        _, _, _, _, _, _, _, _, _, ibuff, _ = ard.aggregation_request_data(landhru, iset, gpatch, zip=False, data_i4_2d_in1=patchdata)
                        types = ibuff
                        del ibuff
                    else:
                        _, _, _, _, _, _, _, _, _, ibuff, _  = ard.aggregation_request_data(landelm, iset, gpatch, zip=False, data_i4_2d_in1=patchdata)
                        types = ibuff
                        # for ipxl in range(ipxstt, ipxend):
                        #     if types[ipxl] != types[ipxl - 1]:
                        #         print(types[ipxl], types[ipxl-1], ipxl,'-+-+-+-+-+-')

                        del ibuff
                else:
                    types = SITE_landtype

                if self.nl_colm['CATCHMENT']:
                    if landhru.settyp[iset] <= 0:
                        types[ipxstt:ipxend] = var_global.WATERBODY

                    types = [var_global.WATERBODY if x == 0 else x for x in types]
                    types = [10 if x == 11 else x for x in types]

                if (self.nl_colm['DEF_USE_PFT'] and (not self.nl_colm['DEF_SOLO_PFT'])) or self.nl_colm['DEF_FAST_PC']:
                    for ipxl in range(ipxstt, ipxend):
                        if types[ipxl] > 0:
                            if self.const_lc.patchtypes[types[ipxl]] == 0:
                                types[ipxl] = 0
                # if iset == 0:
                #     for i in range(50):
                #         print(types[i], '------1----types---------')

                order = list(range(ipxstt, ipxend ))
                keys = np.lexsort((order, types))
                order = [order[i] for i in keys]
                types = [types[i] for i in keys]

                # if iset == 0:
                #     for i in range(50):
                #         print(types[i], '------2----types---------')

                # order = CoLM_Utils.quicksort(npxl, types, order)
                index = 0
                for i in order:
                    self.mesh.mesh[ie].ilon[ipxstt + index] = self.mesh.mesh[ie].ilon[i]
                    self.mesh.mesh[ie].ilat[ipxstt + index] = self.mesh.mesh[ie].ilat[i]
                    index += 1

                if self.nl_colm['DEF_USE_DOMINANT_PATCHTYPE']:
                    npxl_types = np.zeros(max(types),dtype=int)
                    for ipxl in range(ipxstt, ipxend):
                        npxl_types[types[ipxl]] += 1

                    if any(types>0):
                        iloc = [i for i, x in enumerate(types) if x > 0][0] + ipxstt - 1
                        dominant_type = np.argmax(npxl_types)
                        types[iloc:ipxend] = dominant_type

                    del npxl_types

                for ipxl in range(ipxstt, ipxend):
                    # if ipxl == ipxstt and iset==0:
                    #     for i in range(50):
                    #         print(types[i],'------3----types---------')
                    if ipxl == ipxstt:
                        self.numpatch += 1
                        eindex_tmp[self.numpatch] = self.mesh.mesh[ie].indx
                        settyp_tmp[self.numpatch] = types[ipxl]
                        ipxstt_tmp[self.numpatch] = ipxl
                        ielm_tmp[self.numpatch] = ie

                        # print(ipxl, self.mesh.mesh[ie].indx,'------first------------')
                    elif types[ipxl] != types[ipxl-1]:
                        # print(types[ipxl], ipxend, '-+-+-+-+-+-')
                        # print(types[ipxl], types[ipxl-1], ipxl,'-+-+-+-+-+-')
                        ipxend_tmp[self.numpatch] = ipxl
                        self.numpatch += 1
                        eindex_tmp[self.numpatch] = self.mesh.mesh[ie].indx
                        settyp_tmp[self.numpatch] = types[ipxl]
                        ipxstt_tmp[self.numpatch] = ipxl
                        ielm_tmp[self.numpatch] = ie

                        # print(ipxl, self.mesh.mesh[ie].indx,'------11------------')

                ipxend_tmp[self.numpatch] = ipxend
                del types
                del order
            # for i in ipxend_tmp:
            #     print(i,'++++-+------------')

            if self.numpatch > 0:
                self.landpatch.eindex = eindex_tmp[0:self.numpatch]
                self.landpatch.ipxstt = ipxstt_tmp[0:self.numpatch]
                self.landpatch.ipxend = ipxend_tmp[0:self.numpatch]
                self.landpatch.settyp = settyp_tmp[0:self.numpatch]
                self.landpatch.ielm = ielm_tmp[0:self.numpatch]

            if numset > 0:
                del eindex_tmp
                del ipxstt_tmp
                del ipxend_tmp
                del settyp_tmp
                del ielm_tmp
            if self.nl_colm['USEMPI']:
                pass
                # aggregation_worker_done()

        self.landpatch.nset = self.numpatch
        self.landpatch.set_vecgs()

        if self.nl_colm['DEF_LANDONLY']:
            if self.mpi.p_is_worker and self.numpatch > 0:
                msk = [self.landpatch.settyp != 0][0]
                self.numpatch = self.landpatch.pset_pack(msk)
            if msk is not None:
                del msk

        if not self.nl_colm['URBAN_MODEL'] and self.nl_colm['CROP']:
            if self.nl_colm['USEMPI']:
                pass
                # if p_is_worker:
                #     npatch_glb = mpi_reduce(self.numpatch, MPI_SUM, p_root, p_comm_worker, p_err)
                #     if p_iam_worker == 0:
                #         print('Total:', npatch_glb, 'patches.')
                # mpi_barrier(p_comm_glb, p_err)
            else:
                print('Total:', self.numpatch, 'patches.')

            self.elm_patch.build(landelm, self.landpatch, use_frac=True)
            if self.nl_colm['CATCHMENT']:
                self.hru_patch = SubsetType(self.mesh, self.pixel)
                self.hru_patch.build(landhru, self.landpatch, use_frac=True)

            self.write_patchfrac(self.nl_colm['DEF_dir_landdata'], lc_year)

    def write_patchfrac(self, dir_landdata, lc_year):
        cyear = lc_year
        os.system('mkdir -p ' + dir_landdata + '/self.landpatch/' + cyear)

        lndname = dir_landdata + '/self.landpatch/' + cyear + '/patchfrac_elm.nc'
        self.vectorOnes.ncio_create_file_vector(lndname, self.landpatch)
        self.vectorOnes.ncio_define_dimension_vector(lndname, self.landpatch, 'patch')
        self.vectorOnes.ncio_write_vector(lndname, 'patchfrac_elm', 'patch', self.landpatch, self.elm_patch.subfrc,
                                          self.nl_colm['DEF_Srfdata_CompressLevel'])

        if self.nl_colm['CATCHMENT']:
            lndname = dir_landdata + '/self.landpatch/' + cyear + '/patchfrac_hru.nc'
            self.vectorOnes.ncio_create_file_vector(lndname, self.landpatch)
            self.vectorOnes.ncio_define_dimension_vector(lndname, self.landpatch, 'patch')
            self.vectorOnes.ncio_write_vector(lndname, 'patchfrac_hru', 'patch', self.landpatch, self.hru_patch.subfrc,
                                              self.nl_colm['DEF_Srfdata_CompressLevel'])
