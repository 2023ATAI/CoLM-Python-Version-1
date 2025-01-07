#------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
from CoLM_Pixelset import Pixelset_type, SubsetType
from CoLM_DataType import DataType
from CoLM_NetCDFBlock import NetCDFBlock
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector

import CoLM_Utils
import numpy as np
import os

class CoLM_LandPatch(object):
    def __init__(self, nl_colm, mpi, gblock, mesh, const_lc) -> None:
        self.nl_colm = nl_colm
        self.const_lc = const_lc
        self.mpi = mpi
        self.gblock = gblock
        self.mesh = mesh
        self.landpatch = Pixelset_type(nl_colm['USEMPI'], mpi, gblock, self.mesh)
        self.vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)

    def landpatch_build(self, lc_year, SITE_landtype, var_global, landelm, landhru, gpatch):
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
            
            self.landpatch.eindex = np.zeros(self.numpatch)
            self.landpatch.ipxstt = np.zeros(self.numpatch)
            self.landpatch.ipxend = np.zeros(self.numpatch)
            self.landpatch.settyp = np.zeros(self.numpatch)
            self.landpatch.ielm   = np.zeros(self.numpatch)

            self.landpatch.eindex[:] = 1
            self.landpatch.ielm  [:] = 1
            self.landpatch.ipxstt[:] = 1
            self.landpatch.ipxend[:] = 1
            self.landpatch.settyp[:] = SITE_landtype

            self.landpatch.nset = self.numpatch
            self.landpatch.set_vecgs()

            return
        

        if  self.nl_colm['USEMPI']:
            pass
            # mpi_barrier(p_comm_glb, p_err)

        # Define SinglePoint
        if not self.nl_colm['SinglePoint']:
            if self.mpi.p_is_io:
                dt =  DataType(self.gblock)
                patchdata = dt.allocate_block_data(gpatch)

                if self.nl_colm['LULC_USGS']:
                    # add parameter input for time year
                    file_patch = self.nl_colm['DEF_dir_rawdata'] + 'landtypes/landtype-igbp-modis-' + cyear + '.nc'
                else:
                    #TODO: need usgs land cover type data
                    file_patch = self.nl_colm['DEF_dir_rawdata'] + '/landtypes/landtype_usgs_update.nc'
                
                netblock = NetCDFBlock(file_patch, 'landtype', gpatch, patchdata)
                netblock.ncio_read_block(self.gblock)
                patchdata = netblock.rdata

                if self.nl_colm['USEMPI']:
                    pass
                    # aggregation_data_daemon(gpatch, data_i4_2d_in1=patchdata)

        if self.mpi.p_is_worker:
            numset = landhru.numhru if self.nl_colm['CATCHMENT'] else self.mesh.numelm

            if numset > 0:
                N_land_classification = 24 # GLCC USGS number of land cover category
                if self.nl_colm['LULC_IGBP']: 
                    N_land_classification = 17 # MODIS IGBP number of land cover category
                eindex_tmp = np.zeros(numset * N_land_classification)
                settyp_tmp = np.zeros(numset * N_land_classification)
                ipxstt_tmp = np.zeros(numset * N_land_classification)
                ipxend_tmp = np.zeros(numset * N_land_classification)
                ielm_tmp = np.zeros(numset * N_land_classification)

            self.numpatch = 0

            for iset in range(numset):
                if self.nl_colm['CATCHMENT']:
                    ie = landhru['ielm'][iset ]
                    ipxstt = landhru['ipxstt'][iset ]
                    ipxend = landhru['ipxend'][iset ]
                else:
                    ie = landelm['ielm'][iset ]
                    ipxstt = landelm['ipxstt'][iset ]
                    ipxend = landelm['ipxend'][iset ]

                npxl = ipxend - ipxstt + 1
                types = np.zeros(ipxend - ipxstt)
                ard = AggregationRequestData(self.mpi, self.mesh, self.gblock.pixel)

                if not self.nl_colm['SinglePoint']:
                    if self.nl_colm['CATCHMENT']:
                        types[:] = ard. aggregation_request_data (landhru, iset, gpatch, zip = False, \
                                                       data_i4_2d_in1 = patchdata, data_i4_2d_out1 = ibuff)
                    else:
                        types[:] = ard. aggregation_request_data (landelm, iset, gpatch, zip = False, \
                                                        data_i4_2d_in1 = patchdata, data_i4_2d_out1 = ibuff)
                else:
                    types = SITE_landtype

                if self.nl_colm['CATCHMENT']:
                    if landhru.settyp[iset] <= 0:
                        types[ipxstt:ipxend] = var_global.WATERBODY

                    types = [var_global.WATERBODY if x == 0 else x for x in types]
                    types = [10 if x == 11 else x for x in types]

                if (self.nl_colm['DEF_USE_PFT'] and (not self.nl_colm['DEF_SOLO_PFT'])) or self.nl_colm['DEF_FAST_PC']:
                    for ipxl in range(ipxstt, ipxend+1):
                        if types[ipxl] > 0:
                            if self.const_lc.patchtypes[types[ipxl]] == 0:
                                types[ipxl] = 1

                order = list(range(ipxstt, ipxend + 1))

                # CoLM_Utils.quicksort (npxl, types, order)
                keys = np.lexsort((order, types))
                order = [order[i] for i in keys]
                types = [types[i] for i in keys]

                self.mesh[ie].ilon[ipxstt:ipxend+1] = self.mesh[ie].ilon[order]
                self.mesh[ie].ilat[ipxstt:ipxend+1] = self.mesh[ie].ilat[order]

                if self.nl_colm['DEF_USE_DOMINANT_PATCHTYPE']:
                    npxl_types = np.zeros(max(types))
                    for ipxl in range(ipxstt, ipxend + 1):
                        npxl_types[types[ipxl]] += 1

                    if any(types):
                        iloc = types.index(True) + ipxstt - 1
                        dominant_type = npxl_types.index(max(npxl_types))
                        types[iloc:ipxend + 1] = dominant_type
                    
                    del npxl_types

                for ipxl in range(ipxstt, ipxend + 1):
                    if ipxl == ipxstt:
                        self.numpatch += 1
                        eindex_tmp[self.numpatch - 1] = self.mesh[ie].indx
                        settyp_tmp[self.numpatch - 1] = types[ipxl]
                        ipxstt_tmp[self.numpatch - 1] = ipxl
                        ielm_tmp[self.numpatch - 1] = ie
                    elif types[ipxl] != types[ipxl - 1]:
                        ipxend_tmp[self.numpatch - 1] = ipxl - 1

                        self.numpatch += 1
                        eindex_tmp[self.numpatch - 1] = self.mesh[ie].indx
                        settyp_tmp[self.numpatch - 1] = types[ipxl]
                        ipxstt_tmp[self.numpatch - 1] = ipxl
                        ielm_tmp[self.numpatch - 1] = ie

                ipxend_tmp[self.numpatch - 1] = ipxend
                del types
                del order

            if self.numpatch > 0:
                self.landpatch.eindex = eindex_tmp[0:self.numpatch-1]
                self.landpatch.ipxstt = ipxstt_tmp[0:self.numpatch-1]
                self.landpatch.ipxend = ipxend_tmp[0:self.numpatch-1]
                self.landpatch.settyp = settyp_tmp[0:self.numpatch-1]
                self.landpatch.ielm   = ielm_tmp[0:self.numpatch-1]

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
                msk = [self.landpatch.settyp != 0 ]
            self.landpatch.pset_pack(msk, self.numpatch)    
            if msk:
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
            self.elm_patch = SubsetType(self.mesh, self.gblock.pixel)
            self.elm_patch.build(landelm, self.landpatch, use_frac=True)
            if self.nl_colm['CATCHMENT']:
                self.hru_patch = SubsetType(self.mesh, self.gblock.pixel)
                self.hru_patch.build(landhru, self.landpatch, use_frac=True)

            self.write_patchfrac(self.nl_colm['DEF_dir_landdata'], lc_year)


    def write_patchfrac(self, dir_landdata, lc_year):
        cyear = lc_year
        os.system('mkdir -p ' + dir_landdata + '/self.landpatch/' + cyear)

        lndname = dir_landdata + '/self.landpatch/' + cyear + '/patchfrac_elm.nc'
        self.vectorOnes.ncio_create_file_vector(lndname, self.landpatch)
        self.vectorOnes.ncio_define_dimension_vector(lndname, self.landpatch, 'patch')
        self.vectorOnes.ncio_write_vector(lndname, 'patchfrac_elm', 'patch', self.landpatch, self.elm_patch.subfrc, self.nl_colm['DEF_Srfdata_CompressLevel'])

        if self.nl_colm['CATCHMENT']:
            lndname = dir_landdata + '/self.landpatch/' + cyear + '/patchfrac_hru.nc'
            self.vectorOnes.ncio_create_file_vector(lndname, self.landpatch)
            self.vectorOnes.ncio_define_dimension_vector(lndname, self.landpatch, 'patch')
            self.vectorOnes.ncio_write_vector(lndname, 'patchfrac_hru', 'patch', self.landpatch, self.hru_patch.subfrc, self.nl_colm['DEF_Srfdata_CompressLevel'])

