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
# from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
# from CoLM_DataType import DataType
# from CoLM_AggregationRequestData import AggregationRequestData
# from CoLM_5x5DataReadin import CoLM_5x5DataReadin
import CoLM_Utils


class CoLM_HruVector(object):
    def __init__(self, nl_colm, mpi) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi

        self.totalnumhru = 0
        self.hru_data_address = None
        self.eindx_hru = None
        self.htype_hru = None

    def hru_vector_init(self,  p_np_worker, totalnumelm, numelm,
                        basin_hru, landelm, landhru, landpatch, hru_patch, elm_data_address,
                        eindex_glb, settyp, landcrop):

        nhru_bsn = None
        nhru_bsn_glb = None
        rbuff = None
        hru_dsp_glb = None

        if self.mpi.p_is_worker:
            basin_hru.build(landelm, landhru, use_frac=True)

            if self.nl_colm['CROP']:
                hru_patch.build(landhru, landpatch, use_frac=True, sharedfrac=landcrop.pctshrpch)
            else:
                hru_patch.build(landhru, landpatch, use_frac=True)

            if numelm > 0:
                nhru_bsn = np.empty(numelm, dtype=int)
                nhru_bsn[:] = basin_hru.subend - basin_hru.substt + 1

        if self.mpi.p_is_master:
            self.hru_data_address = [None] * (p_np_worker - 1)

            nhru_bsn_glb = np.zeros(totalnumelm, dtype=int)

            for i in range(len(elm_data_address[0].val)):
                nhru_bsn_glb[elm_data_address[0].val[i]] = nhru_bsn[i]

            if np.sum(nhru_bsn) > 0:
                self.hru_data_address[0] = np.zeros(np.sum(nhru_bsn), dtype=int)

        if self.mpi.p_is_master:
            totalnumhru = np.sum(nhru_bsn_glb)

            hru_dsp_glb = np.zeros(totalnumelm, dtype=int)
            hru_dsp_glb[0] = 0
            for ielm in range(1, totalnumelm):
                hru_dsp_glb[ielm] = hru_dsp_glb[ielm - 1] + nhru_bsn_glb[ielm - 1]

            for iwork in range(p_np_worker):
                if elm_data_address[iwork].val is not None:
                    nelm = len(elm_data_address[iwork].val)
                    hru_dsp_loc = 0
                    for ielm in range(nelm):
                        ielm_glb = elm_data_address[iwork].val[ielm]
                        nhru = nhru_bsn_glb[ielm_glb]
                        if nhru > 0:
                            self.hru_data_address[iwork][hru_dsp_loc:hru_dsp_loc + nhru] = \
                                np.arange(hru_dsp_glb[ielm_glb], hru_dsp_glb[ielm_glb] + nhru + 1) + 1
                            hru_dsp_loc += nhru

        if self.mpi.p_is_master:
            self.eindx_hru = np.empty(totalnumhru, dtype=int)

            for ielm in range(totalnumelm):
                self.eindx_hru[hru_dsp_glb[ielm]:hru_dsp_glb[ielm] + nhru_bsn_glb[ielm] + 1] = \
                    eindex_glb[ielm]

            self.htype_hru = np.zeros(totalnumhru, dtype=int)

            self.htype_hru[self.hru_data_address[0].val[:]] = landhru.settyp[:]

            self.htype_hru = np.abs(self.htype_hru)

        # return self.hru_data_address, self.eindx_hru, self.htype_hru
