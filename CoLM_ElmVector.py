import numpy as np
from CoLM_DataType import Pointer
import CoLM_Utils


class CoLM_ElmVector(object):
    def __init__(self, nl_colm, mpi) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi

        self.totalnumelm = 0
        self.eindex_glb = None
        self.elm_data_address = None

    def elm_vector_init(self, landelm, landpatch, landcrop):
        self.totalnumelm = 0
        numelm_worker = None
        vec_worker_dsp = None
        indexelm = None
        order = None

        if self.nl_colm['CROP']:
            landpatch.elm_patch.build (landelm, landpatch, use_frac = True, sharedfrac = landcrop.pctshrpch)
        else:
            landpatch.elm_patch.build(landelm, landpatch, use_frac=True)

        if self.mpi.p_is_worker:
            if self.nl_colm['USEMPI']:
                pass
            else:
                if landpatch.numelm > 0:
                    self.eindex_glb = landelm.eindex

        if self.mpi.p_is_master:
            if self.nl_colm['USEMPI']:
                pass
            else:
                self.totalnumelm = landpatch.numelm
                self.elm_data_address = [Pointer()]
                self.elm_data_address[0].val = np.zeros(self.totalnumelm)

        if self.nl_colm['USEMPI']:
            pass

        if self.mpi.p_is_master:
            order = [ipxl for ipxl in range(self.totalnumelm)]
            eindex_glb, order = CoLM_Utils.quicksort (self.totalnumelm, eindex_glb, order)

            if self.nl_colm['USEMPI']:
                pass
            else:
                self.elm_data_address[0].val[order] = [i for i in range(self.totalnumelm)]

        if numelm_worker is not None:
            del numelm_worker
        if vec_worker_dsp is not None:
            del vec_worker_dsp
        if indexelm is not None:
            del indexelm
        if order is not None:
            del order

    def elm_vector_final(self):
        if self.eindex_glb is not None: del self.eindex_glb
        if self.elm_data_address is not None: del self.elm_data_address
