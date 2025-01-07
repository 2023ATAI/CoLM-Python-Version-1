import numpy as np

class CoLM_Hydro_Hist(object):
    def __init__(self, mpi, numelm, numhru, spval) -> None:
        self.mpi = mpi
        self.numelm = numelm
        self.numhru = numhru
        self.spval = spval
        self.a_wdsrf_hru = None
        self.a_veloc_hru = None
        self.a_xsubs_bsn = None
        self.a_xsubs_hru = None
        self.a_height_riv = None
        self.a_veloct_riv = None
        self.a_discharge = None

    def hist_basin_init(self):
        numbasin = self.numelm

        if self.mpi.p_is_worker():
            if self.numhru > 0:
                self.a_wdsrf_hru = np.zeros(self.numhru, dtype=np.float64)
                self.a_veloc_hru = np.zeros(self.numhru, dtype=np.float64)
                self.a_xsubs_hru = np.zeros(self.numhru, dtype=np.float64)

            if numbasin > 0:
                self.a_height_riv = np.zeros(numbasin, dtype=np.float64)
                self.a_veloct_riv = np.zeros(numbasin, dtype=np.float64)
                self.a_discharge = np.zeros(numbasin, dtype=np.float64)
                self.a_xsubs_bsn = np.zeros(numbasin, dtype=np.float64)

        self.flush_acc_fluxes_basin()

    def flush_acc_fluxes_basin(self):
        if self.mpi.p_is_worker:
            numbasin = self.numelm

            nac_basin = 0

            if numbasin > 0:
                self.a_height_riv[:] = self.spval
                self.a_veloct_riv[:] = self.spval
                self.a_discharge[:] = self.spval
                self.a_xsubs_bsn[:] = self.spval

            if self.numhru > 0:
                self.a_wdsrf_hru[:] = self.spval
                self.a_veloc_hru[:] = self.spval
                self.a_xsubs_hru[:] = self.spval