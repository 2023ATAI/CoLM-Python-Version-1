import numpy as np

class CoLM_Hydro_Vars_TimeVariables():
    def __init__(self) -> None:
        self.wdsrf_bsn = None     # river or lake water depth [m]
        self.veloc_riv = None     # river velocity [m/s]
        self.momen_riv = None     # unit river momentum [m^2/s]
        self.wdsrf_hru = None     # surface water depth [m]
        self.veloc_hru = None     # surface water velocity [m/s]
        self.momen_hru = None     # unit surface water momentum [m^2/s]

        self.wdsrf_bsn_prev = None # river or lake water depth at previous time step [m]
        self.wdsrf_hru_prev = None # surface water depth at previous time step [m]

    def read_hydro_time_variables(self, file_restart):
        # Number of basins
        numbasin = numelm

        # Reading data
        self. wdsrf_bsn = vector_read_basin(file_restart, self. wdsrf_bsn, numbasin, 'wdsrf_bsn', elm_data_address)
        self.veloc_riv = vector_read_basin(file_restart, self.veloc_riv, numbasin, 'veloc_riv', elm_data_address)
        self.wdsrf_bsn_prev = vector_read_basin(file_restart, self.wdsrf_bsn_prev, numbasin, 'wdsrf_bsn_prev', elm_data_address)

        self.wdsrf_hru = vector_read_basin(file_restart, self.wdsrf_hru, numhru, 'wdsrf_hru', hru_data_address)
        self.veloc_hru = vector_read_basin(file_restart, self.veloc_hru, numhru, 'veloc_hru', hru_data_address)
        self.wdsrf_hru_prev = vector_read_basin(file_restart, self.wdsrf_hru_prev, numhru, 'wdsrf_hru_prev', hru_data_address)

