# Ensure the following libraries are installed: numpy, netCDF4

import numpy as np
import CoLM_TimeManager 
from CoLM_Grid import Grid_type
from CoLM_Mapping_Grid2Pset import MappingGrid2PSet
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
import CoLM_RangeCheck
from CoLM_BGC_Vars_TimeVariables import CoLM_BGC_Vars_TimeVariables

class CoLM_NitrifData(object):
    def __init__(self, nl_colm, idate, landpatch, gblock, mpi, nl_soil, patchclass) -> None:
        self.grid_nitrif = Grid_type(nl_colm, gblock)
        self.mg2p_nitrif = MappingGrid2PSet(nl_colm , gblock, mpi)
        self.netfile = NetCDFFile(mpi)
        self.datatype = DataType(self.gblock)
        self.mpi = mpi
        self.landpatch = landpatch
        self.nl_colm = nl_colm
        self.bgc = CoLM_BGC_Vars_TimeVariables(nl_colm, landpatch, nl_soil)
        self.patchclass = patchclass

    def init_nitrif_data(self, idate):
        """
        Initialize nitrif data by reading latitude and longitude from a NetCDF file.
        """
        file_nitrif = f"{self.nl_colm['DEF_dir_runtime']}/nitrif/CONC_O2_UNSAT/CONC_O2_UNSAT_l01.nc"

        lat = self.netfile.ncio_read_bcast_serial(file_nitrif, 'lat', lat)
        lon = self.netfile.ncio_read_bcast_serial(file_nitrif, 'lon', lon)

        self.grid_nitrif.define_by_center(lat, lon)
        self.mg2p_nitrif.build(self.grid_nitrif, self.landpatch, self.patchclass)

        if lon is not None:
            del lon

        if lat is not None:
            del lat

        month, mday = CoLM_TimeManager.julian2monthday(idate[0], idate[1], month, mday)

        self.update_nitrif_data(month, self.nl_soil)


    def update_nitrif_data(self, month, nl_soil, patchclass):
        """
        Update nitrif data for the given month.
        """
        if self.mpi.p_is_worker:
            self.bgc.tCONC_O2_UNSAT_tmp = np.zeros(self.landpatch.numpatch)
            self.bgc.tO2_DECOMP_DEPTH_UNSAT_tmp = np.zeros(self.landpatch.numpatch)

        if self.mpi.p_is_io:
            self.f_xy_nitrif = self.datatype.allocate_block_data(self.grid_nitrif)

        for nsl in range(nl_soil + 1):
            cx = nsl
            file_nitrif = f"{self.nl_colm['DEF_dir_runtime']}/nitrif/CONC_O2_UNSAT/CONC_O2_UNSAT_l{cx}.nc"
            if self.mpi.p_is_io:
                self.f_xy_nitrif = CoLM_NetCDFBlock.ncio_read_block_time(file_nitrif, 'CONC_O2_UNSAT', self.grid_nitrif, month, self.f_xy_nitrif, self.mpi, self.gblock)

            self.mg2p_nitrif.map_aweighted(self.f_xy_nitrif, self.bgc.tCONC_O2_UNSAT_tmp)

            if self.mpi.p_is_worker:
                if self.landpatch.numpatch > 0:
                    for npatch in range(self.landpatch.numpatch + 1):
                        m = patchclass[npatch]
                        if m == 0:
                            self.bgc.tCONC_O2_UNSAT[nsl, npatch] = 0.0
                        else:
                            self.bgc.tCONC_O2_UNSAT[nsl, npatch] = self.bgc.tCONC_O2_UNSAT_tmp[npatch]
                        if self.bgc.tCONC_O2_UNSAT[nsl, npatch] < 1E-10:
                            self.bgc.tCONC_O2_UNSAT[nsl, npatch] = 0.0

        if self.nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('CONC_O2_UNSAT', self.bgc.tCONC_O2_UNSAT)

        for nsl in range(nl_soil + 1):
            cx = nsl
            file_nitrif = f"{self.nl_colm['DEF_dir_runtime']}/nitrif/O2_DECOMP_DEPTH_UNSAT/O2_DECOMP_DEPTH_UNSAT_l{cx}.nc"
            if self.mpi.p_is_io:
                self.f_xy_nitrif = CoLM_NetCDFBlock.ncio_read_block_time(file_nitrif, 'O2_DECOMP_DEPTH_UNSAT', self.grid_nitrif, month, self.f_xy_nitrif, self.mpi, self.gblock)

            self.mg2p_nitrif.map_aweighted(self.f_xy_nitrif, self.bgc.tO2_DECOMP_DEPTH_UNSAT_tmp)

            if self.mpi.p_is_worker:
                if self.landpatch.numpatch > 0:
                    for npatch in range(self.landpatch.numpatch + 1):
                        m = patchclass[npatch]
                        if m == 0:
                            self.bgc.tO2_DECOMP_DEPTH_UNSAT[nsl - 1, npatch - 1] = 0.0
                        else:
                            self.bgc.tO2_DECOMP_DEPTH_UNSAT[nsl - 1, npatch - 1] = self.bgc.tO2_DECOMP_DEPTH_UNSAT_tmp[npatch - 1]
                        if self.bgc.tO2_DECOMP_DEPTH_UNSAT[nsl - 1, npatch - 1] < 1E-10:
                            self.bgc.tO2_DECOMP_DEPTH_UNSAT[nsl - 1, npatch - 1] = 0.0
        if self.nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('O2_DECOMP_DEPTH_UNSAT', self.bgc.tO2_DECOMP_DEPTH_UNSAT)

        if self.mpi.p_is_worker:
            del self.bgc.tCONC_O2_UNSAT_tmp
            del self.bgc.tO2_DECOMP_DEPTH_UNSAT_tmp

