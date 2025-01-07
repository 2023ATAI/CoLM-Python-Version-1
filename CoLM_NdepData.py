import numpy as np
from CoLM_Grid import Grid_type
from CoLM_Mapping_Grid2Pset import MappingGrid2PSet
from CoLM_DataType import DataType
import CoLM_NetCDFBlock
import CoLM_RangeCheck

class CoLM_NdepData(object):
    def __init__(self, nl_colm, mpi, landpatch, gblock, patchclass) -> None:
        self.gridgrid_ndep_nitrif = Grid_type(nl_colm, gblock)
        self.mg2p_ndep = MappingGrid2PSet(nl_colm , gblock, mpi)
        self.nl_colm = nl_colm
        self.landpatch = landpatch
        self.patchclass = patchclass
        self.mpi = mpi
        self.file_ndep = ''
        self.datatype = DataType(self.gblock)   


# def init_ndep_data_annually(YY):
#     """
#     Open NDEP NetCDF file from DEF_dir_runtime, read latitude and longitude info.
#     Initialize NDEP data read in.
#     """
#     file_ndep = f"{DEF_dir_runtime}/ndep/fndep_colm_hist_simyr1849-2006_1.9x2.5_c100428.nc"

#     lat = np.empty(0)
#     lon = np.empty(0)

#     ncio_read_bcast_serial(file_ndep, 'lat', lat)
#     ncio_read_bcast_serial(file_ndep, 'lon', lon)

#     grid_ndep.define_by_center(lat, lon)
#     mg2p_ndep.build_arealweighted(grid_ndep, landpatch)

#     if len(lon) > 0:
#         lon = None
#     if len(lat) > 0:
#         lat = None

#     update_ndep_data_annually(YY, iswrite=True)


# def init_ndep_data_monthly(YY, MM):
#     """
#     Open NDEP NetCDF file from DEF_dir_runtime, read latitude and longitude info.
#     Initialize NDEP data read in.
#     """
#     file_ndep = f"{DEF_dir_runtime}/ndep/fndep_colm_monthly.nc"

#     lat = np.empty(0)
#     lon = np.empty(0)

#     ncio_read_bcast_serial(file_ndep, 'lat', lat)
#     ncio_read_bcast_serial(file_ndep, 'lon', lon)

#     grid_ndep.define_by_center(lat, lon)
#     mg2p_ndep.build_arealweighted(grid_ndep, landpatch)

#     if len(lon) > 0:
#         lon = None
#     if len(lat) > 0:
#         lat = None

#     update_ndep_data_monthly(YY, MM, iswrite=True)


    def update_ndep_data_annually(self, YY, iswrite):
        """
        Read in the Nitrogen deposition data from CLM5.
        Reference:
        Galloway, J.N., et al. 2004. Nitrogen cycles: past, present, and future. Biogeochem. 70:153-226.
        Original:
        Created by Xingjie Lu and Shupeng Zhang, 2022
        """
        itime = max(min(YY, 2006), 1849) - 1848

        if self.mpi.p_is_io:
            self.f_xy_ndep = self.datatype.allocate_block_data(self.grid_ndep)
            self.f_xy_nitrif = CoLM_NetCDFBlock.ncio_read_block_time(self.file_ndep, 'NDEP_year', self.grid_ndep, itime, self.f_xy_ndep)

        self.mg2p_ndep.grid2pset(self.f_xy_ndep, self.ndep)

        if self.mpi.p_is_worker and iswrite:
            if self.landpatch.numpatch > 0:
                for npatch in range(1, self.landpatch.numpatch + 1):
                    m = self.patchclass(npatch)
                    if m == 0:
                        self.ndep_to_sminn[npatch - 1] = 0.0
                    else:
                        if self.nl_colm['DEF_USE_PN']:
                            self.ndep_to_sminn[npatch - 1] = self.ndep[npatch - 1] / 3600. / 365. / 24. * 5
                        else:
                            self.ndep_to_sminn[npatch - 1] = self.ndep[npatch - 1] / 3600. / 365. / 24.

        if self.nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('ndep', self.ndep)


    def update_ndep_data_monthly(self, YY, MM, iswrite):
        """
        Read in the Nitrogen deposition data from CLM5.
        Reference:
        Galloway, J.N., et al. 2004. Nitrogen cycles: past, present, and future. Biogeochem. 70:153-226.
        Original:
        Created by Xingjie Lu and Shupeng Zhang, 2022
        """
        itime = (max(min(YY, 2006), 1849) - 1849) * 12 + MM

        if self.mpi.p_is_io:
            self.f_xy_ndep = self.datatype.allocate_block_data(self.grid_ndep)
            self.f_xy_nitrif = CoLM_NetCDFBlock.ncio_read_block_time(self.file_ndep, 'NDEP_month', self.griself.grid_ndepd_ndep, itime, self.f_xy_ndep)

        self.mg2p_ndep.grid2pset(self.f_xy_ndep, self.ndep)

        if self.mpi.p_is_worker and iswrite:
            if self.landpatch.numpatch > 0:
                for npatch in range(1, self.landpatch.numpatch + 1):
                    m = self.patchclass(npatch)
                    if m == 0:
                        self.ndep_to_sminn[npatch - 1] = 0.0
                    else:
                        if self.nl_colm['DEF_USE_PN']:
                            self.ndep_to_sminn[npatch - 1] = self.ndep[npatch - 1] / 3600. / 365. / 24. * 5
                        else:
                            self.ndep_to_sminn[npatch - 1] = self.ndep[npatch - 1] / 3600. / 365. / 24.

        if self.nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('ndep', self.ndep)

