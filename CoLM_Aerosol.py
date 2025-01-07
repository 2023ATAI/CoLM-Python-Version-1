import numpy as np
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_DataType import DataType, BlockData
from CoLM_Grid import Grid_type
from CoLM_Mapping_Grid2Pset import MappingGrid2PSet
import CoLM_TimeManager
import CoLM_NetCDFBlock
import CoLM_RangeCheck

class CoLM_Aerosol(object):
    def __init__(self, nl_colm, mpi, gblock, spval):
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.gblock = gblock
        self.use_extrasnowlayers =False
        self.snw_rds_min = 54.526          # minimum allowed snow effective radius (also "fresh snow" value) [microns]
        self.fresh_snw_rds_max = 204.526   # maximum warm fresh snow effective radius [microns]

        self.file_aerosol = ''

        # type(grid_type):: grid_aerosol
        self.grid_aerosol = Grid_type(nl_colm, gblock)
        # type(block_data_real8_2d):: f_aerdep
        self.f_aerdep = BlockData(self.gblock.nxblk, self.gblock.nyblk)
        # type(mapping_grid2pset_type):: mg2p_aerdep
        self.mg2p_aerdep = MappingGrid2PSet(self.nl_colm, self.gblock,mpi, spval)

        self.start_year = 1849
        self.end_year = 2001

        self.month_p = 0

    def AerosolMasses(self, dtime, snl, do_capsnow, h2osno_ice, h2osno_liq, qflx_snwcp_ice, snw_rds,
                      mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4,
                      mss_cnc_bcphi, mss_cnc_bcpho, mss_cnc_ocphi, mss_cnc_ocpho,
                      mss_cnc_dst1, mss_cnc_dst2, mss_cnc_dst3, mss_cnc_dst4, maxsnl):

        """
        Calculate column-integrated aerosol masses, and
        mass concentrations for radiative calculations and output
        (based on new snow level state, after SnowFilter is rebuilt.
        NEEDS TO BE AFTER SnowFiler is rebuilt in Hydrology2, otherwise there
        can be zero snow layers but an active column in filter)
        """

        # !LOCAL VARIABLES:
        snowmass = 0.0          # liquid+ice snow mass in a layer [kg/m2]
        snowcap_scl_fct = 0.0   # temporary factor used to correct for snow capping

        for j in range(maxsnl + 1, -1):

            # layer mass of snow:
            snowmass = h2osno_ice[j] + h2osno_liq[j]

            if not self.use_extrasnowlayers:
                # Correct the top layer aerosol mass to account for snow capping.
                # This approach conserves the aerosol mass concentration
                # (but not the aerosol mass) when snow-capping is invoked

                if j == snl + 1:
                    if do_capsnow:
                        snowcap_scl_fct = snowmass / (snowmass + (qflx_snwcp_ice * dtime))

                        mss_bcpho[j] *= snowcap_scl_fct
                        mss_bcphi[j] *= snowcap_scl_fct
                        mss_ocpho[j] *= snowcap_scl_fct
                        mss_ocphi[j] *= snowcap_scl_fct

                        mss_dst1[j] *= snowcap_scl_fct
                        mss_dst2[j] *= snowcap_scl_fct
                        mss_dst3[j] *= snowcap_scl_fct
                        mss_dst4[j] *= snowcap_scl_fct

            if j >= snl + 1:
                mss_cnc_bcphi[j] = mss_bcphi[j] / snowmass
                mss_cnc_bcpho[j] = mss_bcpho[j] / snowmass

                mss_cnc_ocphi[j] = mss_ocphi[j] / snowmass
                mss_cnc_ocpho[j] = mss_ocpho[j] / snowmass

                mss_cnc_dst1[j] = mss_dst1[j] / snowmass
                mss_cnc_dst2[j] = mss_dst2[j] / snowmass
                mss_cnc_dst3[j] = mss_dst3[j] / snowmass
                mss_cnc_dst4[j] = mss_dst4[j] / snowmass
            else:
                # 01/10/2023, yuan: set empty snow layers to snw_rds_min
                # snw_rds(j) = 0.0
                snw_rds[j] = self.snw_rds_min

                mss_bcpho[j] = 0.0
                mss_bcphi[j] = 0.0
                mss_cnc_bcphi[j] = 0.0
                mss_cnc_bcpho[j] = 0.0

                mss_ocpho[j] = 0.0
                mss_ocphi[j] = 0.0
                mss_cnc_ocphi[j] = 0.0
                mss_cnc_ocpho[j] = 0.0
                
                mss_dst1[j] = 0.0
                mss_dst2[j] = 0.0
                mss_dst3[j] = 0.0
                mss_dst4[j] = 0.0
                mss_cnc_dst1[j] = 0.0
                mss_cnc_dst2[j] = 0.0
                mss_cnc_dst3[j] = 0.0
                mss_cnc_dst4[j] = 0.0

            return snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1, mss_dst2, mss_dst3, mss_dst4, mss_cnc_bcphi,mss_cnc_bcpho,mss_cnc_ocphi,mss_cnc_ocpho,mss_cnc_dst1,mss_cnc_dst2,mss_cnc_dst3,mss_cnc_dst4

    def AerosolDepInit(self):
        if self.nl_colm['DEF_Aerosol_Clim']:
            # climatology data
            self.file_aerosol = f"{self.nl_colm['DEF_dir_runtime']}/aerosol/aerosoldep_monthly_2000_mean_0.9x1.25_c090529.nc"
        else:
            # yearly change data
            self.file_aerosol = f"{self.nl_colm['DEF_dir_runtime']}/aerosol/aerosoldep_monthly_1849-2001_0.9x1.25_c090529.nc"

        netfile = NetCDFFile(self.mpi)
        lat = netfile.ncio_read_bcast_serial (self.file_aerosol, 'lat', )
        lon = netfile.ncio_read_bcast_serial (self.file_aerosol, 'lon', )

        lat, lon = self.grid_aerosol.define_by_center(lat, lon)
        datatype = DataType(self.gblock)
        self.f_aerdep = datatype.allocate_block_data(self.grid_aerosol)
        self.landpatch = self.mg2p_aerdep.build(self.grid_aerosol, self.landpatch)
        self.month_p = -1

    def AerosolDepReadin(self, idate):
        year = idate[0]
        julian_day = idate[1]
        month, mday = CoLM_TimeManager.julian2monthday(year, julian_day)

        if year < self.start_year:
            year = self.start_year
        if year > self.end_year:
            year = self.end_year

        if month == self.month_p:
            return

        self.month_p = month

        if self.nl_colm['DEF_Aerosol_Clim']:
            itime = month
        else:
            itime = (year - self.start_year) * 12 + month

        forc_aerdep = np.zeros((14, self.grid_aerosol.shape[0]))

        # BCPHIDRY , hydrophilic BC dry deposition
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'BCPHIDRY', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[1,:])

        # BCPHODRY , hydrophobic BC dry deposition
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'BCPHODRY', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[2,:])

        # BCDEPWET , hydrophilic BC wet deposition
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'BCDEPWET', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[3,:])

        # OCPHIDRY , hydrophilic OC dry deposition
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'OCPHIDRY', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[4,:])

        # OCPHODRY , hydrophobic OC dry deposition
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'OCPHODRY', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[5,:])

        # OCDEPWET , hydrophilic OC wet deposition
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'OCDEPWET', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[6,:])

        # DSTX01WD , DSTX01 wet deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX01WD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[7,:])

        # DSTX01DD , DSTX01 dry deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX01DD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[8,:])

        # DSTX02WD , DSTX02 wet deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX02WD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[9,:])

        # DSTX02DD , DSTX02 dry deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX02DD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[10,:])

        # DSTX03WD , DSTX03 wet deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX03WD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[11,:])

        # DSTX03DD , DSTX03 dry deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX03DD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[12,:])

        # DSTX04WD , DSTX04 wet deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX04WD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[13,:])

        # DSTX04DD , DSTX04 dry deposition flux at bottom
        self.f_aerdep = CoLM_NetCDFBlock.ncio_read_block_time (self.file_aerosol, 'DSTX04DD', self.grid_aerosol, itime, self.f_aerdep, self.mpi, self.gblock)
        self.mg2p_aerdep.map_aweighted (self.f_aerdep, forc_aerdep[14,:])

        # Optional range check
        if self.nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('aerosol [kg/m/s]', forc_aerdep)