import numpy as np
import CoLM_RangeCheck
import CoLM_Catch_HillslopeFlow
import CoLM_Catch_RiverLakeFlow

class CoLM_Catch_LateralFlow:
    def __init__(self, nl_colm, mpi, landpatch, landhru, mesh, pixel, const_pyhsical, catch_riverlakenetwork,  vtv, hydro_VTV, hydro_fluxes, catch_riverlakeflow):
        self.nl_colm = nl_colm
        self.catch_riverlakeflow = catch_riverlakeflow
        self.catch_riverlakenetwork = catch_riverlakenetwork
        self.vtv = vtv
        self.const_pyhsical = const_pyhsical
        self.mpi = mpi
        self.hydro_VTV = hydro_VTV
        self.hydro_fluxes = hydro_fluxes
        self.landhru = landhru
        self.nsubstep = 20
        self.dt_average = 0.0
        self.landarea = 0.0
        self.patcharea = None
        self.numpatch = landpatch.numpatch  # Set this to the actual number of patches in your data
        self.landpatch = landpatch  # Initialize with actual landpatch data
        self.pixel = pixel  # Initialize with actual pixel data
        self.mesh = mesh  # Initialize with actual mesh data

    def lateral_flow_init(self, lc_year):
        # Initialize neighbour, network, and basin
        element_neighbour_init(lc_year)
        hillslope_network_init()
        river_lake_network_init()
        basin_neighbour_init()

        if self.nl_colm['CoLMDEBUG']:
            if self.mpi.p_is_worker():
                self.patcharea = np.zeros(self.numpatch)
                for ip in range(self.numpatch):
                    self.patcharea[ip] = 0.0
                    ie = self.landpatch.ielm[ip]
                    for ipxl in range(self.landpatch.ipxstt[ip], self.landpatch.ipxend[ip] + 1):
                        self.patcharea[ip] += 1.0e6 * self.areaquad(
                            self.pixel.lat_s[self.mesh[ie].ilat[ipxl]],
                            self.pixel.lat_n[self.mesh[ie].ilat[ipxl]],
                            self.pixel.lon_w[self.mesh[ie].ilon[ipxl]],
                            self.pixel.lon_e[self.mesh[ie].ilon[ipxl]]
                        )

                self.landarea = np.sum(self.patcharea) if self.numpatch > 0 else 0.0
                
                # #ifdef USEMPI
                # CALL mpi_allreduce (MPI_IN_PLACE, self.landarea, 1, MPI_REAL8, MPI_SUM, p_comm_worker, p_err)

    def lateral_flow(self, deltime, wdsrf, lakedepth, rsur, rsub, rnof, grav):
        if self.mpi.p_is_worker:
            numbasin = self.mesh.numelm

            # Calculate self.hydro_VTV.wdsrf_hru
            for i in range(self.landhru.numhru):
                ps = self.landpatch.elm_patch.substt[i]
                pe = self.landpatch.elm_patch.subend[i]
                self.hydro_VTV.wdsrf_hru[i] = np.sum(wdsrf[ps:pe+1] * self.landpatch.elm_patch.subfrc[ps:pe+1])
                self.hydro_VTV.wdsrf_hru[i] /= 1.0e3  # mm to m

            self.hydro_VTV.wdsrf_hru_ta[:] = 0
            self.hydro_fluxes.momen_hru_ta[:] = 0
            self.hydro_VTV.wdsrf_bsn_ta[:] = 0
            self.hydro_fluxes.momen_riv_ta[:] = 0

            if self.landpatch.numpatch > 0:
                wdsrf_p = wdsrf
            
            dt_average = 0.0

            if self.landpatch.numpatch > 0:
                rsur[:] = 0.0
            if numbasin > 0:
                self.hydro_fluxes.discharge[:] = 0.0

            for istep in range(self.nsubstep):
                # (1) Surface flow over hillslopes.
                CoLM_Catch_HillslopeFlow.hillslope_flow(deltime / self.nsubstep, self.mpi, self.mesh.numelm, grav, self.landhru, catch_riverlakenetwork, self.hydro_VTV, self.hydro_fluxes.rsur, self.landpatch.elm_patch)

                # (2) River and Lake flow.
                CoLM_Catch_RiverLakeFlow.river_lake_flow(deltime / self.nsubstep, self.mpi.p_is_worker, self.mesh.numelm, catch_riverlakenetwork, self.hydro_VTV, self.hydro_fluxes, self.const_pyhsical.grav )

                dt_average += deltime / self.nsubstep / self.catch_riverlakeflow.ntimestep_riverlake

            if self.landpatch.numpatch > 0:
                rsur[:] /= deltime
            if numbasin > 0:
                self.hydro_fluxes.discharge[:] /= deltime

            if numbasin > 0:
                self.hydro_VTV.wdsrf_bsn_ta[:] /= deltime
                self.hydro_fluxes.momen_riv_ta[:] /= deltime

                self.hydro_VTV.wdsrf_bsn_ta = np.where(self.hydro_VTV.wdsrf_bsn_ta > 0, self.hydro_fluxes.momen_riv_ta / self.hydro_VTV.wdsrf_bsn_ta, 0.0)

            if self.landhru.numhru > 0:
                self.hydro_VTV.wdsrf_hru_ta /= deltime
                self.hydro_fluxes.momen_hru_ta /= deltime

                self.hydro_VTV.veloc_hru_ta = np.where(self.hydro_VTV.wdsrf_hru_ta > 0, self.hydro_fluxes.momen_hru_ta / self.hydro_VTV.wdsrf_hru_ta, 0.0)

            # Update surface water depth on patches
            for i in range(self.landhru.numhru):
                ps = self.landpatch.elm_patch.substt[i]
                pe = self.landpatch.elm_patch.subend[i]
                wdsrf[ps:pe+1] = self.hydro_VTV.wdsrf_hru[i] * 1.0e3  # m to mm

            if self.landpatch.numpatch > 0:
                self.hydro_fluxes.xwsur[:] = (wdsrf_p[:] - wdsrf[:]) / deltime

            # (3) Subsurface lateral flow.
            subsurface_flow(deltime)

            if self.landpatch.numpatch > 0:
                rnof[:] = rsur[:] + rsub[:]

            for i in range(self.landpatch.numpatch):
                self.vtv.h2osoi[:, i] = (self.vtv.wliq_soisno[:, i] / (self.vtv.dz_soi[:] * self.const_pyhsical.denh2o) + 
                                self.vtv.wice_soisno[:, i] / (self.vtv.dz_soi[:] * self.const_pyhsical.denice))
                self.vtv.wat[i] = np.sum(self.vtv.wice_soisno[:, i] + self.vtv.wliq_soisno[:, i]) + self.ldew[i] + self.vtv.scv[i] + self.wetwat[i]

        if self.nl_colm['RangeCheck']:
            if self.mpi.p_is_worker and self.mpi.p_iam_worker == 0:
                print('\nChecking Lateral Flow Variables ...')
                print(f'River Lake Flow average timestep: {dt_average/self.nsubstep:.5f} seconds')

            CoLM_RangeCheck.check_vector_data('Basin Water Depth   [m]  ', self.hydro_VTV.wdsrf_bsn, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('River Velocity      [m/s]', self.hydro_VTV.wdsrf_bsn, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('HRU Water Depth     [m]  ', self.hydro_VTV.wdsrf_hru, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('HRU Water Velocity  [m/s]', self.hydro_VTV.veloc_hru, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Subsurface bt basin [m/s]', self.hydro_fluxes.xsubs_bsn, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Subsurface bt HRU   [m/s]', self.hydro_fluxes.xsubs_hru, self.mpi, self.nl_colm)
            CoLM_RangeCheck.check_vector_data('Subsurface bt patch [m/s]', self.hydro_fluxes.xsubs_pch, self.mpi, self.nl_colm)

        if self.nl_colm['CoLMDEBUG']:
            if self.mpi.p_is_worker:
                dtolw = 0
                toldis = 0

                if self.landpatch.numpatch > 0:
                    dtolw = np.sum(self.patcharea * self.hydro_fluxes.xwsur) / 1.0e3 * deltime

                if self.mesh.numelm > 0:
                    toldis = np.sum(self.hydro_fluxes.discharge * deltime, where=(self.catch_riverlakenetwork.riverdown == 0) | (self.catch_riverlakenetwork.riverdown == -3))
                    dtolw -= toldis

                if self.mpi.p_iam_worker == 0:
                    print(f'Total surface water error: {dtolw:.5f} (m^3) in area {self.landarea:.3e} (m^2), '
                        f'self.hydro_fluxes.discharge {toldis:.3e} (m^3)')

                dtolw = 0
                if self.landpatch.numpatch > 0:
                    dtolw = np.sum(self.patcharea * self.hydro_fluxes.xwsub) / 1.0e3 * deltime

                if self.mpi.p_iam_worker == 0:
                    print(f'Total ground water error: {dtolw:.5f} (m^3) in area {self.landarea:.3e} (m^2)')

    def lateral_flow_final(self):
        """
        Subroutine to finalize lateral flow components.
        """

        # Call the finalization routines for various network components
        self..hillslope_network_final()
        self.catch_riverlakenetwork.river_lake_network_final()
        
        basin_neighbour_final()

        # Debugging block to deallocate patcharea if allocated
        if self.nl_colm['CoLMDEBUG']:
            if self.patcharea is not None:
                del self.patcharea
