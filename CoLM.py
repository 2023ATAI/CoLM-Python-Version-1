import numpy as np
import sys
import os
import time
import config
import copy
from CoLM_SPMD_Task import CoLM_SPMD_Task
from CoLM_Namelist import CoLM_Namelist
from CoLM_Vars_Global import CoLM_Vars_Global
from CoLM_SingleSrfdata import CoLM_SingleSrfdata
import CoLM_TimeManager
from CoLM_Const_LC import CoLM_Const_LC
from CoLM_Block import Block_type
from CoLM_Pixel import Pixel_type
import CoLM_LandElm
from CoLM_Mesh import Mesh
from CoLM_LandHRU import CoLM_LandHRU
from CoLM_LandPatch import CoLM_LandPatch
from CoLM_LandPFT import CoLM_LandPFT
from CoLM_LandCrop import CoLM_LandCrop
from CoLM_ElmVector import CoLM_ElmVector
import CoLM_SrfdataRestart
from CoLM_SnowSnicar import CoLM_SnowSnicar
from CoLM_LAIReadin import LAI_readin
from CoLM_Hydro_SoilWater import Hydro_SoilWater
from CoLM_Vars_TimeInvariants import Vars_TimeInvariants
from CoLM_Vars_TimeVariables import Vars_TimeVariables
from CoLM_Vars_1DForcing import CoLM_Vars_1DForcing
from CoLM_Vars_2DForcing import CoLM_Vars_2DForcing
from CoLM_Forcing import Forcing
from CoLM_Driver import CoLMDRIVER
from CoLM_Const_PFT import Const_PFT
import CoLM_Hist
from CoLM_Vars_1DFluxes import CoLM_Vars_1DFluxes
# from CoLM_Catch_LateralFlow import CoLM_Catch_LateralFlow
from CoLM_Aerosol import CoLM_Aerosol
from CoLM_NitrifData import CoLM_NitrifData
from CoLM_NdepData import CoLM_NdepData
from CoLM_Vars_1DAccFluxes import CoLM_Vars_1DAccFluxes
from CoLM_HistGridded import CoLM_HistGridded
from CoLM_Const_Physical import CoLM_Const_Physical


def inidata(nlfile, info_define, path_dataset, plat_system='win'):
    start_time = 0
    lc_year = 0
    idate = np.zeros(3, dtype=int)
    try:
        # ifdef USEMPI
        #     CALL
        #     spmd_init()
        # endif
        mpi = CoLM_SPMD_Task(info_define['USEMPI'])

        if mpi.p_is_master:
            start_time = time.time()

        namelist = CoLM_Namelist(nlfile, info_define, mpi)
        nl_colm = namelist.nl_colm
        nl_colm_forcing = namelist.nl_colm_forcing
        nl_colm_history = namelist.nl_colm_history
        casename = namelist.nl_colm['DEF_CASE_NAME']
        dir_landdata = os.path.join(path_dataset, nl_colm['DEF_dir_landdata'])
        dir_restart = nl_colm['DEF_dir_restart']
        dir_forcing = nl_colm['DEF_dir_forcing']
        dir_hist = nl_colm['DEF_dir_history']
        dir_restart = nl_colm['DEF_dir_restart']

        var_global = CoLM_Vars_Global(namelist)

        srfdata = CoLM_SingleSrfdata(nl_colm, var_global, mpi)

        fsrfdata = os.path.join(path_dataset, 'srfdata.nc')
        if 'win' in plat_system:
            fsrfdata = path_dataset + '\\' + '/srfdata.nc'
        if nl_colm['SinglePoint']:
            if not nl_colm['URBAN_MODEL']:
                srfdata.read_surface_data_single(fsrfdata, mksrfdata=False)
            else:
                srfdata.read_urban_surface_data_single(fsrfdata, mksrfdata=False, mkrun=True)

        ## Assigning variables from DEF_simulation_time structure
        deltim = nl_colm['DEF_simulation_time%timestep']
        greenwich = nl_colm['DEF_simulation_time%greenwich']
        s_year = nl_colm['DEF_simulation_time%start_year']
        s_month = nl_colm['DEF_simulation_time%start_month']
        s_day = nl_colm['DEF_simulation_time%start_day']
        s_seconds = nl_colm['DEF_simulation_time%start_sec']
        e_year = nl_colm['DEF_simulation_time%end_year']
        e_month = nl_colm['DEF_simulation_time%end_month']
        e_day = nl_colm['DEF_simulation_time%end_day']
        e_seconds = nl_colm['DEF_simulation_time%end_sec']
        p_year = nl_colm['DEF_simulation_time%spinup_year']
        p_month = nl_colm['DEF_simulation_time%spinup_month']
        p_day = nl_colm['DEF_simulation_time%spinup_day']
        p_seconds = nl_colm['DEF_simulation_time%spinup_sec']
        spinup_repeat = nl_colm['DEF_simulation_time%spinup_repeat']

        isgreenwich = CoLM_TimeManager.initimetype(greenwich, nl_colm['SinglePoint'])

        s_julian = CoLM_TimeManager.monthday2julian(s_year, s_month, s_day)
        e_julian = CoLM_TimeManager.monthday2julian(e_year, e_month, e_day)
        p_julian = CoLM_TimeManager.monthday2julian(p_year, p_month, p_day)

        # Assigning values to date arrays
        sdate = CoLM_TimeManager.Timestamp(s_year, s_julian, s_seconds)
        edate = CoLM_TimeManager.Timestamp(e_year, e_julian, e_seconds)
        pdate = CoLM_TimeManager.Timestamp(p_year, p_julian, p_seconds)

        soilwater = Hydro_SoilWater(nl_colm)

        # Calling initialization functions

        const_lc = CoLM_Const_LC(var_global.N_land_classification, nl_colm, var_global.nl_soil,
                                 var_global.z_soih)

        const_physical = CoLM_Const_Physical()

        # constpft = Const_PFT(nl_colm, var_global.N_PFT, var_global.N_CFT, var_global.nl_soil)
        #
        # constpft.Init_PFT_Const(var_global.zi_soi)

        gblock = Block_type(nl_colm, mpi, srfdata.SITE_lon_location, srfdata.SITE_lat_location)
        pixel = Pixel_type(mpi, nl_colm['USEMPI'])
        pixel.load_from_file(dir_landdata)
        gblock.load_from_file(dir_landdata)

        # Handling conditional assignment for lc_year
        lc_year = s_year if nl_colm['LULCC'] else nl_colm['DEF_LC_YEAR']

        mesh = Mesh(nl_colm, gblock, mpi)
        mesh = CoLM_SrfdataRestart.mesh_load_from_file(mpi, nl_colm, gblock, mesh, dir_landdata, lc_year)

        landelm = CoLM_LandElm.get_land_elm(nl_colm['USEMPI'], mpi, gblock, mesh)
        landelm, numelm = CoLM_SrfdataRestart.pixelset_load_from_file(mpi, nl_colm, gblock, mesh, dir_landdata,
                                                                      'landelm', landelm, lc_year)

        landhru = None
        if nl_colm['CATCHMENT']:
            landhru = CoLM_LandHRU(nl_colm, mpi, gblock, mesh, landelm, pixel)
            landhru, numhru = CoLM_SrfdataRestart.pixelset_load_from_file(mpi, nl_colm, gblock, mesh, dir_landdata,
                                                                          'landhru', landhru, lc_year)

        landpatch = CoLM_LandPatch(nl_colm, mpi, gblock, pixel, mesh, const_lc)
        landpatch.landpatch, landpatch.numpatch = CoLM_SrfdataRestart.pixelset_load_from_file(mpi, nl_colm, gblock,
                                                                                              mesh, dir_landdata,
                                                                                              'landpatch',
                                                                                              landpatch.landpatch,
                                                                                              lc_year)

        landpft = None
        if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
            landpft = CoLM_LandPFT(nl_colm, mpi, gblock, mesh, const_lc, var_global)
            landpft, numpft = CoLM_SrfdataRestart.pixelset_load_from_file(mpi, nl_colm, gblock, mesh, dir_landdata,
                                                                          'landpft', landpft, lc_year)
            landpft.map_patch_to_pft()

        landcrop = None
        if nl_colm['CROP']:
            landcrop = CoLM_LandCrop(mpi, nl_colm, gblock, landpatch, var_global, srfdata, mesh)

        if nl_colm['UNSTRUCTURED'] or nl_colm['CATCHMENT']:
            ev = CoLM_ElmVector(nl_colm, mpi)
            ev.elm_vector_init(landelm, landpatch, landcrop)
            if nl_colm['CATCHMENT']:
                pass
                # hv = CoLM_HruVector(nl_colm, mpi)
                # hv.hru_vector_init()

        sdate = CoLM_TimeManager.adj2end(sdate)
        edate = CoLM_TimeManager.adj2end(edate)
        pdate = CoLM_TimeManager.adj2end(pdate)

        ststamp = copy.copy(sdate)
        etstamp = copy.copy(edate)
        ptstamp = copy.copy(pdate)

        # date in beginning style
        jdate = CoLM_TimeManager.adj2begin(copy.copy(sdate))

        if ptstamp <= ststamp:
            spinup_repeat = 0
        else:
            spinup_repeat = max(0, spinup_repeat)

        VTI = Vars_TimeInvariants(nl_colm, mpi, landpatch, var_global)
        VTV = Vars_TimeVariables(nl_colm, mpi, landpatch, var_global, gblock)

        VTI.read_time_invariants(lc_year, casename, dir_restart, gblock)

        # Read in the model time varying data (model state variables)
        VTV.read_time_variables(jdate, lc_year, casename, dir_restart)

        # Read in SNICAR optical and aging parameters
        ssnicar = CoLM_SnowSnicar(nl_colm, mpi)
        ssnicar.snowOptics_init(nl_colm['DEF_file_snowoptics'])  # SNICAR optical parameters
        ssnicar.snowAge_init(nl_colm['DEF_file_snowaging'])  # SNICAR aging parameters

        # ----------------------------------------------------------------------
        doalb = True
        dolai = True
        dosst = False

        varforcing = CoLM_Vars_1DForcing(nl_colm, mpi, landpatch.numpatch, numelm)
        varforcing2 = CoLM_Vars_2DForcing(mpi, gblock)
        forcing = Forcing(nl_colm, nl_colm_forcing, gblock, landpatch, mesh, pixel, mpi, var_global)
        # Initialize meteorological forcing data module
        varforcing.allocate_1D_Forcing()
        if nl_colm['CROP']:
            forcing.forcing_init(dir_forcing, deltim, ststamp, lc_year, landelm, varforcing,
                                 VTI.patchtype, numelm, landcrop.pctshrpch, etstamp)
        else:
            forcing.forcing_init(dir_forcing, deltim, ststamp, lc_year, landelm, varforcing,
                                 VTI.patchtype, numelm, None, etstamp)
        varforcing2.allocate_2D_Forcing(forcing.gforc)

        # Initialize history data module
        varaccfluxes = CoLM_Vars_1DAccFluxes(nl_colm, mpi, landpatch, var_global.spval)
        histgrid = None
        if nl_colm['CROP']:
            histgrid = CoLM_Hist.hist_init(nl_colm, mpi, var_global, landpatch.landpatch, landelm, gblock, pixel, mesh, mesh.numelm,
                                           var_global.spval,
                                           forcing.gforc, landcrop.pctshrpch, dir_hist, varaccfluxes)
        else:
            histgrid = CoLM_Hist.hist_init(nl_colm, mpi, var_global, landpatch, landelm, gblock, pixel, mesh, mesh.numelm,
                                           var_global.spval,
                                           forcing.gforc, None, dir_hist, varaccfluxes)

        varfluxes = CoLM_Vars_1DFluxes(landpatch.numpatch,var_global.spval)
        varfluxes.allocate_1D_Fluxes(mpi.p_is_worker)

        aerosol = None

        # Initialize aerosol deposition forcing data
        if nl_colm['DEF_Aerosol_Readin']:
            aerosol = CoLM_Aerosol(nl_colm, mpi, gblock)
            aerosol.AerosolDepInit()

        # catch_laterflow = None
        # Initialize lateral flow if defined
        if nl_colm['CatchLateralFlow']:
            pass
            # catch_laterflow = CoLM_Catch_LateralFlow(nl_colm, mpi, landpatch, mesh, pixel)
            # catch_laterflow.lateral_flow_init(lc_year)

        # ======================================================================
        # begin time stepping loop
        # ======================================================================

        istep = 1
        idate = sdate
        itstamp = ststamp

        while itstamp < etstamp:
            year_p, jday_p = jdate.year, jdate.day
            month_p, mday_p = CoLM_TimeManager.julian2monthday(year_p, jday_p)

            if mpi.p_is_master:
                if itstamp < ptstamp:
                    print(
                        f"TIMESTEP = {istep} | DATE = {year_p}-{month_p}-{mday_p}-{jdate.sec} Spinup ({spinup_repeat} repeat left)")
                else:
                    print(f"TIMESTEP = {istep} | DATE = {year_p}-{month_p}-{mday_p}-{jdate.sec}")

            Julian_1day_p = (CoLM_TimeManager.calendarday_date(jdate,isgreenwich) - 1) // 1 * 1 + 1
            Julian_8day_p = (CoLM_TimeManager.calendarday_date(jdate,isgreenwich) - 1) // 8 * 8 + 1

            # Read in the meteorological forcing

            forcing.read_forcing(jdate, dir_forcing, varforcing, varforcing2, numelm, const_physical.rgas,isgreenwich)

            # Read in aerosol deposition forcing data
            if nl_colm['DEF_Aerosol_Readin']:
                aerosol.AerosolDepReadin(jdate)

            # Calendar for NEXT time step
            idate = CoLM_TimeManager.ticktime(deltim, idate)
            itstamp += int(deltim)
            jdate = idate
            jdate = CoLM_TimeManager.adj2begin(jdate)

            month, mday = CoLM_TimeManager.julian2monthday(jdate.year, jdate.day)

            if nl_colm['BGC']:
                nitrif = CoLM_NitrifData(nl_colm, idate, landpatch, gblock, mpi, var_global.nl_soil, VTI.patchclass)

                if nl_colm['DEF_USE_NITRIF']:
                    if month != month_p:
                        nitrif.update_nitrif_data(month)

                ndepdata = CoLM_NdepData(nl_colm, mpi, landpatch, gblock, VTI.patchclass)

                if nl_colm['DEF_NDEP_FREQUENCY'] == 1:
                    if jdate[0] != year_p:
                        ndepdata.update_ndep_data_annually(idate.year, iswrite=True)
                elif nl_colm['DEF_NDEP_FREQUENCY'] == 2:
                    if jdate[0] != year_p or month != month_p:
                        ndepdata.update_ndep_data_monthly(jdate.year, month, iswrite=True)
                else:
                    print(f"ERROR: DEF_NDEP_FREQUENCY should be only 1-2, Current is: {nl_colm['DEF_NDEP_FREQUENCY']}")
                    # CoLM_stop()

            cama_var = None
            # Call colm driver
            if mpi.p_is_worker:
                CoLMDRIVER(nl_colm, nl_colm_forcing, const_physical, gblock, mpi, VTV, VTI, isgreenwich, landpft, cama_var, idate, deltim, dolai, doalb, dosst, VTV, VTI,
                                     forcing, varforcing, varfluxes, const_lc, landpatch.numpatch, forcing, var_global)

            if nl_colm['CatchLateralFlow']:
                pass
                # catch_laterflow.lateral_flow(deltim)

            if nl_colm['DataAssimilation']:
                pass
                # do_DataAssimilation(idate, deltim)

            # Write out the model variables for restart run and the history file
            CoLM_Hist.hist_out(nl_colm, nl_colm_forcing, nl_colm_history, mpi, const_physical, var_global, mesh.numelm, varaccfluxes, var_global.spval, landpatch, gblock, VTV, VTI, forcing,
                               varfluxes, varforcing, histgrid, var_global.nl_soil, var_global.maxsnl, var_global.nvegwcs,
                               var_global.nl_lake, idate, deltim, itstamp, etstamp, ptstamp, dir_hist, casename, histgrid)

            # Get leaf area index
            if nl_colm['DYN_PHENOLOGY']:
                pass
            else:
                if nl_colm['DEF_LAI_CHANGE_YEARLY']:
                    lai_year = jdate[0]
                else:
                    lai_year = nl_colm['DEF_LC_YEAR']

                if nl_colm['DEF_LAI_MONTHLY']:
                    if itstamp < etstamp and month != month_p:
                        LAI_readin(lai_year, month, dir_landdata)
                        if nl_colm['URBAN_MODEL']:
                            pass
                            # UrbanLAI_readin(lai_year, month, dir_landdata)
                else:
                    Julian_8day = (CoLM_TimeManager.calendarday_date(jdate) - 1) // 8 * 8 + 1
                    if itstamp < etstamp and Julian_8day != Julian_8day_p:
                        LAI_readin(jdate[0], Julian_8day, dir_landdata)
                        # or depend on DEF_LAI_CHANGE_YEARLY namelist
                        # LAI_readin(lai_year, Julian_8day, dir_landdata)

            if VTV.save_to_restart(idate, deltim, itstamp, ptstamp):
                if nl_colm['LULCC']:
                    VTV.WRITE_TimeVariables(jdate, jdate[0], casename, dir_restart)
                else:
                    VTV.WRITE_TimeVariables(jdate, lc_year, casename, dir_restart)

            if nl_colm['RangeCheck']:
                VTV.check_TimeVariables()

            if nl_colm['CoLMDEBUG']:
                soilwater.print_VSF_iteration_stat_info(mpi)

            if mpi.p_is_master:
                end_time = time.time()
                time_used = (end_time - start_time)
                if time_used >= 3600:
                    print(
                        f"Time elapsed: {time_used // 3600} hours {time_used % 3600 // 60} minutes {time_used % 60} seconds.")
                elif time_used >= 60:
                    print(f"Time elapsed: {time_used // 60} minutes {time_used % 60} seconds.")
                else:
                    print(f"Time elapsed: {time_used} seconds.")

            if spinup_repeat > 1 and ptstamp <= itstamp:
                spinup_repeat -= 1
                idate = sdate
                jdate = sdate
                itstamp = ststamp
                jdate = CoLM_TimeManager.adj2begin(jdate)
                forcing.forcing_reset()

            istep += 1
        # End of time loop

        VTI.deallocate_TimeInvariants()
        VTV.deallocate_TimeVariables()
        varforcing.deallocate_1D_Forcing()
        varfluxes.deallocate_1D_Fluxes(mpi.p_is_worker)

        if nl_colm['CatchLateralFlow']:
            pass
            # catch_laterflow.lateral_flow_final()

        forcing.forcing_final()
        CoLM_Hist.hist_final(nl_colm, varaccfluxes)

        if nl_colm['SinglePoint']:
            single_Data = CoLM_SingleSrfdata(nl_colm, var_global, mpi)
            single_Data.single_srfdata_final(False)

        if nl_colm['DataAssimilation']:
            pass
            # final_DataAssimilation()

        if mpi.p_is_master:
            print("CoLM Execution Completed.")

        # Format strings for printing
        print("TIMESTEP = " + str(istep) + "| DATE = " +str(jdate.year)+"-" +str(month_p)+ "-" +str(mday_p)+ "-" +str(jdate.sec)+ str(spinup_repeat) + " repeat left")
        print("TIMESTEP = " + str(istep) + " | DATE = " +str(jdate.year)+ "-" +str(month_p)+"-" +str(mday_p)+ "-" +str(jdate.sec))
        print("Time elapsed: " +str(time_used/3600)+ " hours " +str((time_used%3600)/60)+ " minutes " +str(time_used%60)+ "seconds.")
        print("Time elapsed: " +str(time_used/60)+ " minutes " +str(time_used%60)+ " seconds.")
        print("Time elapsed: " +str(time_used)+ " seconds.")
    except KeyError as e:
        print('exception:', e)


if __name__ == '__main__':
    path_root = '/home/liqingliang/ATAI/CoLM/code/dataset'  # 来自配置文件
    path_data = '/data'

    path_define = os.path.join(path_root, '__base__/define.yml')

    plat_sys = sys.platform
    if 'win' in plat_sys:
        path_define = path_root + '\\' + '__base__\\define.yml'
    file_define = config.parse_from_yaml(path_define)

    # make_srfdata('C:\\Users\\zjl\\Desktop\\temp\\dataset\\SinglePoint.yml')
    if file_define['SinglePoint']:
        file_define.update({'USEMPI': False})

    if not file_define['CATCHMENT']:
        file_define.update({'LATERAL_FLOW': False})
        file_define.update({'CatchLateralFlow': False})

    if file_define['LULC_IGBP_PFT']:
        file_define.update({'BGC': False})

    if file_define['BGC']:
        file_define.update({'CROP': False})

    if file_define['VectorInOneFileP']:
        file_define.update({'VectorInOneFileS': False})
    file_define.update({'MODAL_AER': False})
    inidata(path_root, file_define, path_data, plat_sys)
