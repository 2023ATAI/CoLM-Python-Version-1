"""
Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
software         : Initialization of Land Characteristic Parameters and Initial State Variables
Reference        : [1] Dai et al., 2003: The Common Land Model (CoLM). Bull. of Amer. Meter. Soc., 84: 1013-1023
                   [2] Dai et al., 2004: A two-big-leaf model for canopy temperature, photosynthesis
                   and stomatal conductance. Journal of Climate
                   [3] Dai et al., 2014: The Terrestrial Modeling System (TMS).
Args:
    patchtype
    lc_year
Returns:
    patchtype
"""
import numpy as np
import time
from CoLM_TimeManager import TimeManager


class Initialdata(object):
    def __init__(self,namelist,var_global,const_LC,pixel,gblock,mesh,SrfdataRestart,landelm,landpatch) -> None:

        casename = namelist.nl_colm['DEF_CASE_NAME']
        dir_landdata = namelist.nl_colm['DEF_dir_landdata']
        dir_restart = namelist.nl_colm['DEF_dir_restart']
        greenwich = namelist.nl_colm['DEF_simulation_time'].greenwich
        s_year = namelist.nl_colm['DEF_simulation_time'].start_year
        s_month = namelist.nl_colm['DEF_simulation_time'].start_month
        s_day = namelist.nl_colm['DEF_simulation_time'].start_day
        s_seconds = namelist.nl_colm['DEF_simulation_time'].start_sec

        if namelist.nl_colm['SinglePoint']:
            fsrfdata = dir_landdata.strip() + '/srfdata.nc'

        timemanager = TimeManager()
        s_julian = timemanager.monthday2julian(s_year, s_month, s_day)
        idate = [s_year, s_julian, s_seconds]
        idate = timemanager.adj2begin(idate)

        lc_year = namelist.nl_colm['DEF_LC_YEAR']

        # const_PFT= CoLM_Const_PFT()

        pixel.load_from_file(dir_landdata)
        gblock.load_from_file(dir_landdata)
        mesh.mesh_load_from_file (dir_landdata, lc_year)
        SrfdataRestart.pixelset_load_from_file(dir_landdata, 'landelm', landelm, numelm, lc_year)
        SrfdataRestart.pixelset_load_from_file(dir_landdata, 'landpatch', landpatch, numpatch, lc_year)

        initialize(casename, dir_landdata, dir_restart, idate, lc_year, greenwich)

        if namelist.nl_colm['SinglePoint']:
            print('single_srfdata_final ()')

        if self.mpi.p_is_master:
            end_time = time.time()
            time_used = end_time - self.start

            if time_used >= 3600:
                hours = int(time_used // 3600)
                minutes = int((time_used % 3600) // 60)
                seconds = int(time_used % 60)
                print(f"Overall system time used: {hours} hours {minutes} minutes {seconds} seconds.")
            elif time_used >= 60:
                minutes = int(time_used // 60)
                seconds = int(time_used % 60)
                print(f"Overall system time used: {minutes} minutes {seconds} seconds.")
            else:
                print(f"Overall system time used: {time_used} seconds.")

            print('CoLM Initialization Execution Completed')

