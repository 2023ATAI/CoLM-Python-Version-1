import sys
import os

import numpy as np

import config
import time
from CoLM_SPMD_Task import CoLM_SPMD_Task
from CoLM_Namelist import CoLM_Namelist
from CoLM_Vars_Global import CoLM_Vars_Global
from CoLM_SingleSrfdata import CoLM_SingleSrfdata
import CoLM_TimeManager
from CoLM_Const_LC import CoLM_Const_LC
from CoLM_Block import Block_type
from CoLM_Pixel import Pixel_type
import CoLM_SrfdataRestart
import CoLM_LandElm
from CoLM_Mesh import Mesh
from CoLM_LandHRU import CoLM_LandHRU
from CoLM_LandPatch import CoLM_LandPatch
from CoLM_LandPFT import CoLM_LandPFT
from CoLM_ElmVector import CoLM_ElmVector
from CoLM_LandCrop import CoLM_LandCrop
from CoLM_HruVector import CoLM_HruVector
from CoLM_SnowSnicar import CoLM_SnowSnicar
import CoLM_Initialize


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
        casename = namelist.nl_colm['DEF_CASE_NAME']
        dir_landdata = os.path.join(path_dataset, nl_colm['DEF_dir_landdata'])
        dir_restart = nl_colm['DEF_dir_restart']
        greenwich = nl_colm['DEF_simulation_time%greenwich']
        s_year = nl_colm['DEF_simulation_time%start_year']
        s_month = nl_colm['DEF_simulation_time%start_month']
        s_day = nl_colm['DEF_simulation_time%start_day']
        s_seconds = nl_colm['DEF_simulation_time%start_sec']

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
        s_julian = CoLM_TimeManager.monthday2julian(s_year, s_month, s_day)
        idate[0] = s_year
        idate[1] = s_julian
        idate[2] = s_seconds
        CoLM_TimeManager.adj2begin(idate)  # update

        if nl_colm['LULCC']:
            lc_year = idate[0]
        else:
            lc_year = nl_colm['DEF_LC_YEAR']

        var_global = CoLM_Vars_Global(namelist)
        const_lc = CoLM_Const_LC(var_global.N_land_classification, nl_colm, var_global.nl_soil,
                                 var_global.z_soih)

        gblock = Block_type(nl_colm, mpi, srfdata.SITE_lon_location, srfdata.SITE_lat_location)
        pixel = Pixel_type(mpi, nl_colm['USEMPI'])
        pixel.load_from_file(dir_landdata)
        gblock.load_from_file(dir_landdata)

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

        landpatch = CoLM_LandPatch(nl_colm, mpi, gblock, mesh, const_lc)
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

        # if nl_colm['URBAN_MODEL']:
        #     CoLM_SrfdataRestart.pixelset_load_from_file(dir_landdata, 'landurban', landurban, numurban, lc_year)
        #     map_patch_to_urban
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

        # Read in SNICAR optical and aging parameters
        ssnicar = CoLM_SnowSnicar(nl_colm, mpi)
        ssnicar.snowOptics_init(nl_colm['DEF_file_snowoptics'])  # SNICAR optical parameters
        ssnicar.snowAge_init(nl_colm['DEF_file_snowaging'])  # SNICAR aging parameters

        CoLM_Initialize.initialize(mpi, landpatch, landpft, nl_colm,casename, dir_landdata, dir_restart, idate, lc_year,
                                   greenwich, const_lc, var_global, srfdata, gblock, pixel, mesh)

        if nl_colm['SinglePoint']:
            single_Data = CoLM_SingleSrfdata(nl_colm, var_global, mpi)
            single_Data.single_srfdata_final(True)

        # ifdef USEMPI
        # mpi_barrier(p_comm_glb, p_err)

        if mpi.p_is_master:
            end_time = time.time()
            time_used = end_time - start_time

            if time_used >= 3600:
                hours = int(time_used // 3600)
                minutes = int((time_used % 3600) // 60)
                seconds = int(time_used % 60)
                print(f"\nOverall system time used: {hours} hours {minutes} minutes {seconds} seconds.")
            elif time_used >= 60:
                minutes = int(time_used // 60)
                seconds = int(time_used % 60)
                print(f"\nOverall system time used: {minutes} minutes {seconds} seconds.")
            else:
                seconds = int(time_used)
                print(f"\nOverall system time used: {seconds} seconds.")

            print("CoLM Initialization Execution Completed")
    except KeyError as e:
        print('exception:', e)


if __name__ == '__main__':
    path_root = '/home/liqingliang/ATAI/CoLM/code/code/dataset'  # 来自配置文件
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
