from enum import Enum
import time
import sys
import os
import config
# sys.path.append('C:\\Users\\zjl\\Desktop\\temp')

from CoLM_SPMD_Task import CoLM_SPMD_Task
# from CoLM_Grid import gblock
from CoLM_Namelist import  CoLM_Namelist
from CoLM_Vars_Global import CoLM_Vars_Global
from CoLM_Const_LC import CoLM_Const_LC
# from CoLM_Pixel import Pixel_type
from CoLM_Block import Block_type
# from CoLM_LandElm import CoLM_LandElm
from CoLM_Mesh import Mesh
from CoLM_MeshFilter import MeshFilter
from CoLM_SingleSrfdata import CoLM_SingleSrfdata
from CoLM_Grid import  Grid_type

from CoLM_LandElm import CoLM_LandElm
from CoLM_LandPatch import CoLM_LandPatch
# from . import CoLM_SrfdataRestart
from CoLM_LandHRU import CoLM_LandHRU
from CoLM_LandUrban import CoLM_LandUrban
from CoLM_LandCrop import CoLM_LandCrop
from CoLM_SrfdataDiag import CoLm_SrfdataDiag

class StateTax(Enum):
    GridTypeNone      = 1
    GridTypeBar       = 2
    GridTypeBeat      = 3
    GridTypeBeatDiv2  = 4
    GridTypeBeatDiv4  = 5
    GridTypeBeatDiv8  = 6
    GridTypeBeatDiv16 = 7
    GridTypeBeatDiv32 = 8
    GridTypeBeatDiv3  = 9  # Triplet eighths
    GridTypeBeatDiv6  = 10
    GridTypeBeatDiv12 = 11
    GridTypeBeatDiv24 = 12
    GridTypeBeatDiv5  = 13 # Quintuplet eighths
    GridTypeBeatDiv10 = 14 # GridTypeBeatDiv20 = 1
    GridTypeBeatDiv7  = 15 # Septuplet eighths
    GridTypeBeatDiv14 = 16
    GridTypeBeatDiv28 = 17
    GridTypeTimecode  = 18
    GridTypeMinSec    = 19
    GridTypeCDFrame   = 20

plat_system ='win'

class Mksrfdata(object):
    def __init__(self,nlfile,info_define,path_dataset) -> None:
        #并行计算环境初始化
        mpi = CoLM_SPMD_Task(info_define['USEMPI'])

        if mpi.p_is_master:
            # 开始计时
            self.start = time.perf_counter()

        namelist = CoLM_Namelist(nlfile, info_define, mpi)

        var_global = CoLM_Vars_Global(namelist)
        srfdata = CoLM_SingleSrfdata(namelist.nl_colm,var_global,path_dataset,mpi)
        if namelist.nl_colm['SinglePoint']:
            if not namelist.nl_colm['URBAN_MODEL']:
                srfdata.read_surface_data_single(mksrfdata=True)
            else:
                srfdata.read_urban_surface_data_single ( mksrfdata=True)

        if namelist.nl_colm['USE_srfdata_from_larger_region']:
            pass
            # CALL srfdata_region_clip (DEF_dir_existing_srfdata, DEF_dir_landdata)
            if namelist.nl_colm['USEMPI']:
                pass
                # CALL mpi_barrier (p_comm_glb, p_err)
                # CALL spmd_exit
        if namelist.nl_colm['USE_srfdata_from_3D_gridded_data']:
            # TODO
            # CALL srfdata_retrieve_from_3D_data (DEF_dir_existing_srfdata, DEF_dir_landdata)
            pass
            if namelist.nl_colm['USEMPI']:
                pass
                # CALL mpi_barrier (p_comm_glb, p_err)
                # CALL spmd_exit
            
        dir_rawdata  = namelist.nl_colm['DEF_dir_rawdata']
        dir_landdata = namelist.nl_colm['DEF_dir_landdata']
        lc_year = namelist.nl_colm['DEF_LC_YEAR']

        gblock = Block_type(namelist.nl_colm,mpi,srfdata.SITE_lon_location,srfdata.SITE_lat_location)
        
        const_LC = CoLM_Const_LC(var_global.N_land_classification,namelist.nl_colm,var_global.nl_soil,var_global.z_soih)

        # ...........................................................................
        # 1. Read in or create the modeling grids coordinates and related information
        # ...........................................................................
        gblock.assimilate_gblock ()
        mesh = Mesh(namelist.nl_colm, gblock, mpi)
        # define grid coordinates of mesh
        if namelist.nl_colm['GRIDBASED']:
            mesh.init_gridbased_mesh_grid ()

        if namelist.nl_colm['CATCHMENT']:
            mesh.gridmesh.define_by_name('merit_90m')

        if namelist.nl_colm['UNSTRUCTURED']:
            mesh.gridmesh.define_from_file(namelist.nl_colm['DEF_file_mesh'])
        # print('1')
        # define grid coordinates of mesh filter
        meshfilter = MeshFilter(namelist.nl_colm,gblock,mpi)
        has_mesh_filter = meshfilter.inquire_mesh_filter ()
        if has_mesh_filter:
            print('test.py---138---no finishing')
            meshfilter.grid_filter.define_from_file (namelist.nl_colm['DEF_file_mesh_filter'])
        # print('1')
        # define grid coordinates of hydro units in catchment
        if namelist.nl_colm['CATCHMENT']:
            self.ghru = Grid_type(namelist.nl_colm, gblock, mpi)
            self.ghru.define_by_name('merit_90m')
        if namelist.nl_colm['LULC_USGS']:
            self.gpatch = Grid_type(namelist.nl_colm, gblock, mpi)
            self.gpatch.define_by_name('colm_1km')
        if namelist.nl_colm['LULC_IGBP']:
            self.gpatch = Grid_type(namelist.nl_colm, gblock, mpi)
            self.gpatch.define_by_name('colm_500m')
        if namelist.nl_colm['LULC_IGBP_PFT'] or namelist.nl_colm['LULC_IGBP_PC']:
            self.gpatch = Grid_type(namelist.nl_colm, gblock, mpi)
            self.gpatch.define_by_name('colm_500m')

        #endif
        if namelist.nl_colm[ 'CROP']:
        # define grid for crop parameters
            self.gcrop = Grid_type (namelist.nl_colm, gblock, mpi)
            path = os.path.join(namelist.nl_colm['DEF_dir_rawdata'],'global_CFT_surface_data.nc')
            if 'win' in plat_system:
                path = namelist.nl_colm['DEF_dir_rawdata'] + '\\global_CFT_surface_data.nc'
            self.gcrop.define_from_file(path, 'lat', 'lon')

        # define grid for soil parameters raw data
        gsoil = Grid_type(namelist.nl_colm,gblock, mpi)
        gsoil.define_by_name('colm_500m')

        # define grid for LAI raw data
        gridlai = Grid_type(namelist.nl_colm,gblock, mpi)
        gridlai.define_by_name('colm_500m')

        # define grid for topography
        gtopo = Grid_type(namelist.nl_colm,gblock, mpi)
        gtopo.define_by_name('colm_500m')

        # add by dong, only test for making urban data
        if namelist.nl_colm['URBAN_MODEL']:
            self.gurban =  Grid_type(namelist.nl_colm,gblock, mpi)
            self.gurban.define_by_name('colm_500m')
            self.grid_urban_5km = Grid_type(namelist.nl_colm,gblock, mpi)
            self.grid_urban_5km.define_by_name('colm_5km')
            self.grid_urban_500m = Grid_type(namelist.nl_colm,gblock, mpi)
            self.grid_urban_500m.define_by_name('colm_500m')

        # assimilate grids to build pixels
        if not namelist.nl_colm['SinglePoint']:
            mesh.gridmesh.assimilate_grid ()
        
        if has_mesh_filter:
            meshfilter.grid_filter.assimilate_grid ()

        if namelist.nl_colm['CATCHMENT']:
            self.ghru.assimilate_grid ()
            
        self.gpatch.assimilate_grid ()
        gsoil.assimilate_grid ()
        gridlai.assimilate_grid ()
        if namelist.nl_colm['URBAN_MODEL']:
            self.gurban.assimilate_grid()
            self.grid_urban_500m.assimilate_grid()
            self.grid_urban_5km.assimilate_grid()

        if namelist.nl_colm['CROP']:
            self.gcrop.assimilate_grid ()

        gtopo.assimilate_grid ()

        #  map pixels to grid coordinates
        if not namelist.nl_colm['SinglePoint']:
            mesh.gridmesh.map_to_grid ()

        if has_mesh_filter:
            meshfilter.grid_filter.map_to_grid ()

        if namelist.nl_colm['CATCHMENT']:
            self.ghru.map_to_grid()
        
        self.gpatch.map_to_grid()
        
        gsoil.map_to_grid()

        if namelist.nl_colm['URBAN_MODEL']:
            self.gurban.map_to_grid()
            self.grid_urban_500m.map_to_grid()
            self.grid_urban_5km.map_to_grid()

        if namelist.nl_colm['CROP']:
            self.gcrop.map_to_grid()

        gtopo.map_to_grid()

        # build land elms
        mesh. mesh_build (srfdata)
        landlm = CoLM_LandElm(namelist.nl_colm['USEMPI'], mpi,gblock,mesh) 

        if namelist.nl_colm['GRIDBASED']:
            if not mesh.read_mesh_from_file:
            #  TODO: distinguish USGS and IGBP land cover
                if not namelist.nl_colm['LULC_USGS']:
                    # pass
                    path_mesh = os.path.join(path_dataset ,namelist.nl_colm['DEF_dir_rawdata'] , 'landtype_update.nc')
    
                    if 'win' in plat_system:
                        path_mesh = path_dataset +namelist.nl_colm['DEF_dir_rawdata'] + '\\landtype_update.nc'
                    meshfilter. mesh_filter (landlm, mesh, self.gpatch, path_mesh, 'landtype')
                else:
                    path_mesh = os.path.join(path_dataset ,namelist.nl_colm['DEF_dir_rawdata'] , 'landtypes/landtype-usgs-update.nc')
    
                    if 'win' in plat_system:
                        path_mesh = path_dataset +namelist.nl_colm['DEF_dir_rawdata'] + '\\landtypes/landtype-usgs-update.nc'
                    # pass
                    meshfilter. mesh_filter (landlm, mesh, self.gpatch, path_mesh, 'landtype')

        # Filtering pixels
        if has_mesh_filter:
            pass
            # meshfilter. mesh_filter (landlm , mesh, meshfilter.grid_filter, namelist.nl_colm['DEF_file_mesh_filter'], 'mesh_filter')
        landhru = None
        if namelist.nl_colm['CATCHMENT']:
            landhru = CoLM_LandHRU(namelist.nl_colm['USEMPI'], mpi, gblock, mesh, landlm)

        #  build land patches
        landpatch = CoLM_LandPatch(namelist.nl_colm, mpi, gblock, mesh, const_LC)
        landpatch.landpatch_build(lc_year,srfdata.SITE_landtype, var_global, landlm, landhru,self.gpatch)
        landurban = None
        if namelist.nl_colm['URBAN_MODEL']:
            landurban = CoLM_LandUrban(mpi, namelist.nl_colm, gblock, landpatch,var_global, mesh)
            landurban.landurban_build(lc_year, landpatch,landlm,landhru)
        crop = None
        if namelist.nl_colm['CROP']:
            crop = CoLM_LandCrop(mpi, namelist.nl_colm, gblock, landpatch,var_global, srfdata, mesh)
            crop. landcrop_build (lc_year, landlm,landhru, self.gpatch, self.gcrop)

        # #if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
        # CALL landpft_build(lc_year)
        # #endif


        # ................................................................
        # 2. SAVE land surface tessellation information
        # ................................................................
        path_landdata = os.path.join(path_dataset ,dir_landdata)
        if 'win' in plat_system:
            path_landdata = path_dataset + '\\'+ dir_landdata
        
        if not os.path.exists(path_landdata):
            os.makedirs(path_landdata)        

        gblock.save_to_file    (path_landdata)

        # pixel.save_to_file     (dir_landdata)

        # SrfdataRestart.mesh_save_to_file      (dir_landdata, lc_year)

        # SrfdataRestart = CoLM_SrfdataRestart.MOD_SrfdataRestart()

        # SrfdataRestart. pixelset_save_to_file  (dir_landdata, 'landelm'  , landelm  , lc_year)

        # SrfdataRestart. pixelset_save_to_file  (dir_landdata, 'landpatch', landpatch, lc_year)

        # SrfdataRestart. pixelset_save_to_file  (dir_landdata, 'landpft'  , landpft  , lc_year)

        # # ................................................................
        # # 3. Mapping land characteristic parameters to the model grids
        # # ................................................................
        #ifdef SrfdataDiag
        #if (defined CROP)
        #     CALL elm_patch%build (landelm, landpatch, use_frac = .true., sharedfrac = pctshrpch)
        # #else
        #     CALL elm_patch%build (landelm, landpatch, use_frac = .true.)
        # #endif
        # #ifdef GRIDBASED
        #     CALL gdiag%define_by_copy (gridmesh)
        # #else
        #     CALL gdiag%define_by_ndims(3600,1800)
        # #endif
        sdd = CoLm_SrfdataDiag(namelist.nl_colm, mpi, gblock)

        # sdd.srfdata_diag_init (dir_landdata,landpatch, landlm, elm_patch,landpft,landurban,landhru,crop.pctshrpch,var_global.N_land_classification)
        #endif

        # #TODO: for lulcc, need to run for each year and SAVE to different subdirs

    #    CALL Aggregation_PercentagesPFT  (gpatch , dir_rawdata, dir_landdata, lc_year)

        # CALL Aggregation_LakeDepth       (gpatch , dir_rawdata, dir_landdata, lc_year)

        # CALL Aggregation_SoilParameters  (gsoil,   dir_rawdata, dir_landdata, lc_year)

        # CALL Aggregation_SoilBrightness  (gpatch , dir_rawdata, dir_landdata, lc_year)

        # IF (DEF_USE_BEDROCK) THEN
        #     CALL Aggregation_DBedrock     (gpatch , dir_rawdata, dir_landdata)
        # ENDIF

        # CALL Aggregation_LAI             (gridlai, dir_rawdata, dir_landdata, lc_year)

        # CALL Aggregation_ForestHeight    (gpatch , dir_rawdata, dir_landdata, lc_year)

        # CALL Aggregation_Topography      (gtopo  , dir_rawdata, dir_landdata, lc_year)

        # #ifdef URBAN_MODEL
        # CALL Aggregation_urban (dir_rawdata, dir_landdata, lc_year, &
        #                         grid_urban_5km, grid_urban_500m)
        # #endif

        # ! ................................................................
        # ! 4. Free memories.
        # ! ................................................................

        # #ifdef SinglePoint
        # #if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
        # CALL write_surface_data_single (numpatch, numpft)
        # #else
        # #ifndef URBAN_MODEL
        # CALL write_surface_data_single (numpatch)
        # #else
        # CALL write_urban_surface_data_single (numurban)
        # #endif
        # #endif
        # CALL single_srfdata_final ()
        # #endif

        # #ifdef USEMPI
        # CALL mpi_barrier (p_comm_glb, p_err)
        # #endif

        # IF (p_is_master) THEN
        #     CALL system_clock (end_time, count_rate = c_per_sec)
        #     time_used = (end_time - start_time) / c_per_sec
        #     IF (time_used >= 3600) THEN
        #         write(*,101) time_used/3600, mod(time_used,3600)/60, mod(time_used,60)
        #         101 format (/, 'Overall system time used:', I4, ' hours', I3, ' minutes', I3, ' seconds.')
        #     ELSEIF (time_used >= 60) THEN
        #         write(*,102) time_used/60, mod(time_used,60)
        #         102 format (/, 'Overall system time used:', I3, ' minutes', I3, ' seconds.')
        #     ELSE
        #         write(*,103) time_used
        #         103 format (/, 'Overall system time used:', I3, ' seconds.')
        #     ENDIF

        #     write(*,*)  'Successful in surface data making.'
        # ENDIF

        # #ifdef USEMPI
        # CALL spmd_exit
        # #endif

if __name__ == '__main__':
    path_root = 'C:\\Users\\zjl\\Desktop\\code\\dataset'#来自配置文件
    path_dataset = 'D:\\database\\landmodel'#来自配置文件

    path_define = os.path.join(path_root ,'__base__/define.yml')
    
    plat_system = sys.platform
    if 'win' in plat_system:
        path_define = path_root + '\\'+ '__base__\\define.yml'
    info_define = config.parse_from_yaml(path_define)
    
    # make_srfdata('C:\\Users\\zjl\\Desktop\\temp\\dataset\\SinglePoint.yml')
    if info_define['SinglePoint']:
        info_define.update({'USEMPI':False})

    if info_define['CATCHMENT']:
        info_define.update({'LATERAL_FLOW':False})

    if info_define['LULC_IGBP_PFT']:
        info_define.update({'BGC':False})

    if info_define['BGC']:
        info_define.update({'CROP':False})
    
    make_srfdata = Mksrfdata(path_root,info_define,path_dataset)