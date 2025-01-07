from enum import Enum
import time
import sys
import os
import config
# sys.path.append('C:\\Users\\zjl\\Desktop\\temp')

from CoLM_SPMD_Task import CoLM_SPMD_Task
from CoLM_Namelist import CoLM_Namelist
from CoLM_Vars_Global import CoLM_Vars_Global
from CoLM_Const_LC import CoLM_Const_LC
from CoLM_Block import Block_type
from CoLM_Mesh import Mesh
from CoLM_MeshFilter import MeshFilter
from CoLM_SingleSrfdata import CoLM_SingleSrfdata
from CoLM_Grid import Grid_type
from CoLM_Pixel import Pixel_type
import CoLM_LandElm
from CoLM_LandPatch import CoLM_LandPatch
from CoLM_LandHRU import CoLM_LandHRU
from CoLM_LandUrban import CoLM_LandUrban
from CoLM_LandPFT import CoLM_LandPFT
from CoLM_LandCrop import CoLM_LandCrop
from CoLM_SrfdataDiag import CoLm_SrfdataDiag
import CoLM_SrfdataRestart
import Aggregation_PercentagesPFT
import Aggregation_LakeDepth
import Aggregation_SoilBrightness
import Aggregation_Topography
import Aggregation_SoilParameters
import Aggregation_LAI
import Aggregation_ForestHeight
import netCDF4

class StateTax(Enum):
    GridTypeNone = 1
    GridTypeBar = 2
    GridTypeBeat = 3
    GridTypeBeatDiv2 = 4
    GridTypeBeatDiv4 = 5
    GridTypeBeatDiv8 = 6
    GridTypeBeatDiv16 = 7
    GridTypeBeatDiv32 = 8
    GridTypeBeatDiv3 = 9  # Triplet eighths
    GridTypeBeatDiv6 = 10
    GridTypeBeatDiv12 = 11
    GridTypeBeatDiv24 = 12
    GridTypeBeatDiv5 = 13  # Quintuplet eighths
    GridTypeBeatDiv10 = 14  # GridTypeBeatDiv20 = 1
    GridTypeBeatDiv7 = 15  # Septuplet eighths
    GridTypeBeatDiv14 = 16
    GridTypeBeatDiv28 = 17
    GridTypeTimecode = 18
    GridTypeMinSec = 19
    GridTypeCDFrame = 20


def mksrfdata(nlfile, info_define, path_dataset, plat_system='win'):
    pixel = None
    gblock = None
    mesh = None
    gsoil = None
    gridlai = None
    gtopo = None

    ghru = None
    gpatch = None
    gcrop = None
    gurban = None
    gurban = None
    grid_urban_5km = None
    grid_urban_5km = None
    grid_urban_500m = None
    grid_urban_500m = None
    start_time = 0

    try:
        # 并行计算环境初始化
        mpi = CoLM_SPMD_Task(info_define['USEMPI'])

        if mpi.p_is_master:
            # 开始计时
            start_time = time.time()

        namelist = CoLM_Namelist(nlfile, info_define, mpi)

        var_global = CoLM_Vars_Global(namelist)
        srfdata = CoLM_SingleSrfdata(namelist.nl_colm, var_global, mpi)

        if 'win' in plat_system:
            path_file = path_dataset + '\\' + namelist.nl_colm['SITE_fsrfdata']

        if namelist.nl_colm['SinglePoint']:
            path_file = os.path.join(path_dataset, namelist.nl_colm['SITE_fsrfdata'])
            if not namelist.nl_colm['URBAN_MODEL']:
                srfdata.read_surface_data_single(path_file, mksrfdata=True)
            else:
                srfdata.read_urban_surface_data_single(path_file, mksrfdata=True)

        if namelist.nl_colm['USE_srfdata_from_larger_region']:
            pass
            # CALL srfdata_region_clip (DEF_dir_existing_srfdata, DEF_dir_landdata)
            if namelist.nl_colm['USEMPI']:
                pass
                # CALL mpi_barrier (p_comm_glb, p_err)
                # CALL spmd_exit
        if namelist.nl_colm['USE_srfdata_from_3D_gridded_data']:
            # CALL srfdata_retrieve_from_3D_data (DEF_dir_existing_srfdata, DEF_dir_landdata)
            pass
            if namelist.nl_colm['USEMPI']:
                pass
                # CALL mpi_barrier (p_comm_glb, p_err)
                # CALL spmd_exit

        dir_rawdata = namelist.nl_colm['DEF_dir_rawdata']
        if 'win' in sys.platform:
            path_split = dir_rawdata.split('/')
            path_win = ''
            for p in path_split:
                path_win += "\\" + p
            dir_rawdata = path_dataset + path_win
        else:
            dir_rawdata = os.path.join(path_dataset, namelist.nl_colm['DEF_dir_rawdata'])
        dir_landdata = namelist.nl_colm['DEF_dir_landdata']
        if 'win' in sys.platform:
            path_split = dir_landdata.split('/')
            path_win = ''
            for p in path_split:
                path_win += "\\" + p
            dir_landdata = path_dataset + path_win
        else:
            dir_landdata = os.path.join(path_dataset, namelist.nl_colm['DEF_dir_landdata'])
        lc_year = namelist.nl_colm['DEF_LC_YEAR']

        gblock = Block_type(namelist.nl_colm, mpi, srfdata.SITE_lon_location, srfdata.SITE_lat_location)
        const_lc = CoLM_Const_LC(var_global.N_land_classification, namelist.nl_colm, var_global.nl_soil,
                                 var_global.z_soih)

        edges = namelist.nl_colm['DEF_domain%edges']
        edgen = namelist.nl_colm['DEF_domain%edgen']
        edgew = namelist.nl_colm['DEF_domain%edgew']
        edgee = namelist.nl_colm['DEF_domain%edgee']
        # ...........................................................................
        # 1. Read in or create the modeling grids coordinates and related information
        # ...........................................................................

        # pixel_test = netCDF4.Dataset('/data/Guangdong/Guandong/landdata/pixel.nc', 'r')
        # a = pixel_test['lat_s'][:]
        # b = pixel_test['lat_n'][:]
        # c = pixel_test['lon_w'][:]
        # d = pixel_test['lon_e'][:]

        pixel = Pixel_type(mpi, namelist.nl_colm['USEMPI'])
        pixel.set_edges(edges, edgen, edgew, edgee)

        pixel.assimilate_gblock(gblock)

        mesh = Mesh(namelist.nl_colm, gblock, mpi)

        # define grid coordinates of mesh
        if namelist.nl_colm['GRIDBASED']:
            mesh.init_gridbased_mesh_grid()
        if namelist.nl_colm['CATCHMENT']:
            mesh.gridmesh.define_by_name('merit_90m')
        if namelist.nl_colm['UNSTRUCTURED']:
            mesh.gridmesh.define_from_file(namelist.nl_colm['DEF_file_mesh'])

        # define grid coordinates of mesh filter
        meshfilter = MeshFilter(namelist.nl_colm, gblock, mpi)
        has_mesh_filter = meshfilter.inquire_mesh_filter()

        if has_mesh_filter:
            meshfilter.grid_filter.define_from_file(namelist.nl_colm['DEF_file_mesh_filter'])

        # define grid coordinates of hydro units in catchment
        if namelist.nl_colm['CATCHMENT']:
            ghru = Grid_type(namelist.nl_colm, gblock)
            ghru.define_by_name('merit_90m')
        if namelist.nl_colm['LULC_USGS']:
            gpatch = Grid_type(namelist.nl_colm, gblock)
            gpatch.define_by_name('colm_1km')
        if namelist.nl_colm['LULC_IGBP']:
            gpatch = Grid_type(namelist.nl_colm, gblock)
            gpatch.define_by_name('colm_500m')
        if namelist.nl_colm['LULC_IGBP_PFT'] or namelist.nl_colm['LULC_IGBP_PC']:
            gpatch = Grid_type(namelist.nl_colm, gblock)
            gpatch.define_by_name('colm_500m')

        # endif
        if namelist.nl_colm['CROP']:
            # define grid for crop parameters
            gcrop = Grid_type(namelist.nl_colm, gblock)
            path = os.path.join(dir_rawdata, 'global_CFT_surface_data.nc')
            if 'win' in plat_system:
                path = dir_rawdata + '\\global_CFT_surface_data.nc'
            gcrop.define_from_file(path, 'lat', 'lon')

        # define grid for soil parameters raw data
        gsoil = Grid_type(namelist.nl_colm, gblock)
        gsoil.define_by_name('colm_500m')

        # define grid for LAI raw data
        gridlai = Grid_type(namelist.nl_colm, gblock)
        gridlai.define_by_name('colm_500m')

        # define grid for topography
        gtopo = Grid_type(namelist.nl_colm, gblock)
        gtopo.define_by_name('colm_500m')

        # add by dong, only test for making urban data
        if namelist.nl_colm['URBAN_MODEL']:
            gurban = Grid_type(namelist.nl_colm, gblock)
            gurban.define_by_name('colm_500m')
            grid_urban_5km = Grid_type(namelist.nl_colm, gblock)
            grid_urban_5km.define_by_name('colm_5km')
            grid_urban_500m = Grid_type(namelist.nl_colm, gblock)
            grid_urban_500m.define_by_name('colm_500m')

        # print('---------------1')
        # assimilate grids to build pixels
        if not namelist.nl_colm['SinglePoint']:
            pixel.assimilate_grid(mesh.gridmesh)

        if has_mesh_filter:
            # print('---------------1')
            pixel.assimilate_grid(meshfilter.grid_filter)

        if namelist.nl_colm['CATCHMENT']:
            # print('---------------1')
            pixel.assimilate_grid(ghru)
        # print('---------------2')
        pixel.assimilate_grid(gpatch)
        # print('---------------3')
        pixel.assimilate_grid(gsoil)
        pixel.assimilate_grid(gridlai)
        if namelist.nl_colm['URBAN_MODEL']:
            pixel.assimilate_grid(gurban)
            pixel.assimilate_grid(grid_urban_500m)
            pixel.assimilate_grid(grid_urban_5km)

        if namelist.nl_colm['CROP']:
            pixel.assimilate_grid(gcrop)

        pixel.assimilate_grid(gtopo)

        #  map pixels to grid coordinates
        if not namelist.nl_colm['SinglePoint']:
            pixel.map_to_grid(mesh.gridmesh)

        if has_mesh_filter:
            pixel.map_to_grid(meshfilter.grid_filter)

        if namelist.nl_colm['CATCHMENT']:
            pixel.map_to_grid(ghru)

        gpatch = pixel.map_to_grid(gpatch)
        gsoil = pixel.map_to_grid(gsoil)
        gridlai = pixel.map_to_grid(gridlai)

        if namelist.nl_colm['URBAN_MODEL']:
            gurban = pixel.map_to_grid(gurban)
            grid_urban_500m = pixel.map_to_grid(grid_urban_500m)
            grid_urban_5km = pixel.map_to_grid(grid_urban_5km)

        if namelist.nl_colm['CROP']:
            gcrop = pixel.map_to_grid(gcrop)

        gtopo = pixel.map_to_grid(gtopo)

        # build land elms
        mesh.mesh_build(srfdata, pixel)
        # m0= mesh.mesh[0].ilat
        # for i in range(len(m0)):
        #     print(m0[i],'****************')
        landlm = CoLM_LandElm.get_land_elm(namelist.nl_colm['USEMPI'], mpi, gblock, mesh)

        if namelist.nl_colm['GRIDBASED']:
            if not mesh.read_mesh_from_file:
                if not namelist.nl_colm['LULC_USGS']:
                    path_mesh = os.path.join(dir_rawdata, 'landtype_update.nc')

                    if 'win' in plat_system:
                        path_mesh = dir_rawdata + 'landtype_update.nc'
                    # print('---------------------------------------')
                    landlm, mesh = meshfilter.mesh_filter(landlm, mesh, gpatch, path_mesh, 'landtype', pixel)
                    # return
                else:
                    path_mesh = os.path.join(dir_rawdata, 'landtypes/landtype-usgs-update.nc')

                    if 'win' in plat_system:
                        path_mesh = dir_rawdata + 'landtypes\\landtype-usgs-update.nc'
                    # pass
                    landlm, mesh = meshfilter.mesh_filter(landlm, mesh, gpatch, path_mesh, 'landtype', pixel)

        # Filtering pixels
        if has_mesh_filter:
            landlm, mesh = meshfilter.mesh_filter(landlm, mesh, meshfilter.grid_filter,
                                                  namelist.nl_colm['DEF_file_mesh_filter'],
                                                  'mesh_filter', pixel)

        landhru = None
        if namelist.nl_colm['CATCHMENT']:
            landhru = CoLM_LandHRU(namelist.nl_colm, mpi, gblock, mesh, landlm, pixel)
        # return
        #  build land patches
        landpatch = CoLM_LandPatch(namelist.nl_colm, mpi, gblock, mesh, const_lc)

        landpatch.landpatch_build(lc_year, srfdata.SITE_landtype, var_global, landlm, landhru, gpatch, pixel)
        # for i in landpatch.landpatch.ipxend:
        #     print(i, '------ipxend------------')
        # return
        landurban = None
        if namelist.nl_colm['URBAN_MODEL']:
            landurban = CoLM_LandUrban(mpi, namelist.nl_colm, gblock, landpatch, var_global, mesh)
            landurban.landurban_build(lc_year, landpatch, landlm, landhru)

        crop = None
        if namelist.nl_colm['CROP']:
            crop = CoLM_LandCrop(mpi, namelist.nl_colm, gblock, landpatch, var_global, srfdata, mesh)
            crop.landcrop_build(lc_year, landlm, landhru, gpatch, gcrop)
        landpft = None
        if namelist.nl_colm['LULC_IGBP_PFT'] or namelist.nl_colm['LULC_IGBP_PC']:
            landpft = CoLM_LandPFT(namelist.nl_colm, mpi, gblock, mesh, const_lc, var_global)
            landpft.landpft_build(lc_year, srfdata.SITE_pctpfts, srfdata.SITE_pfttyp, landpatch, landpatch.numpatch,
                                  crop.cropclass,
                                  landpft, gpatch)

        # ................................................................
        # 2. SAVE land surface tessellation information
        # ................................................................
        path_landdata = dir_landdata

        if not os.path.exists(path_landdata):
            os.makedirs(path_landdata)

        gblock.save_to_file(path_landdata)

        pixel.save_to_file(path_landdata)

        CoLM_SrfdataRestart.mesh_save_to_file(mpi, namelist.nl_colm, gblock, mesh, path_landdata, lc_year)

        CoLM_SrfdataRestart.pixelset_save_to_file(mpi, namelist.nl_colm, gblock, mesh, path_landdata, 'landelm',
                                                  landlm, lc_year)
        if namelist.nl_colm['CATCHMENT']:
            CoLM_SrfdataRestart.pixelset_save_to_file(mpi, namelist.nl_colm, gblock, mesh, path_landdata, 'landhru',
                                                      landhru.landhru, lc_year)

        CoLM_SrfdataRestart.pixelset_save_to_file(mpi, namelist.nl_colm, gblock, mesh, path_landdata, 'landpatch',
                                                  landpatch.landpatch, lc_year)

        if namelist.nl_colm['LULC_IGBP_PFT'] or namelist.nl_colm['LULC_IGBP_PC']:
            CoLM_SrfdataRestart.pixelset_save_to_file(mpi, namelist.nl_colm, gblock, mesh, path_landdata, 'landpft',
                                                      landpft, lc_year)

        # ................................................................
        # 3. Mapping land characteristic parameters to the model grids
        # ................................................................
        sdd = None
        if namelist.nl_colm['SrfdataDiag']:
            sdd = CoLm_SrfdataDiag(namelist.nl_colm, mpi, gblock)
            if namelist.nl_colm['CROP']:
                landpatch.elm_patch.build(landlm, landpatch, use_frac=True, sharedfrac=crop.pctshrpch)
            else:
                landpatch.elm_patch.build(landlm, landpatch, use_frac=True)

            if namelist.nl_colm['GRIDBASED']:
                sdd.gdiag.define_by_copy(mesh.gridmesh)
            else:
                sdd.gdiag.define_by_ndims(3600, 1800)

            sdd.srfdata_diag_init(dir_landdata, landpatch, landlm, landpatch.elm_patch, landpft, landurban, landhru,
                                  crop.pctshrpch, var_global.N_land_classification, pixel, mesh)

        # #TODO: for lulcc, need to run for each year and SAVE to different subdirs

        Aggregation_PercentagesPFT.aggregation_percentagespft(var_global, mpi, namelist.nl_colm, gblock, mesh, pixel,
                                                              sdd, srfdata, crop, landpft, landpatch, gpatch,
                                                              dir_rawdata, dir_landdata, lc_year)

        Aggregation_LakeDepth.aggregation_lakedepth(gpatch, dir_rawdata, dir_landdata, lc_year, namelist.nl_colm, mpi,
                                                    gblock, mesh, pixel, var_global, landpatch, sdd, srfdata)

        Aggregation_SoilParameters.aggregation_soilparameters(gsoil, dir_rawdata, dir_landdata, lc_year,
                                                              namelist.nl_colm, mpi,
                                                              gblock, mesh, pixel, var_global, landpatch, srfdata)

        Aggregation_SoilBrightness.aggregation_soilbrightness(namelist.nl_colm, mpi, gblock, gpatch, dir_rawdata,
                                                              dir_landdata, lc_year, landpatch,
                                                              mesh, pixel, sdd, srfdata)

        if namelist.nl_colm['DEF_USE_BEDROCK']:
            pass
        #     CALL Aggregation_DBedrock     (gpatch , dir_rawdata, dir_landdata)
        # ENDIF

        Aggregation_LAI.aggregation_lai(namelist.nl_colm, mpi, gblock, gridlai, dir_rawdata, dir_landdata, lc_year,
                                        landpatch, landpft, srfdata, mesh, pixel, var_global)

        Aggregation_ForestHeight.aggregation_forestheight(namelist.nl_colm, mpi, gblock, gpatch, dir_rawdata,
                                                          dir_landdata, lc_year, landpatch, landpft,
                                                          mesh, pixel, srfdata, var_global)

        Aggregation_Topography.aggregation_topography(namelist.nl_colm, mpi, gblock, gtopo, dir_rawdata, dir_landdata,
                                                      lc_year, landpatch, mesh, pixel, sdd, srfdata)

        if namelist.nl_colm['URBAN_MODEL']:
            pass
            # CALL Aggregation_urban (dir_rawdata, dir_landdata, lc_year, &
            #                         grid_urban_5km, grid_urban_500m)

        # ! ................................................................
        # ! 4. Free memories.
        # ! ................................................................
        if namelist.nl_colm['SinglePoint']:
            if namelist.nl_colm['LULC_IGBP_PFT'] or namelist.nl_colm['LULC_IGBP_PC']:
                srfdata.write_surface_data_single(landpatch.numpatch, landpft.numpft, mksrfdata=True)
            else:
                if not namelist.nl_colm['URBAN_MODEL']:
                    srfdata.write_surface_data_single(path_dataset, landpatch.numpatch, mksrfdata=True)
                else:
                    srfdata.write_urban_surface_data_single(path_dataset, landurban.numurban)
            srfdata.single_srfdata_final(mksrfdata=True)

        if namelist.nl_colm['USEMPI']:
            pass
            # CALL mpi_barrier (p_comm_glb, p_err)

        if pixel is not None:
            pixel.pixel_free_mem()
        if gblock is not None:
            gblock.block_free_mem()
        if mesh is not None:
            mesh.mesh_free_mem()
        if ghru is not None:
            ghru.grid_free_men()
        if gpatch is not None:
            gpatch.grid_free_men()
        if gcrop is not None:
            gcrop.grid_free_men()
        if gsoil is not None:
            gsoil.grid_free_men()
        if gridlai is not None:
            gridlai.grid_free_men()
        if gtopo is not None:
            gtopo.grid_free_men()
        if gurban is not None:
            gurban.grid_free_men()
        if grid_urban_5km is not None:
            grid_urban_5km.grid_free_men()
        if grid_urban_500m is not None:
            grid_urban_500m.grid_free_men()
        # landlm.

        if mpi.p_is_master:
            end_time = time.time()
            time_used = end_time - start_time

            if time_used >= 3600:
                hours = int(time_used / 3600)
                minutes = int((time_used % 3600) / 60)
                seconds = int(time_used % 60)
                print(f"Overall system time used: {hours} hours {minutes} minutes {seconds} seconds.")
            elif time_used >= 60:
                minutes = int(time_used / 60)
                seconds = int(time_used % 60)
                print(f"Overall system time used: {minutes} minutes {seconds} seconds.")
            else:
                seconds = int(time_used)
                print(f"Overall system time used: {seconds} seconds.")

        print("Successful in surface data making.")

        if namelist.nl_colm['USEMPI']:
            pass
        # CALL spmd_exit
    except KeyError as e:
        print(e)
        if pixel is not None:
            pixel.pixel_free_mem()
        if gblock is not None:
            gblock.block_free_mem()
        if pixel is not None:
            mesh.mesh_free_mem()
        if ghru is not None:
            ghru.grid_free_men()
        if gpatch is not None:
            gpatch.grid_free_men()
        if gcrop is not None:
            gcrop.grid_free_men()
        if gsoil is not None:
            gsoil.grid_free_men()
        if gridlai is not None:
            gridlai.grid_free_men()
        if gtopo is not None:
            gtopo.grid_free_men()
        if gurban is not None:
            gurban.grid_free_men()
        if grid_urban_5km is not None:
            grid_urban_5km.grid_free_men()
        if grid_urban_500m is not None:
            grid_urban_500m.grid_free_men()


if __name__ == '__main__':
    path_root = '/home/liqingliang/ATAI/CoLM/code/code/dataset'  # 来自配置文件
    # path_data = '/home/zjl/data/landmodel1'  # 来自配置文件
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

    if file_define['LULC_IGBP_PFT']:
        file_define.update({'BGC': False})

    if file_define['BGC']:
        file_define.update({'CROP': False})

    mksrfdata(path_root, file_define, path_data, plat_sys)
