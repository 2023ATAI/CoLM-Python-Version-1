# ----------------------------------------------------------------------
# DESCRIPTION:
# Aggregate lake depth of multiple pixels within a lake patch based on Global land cover types
# (updated with the specific dataset)
#
# Global Lake Coverage and Lake Depth (1km resolution)
#   (http://nwpi.krc.karelia.run/flake/)
#    Lake depth data legend
#    Value   Description
# 0       no lake indicated in this pixel
# 1       no any information about this lake and set the default value of 10 m
# 2       no information about depth for this lake and set the default value of 10 m
# 3       have the information about lake depth in this pixel
# 4       this is the river pixel according to our map, set the default value of 3 m
#
# REFERENCE:
# Kourzeneva, E., H. Asensio, E. Martin, and S. Faroux, 2012: Global gridded dataset of lake coverage and lake depth
# for USE in numerical weather prediction and climate modelling. Tellus A, 64, 15640.
#
#
# REVISIONS:
# ----------------------------------------------------------------------
import os
import sys
import numpy as np
import CoLM_NetCDFBlock
import CoLM_RangeCheck
import CoLM_Utils
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_DataType import DataType
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector


def aggregation_soilparameters(gland, dir_rawdata, dir_model_landdata, lc_year,
                               nl_colm, mpi, gblock, mesh, pixel, var_global, landpatch, srfdataDiag):
    cyear = str(lc_year)
    numpatch = landpatch.numpatch
    area_one = np.zeros(numpatch)
    vf_quartz_mineral_s_one = None
    vf_quartz_mineral_s_patches = None
    vf_quartz_mineral_s_grid = None
    SITE_soil_vf_quartz_mineral = None
    vf_gravels_s_patches = None
    vf_sand_s_patches = None
    vf_om_s_patches = None
    BA_alpha_patches = None
    BA_beta_patches = None
    vf_gravels_s_grid = None
    vf_sand_s_grid = None
    vf_om_s_grid = None
    SITE_soil_vf_gravels = None
    SITE_soil_vf_om = None
    SITE_soil_vf_sand = None
    SITE_soil_BA_alpha = None
    SITE_soil_BA_beta = None
    wf_gravels_s_grid = None
    wf_gravels_s_patches = None
    SITE_soil_wf_gravels = None
    wf_sand_s_patches = None
    wf_sand_s_grid = None
    L_vgm_grid = None
    L_vgm_patches = None
    SITE_soil_L_vgm = None
    SITE_soil_wf_sand = None
    theta_r_grid = None
    alpha_vgm_grid = None
    n_vgm_grid = None
    theta_s_grid = None
    theta_r_patches = None
    alpha_vgm_patches = None
    n_vgm_patches = None
    theta_s_patches = None
    SITE_soil_theta_r = None
    SITE_soil_alpha_vgm = None
    SITE_soil_n_vgm = None
    SITE_soil_theta_s = None
    psi_s_grid = None
    lambda_grid = None
    psi_s_patches = None
    lambda_patches = None
    SITE_soil_psi_s = None
    SITE_soil_lambda = None
    k_s_grid = None
    k_s_patches = None
    SITE_soil_k_s = None
    csol_grid = None
    SITE_soil_csol = None
    tksatu_grid = None
    csol_patches = None
    tksatu_patches = None
    SITE_soil_tksatu = None
    tksatf_patches = None
    tksatf_grid = None
    SITE_soil_tksatf = None
    tkdry_grid = None
    tkdry_patches = None
    SITE_soil_tkdry = None
    k_solids_grid = None
    k_solids_patches = None
    SITE_soil_k_solids = None
    OM_density_s_grid = None
    OM_density_s_patches = None
    BD_all_s_grid = None
    BD_all_s_patches = None
    SITE_soil_OM_density = None
    SITE_soil_BD_all = None
    psi_s_one = None
    lambda_one = None
    vf_gravels_s_one = None
    vf_om_s_one = None
    vf_sand_s_one = None
    wf_gravels_s_one = None
    wf_sand_s_one = None
    OM_density_s_one = None
    BD_all_s_one = None
    theta_s_one = None
    theta_r_one = None
    alpha_vgm_one = None
    L_vgm_one = None
    n_vgm_one = None
    csol_one = None
    tksatu_one = None
    tkdry_one = None
    tksatf_one = None
    k_solids_one = None
    k_s_one = None
    dt = None

    nv = 4
    nc = 3
    # nb = 2
    xv = np.zeros(nv)
    xc = np.zeros(nc)
    # xb = np.zeros(nb)

    npointw = 24
    npointb = 20
    xdat = np.array([1., 5., 10., 20., 30., 40., 50., 60., 70., 90., 110., 130., 150.,
                     170., 210., 300., 345., 690., 1020., 5100., 15300., 20000., 100000., 1000000.], dtype=np.float64)
    xdatsr = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                       0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float64)

    # Variables
    ydatc = None
    ydatv = None
    ydatb = None

    # Parameters for Levenberg-Marquardt algorithm in MINPACK library
    factor = 0.1
    ftol = 1.0e-5
    xtol = 1.0e-4
    gtol = 0.0
    mode = 1
    nprint = 0

    if nl_colm['SrfdataDiag']:
        typpatch = np.zeros(nl_colm['N_land_classification'] + 1)
        ityp = 0

    lc_year_str = str(lc_year)
    landdir = os.path.join(dir_model_landdata, 'soil',lc_year_str)
    if 'win' in sys.platform:
        landdir = dir_model_landdata + '\\' + 'soil'+'\\'+lc_year_str

    #   ---------------------------------------------------------------
    #   aggregate the soil parameters from the resolution of raw data to modelling resolution
    #   ---------------------------------------------------------------
    if nl_colm['USEMPI']:
        pass
    if mpi.p_is_worker:
        print('Aggregate Soil Parameters ...')
        if not os.path.exists(landdir):
            os.makedirs(landdir.strip())
    if nl_colm['USEMPI']:
        pass
    if nl_colm['SinglePoint']:
        if nl_colm['USE_SITE_soilparameters']:
            return
        else:
            SITE_soil_vf_quartz_mineral = np.zeros(var_global.nl_soil)
            SITE_soil_vf_gravels = np.zeros(var_global.nl_soil)
            SITE_soil_vf_om = np.zeros(var_global.nl_soil)
            SITE_soil_vf_sand = np.zeros(var_global.nl_soil)
            SITE_soil_wf_gravels = np.zeros(var_global.nl_soil)
            SITE_soil_wf_sand = np.zeros(var_global.nl_soil)
            SITE_soil_OM_density = np.zeros(var_global.nl_soil)
            SITE_soil_BD_all = np.zeros(var_global.nl_soil)
            SITE_soil_theta_s = np.zeros(var_global.nl_soil)
            SITE_soil_psi_s = np.zeros(var_global.nl_soil)
            SITE_soil_lambda = np.zeros(var_global.nl_soil)
            if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                SITE_soil_theta_r = np.zeros(var_global.nl_soil)
                SITE_soil_alpha_vgm = np.zeros(var_global.nl_soil)
                SITE_soil_L_vgm = np.zeros(var_global.nl_soil)
                SITE_soil_n_vgm = np.zeros(var_global.nl_soil)
            SITE_soil_k_s = np.zeros(var_global.nl_soil)
            SITE_soil_csol = np.zeros(var_global.nl_soil)
            SITE_soil_tksatu = np.zeros(var_global.nl_soil)
            SITE_soil_tksatf = np.zeros(var_global.nl_soil)
            SITE_soil_tkdry = np.zeros(var_global.nl_soil)
            SITE_soil_k_solids = np.zeros(var_global.nl_soil)
            SITE_soil_BA_alpha = np.zeros(var_global.nl_soil)
            SITE_soil_BA_beta = np.zeros(var_global.nl_soil)
    if mpi.p_is_worker:
        vf_quartz_mineral_s_patches = np.zeros(numpatch)
        vf_gravels_s_patches = np.zeros(numpatch)
        vf_om_s_patches = np.zeros(numpatch)
        vf_sand_s_patches = np.zeros(numpatch)
        wf_gravels_s_patches = np.zeros(numpatch)
        wf_sand_s_patches = np.zeros(numpatch)
        OM_density_s_patches = np.zeros(numpatch)
        BD_all_s_patches = np.zeros(numpatch)
        theta_s_patches = np.zeros(numpatch)
        psi_s_patches = np.zeros(numpatch)
        lambda_patches = np.zeros(numpatch)
        if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            theta_r_patches = np.zeros(numpatch)
            alpha_vgm_patches = np.zeros(numpatch)
            L_vgm_patches = np.zeros(numpatch)
            n_vgm_patches = np.zeros(numpatch)
        k_s_patches = np.zeros(numpatch)
        csol_patches = np.zeros(numpatch)
        tksatu_patches = np.zeros(numpatch)
        tksatf_patches = np.zeros(numpatch)
        tkdry_patches = np.zeros(numpatch)
        k_solids_patches = np.zeros(numpatch)
        BA_alpha_patches = np.zeros(numpatch)
        BA_beta_patches = np.zeros(numpatch)

    for nsl in range(8):
        c = str(nsl+1)
        # (1) volumetric fraction of quartz within mineral soil
        if mpi.p_is_io:
            dt = DataType(gblock)
            vf_quartz_mineral_s_grid = dt.allocate_block_data(gland)

            lndname = os.path.join(dir_rawdata, 'soil', "vf_quartz_mineral_s.nc")
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + "vf_quartz_mineral_s.nc"

            vf_quartz_mineral_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'vf_quartz_mineral_s_l' + c, mpi,
                                                                        gblock, gland, vf_quartz_mineral_s_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, vf_quartz_mineral_s_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(
                        landpatch.landpatch,ipatch, gland, zip=nl_colm['USE_zip_for_aggregation'],
                        data_r8_2d_in1=vf_quartz_mineral_s_grid,
                        area=area_one)
                    vf_quartz_mineral_s_patches[ipatch] = np.sum(
                        vf_quartz_mineral_s_one * (area_one / np.sum(area_one)))
                else:
                    vf_quartz_mineral_s_patches[ipatch] = -1.0e36

                if np.isnan(vf_quartz_mineral_s_patches[ipatch]):
                    print("NAN appears in vf_quartz_mineral_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

            if nl_colm['USEMPI']:
                pass

        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('vf_quartz_mineral_s lev' + c, vf_quartz_mineral_s_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'vf_quartz_mineral_s_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'vf_quartz_mineral_s_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'vf_quartz_mineral_s_l' + c.strip() + '_patches.nc', 'patch',
                                         landpatch.landpatch, vf_quartz_mineral_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = dir_model_landdata + '/diag/soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(vf_quartz_mineral_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'vf_quartz_mineral_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_vf_quartz_mineral[nsl] = vf_quartz_mineral_s_patches[0]
        # (2)volumetric fraction of gravels
        # (3) volumetric fraction of sand
        # (4) volumetric fraction of organic matter
        # with the parameter alpha and beta in the Balland V. and P. A. Arp (2005) model

        if mpi.p_is_io:
            dt = DataType(gblock)
            vf_gravels_s_grid = dt.allocate_block_data(gland)

            lndname = os.path.join(dir_rawdata, 'soil', "vf_gravels_s.nc")
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + "vf_gravels_s.nc"
            vf_gravels_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'vf_gravels_s_l' + c.strip(), mpi, gblock,
                                                                 gland, vf_gravels_s_grid)

            dt = DataType(gblock)
            vf_sand_s_grid = dt.allocate_block_data(gland)

            lndname = os.path.join(dir_rawdata, 'soil', "vf_sand_s.nc")
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + "vf_sand_s.nc"
            vf_sand_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'vf_sand_s_l' + c.strip(), mpi, gblock, gland,
                                                              vf_sand_s_grid)

            dt = DataType(gblock)
            vf_om_s_grid = dt.allocate_block_data(gland)

            lndname = os.path.join(dir_rawdata, 'soil', "vf_om_s.nc")
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + "vf_om_s.nc"
            vf_om_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'vf_om_s_l' + c.strip(), mpi, gblock, gland,
                                                            vf_om_s_grid)
            if nl_colm['USEMPI']:
                pass
        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, vf_gravels_s_one, vf_sand_s_one, vf_om_s_one, _, _, _, _, _, _, _ = ard.aggregation_request_data(
                        landpatch.landpatch, ipatch,
                        gland, zip=nl_colm['USE_zip_for_aggregation'],
                        data_r8_2d_in1=vf_gravels_s_grid,
                        data_r8_2d_in2=vf_sand_s_grid,
                        data_r8_2d_in3=vf_om_s_grid,
                        area=area_one)

                    vf_gravels_s_patches[ipatch] = np.sum(vf_gravels_s_one * (area_one / np.sum(area_one)))
                    vf_sand_s_patches[ipatch] = np.sum(vf_sand_s_one * (area_one / np.sum(area_one)))
                    vf_om_s_patches[ipatch] = np.sum(vf_om_s_one * (area_one / np.sum(area_one)))
                    # the parameter values of Balland and Arp (2005) Ke-Sr relationship,
                    # modified by Barry-Macaulay et al.(2015), Evaluation of soil thermal conductivity models
                    BA_alpha_one = np.zeros(len(area_one))
                    BA_beta_one = np.zeros(len(area_one))

                    if np.all (vf_gravels_s_one + vf_sand_s_one)> 0.4:
                        BA_alpha_one[:] = 0.38
                        BA_beta_one[:] = 35.0
                    elif np.all(vf_gravels_s_one + vf_sand_s_one) > 0.25:
                        BA_alpha_one[:] = 0.24
                        BA_beta_one[:] = 26.0
                    else:
                        BA_alpha_one[:] = 0.2
                        BA_beta_one[:] = 10.0

                    BA_alpha_patches[ipatch] = CoLM_Utils.median(BA_alpha_one, var_global.spval)
                    BA_beta_patches[ipatch] = CoLM_Utils.median(BA_beta_one, var_global.spval)

                    del BA_alpha_one
                    del BA_beta_one
                else:
                    vf_gravels_s_patches[ipatch] = -1.0e36
                    vf_sand_s_patches[ipatch] = -1.0e36
                    vf_om_s_patches[ipatch] = -1.0e36
                    BA_alpha_patches[ipatch] = -1.0e36
                    BA_beta_patches[ipatch] = -1.0e36

                if np.isnan(vf_gravels_s_patches[ipatch]):
                    print("NAN appears in vf_gravels_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(vf_sand_s_patches[ipatch]):
                    print("NAN appears in vf_sand_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(vf_om_s_patches[ipatch]):
                    print("NAN appears in vf_om_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('vf_gravels_s lev' + c, vf_gravels_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('vf_sand_s lev' + c, vf_sand_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('vf_om_s lev' + c, vf_om_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('BA_alpha lev' + c, BA_alpha_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('BA_beta lev' + c, BA_beta_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'vf_gravels_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'vf_gravels_s_l' + c + '_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'vf_gravels_s_l' + c + '_patches', 'patch', landpatch.landpatch,
                                          vf_gravels_s_patches,
                                          nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(vf_gravels_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'vf_gravels_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_vf_gravels[nsl] = vf_gravels_s_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'vf_sand_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'vf_sand_s_l' + c + '_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'vf_sand_s_l' + c + '_patches', 'patch',
                                          landpatch.landpatch, vf_sand_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(vf_sand_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'vf_sand_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_vf_sand[nsl] = vf_sand_s_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'vf_om_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'vf_om_s_l' + c + '_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'vf_om_s_l' + c + '_patches', 'patch',
                                          landpatch.landpatch, vf_om_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(vf_om_s_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'vf_om_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_vf_om[nsl] = vf_om_s_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'BA_alpha_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'BA_alpha_l' + c + '_patches.nc'
            vector_ones = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vector_ones.ncio_create_file_vector(lndname, landpatch.landpatch)
            vector_ones.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vector_ones.ncio_write_vector(lndname, 'BA_alpha_l' + c + '_patches', 'patch',
                                          landpatch.landpatch, BA_alpha_patches, nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(BA_alpha_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'BA_alpha_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_BA_alpha[nsl] = BA_alpha_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'BA_beta_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'BA_beta_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'BA_beta_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         BA_beta_patches, nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]

                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(BA_beta_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'BA_beta_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_BA_beta[nsl] = BA_beta_patches[0]

        # (5) gravimetric fraction of gravels
        if mpi.p_is_io:
            dt = DataType(gblock)
            wf_gravels_s_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil' , 'wf_gravels_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'wf_gravels_s.nc'
            wf_gravels_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'wf_gravels_s_l' + c.strip(), mpi, gblock,
                                                                 gland, wf_gravels_s_grid)
            if nl_colm['USEMPI']:
                pass
        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, wf_gravels_s_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch,
                                                                                                         ipatch, gland,
                                                                                                         zip=nl_colm['USE_zip_for_aggregation'],
                                                                                                         data_r8_2d_in1=wf_gravels_s_grid,
                                                                                                         area=area_one)
                    wf_gravels_s_patches[ipatch] = np.sum(
                        wf_gravels_s_one * (area_one / np.sum(area_one)))
                else:
                    wf_gravels_s_patches[ipatch] = -1.0e36

                if np.isnan(wf_gravels_s_patches[ipatch]):
                    print("NAN appears in wf_gravels_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('wf_gravels_s lev ', wf_gravels_s_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'wf_gravels_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'wf_gravels_s_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'wf_gravels_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         wf_gravels_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(wf_gravels_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'wf_gravels_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_wf_gravels[nsl] = wf_gravels_s_patches[0]

        # (6) gravimetric fraction of sand
        if mpi.p_is_io:
            dt = DataType(gblock)
            wf_sand_s_grid = dt.allocate_block_data(gland)

            lndname = os.path.join(dir_rawdata, 'soil', 'wf_sand_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'wf_sand_s.nc'
            wf_sand_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'wf_sand_s_l' + c.strip(), mpi, gblock, gland,
                                                              wf_sand_s_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, wf_sand_s_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch, gland,
                                                                 zip=nl_colm['USE_zip_for_aggregation'],
                                                                 data_r8_2d_in1=wf_sand_s_grid, area=area_one)
                    wf_sand_s_patches[ipatch] = np.sum(
                        wf_sand_s_one * (area_one / np.sum(area_one)))
                else:
                    wf_sand_s_patches[ipatch] = -1.0e36

                if np.isnan(wf_sand_s_patches[ipatch]):
                    print("NAN appears in wf_sand_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('wf_sand_s lev ' + c, wf_sand_s_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'wf_sand_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'wf_sand_s_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'wf_sand_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         wf_sand_s_patches, 1)
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(wf_sand_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'wf_sand_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_wf_sand[nsl] = wf_sand_s_patches[0]

        if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            # (7) VGM's pore-connectivity parameter (L)
            if mpi.p_is_io:
                dt = DataType(gblock)
                L_vgm_grid = dt.allocate_block_data(gland)
                lndname = os.path.join(dir_rawdata, 'soil', 'VGM_L.nc')
                if 'win' in sys.platform:
                    lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'VGM_L.nc'
                L_vgm_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'VGM_L_l' + c.strip(), mpi, gblock, gland,
                                                              L_vgm_grid)

                theta_r_grid = dt.allocate_block_data(gland)
                lndname = os.path.join(dir_rawdata, 'soil', 'VGM_theta_r.nc')
                if 'win' in sys.platform:
                    lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'VGM_theta_r.nc'
                theta_r_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'VGM_theta_r_l' + c.strip(), mpi, gblock,
                                                                gland, theta_r_grid)

                alpha_vgm_grid = dt.allocate_block_data(gland)
                lndname = os.path.join(dir_rawdata, 'soil', 'VGM_alpha.nc')
                if 'win' in sys.platform:
                    lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'VGM_alpha.nc'
                alpha_vgm_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'VGM_alpha_l' + c.strip(), mpi, gblock,
                                                                  gland,
                                                                  alpha_vgm_grid)

                n_vgm_grid = dt.allocate_block_data(gland)
                lndname = os.path.join(dir_rawdata, 'soil', 'VGM_n.nc')
                if 'win' in sys.platform:
                    lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'VGM_n.nc'
                n_vgm_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'VGM_n_l' + c.strip(), mpi, gblock, gland,
                                                              n_vgm_grid)

                theta_s_grid = dt.allocate_block_data(gland)
                lndname = os.path.join(dir_rawdata, 'soil', 'theta_s.nc')
                if 'win' in sys.platform:
                    lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'theta_s.nc'
                theta_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'theta_s_l' + c.strip(), mpi, gblock, gland,
                                                                theta_s_grid)

                k_s_grid = dt.allocate_block_data(gland)
                lndname = os.path.join(dir_rawdata, 'soil', 'k_s.nc')
                if 'win' in sys.platform:
                    lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'k_s.nc'
                k_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'k_s_l' + c.strip(), mpi, gblock, gland,
                                                            k_s_grid)
                if nl_colm['USEMPI']:
                    pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, theta_r_one, alpha_vgm_one, n_vgm_one, theta_s_one, k_s_one, L_vgm_one, _, _, _, _ = ard.aggregation_request_data(
                        landpatch.landpatch, ipatch,
                        gland, zip=nl_colm['USE_zip_for_aggregation'],
                        data_r8_2d_in1=theta_r_grid,
                        data_r8_2d_in2=alpha_vgm_grid,
                        data_r8_2d_in3=n_vgm_grid,
                        data_r8_2d_in4=theta_s_grid,
                        data_r8_2d_in5=k_s_grid,
                        data_r8_2d_in6=L_vgm_grid,
                        area=area_one)
                    theta_r_patches[ipatch] = np.sum(theta_r_one * (area_one / sum(area_one)))
                    alpha_vgm_patches[ipatch] = CoLM_Utils.median(alpha_vgm_one, var_global.spval)
                    n_vgm_patches[ipatch] = CoLM_Utils.median(n_vgm_one, var_global.spval)
                    theta_s_patches[ipatch] = np.sum(theta_s_one * (area_one / sum(area_one)))
                    k_s_patches[ipatch] = np.prod(np.power(k_s_one, area_one / np.sum(area_one)))
                    L_vgm_patches[ipatch] = CoLM_Utils.median(L_vgm_one, var_global.spval)

                    if nl_colm['DEF_USE_SOILPAR_UPS_FIT']:
                        npp = len(theta_r_one)
                        if npp > 1:
                            ydatv = np.zeros((npp, npointw))
                            ydatvks = np.zeros((npp, npointw))
                            THETA = np.zeros(npointw)
                            fjacv = np.zeros((npointw, nv))
                            fvecv = np.zeros(npointw)
                            # Populate arrays
                            for LL in range(npp):
                                THETA = (1 + (alpha_vgm_one[LL] * xdat) ** n_vgm_one[LL]) ** (1.0 / n_vgm_one[LL] - 1)
                                ydatv[LL, :] = theta_r_one[LL] + (theta_s_one[LL] - theta_r_one[LL]) * THETA
                                ydatvks[LL, :] = k_s_one[LL] * THETA ** L_vgm_one[LL] * (
                                        1 - (1 - THETA ** (n_vgm_one[LL] / (n_vgm_one[LL] - 1))) ** (
                                        1.0 - 1.0 / n_vgm_one[LL])) ** 2
                            # Fitting van Genuchten SW retention parameters
                            ldfjac = npointw
                            xv[0] = theta_r_patches[ipatch]
                            xv[1] = alpha_vgm_patches[ipatch]
                            xv[2] = n_vgm_patches[ipatch]
                            xv[3] = k_s_patches[ipatch]
                            maxfev = 100 * (nv + 1)
                            isiter = 1
                            xv, isiter = CoLM_Utils.lmder(npointw, nv, xv, fvecv, fjacv, ldfjac, ftol, xtol, gtol, maxfev,
                              mode, factor, nprint,
                              xdat, npointw, ydatv, ydatvks, np, theta_s_patches[ipatch],
                            k_s_patches[ipatch], isiter, L_vgm_patches[ipatch])

                            if xv[0] >= 0.0 and xv[0] <= theta_s_patches[ipatch] and xv[1] >= 1.0e-5 and xv[
                                1] <= 1.0 and xv[2] >= 1.1 and xv[2] <= 10.0 and xv[3] > 0.0 and xv[
                                3] <= 1.0e7 and isiter == 1:
                                theta_r_patches[ipatch] = xv[0]
                                alpha_vgm_patches[ipatch] = xv[1]
                                n_vgm_patches[ipatch] = xv[2]
                                k_s_patches[ipatch] = xv[3]
                            # Deallocate memory
                            del ydatv, ydatvks, THETA, fjacv, fvecv


                else:
                    theta_r_patches[ipatch] = -1.0e36
                    alpha_vgm_patches[ipatch] = -1.0e36
                    n_vgm_patches[ipatch] = -1.0e36
                    theta_s_patches[ipatch] = -1.0e36
                    k_s_patches[ipatch] = -1.0e36
                    L_vgm_patches[ipatch] = -1.0e36

                if np.isnan(L_vgm_patches[ipatch]):
                    print("NAN appears in L_vgm_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(alpha_vgm_patches[ipatch]):
                    print("NAN appears in alpha_vgm_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(n_vgm_patches[ipatch]):
                    print("NAN appears in n_vgm_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(theta_s_patches[ipatch]):
                    print("NAN appears in theta_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(k_s_patches[ipatch]):
                    print("NAN appears in k_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(L_vgm_patches[ipatch]):
                    print("NAN appears in L_vgm_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('theta_r lev' + c, theta_r_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('alpha VGM lev' + c, alpha_vgm_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('n VGM lev' + c, n_vgm_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('theta_s lev' + c, theta_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('k_s lev' + c, k_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('L VGM lev' + c, L_vgm_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'theta_r_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'theta_r_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'theta_r_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         theta_r_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(theta_r_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'theta_r_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_theta_r[nsl] = theta_r_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'alpha_vgm_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'alpha_vgm_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'alpha_vgm_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         alpha_vgm_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(alpha_vgm_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'alpha_vgm_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_alpha_vgm[nsl] = alpha_vgm_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'n_vgm_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'n_vgm_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'n_vgm_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         n_vgm_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(n_vgm_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'n_vgm_l' + c.strip(), compress=1, write_mode='one')
        else:
            SITE_soil_n_vgm[nsl] = n_vgm_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'theta_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'theta_s_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'theta_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         theta_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(theta_s_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'theta_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_theta_s[nsl] = theta_s_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'k_s_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'k_s_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'k_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         k_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(k_s_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'k_s_l' + c.strip(), compress=1, write_mode='one')
        else:
            SITE_soil_k_s[nsl] = k_s_patches[0]

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'L_vgm_l' + c + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'L_vgm_l' + c + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'L_vgm_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         L_vgm_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(L_vgm_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'L_vgm_l' + c.strip(), compress=1, write_mode='one')
        else:
            SITE_soil_L_vgm[nsl] = L_vgm_patches[0]

        # (8) VGM's residual water content (theta_r) [cm3/cm3]
        # (9) VGM's parameter corresponding approximately to the inverse of the air-entry value (alpha)
        # (10) VGM's shape parameter (n)
        # (11) saturated water content [cm3/cm3]
        if mpi.p_is_io:
            dt = DataType(gblock)
            theta_r_grid = dt.allocate_block_data(gland)

            lndname = os.path.join(dir_rawdata, 'soil', 'theta_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'theta_s.nc'

            theta_r_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'theta_s_l' + c.strip(), mpi, gblock, gland,
                                                            theta_r_grid)

            k_s_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'k_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'k_s.nc'
            k_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'k_s_l' + c.strip(), mpi, gblock, gland,
                                                        k_s_grid)

            psi_s_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'psi_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'psi_s.nc'
            psi_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'psi_s_l' + c.strip(), mpi, gblock, gland,
                                                          psi_s_grid)

            lambda_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'lambda.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'lambda.nc'
            lambda_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'lambda_l' + c.strip(), mpi, gblock, gland,
                                                           lambda_grid)
            if nl_colm['USEMPI']:
                pass
        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, theta_s_one, k_s_one, psi_s_one, lambda_one,_,_,_,_,_, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                               gland,
                                                                                               zip=nl_colm['USE_zip_for_aggregation'],
                                                                                               data_r8_2d_in1=theta_r_grid,
                                                                                               data_r8_2d_in2=k_s_grid,
                                                                                               data_r8_2d_in3=psi_s_grid,
                                                                                               data_r8_2d_in4=lambda_grid,
                                                                                               area=area_one)

                    theta_r_patches[ipatch] = np.sum(theta_s_one * (area_one / np.sum(area_one)))
                    k_s_patches[ipatch] = np.prod(k_s_one ** (area_one / sum(area_one)))
                    psi_s_patches[ipatch] = CoLM_Utils.median(psi_s_one, var_global.spval)
                    lambda_patches[ipatch] = CoLM_Utils.median(lambda_one, var_global.spval)

                    if nl_colm['DEF_USE_SOILPAR_UPS_FIT']:
                        npp = len(psi_s_one)
                        # print (npp, len(theta_s_one), len(psi_s_one), len(lambda_one), '--------np--------------')
                        if npp > 1:
                            ydatc = np.zeros((npp, npointw - 7))
                            ydatcks = np.zeros((npp, npointw - 7))
                            fjacc = np.zeros((npointw - 7, nc))
                            fvecc = np.zeros(npointw - 7)
                            # Populate arrays
                            # print(npp,len(theta_s_one),'-------soil--------')
                            for LL in range(npp):
                                ydatc[LL,:] = (-1.0 * xdat[7:] / psi_s_one[LL]) ** (-1.0 * lambda_one[LL]) * theta_s_one[
                                    LL]
                                ydatcks[LL,:] = (-1.0 * xdat[7:] / psi_s_one[LL]) ** (-3.0 * lambda_one[LL] - 2) * \
                                              k_s_one[LL]
                            ldfjac = npointw - 7
                            xc[0] = psi_s_patches[ipatch]
                            xc[1] = lambda_patches[ipatch]
                            xc[2] = k_s_patches[ipatch]
                            maxfev = 100 * (nc + 1)
                            isiter = 1

                            xc, isiter = CoLM_Utils.lmder(npointw - 7, nc, xc, fvecc, fjacc, ldfjac, ftol, xtol, gtol, maxfev,
                            mode, factor, nprint, xdat[7:], npointw - 7, ydatc, ydatcks, np, theta_s_patches[ipatch], k_s_patches[ipatch], isiter)

                            if xc[0] >= -300.0 and xc[0] < 0.0 and xc[1] > 0.0 and xc[1] <= 1.0 and xc[2] > 0.0 and xc[
                                2] <= 1.0e7 and isiter == 1:
                                psi_s_patches[ipatch] = xc[0]
                                lambda_patches[ipatch] = xc[1]
                                k_s_patches[ipatch] = xc[2]

                            del ydatc, ydatcks, fjacc, fvecc
                else:
                    theta_r_patches[ipatch] = -1.0e36
                    alpha_vgm_patches[ipatch] = -1.0e36
                    n_vgm_patches[ipatch] = -1.0e36
                    theta_s_patches[ipatch] = -1.0e36

                if np.isnan(theta_r_patches[ipatch]):
                    print("NAN appears in theta_r_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(alpha_vgm_patches[ipatch]):
                    print("NAN appears in alpha_vgm_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(n_vgm_patches[ipatch]):
                    print("NAN appears in n_vgm_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

                if np.isnan(theta_s_patches[ipatch]):
                    print("NAN appears in theta_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])

            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('theta_r lev '+c, theta_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('alpha VGM lev '+c, k_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('n VGM lev '+c, psi_s_patches, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('theta_s lev '+c, lambda_patches, mpi, nl_colm)

        if not nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
            if not nl_colm['SinglePoint']:
                lndname = os.path.join(landdir, 'theta_s_l' + c + '_patches.nc')
                if 'win' in sys.platform:
                    lndname = landdir + '\\' + 'theta_s_l' + c + '_patches.nc'
                vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
                vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                vectorOnes.ncio_write_vector(lndname, 'theta_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                             theta_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])
                if nl_colm['SrfdataDiag']:
                    typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                    lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                    if 'win' in sys.platform:
                        lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'

                    srfdataDiag.srfdata_map_and_write(theta_s_patches, landpatch.landpatch.settyp, typpatch,
                                                      srfdataDiag.m_patch2diag,
                                                      -1.0e36, lndname, 'theta_r_l' + c.strip(), compress=1,
                                                      write_mode='one')
            else:

                SITE_soil_theta_s[nsl] = theta_s_patches[0]

            if not nl_colm['SinglePoint']:
                lndname = os.path.join(landdir, 'k_s_l' + c + '_patches.nc')
                if 'win' in sys.platform:
                    lndname = landdir + '\\' + 'k_s_l' + c + '_patches.nc'
                vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
                vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                vectorOnes.ncio_write_vector(lndname, 'k_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                             k_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])

                if nl_colm['SrfdataDiag']:
                    typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                    lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                    if 'win' in sys.platform:
                        lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                    srfdataDiag.srfdata_map_and_write(k_s_patches, landpatch.landpatch.settyp, typpatch,
                                                      srfdataDiag.m_patch2diag,
                                                      -1.0e36, lndname, 'k_s_l' + c.strip(), compress=1,
                                                      write_mode='one')
            else:
                SITE_soil_k_s[nsl] = k_s_patches[0]

            if not nl_colm['SinglePoint']:
                lndname = os.path.join(landdir, 'psi_s_l' + c + '_patches.nc')
                if 'win' in sys.platform:
                    lndname = landdir + '\\' + 'psi_s_l' + c + '_patches.nc'
                vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
                vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                vectorOnes.ncio_write_vector(lndname, 'psi_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                             psi_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])

                if nl_colm['SrfdataDiag']:
                    typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                    lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                    if 'win' in sys.platform:
                        lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                    srfdataDiag.srfdata_map_and_write(psi_s_patches, landpatch.landpatch.settyp, typpatch,
                                                      srfdataDiag.m_patch2diag,
                                                      -1.0e36, lndname, 'psi_s_l' + c.strip(), compress=1,
                                                      write_mode='one')
            else:
                SITE_soil_psi_s[nsl] = psi_s_patches[0]

            if not nl_colm['SinglePoint']:
                lndname = os.path.join(landdir, 'lambda_l' + c + '_patches.nc')
                if 'win' in sys.platform:
                    lndname = landdir + '\\' + 'lambda_l' + c + '_patches.nc'
                vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
                vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
                vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
                vectorOnes.ncio_write_vector(lndname, 'lambda_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                             lambda_patches, nl_colm['DEF_Srfdata_CompressLevel'])

                if nl_colm['SrfdataDiag']:
                    typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                    lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                    if 'win' in sys.platform:
                        lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                    srfdataDiag.srfdata_map_and_write(lambda_patches, landpatch.landpatch.settyp, typpatch,
                                                      srfdataDiag.m_patch2diag,
                                                      -1.0e36, lndname, 'lambda_l' + c.strip(), compress=1,
                                                      write_mode='one')
            else:
                SITE_soil_lambda[nsl] = lambda_patches[0]

        # (11) saturated water content [cm3/cm3]
        # (12) matric potential at saturation (psi_s) [cm]
        # (13) pore size distribution index [dimensionless]
        if mpi.p_is_io:
            dt = DataType(gblock)
            csol_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'csol.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'csol.nc'
            csol_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'csol_l' + c.strip(), mpi, gblock, gland,
                                                         csol_grid)

            if nl_colm['USEMPI']:
                pass
        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, csol_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                 gland, zip=nl_colm['USE_zip_for_aggregation'],
                                                                                                 data_r8_2d_in1=csol_grid,
                                                                                                 area=area_one)
                    csol_patches[ipatch] = np.sum(csol_one * (area_one / np.sum(area_one)))
                else:
                    csol_patches[ipatch] = -1.0e36

                if np.isnan(csol_patches[ipatch]):
                    print("NAN appears in theta_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass

        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('csol lev ', csol_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'csol_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'csol_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'csol_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         csol_patches, nl_colm['DEF_Srfdata_CompressLevel'])

            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(csol_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'csol_l' + c.strip(), compress=1, write_mode='one')
        else:
            SITE_soil_csol[nsl] = csol_patches[0]

        # (14) saturated hydraulic conductivity [cm/day]
        if mpi.p_is_io:
            dt = DataType(gblock)
            tksatu_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'tksatu.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'tksatu.nc'
            tksatu_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'tksatu_l' + c.strip(), mpi, gblock, gland,
                                                           tksatu_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, tksatu_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                   gland,
                                                                                                   zip=nl_colm['USE_zip_for_aggregation'],
                                                                                                   data_r8_2d_in1=tksatu_grid,
                                                                                                   area=area_one)
                    tksatu_patches[ipatch] = np.prod(tksatu_one ** (area_one / np.sum(area_one)))
                else:
                    tksatu_patches[ipatch] = -1.0e36

                if np.isnan(tksatu_patches[ipatch]):
                    print("NAN appears in tksatu_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('tksatu lev ' + c, tksatu_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'tksatu_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'tksatu_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'tksatu_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         tksatu_patches,
                                         nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(tksatu_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'tksatu_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_tksatu[nsl] = tksatu_patches[0]

        # (15) heat capacity of soil solids [J/(m3 K)]
        if mpi.p_is_io:
            dt = DataType(gblock)
            tksatf_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'tksatf.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'tksatf.nc'
            tksatf_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'tksatf_l' + c.strip(), mpi, gblock, gland,
                                                           tksatf_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, tksatf_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                   gland,
                                                                                                   zip=nl_colm[
                                                                                                       'USE_zip_for_aggregation'],
                                                                                                   data_r8_2d_in1=tksatf_grid,
                                                                                                   area=area_one)
                    tksatf_patches[ipatch] = np.prod(tksatf_one ** (area_one / np.sum(area_one)))
                else:
                    tksatf_patches[ipatch] = -1.0e36

                if np.isnan(tksatf_patches[ipatch]):
                    print("NAN appears in tksatf_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('tksatf lev' + c, tksatf_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'tksatf_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'tksatf_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'tksatf_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         tksatf_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(tksatf_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'tksatf_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_tksatf[nsl] = tksatf_patches[0]

        # (16) thermal conductivity of unfrozen saturated soil [W/m-K]
        if mpi.p_is_io:
            dt = DataType(gblock)
            tkdry_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'tkdry.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'tkdry.nc'
            tkdry_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'tkdry_l' + c.strip(), mpi, gblock, gland,
                                                          tkdry_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, tkdry_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                  gland,
                                                                                                  zip=nl_colm['USE_zip_for_aggregation'],
                                                                                                  data_r8_2d_in1=tkdry_grid,
                                                                                                  area=area_one)
                    tkdry_patches[ipatch] = np.prod(np.power(tkdry_one, area_one / np.sum(area_one)))
                else:
                    tkdry_patches[ipatch] = -1.0e36

                if np.isnan(tkdry_patches[ipatch]):
                    print("NAN appears in tksatu_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('tkdry lev'+c, tkdry_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'tkdry_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'tkdry_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'tkdry_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         tkdry_patches, 1)
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(tkdry_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'tkdry_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_tkdry[nsl] = tkdry_patches[0]

        # (17) thermal conductivity of frozen saturated soil [W/m-K]
        if mpi.p_is_io:
            dt = DataType(gblock)
            k_solids_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'k_solids.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'k_solids.nc'
            k_solids_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'k_solids_l' + c.strip(), mpi, gblock, gland,
                                                             k_solids_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, k_solids_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                     gland,
                                                                                                     zip=nl_colm[
                                                                                                         'USE_zip_for_aggregation'],
                                                                                                     data_r8_2d_in1=k_solids_grid,
                                                                                                     area=area_one)
                    k_solids_patches[ipatch] = np.prod(np.power(k_solids_one, area_one / np.sum(area_one)))
                else:
                    k_solids_patches[ipatch] = -1.0e36

                if np.isnan(k_solids_patches[ipatch]):
                    print("NAN appears in tksatf_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('k_solids lev'+c, k_solids_patches, mpi, nl_colm)
        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'k_solids_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'k_solids_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'k_solids_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         k_solids_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(tksatf_patches, landpatch.landpatch.settyp, typpatch, srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'k_solids_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_k_solids[nsl] = k_solids_patches[0]

        # (18) thermal conductivity for dry soil [W/(m-K)]
        if mpi.p_is_io:
            dt = DataType(gblock)
            OM_density_s_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'OM_density_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'OM_density_s.nc'
            OM_density_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'OM_density_s_l' + c.strip(), mpi, gblock, gland,
                                                                 OM_density_s_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, OM_density_s_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch,
                                                                                                         ipatch,
                                                                                                         gland,
                                                                                                         zip=nl_colm[
                                                                                                             'USE_zip_for_aggregation'],
                                                                                                         data_r8_2d_in1=OM_density_s_grid,
                                                                                                         area=area_one)
                    OM_density_s_patches[ipatch] = sum(OM_density_s_one * (area_one / sum(area_one)))
                else:
                    OM_density_s_patches[ipatch] = -1.0e36

                if np.isnan(OM_density_s_patches[ipatch]):
                    print("NAN appears in OM_density_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('OM_density_s lev', OM_density_s_patches, mpi, nl_colm)

        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'OM_density_s_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'OM_density_s_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'OM_density_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         OM_density_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(OM_density_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'OM_density_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_OM_density[nsl] = OM_density_s_patches[0]

        # (19) thermal conductivity of soil solids [W/m-K]
        if mpi.p_is_io:
            dt = DataType(gblock)
            BD_all_s_grid = dt.allocate_block_data(gland)
            lndname = os.path.join(dir_rawdata, 'soil', 'BD_all_s.nc')
            if 'win' in sys.platform:
                lndname = dir_rawdata + '\\' + 'soil' + '\\' + 'BD_all_s.nc'
            BD_all_s_grid = CoLM_NetCDFBlock.ncio_read_block(lndname, 'BD_all_s_l' + c.strip(), mpi, gblock, gland,
                                                             BD_all_s_grid)
            if nl_colm['USEMPI']:
                pass

        if mpi.p_is_worker:
            for ipatch in range(numpatch):
                L = landpatch.landpatch.settyp[ipatch]
                if L != 0:
                    ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, pixel)
                    area_one, BD_all_s_one, _, _, _, _, _, _, _, _, _ = ard.aggregation_request_data(landpatch.landpatch, ipatch,
                                                                                                     gland,
                                                                                                     zip=nl_colm[
                                                                                                         'USE_zip_for_aggregation'],
                                                                                                     data_r8_2d_in1=BD_all_s_grid,
                                                                                                     area=area_one)
                    BD_all_s_patches[ipatch] = sum(BD_all_s_one * (area_one / sum(area_one)))
                else:
                    BD_all_s_patches[ipatch] = -1.0e36

                if np.isnan(BD_all_s_patches[ipatch]):
                    print("NAN appears in BD_all_s_patches.")
                    print(landpatch.landpatch.eindex[ipatch], landpatch.landpatch.settyp[ipatch])
            if nl_colm['USEMPI']:
                pass
        if nl_colm['USEMPI']:
            pass
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('BD_all_s lev' + c, BD_all_s_patches, mpi, nl_colm)
        if not nl_colm['SinglePoint']:
            lndname = os.path.join(landdir, 'BD_all_s_l' + c.strip() + '_patches.nc')
            if 'win' in sys.platform:
                lndname = landdir + '\\' + 'BD_all_s_l' + c.strip() + '_patches.nc'
            vectorOnes = CoLM_NetCDFVector(nl_colm, mpi, gblock)
            vectorOnes.ncio_create_file_vector(lndname, landpatch.landpatch)
            vectorOnes.ncio_define_dimension_vector(lndname, landpatch.landpatch, 'patch')
            vectorOnes.ncio_write_vector(lndname, 'BD_all_s_l' + c.strip() + '_patches.nc', 'patch', landpatch.landpatch,
                                         BD_all_s_patches, nl_colm['DEF_Srfdata_CompressLevel'])
            if nl_colm['SrfdataDiag']:
                typpatch = [ityp for ityp in range(nl_colm['N_land_classification'])]
                lndname = os.path.join(dir_model_landdata, 'diag', 'soil_parameters_' + cyear + '.nc')
                if 'win' in sys.platform:
                    lndname = dir_model_landdata + '\\' + 'diag' + '\\' + 'soil_parameters_' + cyear + '.nc'
                srfdataDiag.srfdata_map_and_write(BD_all_s_patches, landpatch.landpatch.settyp, typpatch,
                                                  srfdataDiag.m_patch2diag,
                                                  -1.0e36, lndname, 'BD_all_s_l' + c.strip(), compress=1,
                                                  write_mode='one')
        else:
            SITE_soil_BD_all[nsl] = BD_all_s_patches[0]

        # (20) OM_density [kg/m3]
    if mpi.p_is_worker:
        del vf_quartz_mineral_s_patches
        del vf_gravels_s_patches
        del vf_om_s_patches
        del vf_sand_s_patches
        del wf_gravels_s_patches
        del wf_sand_s_patches
        del OM_density_s_patches
        del BD_all_s_patches
        del theta_s_patches
        del psi_s_patches
        del lambda_patches

        if theta_r_patches is not None: del theta_r_patches
        if alpha_vgm_patches is not None: del alpha_vgm_patches
        if L_vgm_patches is not None: del L_vgm_patches
        if n_vgm_patches is not None: del n_vgm_patches
        if k_s_patches is not None: del k_s_patches
        if csol_patches is not None: del csol_patches
        if tksatu_patches is not None: del tksatu_patches
        if tkdry_patches is not None: del tkdry_patches
        if tksatf_patches is not None: del tksatf_patches
        if k_solids_patches is not None: del k_solids_patches
        if BA_alpha_patches is not None: del BA_alpha_patches
        if BA_beta_patches is not None: del BA_beta_patches
        if vf_quartz_mineral_s_one is not None: del vf_quartz_mineral_s_one
        if vf_gravels_s_one is not None: del vf_gravels_s_one
        if vf_om_s_one is not None: del vf_om_s_one
        if vf_sand_s_one is not None: del vf_sand_s_one
        if wf_gravels_s_one is not None: del wf_gravels_s_one
        if wf_sand_s_one is not None: del wf_sand_s_one
        if OM_density_s_one is not None: del OM_density_s_one
        if BD_all_s_one is not None: del BD_all_s_one
        if theta_s_one is not None: del theta_s_one
        if psi_s_one is not None: del psi_s_one
        if lambda_one is not None: del lambda_one
        if theta_r_one is not None: del theta_r_one
        if alpha_vgm_one is not None: del alpha_vgm_one
        if L_vgm_one is not None: del L_vgm_one
        if n_vgm_one is not None: del n_vgm_one
        if k_s_one is not None: del k_s_one
        if csol_one is not None: del csol_one
        if tksatu_one is not None: del tksatu_one
        if tkdry_one is not None: del tkdry_one
        if tksatf_one is not None: del tksatf_one
        if k_solids_one is not None: del k_solids_one
        if area_one is not None: del area_one
