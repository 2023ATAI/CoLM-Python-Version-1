import numpy as np
import sys
import os

import CoLM_Utils
from CoLM_Vars_TimeInvariants import Vars_TimeInvariants
from CoLM_Vars_TimeVariables import Vars_TimeVariables
import CoLM_LakeDepthReadin
from CoLM_SoilParametersReadin import SoilParametersReadin
from CoLM_Hydro_SoilFunction import Hydro_SoilFunction
import CoLM_HtopReadin
from CoLM_OrbCoszen import orb_coszen
from CoLM_DataType import DataType
from CoLM_NetCDFSerial import NetCDFFile
import CoLM_NetCDFBlock
from CoLM_Grid import  Grid_type
from CoLM_LAIEmpirical import  LAI_empirical
from CoLM_LAIReadin import  LAI_readin
import CoLM_RangeCheck
import CoLM_TimeManager
from CoLM_Mapping_Pset2Grid import MappingPset2Grid
# from CoLM_Const_Physical import CoLM_Const_Physical
from CoLM_IniTimeVariable import IniTimeVariable


def initialize(mpi, const_physical, landpatch, landpft, nl_colm,casename, dir_landdata, dir_restart,
        idate, lc_year, greenwich, const_LC, var_global, srfdata, gblock, pixel, mesh, lulcc_call=False):

    """
    Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
    software         : Initialization routine for land surface model

    Args:
        patchtype
        lc_year
    Returns:
        patchtype

    """
    # const_physical = CoLM_Const_Physical()
    netcdf_file = NetCDFFile(mpi)
    nl_soil = var_global.nl_soil
    dir_restart = dir_restart
    idate = idate
    lc_year = lc_year
    greenwich = greenwich
    const_LC = const_LC
    var_global = var_global
    srfdata = srfdata
    gblock = gblock
    gsoil = Grid_type(nl_colm, gblock)
    gsnow = Grid_type(nl_colm, gblock)
    gcn = Grid_type(nl_colm, gblock)
    gwtd = Grid_type(nl_colm, gblock)
    msoil2p = MappingPset2Grid(mpi, nl_colm, gblock)
    ms2p = MappingPset2Grid(mpi, nl_colm, gblock)
    mc2p = MappingPset2Grid(mpi, nl_colm, gblock)
    mc2f = MappingPset2Grid(mpi, nl_colm, gblock)
    msnow2p = MappingPset2Grid(mpi, nl_colm, gblock)
    zi_soimm = np.zeros(var_global.nl_soil, dtype=np.float64)
    vliq_r = np.zeros(var_global.nl_soil, dtype=np.float64)
    if nl_colm['Campbell_SOIL_MODEL']:
        nprms = 1
    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        nprms = 5
    prms = np.zeros((nprms, var_global.nl_soil), dtype=np.float64)
    use_wtd = False
    nl_soil_ini = 0
# ----------------------------------------------------------------------------------------------------------------------
# Allocates memory for CoLM 1d [ ] variables
# ----------------------------------------------------------------------------------------------------------------------

    VTI = Vars_TimeInvariants(nl_colm, mpi,landpatch,var_global)
    VTV = Vars_TimeVariables (nl_colm, mpi,landpatch,var_global,gblock)

#     allocate_TimeInvariants()
#     allocate_TimeVariables()

    #----------------------------------------------------------------------------------------------------------------------
#1. INITIALIZE TIME INVARIANT VARIABLES
#----------------------------------------------------------------------------------------------------------------------
    if mpi.p_is_worker:  # Equivalent to "if (p_is_worker) then"
        if landpatch.numpatch > 0:
            patchclass = landpatch.landpatch.settyp
            patchmask = True   # Initialize patchmask as a list of True values

        for ipatch in range(landpatch.numpatch):
            m = patchclass[ipatch]
            VTI.patchtype[ipatch] = const_LC.patchtypes[m - 1]

            if nl_colm['DEF_URBAN_ONLY'] and m.ne.URBAN:
                patchmask[ipatch] = False  # Set the patchmask to False if condition is met
                continue  # Skip the rest of the loop for this patch if condition is met
        landpatch.landpatch.get_lonlat_radian(VTI.patchlonr, VTI.patchlatr, var_global.pi, pixel)

        if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
            if landpft.numpft > 0:
                pftclass = landpft.settyp

    if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
        pass
        # pct_readin(dir_landdata, lc_year)

# ----------------------------------------------------------------------------------------------------------------------
# (1.1) Ponding water
# ----------------------------------------------------------------------------------------------------------------------
    if nl_colm['DEF_USE_BEDROCK']:
        pass

    if mpi.p_is_worker:
        if landpatch.numpatch > 0:
            VTV.wdsrf[:] = 0    # Surface water depth
            VTV.wetwat[:] = 0
            # Assign values to wetwat where patchtype equals 2 (for wetland)
            VTV.wetwat[VTI.patchtype == 2] = 200.0
# ----------------------------------------------------------------------------------------------------------------------
# (1.2) Lake depth and layers' thickness
# ----------------------------------------------------------------------------------------------------------------------

    # lakedepth, VTI = CoLM_LakeDepthReadin.lakedepth_readin(dir_landdata, lc_year, var_global.nl_lake, VTI)
    VTI = CoLM_LakeDepthReadin.lakedepth_readin(nl_colm, landpatch, mpi, gblock, dir_landdata, lc_year, var_global.nl_lake, VTI)

# ----------------------------------------------------------------------------------------------------------------------
# (1.3) Read in the soil parameters of the patches of the gridcells
# ----------------------------------------------------------------------------------------------------------------------
    SP = SoilParametersReadin(nl_colm, landpatch, mpi, var_global)
    VTI = SP.soil_parameters_readin(dir_landdata, lc_year, srfdata, VTI, gblock)
    HS = Hydro_SoilFunction(nl_colm)

    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                VTI.psi0[:] = -1.0
                for ipatch in range(landpatch.numpatch):
                    for i in range(var_global.nl_soil):
                        VTI.sc_vgm[i,ipatch], VTI.fc_vgm[i,ipatch] = (
                            HS.get_derived_parameters_vGM(VTI.psi0[i,ipatch],VTI.alpha_vgm[i,ipatch], VTI.n_vgm[i,ipatch]))

# ----------------------------------------------------------------------------------------------------------------------
# (1.4) Plant time-invariant variables
# ----------------------------------------------------------------------------------------------------------------------
#   read global tree top height from nc file

    CoLM_HtopReadin.HTOP_readin(nl_colm, landpatch, mpi, gblock, dir_landdata, lc_year, srfdata, patchclass, VTI, const_LC)

    # Check if URBAN_MODEL is defined
    if nl_colm['URBAN_MODEL']:
        pass
# ----------------------------------------------------------------------------------------------------------------------
# (1.5) Initialize TUNABLE constants
# ----------------------------------------------------------------------------------------------------------------------
    zlnd = 0.01  # Roughness length for soil [m]
    zsno = 0.0024  # Roughness length for snow [m]
    csoilc = 0.004  # Drag coefficient for soil under canopy [-]
    dewmx = 0.1  # Maximum dew
    wtfact = 0.38  # Maximum saturated fraction (global mean; see Niu et al., 2005)
    capr = 0.34  # Tuning factor to turn first layer T into surface T
    cnfac = 0.5  # Crank Nicholson factor between 0 and 1
    ssi = 0.033  # Irreducible water saturation of snow
    wimp = 0.05  # Water impremeable if porosity less than wimp
    pondmx = 10.0  # Ponding depth (mm)
    smpmax = -1.5e5  # Wilting point potential in mm
    smpmin = -1.e8  # Restriction for min of soil poten. (mm)
    trsmx0 = 2.e-4  # Max transpiration for moist soil+100% veg. [mm/s]
    tcrit = 2.5  # Critical temp. to determine rain or snow
    wetwatmax = 200.0  # Maximum wetland water (mm)
    if nl_colm['BGC']:
        print('need to finish the code for BGC')
# ----------------------------------------------------------------------------------------------------------------------
# (1.6) Write out as a restart file [histTimeConst]
# ----------------------------------------------------------------------------------------------------------------------
    # Check if RangeCheck is defined
    if nl_colm['RangeCheck']:
        VTI.check_TimeInvariants()

    # Write TimeInvariants
    VTI.WRITE_TimeInvariants(lc_year, casename, dir_restart, landpatch, gblock)

    # Check if USEMPI is defined
    if nl_colm['USEMPI']:
        pass

    # Check if it is the master process
    if mpi.p_is_master:
        print('Successfully Initialize the Land Time-Invariants', file=sys.stdout)
# ----------------------------------------------------------------------------------------------------------------------
# [2] INITIALIZE TIME-VARYING VARIABLES
# as subgrid vectors of length [numpatch]
# initial run: create the time-varying variables based on :
#              i) observation (NOT CODING CURRENTLY), or
#             ii) some already-known information (NO CODING CURRENTLY), or
#            iii) arbitrarily
# ----------------------------------------------------------------------------------------------------------------------
    # ............................
    # 2.1 current time of model run
    # ............................
    greenwich = CoLM_TimeManager.initimetype(greenwich, nl_colm['SinglePoint'])

    if mpi.p_is_master:
        if not greenwich:
            print("\033[1;31mNotice: greenwich false, local time is used.\033[0m")
    # ................................
    # 2.2 cosine of solar zenith angle
    # ................................
    calday = CoLM_TimeManager.calendarday_date(idate, greenwich)

    if mpi.p_is_worker:
        for i in range(landpatch.numpatch):
            VTV.coszen[i] = orb_coszen(calday, VTI.patchlonr[i], VTI.patchlatr[i])
    # ...........................................
    # 2.3 READ in or GUSSES land state information
    # ...........................................
    # for SOIL INIT of water, temperature, snow depth
    dt = DataType(gblock)
    if nl_colm['DEF_USE_SoilInit']:
        fsoildat = nl_colm['DEF_file_soil_init']
        if mpi.p_is_master:
            use_soilini = os.path.exists(fsoildat)

        # Broadcast the value of use_soilini to all processes if MPI is used
        if nl_colm['USEMPI']:
            pass

        if use_soilini:
            # Read soil_z from the file
            soil_z = netcdf_file.ncio_read_bcast_serial(fsoildat, 'soil_z')
            nl_soil_ini = len(soil_z)

            gsoil.define_from_file(fsoildat, latname='lat', lonname='lon')

            month, mday = CoLM_TimeManager.julian2monthday(idate[0], idate[1])

            if mpi.p_is_master:
                missing_value = netcdf_file.ncio_get_attr(fsoildat, 'zwt', 'missing_value')

            if mpi.p_is_io:
                # soil layer temperature (K)
                soil_t_grid = dt.allocate_block_data2d(gsoil, nl_soil_ini)
                soil_t_grid = CoLM_NetCDFBlock.ncio_read_block_time6(fsoildat, 'soiltemp', gsoil, nl_soil_ini, month, soil_t_grid, mpi, gblock)
                # soil layer wetness (-)
                soil_w_grid = dt.allocate_block_data2d(gsoil, nl_soil_ini)
                soil_w_grid = CoLM_NetCDFBlock.ncio_read_block_time6(fsoildat, 'soilwat', gsoil, nl_soil_ini, month, soil_t_grid, mpi, gblock)
                # snow depth (m)
                zwt_grid = dt.allocate_block_data(gsoil)
                zwt_grid = CoLM_NetCDFBlock.ncio_read_block(fsoildat, 'zwt', gsoil, month, zwt_grid, mpi, gblock)

            # gsoil.define_from_file(fsoildat)

            if mpi.p_is_worker:
                if landpatch.numpatch > 0:
                    nl_soil_ini = var_global.nl_soil
                    soil_t = np.zeros((nl_soil_ini, landpatch.numpatch))
                    soil_w = np.zeros((nl_soil_ini, landpatch.numpatch))
                    validval = np.zeros(landpatch.numpatch)
            msoil2p.build(gsoil, landpatch)
            msoil2p.map_aweighted(soil_t_grid, nl_soil_ini, soil_t)
            msoil2p.map_aweighted(soil_w_grid, nl_soil_ini, soil_w)
            msoil2p.map_aweighted(zwt_grid, VTV.zwt)

            if mpi.p_is_worker:
                for i in range(landpatch.numpatch):
                    if not validval[i]:
                        if VTI.patchtype[i]==3:
                            soil_t[:,0] = 250.0
                        else:
                            soil_t[:, i] = 280.0

                        soil_w[:, i] = 1.0
                        VTV.zwt[i] = 0.0
            if validval is not None:
                del validval

    else:
        use_soilini = False

    if not use_soilini:
        # not used, just for filling arguments
        if mpi.p_is_worker:
            soil_z = np.zeros(var_global.nl_soil)
            # snow_d = np.zeros(landpatch.numpatch)
            soil_t = np.zeros((var_global.nl_soil, landpatch.numpatch))
            soil_w = np.zeros((var_global.nl_soil, landpatch.numpatch))

        if nl_colm['BGC']:
            pass
            # if nl_colm['DEF_USE_CN_INIT']:
            #     fcndat = nl_colm['DEF_file_cn_init']
            #     if mpi.p_is_master:
            #         use_cnini = os.path.exists(fcndat)
            #         if use_cnini:
            #             print('Use water table depth and derived equilibrium state to initialize soil water content:',fcndat)
            #         else:
            #             print('no initial data for biogeochemistry: ',fcndat)
            #
            #     if nl_colm['USEMPI']:
            #         pass
            #
            #     if use_cnini:
            #         gcn.define_from_file(fcndat, "lat", "lon")
            #         mc2p.build(gcn, landpatch)
            #         mc2f.build(gcn, landpft)
            #
            #         if mpi.p_is_io:
            #             # soil layer litter & carbon(gCm - 3)
            #
            #             litr1c_grid = dt.allocate_block_data(gcn, nl_soil)
            #             litr1c_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'litr1c_vr',mpi, gcn, nl_soil, litr1c_grid)
            #
            #             litr2c_grid = dt.allocate_block_data(gcn, nl_soil)
            #             litr2c_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'litr2c_vr',mpi, gcn, nl_soil, litr2c_grid)
            #
            #             litr3c_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             litr3c_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'litr3c_vr',mpi, gcn, nl_soil, litr3c_grid)
            #
            #
            #             cwdc_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             cwdc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'cwdc_vr',mpi, gcn, nl_soil, cwdc_grid)
            #
            #
            #             soil1c_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             soil1c_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil1c_vr', mpi,gcn, nl_soil, soil1c_grid)
            #
            #
            #             soil2c_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             soil2c_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil2c_vr',mpi, gcn, nl_soil, soil2c_grid)
            #
            #
            #             soil3c_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             soil3c_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil3c_vr',mpi, gcn, nl_soil, soil3c_grid)
            #
            #             # soil layer litter & nitrogen(gNm - 3)
            #
            #             litr1n_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             litr1n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'litr1n_vr', mpi, gcn, nl_soil, litr1n_grid)
            #
            #
            #             litr2n_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             litr2n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'litr2n_vr',mpi,  gcn, nl_soil, litr2n_grid)
            #
            #
            #             litr3n_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             litr3n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'litr3n_vr',mpi,  gcn, nl_soil, litr3n_grid)
            #
            #
            #             cwdn_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             cwdn_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'cwdn_vr',mpi,  gcn, nl_soil, cwdn_grid)
            #
            #
            #             soil1n_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             soil1n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil1n_vr', mpi, gcn, nl_soil, soil1n_grid)
            #
            #
            #             soil2n_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             soil2n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil2n_vr',mpi,  gcn, nl_soil, soil2n_grid)
            #
            #
            #             soil3n_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             soil3n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil3n_vr', mpi, gcn, nl_soil, soil3n_grid)
            #
            #
            #             # soil3n_grid = dt.allocate_block_data(gcn, nl_soil)
            #             #
            #             # soil3n_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'soil3n_vr', gcn, nl_soil, soil3n_grid)
            #
            #
            #             smin_nh4_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             smin_nh4_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'smin_nh4_vr',mpi,  gcn, nl_soil, smin_nh4_grid)
            #
            #
            #             smin_no3_grid = dt.allocate_block_data(gcn, nl_soil)
            #
            #             smin_no3_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'smin_no3_vr', mpi, gcn, nl_soil, smin_no3_grid)
            #
            #
            #             leafc_grid = dt.allocate_block_data(gcn)
            #
            #             leafc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'leafc', mpi, gcn, leafc_grid)
            #
            #
            #             leafc_storage_grid = dt.allocate_block_data(gcn)
            #
            #             leafc_storage_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'leafc_storage', mpi, gcn, leafc_storage_grid)
            #
            #
            #             frootc_grid = dt.allocate_block_data(gcn)
            #
            #             frootc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'frootc',mpi,  gcn, frootc_grid)
            #
            #
            #             frootc_storage_grid = dt.allocate_block_data(gcn)
            #
            #             frootc_storage_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'frootc_storage', mpi, gcn, frootc_storage_grid)
            #
            #
            #             livestemc_grid = dt.allocate_block_data(gcn, )
            #
            #             livestemc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'livestemc',mpi,  gcn, livestemc_grid)
            #
            #
            #             deadstemc_grid = dt.allocate_block_data(gcn, )
            #
            #             deadstemc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'deadstemc', mpi, gcn, deadstemc_grid)
            #
            #
            #             livecrootc_grid = dt.allocate_block_data(gcn, )
            #
            #             livecrootc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'livecrootc',mpi,  gcn, livecrootc_grid)
            #
            #
            #             deadcrootc_grid = dt.allocate_block_data(gcn, )
            #
            #             deadcrootc_grid = CoLM_NetCDFBlock.ncio_read_block(fcndat, 'deadcrootc', mpi, gcn, deadcrootc_grid)
            #
            #         if mpi.p_is_worker:
            #             litr1c_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             litr2c_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             litr3c_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             cwdc_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             soil1c_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             soil2c_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             soil3c_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             litr1n_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             litr2n_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             litr3n_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             cwdn_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             soil1n_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             soil2n_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             soil3n_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             min_nh4_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             min_no3_vr= np.zeros((nl_soil, landpatch.numpatch))
            #             leafcin_p= np.zeros((landpft.numpft))
            #             leafc_storagein_p= np.zeros((landpft.numpft))
            #             frootcin_p= np.zeros((landpft.numpft))
            #             frootc_storagein_p= np.zeros((landpft.numpft))
            #             livestemcin_p= np.zeros((landpft.numpft))
            #             deadstemcin_p= np.zeros((landpft.numpft))
            #             livecrootcin_p= np.zeros((landpft.numpft))
            #             deadcrootcin_p= np.zeros((landpft.numpft))
            #
            #
            #         mc2p.map_aweighted(litr1c_grid, nl_soil, litr1c_vr)
            #
            #         mc2p.map_aweighted(litr2c_grid, nl_soil, litr2c_vr)
            #
            #         mc2p.map_aweighted(litr3c_grid, nl_soil, litr3c_vr)
            #
            #         mc2p.map_aweighted(cwdc_grid, nl_soil, cwdc_vr)
            #
            #         mc2p.map_aweighted(soil1c_grid, nl_soil, soil1c_vr)
            #
            #         mc2p.map_aweighted(soil2c_grid, nl_soil, soil2c_vr)
            #
            #         mc2p.map_aweighted(soil3c_grid, nl_soil, soil3c_vr)
            #
            #         mc2p.map_aweighted(litr1n_grid, nl_soil, litr1n_vr)
            #
            #         mc2p.map_aweighted(litr2n_grid, nl_soil, litr2n_vr)
            #
            #         mc2p.map_aweighted(litr3n_grid, nl_soil, litr3n_vr)
            #
            #         mc2p.map_aweighted(cwdn_grid, nl_soil, cwdn_vr)
            #
            #         mc2p.map_aweighted(soil1n_grid, nl_soil, soil1n_vr)
            #
            #         mc2p.map_aweighted(soil2n_grid, nl_soil, soil2n_vr)
            #
            #         mc2p.map_aweighted(soil3n_grid, nl_soil, soil3n_vr)
            #
            #         mc2p.map_aweighted(smin_nh4_grid, nl_soil, min_nh4_vr)
            #
            #         mc2p.map_aweighted(smin_no3_grid, nl_soil, min_no3_vr)
            #
            #         mc2f.map_aweighted(leafc_grid, leafcin_p)
            #
            #         mc2f.map_aweighted(leafc_storage_grid, leafc_storagein_p)
            #
            #         mc2f.map_aweighted(frootc_grid, frootcin_p)
            #
            #         mc2f.map_aweighted(frootc_storage_grid, frootc_storagein_p)
            #
            #         mc2f.map_aweighted(livestemc_grid, livestemcin_p)
            #
            #         mc2f.map_aweighted(deadstemc_grid, deadstemcin_p)
            #
            #         mc2f.map_aweighted(livecrootc_grid, livecrootcin_p)
            #
            #         mc2f.map_aweighted(deadcrootc_grid, deadcrootcin_p)
            #
            #         if mpi.p_is_worker:
            #             for i in range(0, landpatch.numpatch):
            #                 ps = landpft.patch_pft_s[i]
            #                 pe = landpft.patch_pft_e[i]
            #                 for nsl in range(0, nl_soil):
            #                     decomp_cpools_vr[nsl,i_met_lit,i] = litr1c_vr[nsl,i]
            #                     decomp_cpools_vr[nsl,i_cel_lit,i] = litr2c_vr[nsl,i]
            #                     decomp_cpools_vr[nsl,i_lig_lit,i] = litr3c_vr[nsl,i]
            #                     decomp_cpools_vr[nsl,i_cwd,i] = cwdc_vr[nsl,i]
            #                     decomp_cpools_vr[nsl,i_soil1,i] = soil1c_vr[nsl,i]
            #                     decomp_cpools_vr[nsl,i_soil2,i] = soil2c_vr[nsl,i]
            #                     decomp_cpools_vr[nsl,i_soil3,i] = soil3c_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_met_lit,i] = litr1n_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_cel_lit,i] = litr2n_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_lig_lit,i] = litr3n_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_cwd,i] = cwdn_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_soil1,i] = soil1n_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_soil2,i] = soil2n_vr[nsl,i]
            #                     decomp_npools_vr[nsl,i_soil3,i] = soil3n_vr[nsl,i]
            #                     smin_nh4_vr[nsl,i] = min_nh4_vr[nsl,i]
            #                     smin_no3_vr[nsl,i] = min_no3_vr[nsl,i]
            #                     sminn_vr[nsl,i] = min_nh4_vr[nsl,i] + min_no3_vr[nsl,i]
            #
            #                 if patchtype[i] == 0:
            #                     for m in range(ps, pe):
            #                         ivt = pftclass[m]
            #                         if isevg[ivt]:
            #                             leafc_p[m] = leafcin_p[m]
            #                             frootc_p[m] = frootcin_p[m]
            #                         else:
            #                             leafc_p[m] = leafcin_p[m]
            #                             leafc_storage_p[m] = leafc_storagein_p[m]
            #                             frootc_p[m] = frootcin_p[m]
            #                             frootc_storage_p[m] = frootc_storagein_p[m]
            #
            #                         if woody[ivt] == 1:
            #                             deadstemc_p[m] = deadstemcin_p[m]
            #                             livestemc_p[m] = livestemcin_p[m]
            #                             deadcrootc_p[m] = deadcrootcin_p[m]
            #                             livecrootc_p[m] = livecrootcin_p[m]
            # else:
            #     use_cnini = False

        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                snow_d = np.zeros(landpatch.numpatch)

            if nl_colm['DEF_USE_SnowInit']:
                fsnowdat = nl_colm['DEF_file_SnowInit']
                if mpi.p_is_master:
                    use_snowini = os.path.exists(fsnowdat)

                # use_snowini = comm.bcast(use_snowini, root=p_root)

                if use_snowini:
                    gsnow.define_from_file(fsnowdat, latname='lat', lonname='lon')

                    month, mday = CoLM_TimeManager.julian2monthday(idate[0], idate[1])

                    if mpi.p_is_master:
                        missing_value = netcdf_file.ncio_get_attr(fsnowdat, 'snowdepth', 'missing_value')

                    # missing_value = comm.bcast(missing_value, root=p_root)

                    if mpi.p_is_io:
                        snow_d_grid = dt.allocate_block_data(gsnow)
                        snow_d_grid = CoLM_NetCDFBlock.ncio_read_block_time(fsnowdat, 'snowdepth', gsnow, month, snow_d_grid, mpi, gblock)

                    if mpi.p_is_worker:
                        if landpatch.numpatch > 0:
                            validval = np.zeros(landpatch.numpatch, dtype=bool)

                    msnow2p.build(gsnow, landpatch, snow_d_grid, missing_value, validval)
                    msnow2p.map_aweighted(snow_d_grid, snow_d)

                    if mpi.p_is_worker:
                        snow_d[~validval] = 0.0

                    if validval is not None:
                        del validval
            else:
                use_snowini = False

            fwtd = os.path.join(nl_colm['DEF_dir_runtime'], 'wtd.nc')

            if mpi.p_is_master:
                use_wtd = os.path.exists(fwtd)
                if use_soilini:
                    use_wtd = False
                if use_wtd:
                    print('\nUse water table depth and derived equilibrium state '
                        'to initialize soil water content:', fwtd)
        
            if use_wtd:
                month, mday = CoLM_TimeManager.julian2monthday(idate[0], idate[1])
                gwtd.define_from_file(fwtd)
                if mpi.p_is_io:
                    wtd_xy = dt.allocate_block_data(gwtd)
                    wtd_xy = CoLM_NetCDFBlock.ncio_read_block_time(fwtd, 'wtd', gwtd, month, wtd_xy, mpi, gblock)
                m_wtd2p = MappingPset2Grid(mpi,nl_colm,gblock)
                m_wtd2p.build (gwtd, landpatch.landpatch, pixel, mesh.mesh)
                VTV.zwt = m_wtd2p.map_aweighted_2d(wtd_xy, VTV.zwt, var_global.spval)
    # ...................
    # 2.4 LEAF area index
    # ...................
    if nl_colm['DYN_PHENOLOGY']:
        #! CREAT fraction of vegetation cover, greenness, leaf area index, stem index
        if mpi.p_is_worker:
            for i in range (landpatch.numpatch):
                if use_soilini:
                    for nsl in range(var_global.nl_soil):
                        VTV.t_soisno[nsl, i] = CoLM_Utils.polint(soil_z,soil_t[:,i],nl_soil_ini,var_global.z_soi[nsl])
                else:
                    VTV.t_soisno[0:, i] = 283.
            VTV.tlai = 0.0;  VTV.tsai[:] = 0.0; VTV.green[:] = 0.0; VTV.fveg[:] = 0.0
            for i in range (landpatch.numpatch):
                # ! Call Ecological Model()
                ltyp = VTI.patchtype[i]
                if ltyp > 0:
                    VTV.tlai[i], VTV.tsai[i], VTV.fveg[i], VTV.green[i] = LAI_empirical(nl_colm, ltyp, var_global.nl_soil, const_LC.rootfr[0:, i], VTV.t_soisno[0:, i])
    else:
        year = idate[0]
        jday = idate[1]

        if nl_colm['DEF_LAI_MONTHLY']:
            month, mday = CoLM_TimeManager.julian2monthday(year, jday)
            if nl_colm['DEF_LAI_CHANGE_YEARLY']:
                # 08/03/2019, yuan: read global LAI/SAI data
                LAI_readin(nl_colm, year, month, dir_landdata, srfdata, mpi, landpatch, patchclass, VTV, const_LC)
                # if URBAN_MODEL is defined, read UrbanLAI data
                if nl_colm['URBAN_MODEL']:
                    pass
            else:
                LAI_readin(nl_colm, lc_year, month, dir_landdata, srfdata, mpi, landpatch, patchclass, VTV, const_LC)
                # if URBAN_MODEL is defined, read UrbanLAI data
                if nl_colm['URBAN_MODEL']:
                    pass
        else:
            Julian_8day = int(CoLM_TimeManager.calendarday_date(idate) - 1) // 8 * 8 + 1
            LAI_readin(nl_colm, year, Julian_8day, dir_landdata, srfdata, mpi, landpatch, patchclass, VTV, const_LC)

        # if RangeCheck is defined, check LAI and SAI vector data
        if nl_colm['RangeCheck']:
            CoLM_RangeCheck.check_vector_data('LAI ', VTV.tlai, mpi, nl_colm)
            CoLM_RangeCheck.check_vector_data('SAI ', VTV.tsai, mpi, nl_colm)

        if nl_colm['CROP']:
            pass

    # ..............................................................................
    # 2.5 initialize time-varying variables, as subgrid vectors of length [numpatch]
    # ..............................................................................
    # ------------------------------------------
    # PLEASE
    # PLEASE UPDATE
    # PLEASE UPDATE when have the observed lake status
    if mpi.p_is_worker:
        VTV.t_lake      [:,:] = 285.
        VTV.lake_icefrac[:,:] = 0.
        VTV.savedtke1   [:]   = const_physical.tkwat
        z_soisno = np.full((var_global.nl_soil - var_global.maxsnl - 1, landpatch.numpatch), var_global.spval)
        dz_soisno = np.full((var_global.nl_soil - var_global.maxsnl - 1, landpatch.numpatch), var_global.spval)
        for i in range (landpatch.numpatch):
            z_soisno[:nl_soil, i] = var_global.z_soi[:nl_soil]
            dz_soisno[:nl_soil, i] = var_global.dz_soi[:nl_soil]

        zc_soimm = var_global.z_soi * 1000.0
        zi_soimm[0] = 0
        zi_soimm[0:var_global.nl_soil+1] = var_global.zi_soi * 1000.0

        for i in range(landpatch.numpatch):
            m = patchclass[i]

            if use_wtd:
                zwtmm = VTV.zwt[i] * 1000.0
            if nl_colm['Campbell_SOIL_MODEL']:
                vliq_r[:] = 0.
                prms[0,:nl_soil] = VTI.bsw[0:nl_soil,i]
            if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
                vliq_r[:] = VTI.theta_r[:nl_soil, i]
                prms[0, :nl_soil] = VTI.alpha_vgm[:nl_soil, i]
                prms[1, :nl_soil] = VTI.n_vgm[:nl_soil, i]
                prms[2, :nl_soil] = VTI.L_vgm[:nl_soil, i]
                prms[3, :nl_soil] = VTI.sc_vgm[:nl_soil, i]
                prms[4, :nl_soil] = VTI.fc_vgm[:nl_soil, i]

            # Initialize time-varying variables
            # ITV.iniTimeVar()
            init_time = IniTimeVariable(mpi, nl_colm, landpatch.landpatch, var_global, VTV)
            if nl_colm['BGC']:
                pass
               #  init_time.ini_time_Var(i, patchtype[i]
               # ,porsl[1:,i],psi0[1:,i],hksati[1:,i]
               # ,soil_s_v_alb[i],soil_d_v_alb[i],soil_s_n_alb[i],soil_d_n_alb[i]
               # ,z0m[i],zlnd,htop[i],z0mr[m],chil[m],rho[1:,1:,m],tau[1:,1:,m]
               # ,z_soisno[maxsnl+1:,i],dz_soisno[maxsnl+1:,i]
               # ,t_soisno[maxsnl+1:,i],wliq_soisno[maxsnl+1:,i],wice_soisno[maxsnl+1:,i]
               # ,smp[1:,i],hk[1:,i],zwt[i],wa[i],vegwp[1:,i],gs0sun[i],gs0sha[i]
               #  , t_grnd[i], tleaf[i], ldew[i], ldew_rain[i], ldew_snow[i], sag[i], scv[i]
               #  , snowdp[i], fveg[i], fsno[i], sigf[i], green[i], lai[i], sai[i], coszen[i]
               #  , snw_rds[:, i], mss_bcpho[:, i], mss_bcphi[:, i], mss_ocpho[:, i], mss_ocphi[:, i]
               #  , mss_dst1[:, i], mss_dst2[:, i], mss_dst3[:, i], mss_dst4[:, i]
               #  , alb[1:, 1:, i], ssun[1:, 1:, i], ssha[1:, 1:, i]
               #  , ssoi[1:, 1:, i], ssno[1:, 1:, i], ssno_lyr[1:, 1:,:, i]
               #  , thermk[i], extkb[i], extkd[i]
               #  , trad[i], tref[i], qref[i], rst[i], emis[i], zol[i], rib[i]
               #  , ustar[i], qstar[i], tstar[i], fm[i], fh[i], fq[i]
               #             , use_cnini, totlitc[i], totsomc[i], totcwdc[i],
               #             decomp_cpools[:, i], decomp_cpools_vr[:,:, i]
               #  , ctrunc_veg[i], ctrunc_soil[i], ctrunc_vr[:, i]
               #  , totlitn[i], totsomn[i], totcwdn[i], decomp_npools[:, i], decomp_npools_vr[:,:, i]
               #  , ntrunc_veg[i], ntrunc_soil[i], ntrunc_vr[:, i]
               #  , totvegc[i], totvegn[i], totcolc[i], totcoln[i], col_endcb[i], col_begcb[i], col_endnb[i], col_begnb[
               #      i]
               #  , col_vegendcb[i], col_vegbegcb[i], col_soilendcb[i], col_soilbegcb[i]
               #  , col_vegendnb[i], col_vegbegnb[i], col_soilendnb[i], col_soilbegnb[i]
               #  , col_sminnendnb[i], col_sminnbegnb[i]
               #  , altmax[i], altmax_lastyear[i], altmax_lastyear_indx[i], lag_npp[i]
               #  , sminn_vr[:, i], sminn[i], smin_no3_vr[:, i], smin_nh4_vr[:, i]
               #  , prec10[i], prec60[i], prec365[i], prec_today[i], prec_daily[:, i], tsoi17[i], rh30[i], accumnstep[
               #      i], skip_balance_check[i]
               #  , decomp0_cpools_vr[:,:, i], decomp0_npools_vr[:,:, i]
               #  , I_met_c_vr_acc[:, i], I_cel_c_vr_acc[:, i], I_lig_c_vr_acc[:, i], I_cwd_c_vr_acc[:, i]
               #  , AKX_met_to_soil1_c_vr_acc[:, i], AKX_cel_to_soil1_c_vr_acc[:, i], AKX_lig_to_soil2_c_vr_acc[:, i], AKX_soil1_to_soil2_c_vr_acc[:, i]
               #  , AKX_cwd_to_cel_c_vr_acc[:, i], AKX_cwd_to_lig_c_vr_acc[:, i], AKX_soil1_to_soil3_c_vr_acc[:, i], AKX_soil2_to_soil1_c_vr_acc[:, i]
               #  , AKX_soil2_to_soil3_c_vr_acc[:, i], AKX_soil3_to_soil1_c_vr_acc[:, i]
               #  , AKX_met_exit_c_vr_acc[:, i], AKX_cel_exit_c_vr_acc[:, i], AKX_lig_exit_c_vr_acc[:, i], AKX_cwd_exit_c_vr_acc[:, i]
               #  , AKX_soil1_exit_c_vr_acc[:, i], AKX_soil2_exit_c_vr_acc[:, i], AKX_soil3_exit_c_vr_acc[:, i]
               #  , diagVX_c_vr_acc[:,:, i], upperVX_c_vr_acc[:,:, i], lowerVX_c_vr_acc[:,:, i]
               #  , I_met_n_vr_acc[:, i], I_cel_n_vr_acc[:, i], I_lig_n_vr_acc[:, i], I_cwd_n_vr_acc[:, i]
               #  , AKX_met_to_soil1_n_vr_acc[:, i], AKX_cel_to_soil1_n_vr_acc[:, i], AKX_lig_to_soil2_n_vr_acc[:, i], AKX_soil1_to_soil2_n_vr_acc[:, i]
               #  , AKX_cwd_to_cel_n_vr_acc[:, i], AKX_cwd_to_lig_n_vr_acc[:, i], AKX_soil1_to_soil3_n_vr_acc[:, i], AKX_soil2_to_soil1_n_vr_acc[:, i]
               #  , AKX_soil2_to_soil3_n_vr_acc[:, i], AKX_soil3_to_soil1_n_vr_acc[:, i]
               #  , AKX_met_exit_n_vr_acc[:, i], AKX_cel_exit_n_vr_acc[:, i], AKX_lig_exit_n_vr_acc[:, i], AKX_cwd_exit_n_vr_acc[:, i]
               #  , AKX_soil1_exit_n_vr_acc[:, i], AKX_soil2_exit_n_vr_acc[:, i], AKX_soil3_exit_n_vr_acc[:, i]
               #  , diagVX_n_vr_acc[:,:, i], upperVX_n_vr_acc[:,:, i], lowerVX_n_vr_acc[:,:, i]
               #  )
            else:
                # print(i, m,'**********************')
                m = m-1
            #互换唯独wliq_soisno
                init_time.ini_time_Var(nl_soil, var_global.zi_soi, var_global.nvegwcs, const_physical.tfrz,const_physical.denh2o,const_physical.denice, var_global.maxsnl,VTV.tlai,VTV.tsai,i, VTI.patchtype[i]
               ,VTI.porsl[:,i],VTI.psi0[:,i],VTI.hksati[:,i]
               ,VTI.soil_s_v_alb[i],VTI.soil_d_v_alb[i],VTI.soil_s_n_alb[i],VTI.soil_d_n_alb[i]
               ,VTV.z0m[i],VTI.zlnd,VTI.htop[i],const_LC.z0mr[m],const_LC.chil[m],const_LC.rho[:,:,m],const_LC.tau[:,:,m]
               ,z_soisno[:var_global.maxsnl+1,i],dz_soisno[:var_global.maxsnl+1,i]
               ,VTV.t_soisno[:var_global.maxsnl+1,i],VTV.wliq_soisno[:var_global.maxsnl+1,i],VTV.wice_soisno[:var_global.maxsnl+1,i]
               ,VTV.smp[:,i],VTV.hk[:,i],VTV.zwt[i],VTV.wa[i],VTV.vegwp[:,i],VTV.gs0sun[i],VTV.gs0sha[i]
                , VTV.t_grnd[i], VTV.tleaf[i], VTV.ldew[i], VTV.ldew_rain[i], VTV.ldew_snow[i], VTV.sag[i], VTV.scv[i]
                , VTV.snowdp[i], VTV.fveg[i], VTV.fsno[i], VTV.sigf[i], VTV.green[i], VTV.lai[i], VTV.sai[i], VTV.coszen[i]
                , VTV.snw_rds[:, i], VTV.mss_bcpho[:, i], VTV.mss_bcphi[:, i], VTV.mss_ocpho[:, i], VTV.mss_ocphi[:, i]
                , VTV.mss_dst1[:, i], VTV.mss_dst2[:, i], VTV.mss_dst3[:, i], VTV.mss_dst4[:, i]
                , VTV.alb[:, :, i], VTV.ssun[:, :, i], VTV.ssha[:, :, i]
                , VTV.ssoi[:, :, i], VTV.ssno[:, :, i], VTV.ssno_lyr[:, :,:, i]
                , VTV.thermk[i], VTV.extkb[i], VTV.extkd[i]
                , VTV.trad[i], VTV.tref[i], VTV.qref[i], VTV.rst[i], VTV.emis[i], VTV.zol[i], VTV.rib[i]
                , VTV.ustar[i], VTV.qstar[i], VTV.tstar[i], VTV.fm[i],VTV.fh[i], VTV.fq[i]
                , use_soilini, nl_soil_ini, soil_z, soil_t[:, i], soil_w[:, i], use_snowini, snow_d[i]
                ,use_wtd, zwtmm, zc_soimm, zi_soimm, vliq_r, nprms, prms)
            # Urban model initialization
            if m == var_global.URBAN:
                pass
        for i in range(landpatch.numpatch):
            VTV.z_sno[var_global.maxsnl + 1:0, i] = z_soisno[var_global.maxsnl + 1:0, i]
            VTV.dz_sno[var_global.maxsnl + 1:0, i] = dz_soisno[var_global.maxsnl + 1:0, i]
    
    
    
    
    if nl_colm['CatchLateralFlow']:
        pass
    # ...............................................................
    # 2.6 Write out the model variables for restart run [histTimeVar]
    # ...............................................................
    if nl_colm['RangeCheck']:
        VTV.check_TimeVariables()

    # Write out time variables if not running LULCC
    if lulcc_call is not None:
        # ! only be called in runing MKINI, LULCC will be executed later
        VTV.WRITE_TimeVariables(idate, lc_year, casename, dir_restart)

    # Barrier synchronization for MPI
    if nl_colm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Successfully Initialize the Land Time-Varying Variables')

    # --------------------------------------------------
    # Deallocates memory for CoLM 1d [numpatch] variables
    # --------------------------------------------------
    if lulcc_call is not None:
        VTI.deallocate_TimeInvariants()
        VTV.deallocate_TimeVariables()
    if z_soisno is not None:
        del z_soisno
    if dz_soisno is not None:
        del dz_soisno
    if soil_w is not None:
        del soil_w
    if soil_z is not None:
        del soil_z
    if snow_d is not None:
        del snow_d
    if soil_t is not None:
        del soil_t
