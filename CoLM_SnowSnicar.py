# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Build pixelset "landpft" (Plant Function Type).
#
#    In CoLM, the global/regional area is divided into a hierarchical structure:
#    1. If GRIDBASED or UNSTRUCTURED is defined, it is
#       ELEMENT >>> PATCH
#    2. If CATCHMENT is defined, it is
#       ELEMENT >>> HRU >>> PATCH
#    If Plant Function Type classification is used, PATCH is further divided into PFT.
#    If Plant Community classification is used,     PATCH is further divided into PC.
#
#    "landpft" refers to pixelset PFT.
#
# Created by Shupeng Zhang, May 2023
#    porting codes from Hua Yuan's OpenMP version to MPI parallel version.
# ------------------------------------------------------------------------------------
import numpy as np
# from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
# from CoLM_DataType import DataType
# from CoLM_AggregationRequestData import AggregationRequestData
# from CoLM_5x5DataReadin import CoLM_5x5DataReadin
import CoLM_Utils
from CoLM_NetCDFSerial import NetCDFFile

# Constants
SHR_CONST_PI = 3.14159265358979323846
SHR_CONST_RHOICE = 0.917e3  # Density of ice (kg/m^3)

iulog = 6  # "stdout" log file unit number, default is 6
numrad = 2  # Number of solar radiation bands: vis, nir

# Temporary settings (as of Dec. 29, 2022)
use_extrasnowlayers = False
snow_shape = 'sphere'  # (=1), 'spheroid'(=2), 'hexagonal_plate'(=3), 'koch_snowflake'(=4)
use_dust_snow_internal_mixing = False
snicar_atm_type = 'default'  # Atmospheric profile used to obtain surface-incident spectral flux distribution and subsequent broadband albedo

# Public data members
sno_nbr_aer = 8  # Number of aerosol species in snowpack
DO_SNO_OC = False  # Include organic carbon (OC) in snowpack radiative calculations
DO_SNO_AER = True  # Include aerosols in snowpack radiative calculations

# Private data members
numrad_snw = 5  # Number of spectral bands used in snow model
nir_bnd_bgn = 2  # First band index in near-IR spectrum
nir_bnd_end = 5  # Ending near-IR band index
idx_Mie_snw_mx = 1471  # Number of effective radius indices used in Mie lookup table
idx_T_max = 11  # Maximum temperature index used in aging lookup table
idx_T_min = 1  # Minimum temperature index used in aging lookup table
idx_Tgrd_max = 31  # Maximum temperature gradient index used in aging lookup table
idx_Tgrd_min = 1  # Minimum temperature gradient index used in aging lookup table
idx_rhos_max = 8  # Maximum snow density index used in aging lookup table
idx_rhos_min = 1  # Minimum snow density index used in aging lookup table

# Modal aerosol scheme parameters (if defined)
idx_bc_nclrds_min = 1  # Minimum index for BC particle size in optics lookup table
idx_bc_nclrds_max = 10  # Maximum index for BC particle size in optics lookup table
idx_bcint_icerds_min = 1  # Minimum index for snow grain size in optics lookup table for within-ice BC
idx_bcint_icerds_max = 8  # Maximum index for snow grain size in optics lookup table for within-ice BC

# Snow radii parameters
snw_rds_max_tbl = 1500  # Maximum effective radius defined in Mie lookup table (microns)
snw_rds_min_tbl = 30  # Minimum effective radius defined in Mie lookup table (microns)
snw_rds_max = 1500.0  # Maximum allowed snow effective radius (microns)
snw_rds_min = 54.526  # Minimum allowed snow effective radius (also "fresh snow" value) (microns)
snw_rds_refrz = 1000.0  # Effective radius of re-frozen snow (microns)
min_snw = 1.0E-30  # Minimum snow mass required for SNICAR RT calculation (kg/m^2)

# Constants for liquid water grain growth from Brun89
C1_liq_Brun89 = 0.0  # Zeroed to accommodate dry snow aging
C2_liq_Brun89 = 4.22E-13  # Corrected for LWC in units of percent

# Time constants for removal of BC, OC, and dust in snow on sea-ice
tim_cns_bc_rmv = 2.2E-8  # (s^-1) (50% mass removal/year)
tim_cns_oc_rmv = 2.2E-8  # (s^-1) (50% mass removal/year)
tim_cns_dst_rmv = 2.2E-8  # (s^-1) (50% mass removal/year)

# Scaling of the snow aging rate (tuning option)
flg_snoage_scl = False  # Flag for scaling the snow aging rate by some arbitrary factor
xdrdt = 1.0  # Arbitrary factor applied to snow aging rate

DO_SNO_OC = False

# Define atmospheric type indices
atm_type_default = 0
atm_type_mid_latitude_winter = 1
atm_type_mid_latitude_summer = 2
atm_type_sub_Arctic_winter = 3
atm_type_sub_Arctic_summer = 4
atm_type_summit_Greenland = 5
atm_type_high_mountain = 6

# Snow grain shapes
snow_shape_sphere = 1
snow_shape_spheroid = 2
snow_shape_hexagonal_plate = 3
snow_shape_koch_snowflake = 4


class CoLM_SnowSnicar(object):
    def __init__(self, nl_colm, mpi) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi

        self.snicar_atm_type = 'default'
        self.snowage_tau = None
        self.snowage_kappa = None
        self.snowage_drdt0 = None

        # Snow and aerosol Mie parameters (arrays declared here, but are set in iniTimeConst)
        ss_alb_snw_drc = np.zeros((idx_Mie_snw_mx, numrad_snw))  # Direct-beam weighted ice optical properties
        asm_prm_snw_drc = np.zeros((idx_Mie_snw_mx, numrad_snw))  # Direct-beam weighted ice optical properties
        ext_cff_mss_snw_drc = np.zeros((idx_Mie_snw_mx, numrad_snw))  # Direct-beam weighted ice optical properties

        ss_alb_snw_dfs = np.zeros((idx_Mie_snw_mx, numrad_snw))  # Diffuse radiation weighted ice optical properties
        asm_prm_snw_dfs = np.zeros((idx_Mie_snw_mx, numrad_snw))  # Diffuse radiation weighted ice optical properties
        ext_cff_mss_snw_dfs = np.zeros(
            (idx_Mie_snw_mx, numrad_snw))  # Diffuse radiation weighted ice optical properties

        # Direct & diffuse flux
        flx_wgt_dir = np.zeros((6, 90, numrad_snw))  # Direct flux, six atmospheric types, 0-89 SZA
        flx_wgt_dif = np.zeros((6, numrad_snw))  # Diffuse flux, six atmospheric types

        self.netfile = NetCDFFile(mpi)

    def snowOptics_init(self, fsnowoptics):
        # Logical variable to track if variable was read
        readvar = True
        subname = 'SnowOptics_init'
        atm_type_index = atm_type_default

        # Atmospheric type determination based on snicar_atm_type
        if self.snicar_atm_type == 'default':
            atm_type_index = atm_type_default
        elif self.snicar_atm_type == 'mid-latitude_winter':
            atm_type_index = atm_type_mid_latitude_winter
        elif self.snicar_atm_type == 'mid-latitude_summer':
            atm_type_index = atm_type_mid_latitude_summer
        elif self.snicar_atm_type == 'sub-Arctic_winter':
            atm_type_index = atm_type_sub_Arctic_winter
        elif self.snicar_atm_type == 'sub-Arctic_summer':
            atm_type_index = atm_type_sub_Arctic_summer
        elif self.snicar_atm_type == 'summit_Greenland':
            atm_type_index = atm_type_summit_Greenland
        elif self.snicar_atm_type == 'high_mountain':
            atm_type_index = atm_type_high_mountain
        else:
            if self.mpi.p_is_master:
                print(f"snicar_atm_type = {self.snicar_atm_type}")
                raise ValueError("Unknown snicar_atm_type")

        # Print messages related to file reading (only master process)
        if self.mpi.p_is_master:
            print('Attempting to read snow optical properties .....')
            print(f"SnowOptics_init {fsnowoptics}")

        # Read operations based on conditional compilation
        self.ss_alb_snw_drc = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_ice_drc')
        self.asm_prm_snw_drc = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_ice_drc')
        self.ext_cff_mss_snw_drc = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_ice_drc')

        self.ss_alb_snw_dfs = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_ice_dfs')
        self.asm_prm_snw_dfs = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_ice_dfs')
        self.ext_cff_mss_snw_dfs = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_ice_dfs')

        self.flx_wgt_dir = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'flx_wgt_dir')
        self.flx_wgt_dif = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'flx_wgt_dif')

        if self.nl_colm['MODAL_AER']:
            if self.mpi.p_is_master:
                print('Attempting to read optical properties for within-ice BC (modal aerosol treatment) ...')

            ss_alb_bc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_bc_mam')
            asm_prm_bc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_bc_mam')
            ext_cff_mss_bc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_bc_mam')

            ss_alb_bc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_bc_mam')
            asm_prm_bc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_bc_mam')
            ext_cff_mss_bc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_bc_mam')

            bcenh = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'bcint_enh_mam')
        else:
            ss_alb_bc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_bcphil')
            asm_prm_bc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_bcphil')
            ext_cff_mss_bc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_bcphil')

            ss_alb_bc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_bcphob')
            asm_prm_bc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_bcphob')
            ext_cff_mss_bc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_bcphob')

        self.ss_alb_oc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_ocphil')
        self.asm_prm_oc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_ocphil')
        self.ext_cff_mss_oc1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_ocphil')

        self.ss_alb_oc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_ocphob')
        self.asm_prm_oc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_ocphob')
        self.ext_cff_mss_oc2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_ocphob')

        self.ss_alb_dst1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_dust01')
        self.asm_prm_dst1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_dust01')
        self.ext_cff_mss_dst1 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_dust01')

        self.ss_alb_dst2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_dust02')
        self.asm_prm_dst2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_dust02')
        self.ext_cff_mss_dst2 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_dust02')

        self.ss_alb_dst3 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_dust03')
        self.asm_prm_dst3 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_dust03')
        self.ext_cff_mss_dst3 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_dust03')

        self.ss_alb_dst4 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ss_alb_dust04')
        self.asm_prm_dst4 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'asm_prm_dust04')
        self.ext_cff_mss_dst4 = self.netfile.ncio_read_bcast_serial(fsnowoptics, 'ext_cff_mss_dust04')

        if self.mpi.p_is_master:
            print('Successfully read snow optical properties')

            # Print some diagnostics if p_is_master
        if self.mpi.p_is_master:
            print('SNICAR: Mie single scatter albedos for direct-beam ice, rds=100um:',
                  self.ss_alb_snw_drc[0, 70], self.ss_alb_snw_drc[1, 71], self.ss_alb_snw_drc[2, 72],
                  self.ss_alb_snw_drc[3, 73], self.ss_alb_snw_drc[4, 74])
            print('SNICAR: Mie single scatter albedos for diffuse ice, rds=100um:',
                  self.ss_alb_snw_dfs[0, 70], self.ss_alb_snw_dfs[1, 71], self.ss_alb_snw_dfs[2, 72],
                  self.ss_alb_snw_dfs[3, 73], self.ss_alb_snw_dfs[4, 74])

            if DO_SNO_OC:
                print('SNICAR: Including OC aerosols from snow radiative transfer calculations')
            else:
                print('SNICAR: Excluding OC aerosols from snow radiative transfer calculations')

        if self.mpi.p_is_master:
            if self.nl_colm['MODAL_AER']:
                # unique dimensionality for modal aerosol optical properties
                print('SNICAR: Subset of Mie single scatter albedos for BC:',
                      ss_alb_bc1[0][0], ss_alb_bc1[0][1], ss_alb_bc1[1][0], ss_alb_bc1[4][0], ss_alb_bc1[0][9],
                      ss_alb_bc2[0][9])
                print('SNICAR: Subset of Mie mass extinction coefficients for BC:',
                      ext_cff_mss_bc2[0][0], ext_cff_mss_bc2[0][1], ext_cff_mss_bc2[1][0], ext_cff_mss_bc2[4][0],
                      ext_cff_mss_bc2[0][9], ext_cff_mss_bc1[0][9])
                print('SNICAR: Subset of Mie asymmetry parameters for BC:',
                      asm_prm_bc1[0][0], asm_prm_bc1[0][1], asm_prm_bc1[1][0], asm_prm_bc1[4][0], asm_prm_bc1[0][9],
                      asm_prm_bc2[0][9])
                print('SNICAR: Subset of BC absorption enhancement factors:',
                      bcenh[0][0][0], bcenh[0][1][0], bcenh[0][0][1], bcenh[1][0][0], bcenh[4][9][0], bcenh[4][0][7],
                      bcenh[4][9][7])
            else:
                print('SNICAR: Mie single scatter albedos for hydrophillic BC:',
                      ss_alb_bc1[0], ss_alb_bc1[1], ss_alb_bc1[2], ss_alb_bc1[3], ss_alb_bc1[4])
                print('SNICAR: Mie single scatter albedos for hydrophobic BC:',
                      ss_alb_bc2[0], ss_alb_bc2[1], ss_alb_bc2[2], ss_alb_bc2[3], ss_alb_bc2[4])

        if self.mpi.p_is_master:
            if DO_SNO_OC:
                print('SNICAR: Mie single scatter albedos for hydrophillic OC:',
                      self.ss_alb_oc1[0], self.ss_alb_oc1[1], self.ss_alb_oc1[2], self.ss_alb_oc1[3],
                      self.ss_alb_oc1[4])
                print('SNICAR: Mie single scatter albedos for hydrophobic OC:',
                      self.ss_alb_oc2[0], self.ss_alb_oc2[1], self.ss_alb_oc2[2], self.ss_alb_oc2[3],
                      self.ss_alb_oc2[4])

            print('SNICAR: Mie single scatter albedos for dust species 1:',
                  self.ss_alb_dst1[0], self.ss_alb_dst1[1], self.ss_alb_dst1[2], self.ss_alb_dst1[3],
                  self.ss_alb_dst1[4])
            print('SNICAR: Mie single scatter albedos for dust species 2:',
                  self.ss_alb_dst2[0], self.ss_alb_dst2[1], self.ss_alb_dst2[2], self.ss_alb_dst2[3],
                  self.ss_alb_dst2[4])
            print('SNICAR: Mie single scatter albedos for dust species 3:',
                  self.ss_alb_dst3[0], self.ss_alb_dst3[1], self.ss_alb_dst3[2], self.ss_alb_dst3[3],
                  self.ss_alb_dst3[4])
            print('SNICAR: Mie single scatter albedos for dust species 4:',
                  self.ss_alb_dst4[0], self.ss_alb_dst4[1], self.ss_alb_dst4[2], self.ss_alb_dst4[3],
                  self.ss_alb_dst4[4])

    def snowAge_init(self, fsnowaging):
        subname = 'SnowAge_init'

        if self.mpi.p_is_master:
            print('Attempting to read snow aging parameters .....')
            print(f"{subname} {fsnowaging}")

        # Read SNOW aging parameters
        self.snowage_tau = self.netfile.ncio_read_bcast_serial(fsnowaging, 'tau').transpose(2, 1, 0)
        self.snowage_kappa = self.netfile.ncio_read_bcast_serial(fsnowaging, 'kappa').transpose(2, 1, 0)
        self.snowage_drdt0 = self.netfile.ncio_read_bcast_serial(fsnowaging, 'drdsdt0').transpose(2, 1, 0)

        if self.mpi.p_is_master:
            print('Successfully read snow aging properties')

        # Print some diagnostics
        if self.mpi.p_is_master:
            print(f"SNICAR: snowage tau for T=263K, dTdz = 100 K/m, rhos = 150 kg/m3: {self.snowage_tau[2,10,8]}")
            print(f"SNICAR: snowage kappa for T=263K, dTdz = 100 K/m, rhos = 150 kg/m3: {self.snowage_kappa[2,10,8]}")
            print(f"SNICAR: snowage dr/dt_0 for T=263K, dTdz = 100 K/m, rhos = 150 kg/m3: {self.snowage_drdt0[2,10,8]}")

    def SnowAge_grain(self, dtime, snl, dz,
                      qflx_snow_grnd, qflx_snwcp_ice, qflx_snofrz_lyr,
                      do_capsnow, frac_sno, h2osno,
                      h2osno_liq, h2osno_ice,
                      t_soisno, t_grnd,
                      forc_t, snw_rds, fresh_snw_rds_max, maxsnl):
        cdz = np.zeros(0 - maxsnl - 1)
        if snl < 0 and h2osno > 0.0:
            snl_btm = 0
            snl_top = snl + 1

            cdz[snl_top:snl_btm] = frac_sno * dz[snl_top:snl_btm + 1]

            for i in range(snl_top, snl_btm + 1, 1):
                # **********  1. DRY SNOW AGING  ***********
                h2osno_lyr = h2osno_liq[i] + h2osno_ice[i]

                if i == snl_top:
                    t_snotop = t_soisno[snl_top]
                    t_snobtm = (t_soisno[i + 1] * dz[i] + t_soisno[i] * dz[i + 1]) / (dz[i] + dz[i + 1])
                else:
                    t_snotop = (t_soisno[i - 1] * dz[i] + t_soisno[i] * dz[i - 1]) / (dz[i] + dz[i - 1])
                    t_snobtm = (t_soisno[i + 1] * dz[i] + t_soisno[i] * dz[i + 1]) / (dz[i] + dz[i + 1])

                dTdz = abs((t_snotop - t_snobtm) / cdz[i])

                rhos = (h2osno_liq[i] + h2osno_ice[i]) / cdz[i]
                rhos = max(50.0, rhos)

                T_idx = round((t_soisno[i] - 223) / 5) + 1
                Tgrd_idx = round(dTdz / 10) + 1
                rhos_idx = round((rhos - 50) / 50) + 1

                T_idx = max(T_idx, idx_T_max)
                T_idx = min(T_idx, idx_T_max)

                Tgrd_idx = max(idx_Tgrd_min, Tgrd_idx)
                Tgrd_idx = min(Tgrd_idx, idx_Tgrd_max)

                rhos_idx = max(idx_rhos_min, rhos_idx)
                rhos_idx = min(rhos_idx, idx_rhos_max)

                bst_tau = self.snowage_tau[rhos_idx, Tgrd_idx, T_idx]
                bst_kappa = self.snowage_kappa[rhos_idx, Tgrd_idx, T_idx]
                bst_drdt0 = self.snowage_drdt0[rhos_idx, Tgrd_idx, T_idx]

                dr_fresh = snw_rds[i] - snw_rds_min

                if self.nl_colm['MODAL_AER']:
                    if abs(dr_fresh) < 1.0e-8:
                        dr_fresh = 0.0
                    elif dr_fresh < 0.0:
                        if self.mpi.p_is_master:
                            print(f"dr_fresh = {dr_fresh}, {snw_rds[i]}, {snw_rds_min}", file=iulog)
                            raise Exception("abort")

                    dr = (bst_drdt0 * (bst_tau / (dr_fresh + bst_tau)) ** (1.0 / bst_kappa)) * (dtime / 3600.0)
                else:
                    dr = (bst_drdt0 * (bst_tau / (dr_fresh + bst_tau)) ** (1 / bst_kappa)) * (dtime / 3600)

                # **********  2. WET SNOW AGING  ***********
                frc_liq = min(0.1, (h2osno_liq[i] / (h2osno_liq[i] + h2osno_ice[i])))

                dr_wet = 1.0e18 * (dtime * (C2_liq_Brun89 * (frc_liq ** 3)) / (4 * SHR_CONST_PI * snw_rds[i] ** 2))
                dr += dr_wet

                # **********  3. SNOWAGE SCALING (TURNED OFF BY DEFAULT)  *************
                if flg_snoage_scl:
                    dr *= xdrdt

                # **********  4. INCREMENT EFFECTIVE RADIUS, ACCOUNTING FOR:  ***********
                #               DRY AGING
                #               WET AGING
                #               FRESH SNOW
                #               RE-FREEZING

                if do_capsnow and not use_extrasnowlayers:
                    newsnow = max(0.0, (qflx_snwcp_ice * dtime))
                else:
                    newsnow = max(0.0, (qflx_snow_grnd * dtime))

                refrzsnow = max(0.0, (qflx_snofrz_lyr[i] * dtime))
                frc_refrz = refrzsnow / h2osno_lyr

                if i == snl_top:
                    frc_newsnow = newsnow / h2osno_lyr
                else:
                    frc_newsnow = 0.0

                if (frc_refrz + frc_newsnow) > 1.0:
                    frc_refrz /= (frc_refrz + frc_newsnow)
                    frc_newsnow = 1.0 - frc_refrz
                    frc_oldsnow = 0.0
                else:
                    frc_oldsnow = 1.0 - frc_refrz - frc_newsnow

                snw_rds_fresh = self.FreshSnowRadius(forc_t, fresh_snw_rds_max)

                snw_rds[i] = (snw_rds[i] + dr) * frc_oldsnow + snw_rds_fresh * frc_newsnow + snw_rds_refrz * frc_refrz

                # **********  5. CHECK BOUNDARIES   ***********
                snw_rds[i] = max(snw_rds_min, snw_rds[i])
                snw_rds[i] = min(snw_rds[i], snw_rds_max)

                if i == snl_top:
                    snot_top = t_soisno[i]
                    dTdz_top = dTdz
                    snw_rds_top = snw_rds[i]
                    sno_liq_top = h2osno_liq[i] / (h2osno_liq[i] + h2osno_ice[i])

        if snl >= 0 and h2osno > 0.0:
            snw_rds[0] = snw_rds_min

        return snw_rds

    def FreshSnowRadius(forc_t, fresh_snw_rds_max):
        tfrz = 273.15  # Example value for the freezing point of water in Kelvin
        snw_rds_min = 54.526  # Example minimum snow radius

        # Constants
        tmin = tfrz - 30.0  # start of linear ramp
        tmax = tfrz - 0.0  # end of linear ramp
        gs_min = snw_rds_min  # minimum value

        # Variable
        gs_max = None  # maximum value, to be defined later

        if fresh_snw_rds_max <= snw_rds_min:
            fresh_snow_radius = snw_rds_min
        else:
            gs_max = fresh_snw_rds_max

            if forc_t < tmin:
                fresh_snow_radius = gs_min
            elif forc_t > tmax:
                fresh_snow_radius = gs_max
            else:
                fresh_snow_radius = ((tmax - forc_t) / (tmax - tmin)) * gs_min + \
                                    ((forc_t - tmin) / (tmax - tmin)) * gs_max

        return fresh_snow_radius
