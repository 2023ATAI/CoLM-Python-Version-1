#-----------------------------------------------------------------------
   # DESCRIPTION:
   #Read in VT.lakedepth and assign lake thickness of each layer.

#-----------------------------------------------------------------------


import numpy as np
# from CoLM_NetCDFVectorOneP import CoLM_NetCDFVectorP
import CoLM_NetCDFVectorBlk

def lakedepth_readin(nl_colm, landpatch, mpi, gblock, dir_landdata, lc_year, nl_lake, VT):
    dzlak = np.array([0.1, 1., 2., 3., 4., 5., 7., 7., 10.45, 10.45], dtype=np.float64)  # m
    zlak = np.array([0.05, 0.6, 2.1, 4.6, 8.1, 12.6, 18.6, 25.6, 34.325, 44.775], dtype=np.float64)
    # --------------------------------------------------------------------------------------------------------------
    #       ! For site simulations with 25 layers, the default thicknesses are (m):
    #       ! real(r8), dimension(25) :: dzlak
    #       ! dzlak = (/0.1,                         & ! 0.1 for layer 1;
    #       !           0.25, 0.25, 0.25, 0.25,      & ! 0.25 for layers 2-5;
    #       !           0.50, 0.50, 0.50, 0.50,      & ! 0.5 for layers 6-9;
    #       !           0.75, 0.75, 0.75, 0.75,      & ! 0.75 for layers 10-13;
    #       !           2.00, 2.00,                  & ! 2 for layers 14-15;
    #       !           2.50, 2.50,                  & ! 2.5 for layers 16-17;
    #       !           3.50, 3.50, 3.50, 3.50,      & ! 2.5 for layers 16-17;
    #       !           5.225, 5.225, 5.225, 5.225/)   ! 5.225 for layers 22-25.
    #       !
    #       ! For lakes with depth d /= 50 m and d >= 1 m,
    #       !                       the top layer is kept at 10 cm and
    #       !                       the other 9 layer thicknesses are adjusted to maintain fixed proportions.
    #       !
    #       ! For lakes with d < 1 m, all layers have equal thickness.
    #       ! ------------------------------------------------------------------------------------------------------
    if nl_colm['SinglePoint']:
        VT.lakedepth = nl_colm['SITE_lakedepth']
    else:
        cyear = f"{lc_year:04d}"
        lndname = f"{dir_landdata.strip()}/lakedepth/{cyear}/lakedepth_patches.nc"
        # net_vp = CoLM_NetCDFVectorP(nl_colm, mpi, gblock)
        VT.lakedepth = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'lakedepth_patches', landpatch.landpatch, VT.lakedepth,nl_colm['USEMPI'],mpi,gblock)

    # Define lake levels
    if mpi.p_is_worker:
        for ipatch in range(landpatch.numpatch):
            # print(landpatch.numpatch, len(VT.dz_lake),'-----------')
            # print(ipatch, len(VT.dz_lake[ipatch][:]), '+++++++')
            if VT.lakedepth[ipatch] < 0.1:
                VT.lakedepth[ipatch] = 0.1
            if 1. < VT.lakedepth[ipatch] < 1000.:
                depthratio = VT.lakedepth[ipatch] / np.sum(dzlak[:nl_lake])
                VT.dz_lake[0, ipatch] = dzlak[0]
                VT.dz_lake[1:nl_lake - 1, ipatch] = dzlak[1:nl_lake - 1] * depthratio
                VT.dz_lake[nl_lake - 1, ipatch] = dzlak[nl_lake - 1] * depthratio - (
                            VT.dz_lake[0, ipatch] - dzlak[0] * depthratio)
            elif 0. < VT.lakedepth[ipatch] <= 1.0:
                # for i in range(len(VT.dz_lake[ipatch][:])):
                VT.dz_lake[:,ipatch] = VT.lakedepth[ipatch] / nl_lake
            else:  # non land water bodies or missing value of the lake depth
                VT.lakedepth[ipatch] = np.sum(dzlak[:nl_lake])
                VT.dz_lake[:nl_lake, ipatch] = dzlak[:nl_lake]

    return VT
