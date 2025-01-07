# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Build pixel_set "land_elm".
#
#    In CoLM, the global/regional area is divided into a hierarchical structure:
#    1. If GRIDBASED or UNSTRUCTURED is defined, it is
#       ELEMENT >>> PATCH
#    2. If CATCHMENT is defined, it is
#       ELEMENT >>> HRU >>> PATCH
#    If Plant Function Type classification is used, PATCH is further divided into PFT.
#    If Plant Community classification is used,     PATCH is further divided into PC.
# 
#    "land_elm" refers to pixel_set ELEMENT.
# ------------------------------------------------------------------------------------
import numpy as np
# from CoLM_SPMD_Task import CoLM_SPMD_Task
from CoLM_Pixelset import Pixelset_type


def get_land_elm(use_mpi, mpi, g_block, mesh):
    if mpi.p_is_master:
        print('Making land elements:')
    land_elm = Pixelset_type(use_mpi, mpi, g_block, mesh)

    if mpi.p_is_worker:
        land_elm.eindex = np.zeros(mesh.numelm)

        # print(mesh.num_elm, len(mesh.mesh), '-----------------')
        land_elm.ipxstt = np.zeros(mesh.numelm, dtype='int')
        land_elm.ipxend = np.zeros(mesh.numelm, dtype='int')
        land_elm.settyp = np.zeros(mesh.numelm, dtype='int')
        land_elm.ielm = np.zeros(mesh.numelm, dtype='int')

        for i_elm in range(mesh.numelm):
            # print(i_elm, mesh.mesh[i_elm].npxl, '---------npxl---------')
            land_elm.eindex[i_elm] = mesh.mesh[i_elm].indx
            land_elm.ipxstt[i_elm] = 0
            land_elm.ipxend[i_elm] = mesh.mesh[i_elm].npxl
            # print (i_elm, mesh.mesh[i_elm].indx, mesh.mesh[i_elm]. npxl, '-------landelm-------------')
            land_elm.settyp[i_elm] = 0
            land_elm.ielm[i_elm] = i_elm

    land_elm.nset = mesh.numelm
    land_elm.set_vecgs()

    if use_mpi:
        # CALL mpi_barrier (p_comm_glb, p_err)
        if mpi.p_is_worker:
            # CALL mpi_reduce (num_elm, n_elm_glb, 1, MPI_INTEGER, MPI_SUM, p_root, p_comm_worker, p_err)
            if mpi.p_iam_worker == 0:
                print('Total: ' + mesh.nelm_glb + ' elements.')

        # CALL mpi_barrier (p_comm_glb, p_err)
    else:
        print('Total: ' + str(mesh.numelm) + ' elements.')
    return land_elm
