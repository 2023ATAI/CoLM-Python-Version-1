# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Mesh filter. 
#    Mesh filter can be used to mask part of region or globe as needed.
#
# Created by Shupeng Zhang, May 2023
# ------------------------------------------------------------------------------------
import os

import numpy as np

from CoLM_Grid import Grid_type
from CoLM_DataType import DataType
from CoLM_AggregationRequestData import AggregationRequestData
import CoLM_LandElm
import CoLM_NetCDFBlock


class MeshFilter(object):
    def __init__(self, nl_colm, gblock, mpi) -> None:
        self.gblock = gblock
        self.has_mesh_filter = False
        self.grid_filter = Grid_type(nl_colm, gblock)
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.USEMPI = nl_colm['USEMPI']

    def inquire_mesh_filter(self):
        f_exists = False
        if self.mpi.p_is_master:
            if os.path.exists(self.nl_colm['DEF_file_mesh_filter']):
                f_exists = True

            if not f_exists:
                print('Mesh Filter not used: file ' + self.nl_colm['DEF_file_mesh_filter'])
            else:
                print('Mesh Filter from file ' + self.nl_colm['DEF_file_mesh_filter'])
        if self.nl_colm['USEMPI']:
            pass
            # call mpi_bcast (f_exists, 1, MPI_LOGICAL, p_root, p_comm_glb, p_err)

        self.has_mesh_filter = f_exists
        return f_exists

    def mesh_filter(self, landelm, mesh, gridf, ffilter, fvname, pixel):
        if self.USEMPI:
            pass
            # CALL mpi_barrier(p_comm_glb, p_err)

        if self.mpi.p_is_master:
            print('Filtering pixels ...')
        datafilter = None
        if self.mpi.p_is_io:
            data_type = DataType(self.gblock)
            datafilter = data_type.allocate_block_data(gridf)
            # v = datafilter.blk[311,77].val

            datafilter = CoLM_NetCDFBlock.ncio_read_block(ffilter, fvname, self.mpi, self.gblock, gridf, datafilter)
            # v = datafilter.blk[311, 77].val
            if self.USEMPI:
                pass
                # CALL aggregation_data_daemon(gridf, data_i4_2d_in1=datafilter)


        ar = AggregationRequestData(self.USEMPI, self.mpi, mesh.mesh, pixel)

        if self.mpi.p_is_worker:
            jelm = 0
            # ia = []
            # print(mesh.numelm,'*************')
            for ielm in range(mesh.numelm):
                _, _, _, _, _, _, _, _, _, ifilter, _ = ar.aggregation_request_data(pixelset=landelm, iset=ielm, grid_in=gridf, zip=False,
                                                      data_i4_2d_in1=datafilter, filledvalue_i4=-1)
                # return

                filter = [ifilter > 0][0]


                if np.any(filter):

                    # ia.append(ielm)
                    if not np.all(filter):
                        dims = filter.ndim
                        npxl = 0
                        if dims==2:
                            npxl = sum(1 for row in ifilter for elem in row if elem>0)
                        elif dims==1:
                            npxl = sum(1 for elem in ifilter if elem > 0)
                        # npxl = len(filter)
                        # print(npxl,'------------------')
                        xtemp = np.compress(filter,mesh.mesh[ielm].ilon)
                        ytemp = np.compress(filter,mesh.mesh[ielm].ilat)
                        del mesh.mesh[ielm].ilon
                        del mesh.mesh[ielm].ilat

                        mesh.mesh[ielm].npxl = npxl

                        mesh.mesh[ielm].ilon = xtemp
                        mesh.mesh[ielm].ilat = ytemp

                    if jelm != ielm:
                        mesh.mesh[jelm] = mesh.copy_elm(mesh.mesh[ielm])
                    jelm += 1
                # else:
                #     print(ielm,'+++')
                del filter
            mesh.numelm = jelm
            # print(jelm,'++++++++++++++++')

            if self.USEMPI:
                pass
                # CALL aggregation_worker_done()

        if self.mpi.p_is_worker:
            if landelm.eindex is not None:
                del landelm.eindex
            if landelm.ipxstt is not None:
                del landelm.ipxstt
            if landelm.ipxend is not None:
                del landelm.ipxend
            if landelm.settyp is not None:
                del landelm.settyp
            if landelm.ielm is not None:
                del landelm.ielm

        landelm = CoLM_LandElm.get_land_elm(self.nl_colm['USEMPI'], self.mpi, self.gblock, mesh)

        if self.USEMPI:
            pass
        else:
            print('Total:', mesh.numelm, 'elements after mesh filtering.')
        mesh.nelm_blk[:, :] = 0
        if self.mpi.p_is_worker:
            for ielm in range(mesh.numelm):
                mesh.nelm_blk[mesh.mesh[ielm].xblk, mesh.mesh[ielm].yblk] += 1

        if self.USEMPI:
            pass

        return landelm, mesh
