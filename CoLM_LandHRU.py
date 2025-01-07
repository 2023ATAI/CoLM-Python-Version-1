# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Build pixelset "self.landhru".
#
#    In CoLM, the global/regional area is divided into a hierarchical structure:
#    1. If GRIDBASED or UNSTRUCTURED is defined, it is
#       ELEMENT >>> PATCH
#    2. If CATCHMENT is defined, it is
#       ELEMENT >>> HRU >>> PATCH
#    If Plant Function Type classification is used, PATCH is further divided into PFT.
#    If Plant Community classification is used,     PATCH is further divided into PC.
# 
#    "self.landhru" refers to pixelset HRU.
# ------------------------------------------------------------------------------------
import numpy as np
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_DataType import DataType
import CoLM_CatchmentDataReadin
from CoLM_Pixelset import Pixelset_type
from CoLM_AggregationRequestData import AggregationRequestData
from CoLM_Grid import Grid_type
import CoLM_Utils


class CoLM_LandHRU(object):
    def __init__(self, namelist, mpi, gblock, mesh, landelm, pixel) -> None:
        ibuff = None
        self.numhru = 0
        self.mpi = mpi
        nhru_glb = 0
        ghru = Grid_type(namelist.nl_colm, gblock)
        self.landhru = Pixelset_type(namelist['USEMPI'], mpi, gblock, mesh)

        if namelist['USEMPI']:
            pass
            # CALL mpi_barrier (p_comm_glb, p_err)

        if self.mpi.p_is_master:
            print('Making land hydro units :')
            netfile = NetCDFFile(namelist['USEMPI'])
            self.numhru_all_g = netfile.ncio_read_serial(namelist['DEF_CatchmentMesh_data'], 'basin_numhru')
            self.lakeid = netfile.ncio_read_serial(namelist['DEF_CatchmentMesh_data'], 'lake_id')

        if namelist['USEMPI']:
            if self.mpi.p_is_master:
                for iwork in range(self.mpi.p_np_worker - 1):
                    pass

                    # CALL mpi_recv (ncat, 1, MPI_INTEGER4, p_address_worker(iwork), mpi_tag_size, &
                    # p_comm_glb, p_stat, p_err)

                    # if ncat > 0:
                    #     catnum = np.zeros(ncat)
                    #     ibuff = np.zeros (ncat)

                    #     CALL mpi_recv (catnum, ncat, MPI_INTEGER8, p_address_worker(iwork), mpi_tag_data, &
                    #         p_comm_glb, p_stat, p_err)

                    #     nhru = sum(self.numhru_all_g(catnum))
                    #     CALL mpi_send (nhru, 1, MPI_INTEGER4, &
                    #         p_address_worker(iwork), mpi_tag_size, p_comm_glb, p_err) 

                    #     ibuff = self.lakeid(catnum)
                    #     CALL mpi_send (ibuff, ncat, MPI_INTEGER4, &
                    #         p_address_worker(iwork), mpi_tag_data, p_comm_glb, p_err) 

                    #     del catnum
                    #     del ibuff

            # if self.mpi.p_is_worker:
            # CALL mpi_send (numelm, 1, MPI_INTEGER4, p_root, mpi_tag_size, p_comm_glb, p_err)
            # if numelm > 0:
            #     allocate (self.lakeid (numelm))
            #     CALL mpi_send (landelm.eindex, numelm, MPI_INTEGER8, p_root, mpi_tag_data, p_comm_glb, p_err)
            #     CALL mpi_recv (self.numhru, 1,      MPI_INTEGER4, p_root, mpi_tag_size, p_comm_glb, p_stat, p_err)
            #     CALL mpi_recv (self.lakeid, numelm, MPI_INTEGER4, p_root, mpi_tag_data, p_comm_glb, p_stat, p_err)
        else:
            self.numhru = sum(self.numhru_all_g)

        if self.mpi.p_is_master:
            if self.numhru_all_g is not None:
                del self.numhru_all_g
        hrudata = None
        if self.mpi.p_is_io:
            dt = DataType(gblock)
            hrudata = dt.allocate_block_data(ghru)
        hrudata = CoLM_CatchmentDataReadin.catchment_data_read(mpi, gblock, namelist['DEF_CatchmentMesh_data'],
                                                               'ihydrounit2d', ghru, hrudata)

        if namelist['USEMPI']:
            if self.mpi.p_is_io:
                # CALL aggregation_data_daemon (ghru, data_i4_2d_in1 = hrudata)
                pass

        if self.mpi.p_is_worker:
            self.landhru.eindex = np.zeros(self.numhru, dtype='int')
            self.landhru.settyp = np.zeros(self.numhru, dtype='int')
            self.landhru.ipxstt = np.zeros(self.numhru, dtype='int')
            self.landhru.ipxend = np.zeros(self.numhru, dtype='int')
            self.landhru.ielm = np.zeros(self.numhru, dtype='int')

            self.numhru = 0

            for ie in range(mesh.numelm):

                if self.lakeid[ie] > 0:
                    typsgn = -1
                else:
                    typsgn = 1

                npxl = mesh[ie].npxl

                types = np.zeros(npxl - 1)
                ard = AggregationRequestData(namelist['USEMPI'], mpi, mesh, pixel)
                _, _, _, _, _, _, _, _, _, ibuff, _ = ard.aggregation_request_data(landelm, ie, ghru, zip=False,
                                                                                   data_i4_2d_in1=hrudata)

                types = ibuff

                order = [ipxl for ipxl in range(npxl)]
                keys = np.lexsort((order, types))
                order = [order[i] for i in keys]
                types = [types[i] for i in keys]

                # CoLM_Utils.quicksort(npxl, types, order)

                mesh[ie].ilon[0:npxl] = mesh[ie].ilon[order]
                mesh[ie].ilat[0:npxl] = mesh[ie].ilat[order]

                for ipxl in range(npxl):
                    if ipxl == 0:
                        self.landhru.eindex[self.numhru] = mesh[ie].indx
                        self.landhru.settyp[self.numhru] = types[ipxl] * typsgn
                        self.landhru.ipxstt[self.numhru] = ipxl
                        self.landhru.ielm[self.numhru] = ie
                        self.numhru = self.numhru + 1
                    elif types[ipxl] != types[ipxl - 1]:
                        self.landhru.ipxend[self.numhru] = ipxl - 1

                        self.numhru = self.numhru + 1
                        self.landhru.eindex[self.numhru] = mesh[ie].indx
                        self.landhru.settyp[self.numhru] = types[ipxl] * typsgn
                        self.landhru.ipxstt[self.numhru] = ipxl
                        self.landhru.ielm[self.numhru] = ie
                self.landhru.ipxend[self.numhru] = npxl

                del ibuff
                del types
                del order

            if namelist['USEMPI']:
                pass
                # CALL aggregation_worker_done ()

        self.landhru.nset = self.numhru
        self.landhru.set_vecgs()

        if namelist['USEMPI']:
            if self.mpi.p_is_worker:
                # CALL mpi_reduce (self.numhru, nhru_glb, 1, MPI_INTEGER, MPI_SUM, p_root, p_comm_worker, p_err)
                if mpi.p_iam_worker == 0:
                    print('Total: ', nhru_glb, ' hydro units.')

            # CALL mpi_barrier (p_comm_glb, p_err)
        else:
            print('Total: ', self.numhru, ' hydro units.')
        # endif

        if self.lakeid is not None:
            del self.lakeid
