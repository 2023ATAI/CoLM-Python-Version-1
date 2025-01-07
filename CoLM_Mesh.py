import os
from CoLM_Grid import Grid_type
import numpy as np
import CoLM_Utils
from CoLM_DataType import DataType, Pointer
import CoLM_NetCDFBlock

"""
   ! DESCRIPTION:
   !
   !    MESH refers to the set of largest elements in CoLM.
   ! 
   !    In CoLM, the global/regional area is divided into a hierarchical structure:
   !    1. If GRIDBASED or UNSTRUCTURED is defined, it is
   !       ELEMENT >>> PATCH
   !    2. If CATCHMENT is defined, it is
   !       ELEMENT >>> HRU >>> PATCH
   !    If Plant Function Type classification is used, PATCH is further divided into PFT.
   !    If Plant Community classification is used,     PATCH is further divided into PC.
   !
   !    To represent ELEMENT in CoLM, the land surface is first divided into pixels, 
   !    which are rasterized points defined by fine-resolution data.
   ! 
   !    ELEMENT in MESH is set of pixels:
   !    1. If GRIDBASED,    ELEMENT is set of pixels in a longitude-latitude rectangle. 
   !    2. If UNSTRUCTURED, ELEMENT is set of pixels in an irregular area (usually polygon). 
   !    3. If CATCHMENT,    ELEMENT is set of pixels in a catchment whose area is less than
   !       a predefined value. 
   !
   !    If GRIDBASED is defined, MESH is built by using input files containing mask of 
   !    land area or by defining the resolution of longitude-latitude grid.
   !    If CATCHMENT or UNSTRUCTURED is defined, MESH is built by using input files 
   !    containing index of elements.
"""


class Mesh(object):
    def __init__(self, nl_colm, gblock, mpi) -> None:
        self.nl_colm = nl_colm
        self.gblock = gblock
        self.mpi = mpi
        self.read_mesh_from_file = True
        self.datayype = DataType(self.gblock)
        self.gridmesh = Grid_type(self.nl_colm, self.gblock)
        self.numelm = 1
        self.mesh = None
        self.nelm_blk = None


    def init_gridbased_mesh_grid(self):
        if self.mpi.p_is_master:
            if os.path.isfile(self.nl_colm['DEF_file_mesh']):
                self.read_mesh_from_file = True
            else:
                self.read_mesh_from_file = False

        if self.nl_colm['USEMPI']:
            pass
            # mpi_bcast (read_mesh_from_file, 1, MPI_LOGICAL, p_root, p_comm_glb, p_err)

        if self.read_mesh_from_file:
            self.gridmesh.define_from_file(self.nl_colm['DEF_file_mesh'])
        else:
            self.gridmesh.define_by_res(self.nl_colm['DEF_GRIDBASED_lon_res'], self.nl_colm['DEF_GRIDBASED_lat_res'])

    def mesh_build(self, srfdata, pixel):
        elmid = 0
        if self.nl_colm['SinglePoint']:
            self.numelm = 1
            self.mesh = [IrregularElement() for _ in range(1)]
            self.mesh[0].indx = 0
            self.mesh[0].npxl = 0

            self.mesh[0].ilat = np.zeros(1, dtype=int)
            self.mesh[0].ilat[0] = CoLM_Utils.find_nearest_south(srfdata.SITE_lat_location, pixel.nlat, pixel.lat_s)
            srfdata.SITE_lon_location = CoLM_Utils.normalize_longitude(srfdata.SITE_lon_location)

            self.mesh[0].ilon = np.zeros(1, dtype=int)
            self.mesh[0].ilon[0] = CoLM_Utils.find_nearest_west(srfdata.SITE_lon_location, pixel.nlon, pixel.lon_w)
            self.mesh[0].xblk = CoLM_Utils.find_nearest_west(pixel.lon_w[self.mesh[0].ilon[0]], self.gblock.nxblk,
                                                             self.gblock.lon_w)
            self.mesh[0].yblk = CoLM_Utils.find_nearest_south(pixel.lat_s[self.mesh[0].ilat[0]], self.gblock.nyblk,
                                                              self.gblock.lat_s)

            self.nelm_blk = np.zeros((self.gblock.nxblk, self.gblock.nyblk))
            self.nelm_blk[self.mesh[0].xblk, self.mesh[0].yblk] = 1
            return

        datamesh = None
        blkdsp = None
        meshtmp = None
        iaddr = None
        elist = None
        self.mesh = None
        blkcnt = None

        if self.mpi.p_is_io:
            datamesh = self.datayype.allocate_block_data(self.gridmesh)

        if self.nl_colm['GRIDBASED']:
            if self.read_mesh_from_file:
                # net_block = NetCDFBlock(self.nl_colm['DEF_file_mesh'], 'landmask', self.gridmesh, datamesh, self.mpi)
                datamesh = CoLM_NetCDFBlock.ncio_read_block(self.nl_colm['DEF_file_mesh'], 'landmask', self.mpi,
                                                            self.gblock, self.gridmesh, datamesh)
            else:
                datamesh = self.datayype.flush_block_data(1)

        if self.nl_colm['CATCHMENT']:
            pass
            # CALL catchment_data_read (DEF_CatchmentMesh_data, 'icatchment2d', gridmesh, datamesh, spv = -1)
        # endif

        if self.nl_colm['UNSTRUCTURED']:
            pass
            # CALL ncio_read_block (DEF_file_mesh, 'elmindex', gridmesh, datamesh)

        #   Step 1: How many elms in each block?
        nelm = 0
        if self.mpi.p_is_io:

            nelm_worker = np.zeros(self.mpi.p_np_worker,dtype=int)
            elist_worker = [Pointer()] * self.mpi.p_np_worker

            for iworker in range(self.mpi.p_np_worker):
                elist_worker[iworker].val = np.zeros(1000)

            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]

                for yloc in range(self.gridmesh.ycnt[jblk]):
                    for xloc in range(self.gridmesh.xcnt[iblk]):
                        if self.nl_colm['GRIDBASED']:
                            if datamesh.blk[iblk, jblk].val is not None:
                                # print(xloc, yloc, '-------------')
                                if datamesh.blk[iblk, jblk].val[xloc, yloc] > 0 :
                                    xg = (self.gridmesh.xdsp[iblk] + 1) + (xloc + 1)
                                    if xg > self.gridmesh.nlon:
                                        xg -= self.gridmesh.nlon
                                    yg = (self.gridmesh.ydsp[jblk] + 1) + (yloc + 1)
                                    # print(self.gridmesh.ydsp[jblk], yloc)
                                    elmid = self.gridmesh.nlon * (yg - 1) + xg
                                    # if yloc==8 and yloc == 8:
                                    # print(self.gridmesh.nlon, yg, xg ,elmid,'++++++++++++++')
                                else:
                                    elmid = 0

                        if self.nl_colm['CATCHMENT']:
                            pass

                        if self.nl_colm['UNSTRUCTURED']:
                            pass

                        if elmid > 0:
                            iworker = elmid % self.mpi.p_np_worker
                            iloc, _, nelm_worker[iworker], elist_worker[iworker].val = CoLM_Utils.insert_into_sorted_list(elmid, nelm_worker[iworker],
                                                                                  elist_worker[iworker].val)

                            if nelm_worker[iworker] == len(elist_worker[iworker].val):
                                elist_worker = CoLM_Utils.expand_list(elist_worker[iworker].val, 0.2)

                if self.nl_colm['USEMPI']:
                    pass
                #     DO
                #     iworker = 0, p_np_worker - 1
                #     IF(nelm_worker(iworker) > 0)
                #     THEN
                #     idest = p_address_worker(iworker)
                #     smesg(1: 2) = (/ p_iam_glb, nelm_worker(iworker) /)
                #     ! send(01)
                #     CALL
                #     mpi_send(smesg(1: 2), 2, MPI_INTEGER, &
                #     idest, mpi_tag_size, p_comm_glb, p_err)
                #     ENDIF
                # ENDDO

                nelm = nelm + sum(nelm_worker)
                nelm_worker = np.zeros_like(nelm_worker)

            if self.nl_colm['USEMPI']:
                pass
                # DO
                # iworker = 0, p_np_worker - 1
                # idest = p_address_worker(iworker)
                # ! send(02)
                # smesg(1: 2) = (/ p_iam_glb, 0 /)
                # CALL
                # mpi_send(smesg(1: 2), 2, MPI_INTEGER, &
                # idest, mpi_tag_size, p_comm_glb, p_err)
                # ENDDO

            del nelm_worker
            for iworker in range(self.mpi.p_np_worker):
                del elist_worker[iworker].val
            del elist_worker

        if self.nl_colm['USEMPI']:
            pass
            #             IF(p_is_worker)
            #             THEN
            #             nelm = 0
            #             allocate(work_done(0: p_np_io - 1))
            #             work_done(:) =.false.
            #             DO
            #             WHILE(.
            #             not.all(work_done))
            #             ! recv(01, 02)
            #             CALL
            #             mpi_recv(rmesg(1: 2), 2, MPI_INTEGER, &
            #             MPI_ANY_SOURCE, mpi_tag_size, p_comm_glb, p_stat, p_err)
            #
            #             isrc = rmesg(1)
            #             nrecv = rmesg(2)
            #
            #             IF(nrecv > 0)
            #             THEN
            #             nelm = nelm + nrecv
            #         ELSE
            #         work_done(p_itis_io(isrc)) =.true.
            #
            #     ENDIF
            #
            #
            # ENDDO
            #
            # deallocate(work_done)
            # ENDIF
            #
            # CALL
            # mpi_barrier(p_comm_glb, p_err)

        # Step 2: Build pixel list for each elm.

        if self.mpi.p_is_worker:
            if nelm > 0:
                meshtmp = [IrregularElement() for _ in range(nelm)]
                elist = np.zeros(nelm)
                iaddr = np.zeros(nelm,dtype=int)
            nelm = 0

        if self.mpi.p_is_io:
            ysp = 0
            ynp = 0
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]

                if self.gridmesh.xcnt[iblk] <= 0 or self.gridmesh.ycnt[jblk] <= 0:
                    continue
                #
                ylg = self.gridmesh.ydsp[jblk] + 1
                yug = self.gridmesh.ydsp[jblk] + self.gridmesh.ycnt[jblk]

                if self.gridmesh.yinc == 1:
                    ysp = CoLM_Utils.find_nearest_south(self.gridmesh.lat_s[ylg], pixel.nlat, pixel.lat_s)
                    ynp = CoLM_Utils.find_nearest_north(self.gridmesh.lat_n[yug], pixel.nlat, pixel.lat_n)
                else:
                    ysp = CoLM_Utils.find_nearest_south(self.gridmesh.lat_s[yug], pixel.nlat, pixel.lat_s)
                    ynp = CoLM_Utils.find_nearest_north(self.gridmesh.lat_n[ylg], pixel.nlat, pixel.lat_n)

                nyp = ynp - ysp + 1

                xlg = self.gridmesh.xdsp[iblk] + 1
                xug = self.gridmesh.xdsp[iblk] + self.gridmesh.xcnt[iblk]
                if xug > self.gridmesh.nlon:
                    xug -= self.gridmesh.nlon

                xwp = CoLM_Utils.find_nearest_west(self.gridmesh.lon_w[xlg], pixel.nlon, pixel.lon_w)
                if not CoLM_Utils.lon_between_floor(pixel.lon_w[xwp], self.gridmesh.lon_w[xlg],
                                                    self.gridmesh.lon_e[xlg]):
                    xwp = (xwp % pixel.nlon) + 1

                xep = CoLM_Utils.find_nearest_east(self.gridmesh.lon_e[xug], pixel.nlon, pixel.lon_e)
                if not CoLM_Utils.lon_between_ceil(pixel.lon_e[xep], self.gridmesh.lon_w[xug],
                                                   self.gridmesh.lon_e[xug]):
                    xep -= 1
                    if xep == 0:
                        xep = pixel.nlon
                #
                nxp = xep - xwp + 1
                if nxp <= 0:
                    nxp += pixel.nlon

                elist2 = np.zeros((nxp, nyp), dtype=int)
                xlist2 = np.zeros((nxp, nyp), dtype=int)
                ylist2 = np.zeros((nxp, nyp), dtype=int)
                msk2 = np.zeros((nxp, nyp))

                for iy in range(ysp, ynp + 1):
                    yg = self.gridmesh.ygrd[iy]
                    yloc = self.gridmesh.yloc[int(yg)]
                    iyloc = iy - ysp
                    # iyloc = iy - ysp + 1
                    dlatp = pixel.lat_n[iy] - pixel.lat_s[iy]

                    if dlatp < 1.0e-6:
                        elist2[:, iyloc] = 0
                        continue

                    ix = xwp
                    ixloc = 0

                    while True:
                        dlonp = pixel.lon_e[ix] - pixel.lon_w[ix]
                        if dlonp < 0:
                            dlonp += 360.0
                        xg = self.gridmesh.xgrd[ix]
                        xloc = self.gridmesh.xloc[int(xg)]
                        if self.nl_colm['GRIDBASED']:
                            if datamesh.blk[iblk, jblk].val is not None:
                                # print(xloc, yloc, '-------------')
                                if datamesh.blk[iblk, jblk].val[xloc, yloc] > 0:
                                    elmid = int(self.gridmesh.nlon) * (yg + 1 - 1) + xg + 1
                                    # print(self.gridmesh.nlon, xg, yg, elmid, '+++++++++++++++')
                                else:
                                    elmid = 0

                        if self.nl_colm['CATCHMENT']:
                            elmid = datamesh.blk[iblk,jblk].val[xloc,yloc]
                        if self.nl_colm['UNSTRUCTURED']:
                            elmid = datamesh.blk[iblk,jblk].val[xloc,yloc]

                        xlist2[ixloc, iyloc] = ix
                        ylist2[ixloc, iyloc] = iy
                        elist2[ixloc, iyloc] = elmid

                        if dlonp < 1.0e-6:
                            elist2[ixloc, iyloc] = 0
                        if ix == xep:
                            break
                        ix = (ix % pixel.nlon) + 1
                        ixloc += 1
                # if iblkme==0:
                #     for i in range(nyp):
                #         print(ylist2[0,i],'//////////////////')
                if self.nl_colm['USEMPI']:
                    pass
                else:
                    # print(nxp,nyp,'------nxp----nyp------------')
                    for iy in range(nyp):
                        for ix in range(nxp):
                            elmid = elist2[ix][iy]
                            if elmid > 0:
                                # print(elmid, nelm, iloc, '+-*-/-*-')
                                iloc, is_new, nelm, elist = CoLM_Utils.insert_into_sorted_list(elmid, nelm, elist)

                                # npxl = len(np.where(elist2 == elmid))
                                npxl = (elist2 == elmid).sum()

                                if is_new:
                                    if iloc < nelm:
                                        iaddr[iloc + 1:nelm] = iaddr[iloc:nelm - 1]

                                    iaddr[iloc] = nelm - 1
                                    # print(iloc,nelm,'--------------')

                                    meshtmp[iaddr[iloc]].indx = elmid
                                    meshtmp[iaddr[iloc]].npxl = npxl
                                    # print(npxl, elmid, ix, iy, iloc, iaddr[iloc], '*********new***********')
                                else:
                                    meshtmp[iaddr[iloc]].npxl += npxl
                                    # print(npxl, elmid, ix, iy, iloc, iaddr[iloc], '********************')
                                # if iy==0 and ix==0:
                                #     for i in range(nyp):
                                #         print(xlist2[0,i], '//////////////////')
                                xlist = xlist2.transpose()[elist2.transpose()==elmid]
                                # if iy == 0 and ix == 0:
                                #     for i in range(len(xlist)):
                                #             print(xlist[i], '//////////////////')
                                ylist = ylist2.transpose()[elist2.transpose()==elmid]

                                # if iy == 0 and ix == 0:
                                #     for i in range(0, len(ylist),10):
                                #             print(ylist[i], '//////////////////')

                                # xlist = [xlist2[i][j] for i in range(nxp) for j in range(nyp) if msk2[i][j]]
                                # ylist = [ylist2[i][j] for i in range(nxp) for j in range(nyp) if msk2[i][j]]
                                # CoLM_Utils.append_to_list()
                                meshtmp[iaddr[iloc]].ilon.extend(xlist)
                                meshtmp[iaddr[iloc]].ilat.extend(ylist)
                                # if ix==0 and iy==0:
                                #     pass

                                elist2[elist2==elmid] = -1

                                # for i in range(nxp):
                                #     for j in range(nyp):
                                #         if msk2[i][j]:
                                #             elist2[i][j] = -1

                                del xlist
                                del ylist

                del elist2
                del xlist2
                del ylist2
                del msk2
            if self.nl_colm['USEMPI']:
                pass

        if self.nl_colm['USEMPI']:
            pass

        # 如果变量已分配，则释放内存
        if 'elist' in locals() and elist is not None:
            del elist
        if 'iaddr' in locals() and iaddr is not None:
            del iaddr

        # Step 3: Which block each elm locates at.
        if self.mpi.p_is_worker:
            npxl_blk = np.zeros((self.gblock.nxblk, self.gblock.nyblk))  # 假设您需要一个全零数组
            self.nelm_blk = np.zeros((self.gblock.nxblk, self.gblock.nyblk))  # 同样，这里也是全零数组

            for ie in range(nelm):
                npxl_blk[:] = 0
                for ipxl in range(meshtmp[ie].npxl):
                    xp = meshtmp[ie].ilon[ipxl]
                    yp = meshtmp[ie].ilat[ipxl]

                    xg = self.gridmesh.xgrd[xp]
                    yg = self.gridmesh.ygrd[yp]

                    xblk = self.gridmesh.xblk[xg]
                    yblk = self.gridmesh.yblk[yg]

                    npxl_blk[xblk, yblk] += 1

                # iloc_max = np.unravel_index(np.argmax(npxl_blk), np.shape(npxl_blk))
                m = np.argmax(npxl_blk)
                iloc_max = np.zeros(2, dtype=int)
                iloc_max[0], iloc_max[1] = divmod(m, npxl_blk.shape[1])
                # print(ie, '-------')
                meshtmp[ie].xblk = iloc_max[0]
                meshtmp[ie].yblk = iloc_max[1]

                self.nelm_blk[iloc_max[0], iloc_max[1]] += 1

            del npxl_blk

        if self.nl_colm['USEMPI']:
            pass
        # Step 4: IF MPI is used, sending elms from worker to their IO processes.

        if self.mpi.p_is_io:
            blkdsp = np.zeros((self.gblock.nxblk, self.gblock.nyblk), dtype=int)
            for jblk in range(self.gblock.nyblk):
                for iblk in range(self.gblock.nxblk):
                    if (iblk != 0) or (jblk != 0):
                        if iblk == 0:
                            iblk_p = self.gblock.nxblk-1
                            jblk_p = jblk - 1
                        else:
                            iblk_p = iblk - 1
                            jblk_p = jblk

                        if self.gblock.pio[iblk_p, jblk_p] == self.mpi.p_iam_glb:
                            blkdsp[iblk, jblk] = blkdsp[iblk_p, jblk_p] + self.nelm_blk[iblk_p, jblk_p]
                        else:
                            blkdsp[iblk, jblk] = blkdsp[iblk_p, jblk_p]

        if self.nl_colm['USEMPI']:
            pass
        else:
            self.numelm = nelm
            if self.numelm > 0:
                self.mesh = [IrregularElement() for _ in range(self.numelm)]
                blkcnt = np.zeros((self.gblock.nxblk, self.gblock.nyblk), dtype=int)
                for ie in range(self.numelm):
                    xblk = meshtmp[ie].xblk
                    yblk = meshtmp[ie].yblk

                    je = blkdsp[xblk, yblk] + blkcnt[xblk, yblk]
                    # print(xblk, yblk, blkdsp[xblk, yblk], blkcnt[xblk, yblk]+1, je+1, '------------')
                    # print(ie,je,len(self.mesh),len(meshtmp),'++++++++')
                    self.mesh[je] = self.copy_elm(meshtmp[ie])
                    blkcnt[xblk, yblk] += 1

        # Step 4-2: sort elms.
        if self.mpi.p_is_io:
            if meshtmp is not None:
                for ie in range(len(meshtmp)):
                    if meshtmp[ie].ilon is not None:
                        del meshtmp[ie].ilon
                    if meshtmp[ie].ilat is not None:
                        del meshtmp[ie].ilat
                del meshtmp

            if self.numelm > 0:
                meshtmp = [self.copy_elm(self.mesh[ie]) for ie in range(self.numelm)]

                for iblkme in range(self.gblock.nblkme):
                    iblk = self.gblock.xblkme[iblkme]
                    jblk = self.gblock.yblkme[iblkme]
                    if blkcnt[iblk, jblk] > 0:
                        elmindx = np.zeros(blkcnt[iblk, jblk],dtype=int)
                        order = [i for i in range(blkcnt[iblk, jblk])]
                        # print(order,'-------------------1')
                        for ie in range(blkdsp[iblk, jblk], blkdsp[iblk, jblk] + blkcnt[iblk, jblk]):
                            # print(ie,blkdsp[iblk, jblk],ie - blkdsp[iblk, jblk], len(elmindx),len(self.mesh),'++++++++++++')
                            elmindx[ie - blkdsp[iblk, jblk]] = self.mesh[ie].indx
                        # if iblkme==0:
                        #     for i in range(len(elmindx)):
                        #         print(elmindx[i],'-------++++------')
                        keys = np.lexsort((order,elmindx))
                        order = [order[i] for i in keys]
                        elmindx = [elmindx[i] for i in keys]
                        # if iblkme==0:
                        #     for i in range(len(order)):
                        #         print(order[i],'----------------')

                        # order,_ = CoLM_Utils.quicksort(blkcnt[iblk, jblk], elmindx, order)
                        # print(order,'---------------2')

                        for ie in range(blkcnt[iblk, jblk]):
                            # print(blkdsp[iblk, jblk] + ie, blkdsp[iblk, jblk] + order[ie], ie, order[ie],'-----------------------')
                            self.mesh[blkdsp[iblk, jblk] + ie] = self.copy_elm(
                                meshtmp[blkdsp[iblk, jblk] + order[ie]])

                        del elmindx
                        del order

        if blkdsp is not None:
            del blkdsp
        if blkcnt is not None:
            del blkcnt

        if meshtmp is not None:
            for ie in range(len(meshtmp)):
                if meshtmp[ie].ilon is not None:
                    del meshtmp[ie].ilon
                if meshtmp[ie].ilat is not None:
                    del meshtmp[ie].ilat
            del meshtmp

        # Step 5: IF MPI is used, scatter elms from IO to workers.
        if self.nl_colm['USEMPI']:
            pass
            # CALL  scatter_mesh_from_io_to_worker()

        if self.mpi.p_is_master:
            print('Making mesh elements :')

        if self.nl_colm['USEMPI']:
            pass
        else:
            print('Total: ', self.numelm, ' elements.')

    def copy_elm(self, elm_from):
        elm_to = IrregularElement()
        elm_to.indx = elm_from.indx
        elm_to.npxl = elm_from.npxl
        elm_to.xblk = elm_from.xblk
        elm_to.yblk = elm_from.yblk

        if elm_to.ilat is not None:
            del elm_to.ilat
        if elm_to.ilon is not None:
            del elm_to.ilon

        elm_to.ilon = elm_from.ilon.copy()
        elm_to.ilat = elm_from.ilat.copy()
        return elm_to

    def mesh_free_mem(self):
        if self.mesh is not None:
            for ie in range(self.numelm):
                if self.mesh[ie].ilon is not None:
                    del self.mesh[ie].ilon
                if self.mesh[ie].ilat is not None:
                    del self.mesh[ie].ilat
            del self.mesh


class IrregularElement:
    def __init__(self):
        self.indx = 0
        self.xblk = 0
        self.yblk = 0
        self.npxl = 0
        self.ilon = []
        self.ilat = []
