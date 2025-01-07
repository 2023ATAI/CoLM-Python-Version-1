from CoLM_Grid import Grid_type, GridList
from CoLM_DataType import Pointer
import numpy as np
import CoLM_Utils


class MappingGrid2PSet:
    def __init__(self, nl_colm, gblock, mpi, spval):
        self.grid = Grid_type(nl_colm, gblock)
        self.mpi = mpi
        self.npset = 0
        self.spval = spval
        self.glist = []
        self.address = []
        self.gweight = []

    # def build(self, grid, npset, glist, address, gweight):
    #     self.grid = grid
    #     self.npset = npset
    #     self.glist = glist
    #     self.address = address
    #     self.gweight = gweight

    # ------------------------------------------
    def build(self, fgrid, pixelset, mesh, pixel, gfilter=None, missing_value=None, pfilter=None):
        """
        DESCRIPTION:

            Mapping data types and subroutines from gridded data to vector data
            defined on pixelset.

            Notice that:
            1. A mapping can be built with method mapping.build.
            2. Area weighted mapping is carried out.
            3. For 2D gridded data, dimensions are from [lon,lat] to [vector].
            4. For 3D gridded data, dimensions are from [d,lon,lat] to [d,vector].

        Created by Shupeng Zhang, May 2023
        """

        # Local variables
        afrac = []
        gfrom = []
        list_lat = []
        ng_lat = None
        ys = None
        yn = None
        xw = None
        xe = None
        xlist = None
        ylist = None
        ipt = None
        msk = None

        ie = None
        iset = None
        ng = None
        ig = None
        ng_all = None
        iloc = None
        ng0 = None
        npxl = None
        ipxl = None
        ilat = None
        ilon = None
        iworker = None
        iproc = None
        iio = None
        idest = None
        isrc = None
        nrecv = None
        rmesg = [0, 0]
        smesg = [0, 0]
        iy = None
        ix = None
        xblk = None
        yblk = None
        xloc = None
        yloc = None
        lat_s = None
        lat_n = None
        lon_w = None
        lon_e = None
        area = None
        is_new = None

        # Assuming some MPI-related functionality
        # if USEMPI:
        #     mpi_barrier(p_comm_glb, p_err)

        if self.mpi.p_is_master:
            print(
                f"Making mapping from grid to pixel set: {fgrid.nlat} grids in latitude, {fgrid.nlon} grids in longitude")

        self.grid.xblk = fgrid.xblk
        self.grid.yblk = fgrid.yblk
        self.grid.xloc = fgrid.xloc
        self.grid.yloc = fgrid.yloc

        self.npset = pixelset.nset
        if self.mpi.p_is_worker:
            for i in range(pixelset.nset):
                afrac.append(Pointer())
                gfrom.append(GridList(0))

            ys = np.zeros(pixel.nlat, dtype=int)
            yn = np.zeros(pixel.nlat, dtype=int)
            xw = np.zeros(pixel.nlon, dtype=int)
            xe = np.zeros(pixel.nlon, dtype=int)

            for ilat in range(pixel.nlat):
                ys[ilat] = CoLM_Utils.find_nearest_south(pixel.lat_s[ilat], fgrid.nlat, fgrid.lat_s)
                yn[ilat] = CoLM_Utils.find_nearest_north(pixel.lat_n[ilat], fgrid.nlat, fgrid.lat_n)

            for ilon in range(pixel.nlon):
                xw[ilon] = CoLM_Utils.find_nearest_west(pixel.lon_w[ilon], fgrid.nlon, fgrid.lon_w)
                xe[ilon] = CoLM_Utils.find_nearest_east(pixel.lon_e[ilon], fgrid.nlon, fgrid.lon_e)

            for i in range(fgrid.nlat):
                p = Pointer()
                p.val = np.zeros(100)
                list_lat.append(p)

            ng_lat = np.zeros(fgrid.nlat, dtype=int)

            for iset in range(pixelset.nset):
                ie = pixelset.ielm[iset]
                npxl = pixelset.ipxend[iset] - pixelset.ipxstt[iset] + 1

                afrac[iset].val = np.zeros(npxl)
                gfrom[iset].ilat = np.zeros(npxl, dtype=int)
                gfrom[iset].ilon = np.zeros(npxl, dtype=int)
                gfrom[iset].ng = 0
                # for ipxl in range(pixelset.ipxstt[iset] - 1, pixelset.ipxend[iset]):
                #     print(mesh[ie].ilon[ipxl],'------ilat------')

                # print(pixelset.ipxstt[iset]-1, pixelset.ipxend[iset],'-------ipxend---------')
                for ipxl in range(pixelset.ipxstt[iset] - 1, pixelset.ipxend[iset]):
                    # print(ipxl,pixelset.ipxstt[iset - 1] - 1,pixelset.ipxend[iset - 1],'-------------')
                    ilat = mesh[ie].ilat[ipxl] - 1
                    ilon = mesh[ie].ilon[ipxl] - 1

                    # print('***************', ilat, len(ys), len(yn))

                    for iy in range(ys[ilat], yn[ilat]+1, fgrid.yinc):
                        # if ipxl ==3473:
                        #     print(iy, ilat, ys[ilat], yn[ilat] + 1, fgrid.yinc, '--------')
                        lat_s = max(fgrid.lat_s[iy], pixel.lat_s[ilat])
                        lat_n = min(fgrid.lat_n[iy], pixel.lat_n[ilat])
                        # print(iy, fgrid.lat_n[iy], ilat, pixel.lat_n[ilat], '----fgrid-----')
                        # print(ys[ilat],yn[ilat],lat_s,lat_n,ilat,ilon,'----fgrid-----')
                        # print(ilat,ilon,lat_s,lat_n)
                        # print(fgrid.lat_s[iy],pixel.lat_s[ilat],fgrid.lat_n[iy],pixel.lat_n[ilat])

                        if (lat_n - lat_s) < 1.0e-6:
                            continue

                        ix = xw[ilon]
                        while True:
                            if ix == xw[ilon]:
                                lon_w = pixel.lon_w[ilon]
                            else:
                                lon_w = fgrid.lon_w[ix]

                            if ix == xe[ilon]:
                                lon_e = pixel.lon_e[ilon]
                            else:
                                lon_e = fgrid.lon_e[ix]

                            if lon_e > lon_w:
                                if (lon_e - lon_w) < 1.0e-6:
                                    if ix == xe[ilon]:
                                        break
                                    ix = (ix % (fgrid.nlon - 1)) + 1
                                    continue
                            else:
                                if (lon_e + 360.0 - lon_w) < 1.0e-6:
                                    if ix == xe[ilon]:
                                        break
                                    ix = (ix % (fgrid.nlon - 1)) + 1
                                    continue

                            area = CoLM_Utils.areaquad(lat_s, lat_n, lon_w, lon_e)

                            gfrom[iset].ng, gfrom[iset].ilon, gfrom[
                                iset].ilat, iloc, is_new = CoLM_Utils.insert_into_sorted_list2(ix, iy, gfrom[iset].ng,
                                                                                               gfrom[iset].ilon,
                                                                                               gfrom[iset].ilat)

                            if is_new:
                                if iloc < gfrom[iset].ng:
                                    afrac[iset].val[iloc + 1:gfrom[iset].ng] = \
                                        afrac[iset].val[iloc:gfrom[iset].ng - 1]

                                afrac[iset].val[iloc] = area
                            else:
                                afrac[iset].val[iloc] += area

                            if gfrom[iset].ng == len(gfrom[iset].ilat):
                                gfrom[iset].ilat = CoLM_Utils.expand_list(gfrom[iset].ilat, 0.2)
                                gfrom[iset].ilon = CoLM_Utils.expand_list(gfrom[iset].ilon, 0.2)
                                afrac[iset].val = CoLM_Utils.expand_list(afrac[iset].val, 0.2)

                            # print(ix,iy,ng_lat[iy],iloc,'====sort=======')
                            ng_lat[iy], list_lat[iy].val, iloc, is_new = CoLM_Utils.insert_into_sorted_list1(ix,
                                                                                                             ng_lat[iy],
                                                                                                             list_lat[
                                                                                                                 iy].val)

                            if ng_lat[iy] == len(list_lat[iy].val):
                                list_lat[iy].val = CoLM_Utils.expand_list(list_lat[iy].val, 0.2)

                            if ix == xe[ilon]:
                                break
                            ix = (ix % (fgrid.nlon - 1)) + 1

                # Deallocate (free) variables
            del ys
            del yn
            del xw
            del xe

            # Compute the total number of elements in xlist and ylist

            ng_all = np.sum(ng_lat)
            # print(ng_all,'-----------ng_all-----------')
            # Allocate (initialize) xlist and ylist
            xlist = np.zeros(ng_all, dtype=int)
            ylist = np.zeros(ng_all, dtype=int)

            # Initialize the index counter
            ig = 0

            # Fill xlist and ylist with values
            for iy in range(fgrid.nlat):
                for ix in range(ng_lat[iy]):
                    xlist[ig] = list_lat[iy].val[ix]
                    ylist[ig] = iy
                    ig += 1

            # Deallocate ng_lat
            del ng_lat

            # Deallocate list_lat
            for iy in range(fgrid.nlat):
                list_lat[iy].val = None
            list_lat = None

        # Deallocate (free) self%glist if already allocated
        # Allocate (initialize) self%glist
        # print(self.mpi.p_np_io,'-------p_np_io--------')
        for i in range(self.mpi.p_np_io):
            self.glist.append(GridList(ng))

        for iproc in range(self.mpi.p_np_io):
            # if USEMPI:
            #     # Determine the mask and count based on ipt and p_address_io
            #     msk = (ipt == p_address_io[iproc])
            #     ng = np.sum(msk)
            # else:
            ng = ng_all

            self.glist[iproc].ilat = np.zeros(ng, dtype=int)
            self.glist[iproc].ilon = np.zeros(ng, dtype=int)
            self.glist[iproc].ng = 0

        # Fill self%glist with appropriate values
        for ig in range(ng_all):
            # if USEMPI:
            #     iproc = p_itis_io[ipt[ig - 1]]
            # else:
            iproc = 0
            ng = self.glist[iproc].ng
            self.glist[iproc].ilon[ng] = xlist[ig]
            self.glist[iproc].ilat[ng] = ylist[ig]
            self.glist[iproc].ng += 1

        # Deallocate (free) xlist and ylist
        del xlist
        del ylist

        # Handling missing values
        if missing_value is not None:
            if self.mpi.p_is_io:
                for iproc in range(self.mpi.p_np_worker):
                    if self.glist[iproc].ng > 0:
                        msk = np.zeros(self.glist[iproc].ng, dtype=bool)

                        for ig in range(self.glist[iproc].ng):
                            ilon = self.glist[iproc].ilon[ig]
                            ilat = self.glist[iproc].ilat[ig]
                            xblk = self.grid.xblk[ilon]
                            yblk = self.grid.yblk[ilat]
                            xloc = self.grid.xloc[ilon]
                            yloc = self.grid.yloc[ilat]

                            msk[ig] = gfilter.blk[xblk, yblk].val[xloc, yloc] != missing_value

                        if np.any( not msk):
                            # self.glist[iproc].ng = np.sum(msk)
                            self.glist[iproc].ng = np.sum(msk)
                            if self.glist[iproc].ng > 0:
                                xlist = np.compress(msk, self.glist[iproc].ilon)
                                ylist = np.compress(msk, self.glist[iproc].ilat)

                                # print('********************')

                                # Reallocate and update the lists
                                self.glist[iproc].ilon = xlist
                                self.glist[iproc].ilat = ylist

                            # Clean up temporary arrays
                            if xlist is not None:
                                del xlist
                            if ylist is not None:
                                del ylist
                        del msk

            # Handle processing for workers
            if self.mpi.p_is_worker:
                for iset in range(pixelset.nset):
                    msk = np.zeros(gfrom[iset].ng, dtype=bool)

                    for ig in range(gfrom[iset].ng):
                        ilon = gfrom[iset].ilon[ig]
                        ilat = gfrom[iset].ilat[ig]
                        xblk = fgrid.xblk[ilon]
                        yblk = fgrid.yblk[ilat]

                        iproc = 0

                        msk[ig] = CoLM_Utils.find_in_sorted_list2(ilon, ilat, self.glist[iproc].ng,
                                                                  self.glist[iproc].ilon, self.glist[iproc].ilat) > 0

                        # Further processing can be done here
                    if pfilter is not None:
                        pfilter[iset] = np.any(msk)

                    ng0 = gfrom[iset].ng
                    gfrom[iset].ng = np.count_nonzero(msk)
                    if np.any(msk) and np.any(~msk):
                        ng = gfrom[iset].ng
                        gfrom[iset].ilon[:ng] = np.compress(msk, gfrom[iset].ilon[:ng0], axis=0)
                        gfrom[iset].ilat[:ng] = np.compress(msk, gfrom[iset].ilat[:ng0], axis=0)
                        afrac[iset].val[:ng] = np.compress(msk, afrac[iset].val[:ng0], axis=0)

                    # Deallocate mask array
                    del msk

        if self.mpi.p_is_worker:
            for i in range(pixelset.nset):
                self.address.append(Pointer())
            for i in range(pixelset.nset):          
                self.gweight.append(Pointer())

            for iset in range(pixelset.nset):
                ng = gfrom[iset].ng
                if ng > 0:
                    self.address[iset].val = np.zeros((2, ng), dtype =int)
                    self.gweight[iset].val = np.zeros(ng)

                    if np.sum(afrac[iset].val[:ng]) < 1.0e-12:
                        self.gweight[iset].val = [1.0 / ng]
                    else:
                        self.gweight[iset].val = afrac[iset].val[:ng] / np.sum(afrac[iset].val[:ng])

                    for ig in range(gfrom[iset].ng):
                        ilon = gfrom[iset].ilon[ig]
                        ilat = gfrom[iset].ilat[ig]
                        # xblk = fgrid.xblk[int(ilon)]
                        # yblk = fgrid.yblk[int(ilat)]

                        # Further processing can be done here
                        iproc = 0

                        # Update address and find location in sorted list
                        self.address[iset].val[0, ig] = iproc
                        self.address[iset].val[1, ig] = CoLM_Utils.find_in_sorted_list2(ilon, ilat, self.glist[iproc].ng,  self.glist[iproc].ilon, self.glist[iproc].ilat) - 1

            # Deallocate memory
            # for iset in range(pixelset.nset):
            #     del afrac[iset].val
            #     del gfrom[iset].ilon
            #     del gfrom[iset].ilat

            del afrac
            del gfrom

            # if USEMPI:
            #     # Synchronize processes
            #     mpi_barrier(p_comm_glb, p_err)
        return pfilter

    #     # Actual function converted from Fortran
    def map_aweighted(self, gdata, pdata):
        if self.mpi.p_is_io:
            for iproc in range(self.mpi.p_np_worker):
                if self.glist[iproc].ng > 0:
                    gbuff = []
                    for i in range(self.glist[iproc].ng):
                        gbuff.append(Pointer())

                    for ig in range(self.glist[iproc].ng):
                        ilon = self.glist[iproc].ilon[ig]
                        ilat = self.glist[iproc].ilat[ig]
                        xblk = self.grid.xblk[ilon]
                        yblk = self.grid.yblk[ilat]
                        xloc = self.grid.xloc[ilon]
                        yloc = self.grid.yloc[ilat]

                        gbuff[ig] = gdata.blk[xblk, yblk].val[xloc, yloc]

                    # if USEMPI:
                    #     idest = p_address_worker(iproc)
                    #     mpi_send(gbuff, self.glist[iproc].ng, MPI.DOUBLE, idest, mpi_tag_data, comm)
                    #     del gbuff

        if self.mpi.p_is_worker:
            pbuff = []
            for i in range(self.mpi.p_np_io):
                pbuff.append(Pointer())

            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    pbuff[iproc].val = np.zeros(self.glist[iproc].ng, dtype=np.float64)

                    # if USEMPI:
                    #     isrc = p_address_io(iproc)
                    #     mpi_recv(pbuff[iproc], self.glist[iproc].ng, MPI.DOUBLE, isrc, mpi_tag_data, comm, p_err)
                    # else:
                    pbuff[0].val = gbuff
                    del gbuff

            for iset in range(self.npset):
                if self.gweight[iset].val is not None:
                    pdata[iset] = 0.0
                    for ig in range(len(self.gweight[iset].val)):
                        iproc = int(self.address[iset].val[0, ig])
                        iloc = int(self.address[iset].val[1, ig])
                        pdata[iset] += pbuff[iproc].val[iloc] * self.gweight[iset].val[ig]
                else:
                    pdata[iset] = self.spval

            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    del pbuff[iproc]

            del pbuff

        return pdata

# # Actual function converted from Fortran
# def map_g2p_aweighted_3d(self, gdata, ndim1, pdata):
#     comm = MPI.COMM_WORLD
#     p_err = MPI.Status()
#     p_np_worker = comm.Get_size() - 1
#     p_np_io = 1
#     p_address_worker = lambda x: x  # Placeholder function
#     p_address_io = lambda x: x  # Placeholder function

#     # Local variables
#     if self.mpi.p_is_io():
#         for iproc in range(p_np_worker):
#             if self.glist[iproc].ng > 0:
#                 gbuff = np.empty((ndim1, self.glist[iproc].ng), dtype=np.float64)

#                 for ig in range(self.glist[iproc].ng):
#                     ilon = self.glist[iproc].ilon[ig]
#                     ilat = self.glist[iproc].ilat[ig]
#                     xblk = self.grid.xblk[ilon]
#                     yblk = self.grid.yblk[ilat]
#                     xloc = self.grid.xloc[ilon]
#                     yloc = self.grid.yloc[ilat]

#                     gbuff[:, ig] = gdata.blk[xblk, yblk].val[:, xloc, yloc]

#                 if USEMPI:
#                     idest = p_address_worker(iproc)
#                     mpi_send(gbuff, ndim1 * self.glist[iproc].ng, MPI.DOUBLE, idest, mpi_tag_data, comm)
#                     del gbuff

#     if self.mpi.p_is_worker():
#         pbuff = [None] * p_np_io

#         for iproc in range(p_np_io):
#             if self.glist[iproc].ng > 0:
#                 pbuff[iproc] = np.empty((ndim1, self.glist[iproc].ng), dtype=np.float64)

#                 if USEMPI:
#                     isrc = p_address_io(iproc)
#                     mpi_recv(pbuff[iproc], ndim1 * self.glist[iproc].ng, MPI.DOUBLE, isrc, mpi_tag_data, comm, p_err)
#                 else:
#                     pbuff[0] = gbuff
#                     del gbuff

#         for iset in range(1, self.npset + 1):
#             if self.gweight[iset].val is not None:
#                 pdata[:, iset] = 0.0
#                 for ig in range(len(self.gweight[iset].val)):
#                     iproc = self.address[iset].val[0, ig]
#                     iloc = self.address[iset].val[1, ig]
#                     pdata[:, iset] += pbuff[iproc][:, iloc] * self.gweight[iset].val[ig]
#             else:
#                 pdata[:, iset] = spval

#         for iproc in range(p_np_io):
#             if self.glist[iproc].ng > 0:
#                 del pbuff[iproc]

#         del pbuff

#     if USEMPI:
#         mpi_barrier(comm)

#     # Actual function converted from Fortran
#     def map_g2p_max_frequency_2d(self, gdata, pdata):
#         comm = MPI.COMM_WORLD
#         p_err = MPI.Status()
#         p_np_worker = comm.Get_size() - 1
#         p_np_io = 1
#         p_address_worker = lambda x: x  # Placeholder function
#         p_address_io = lambda x: x  # Placeholder function

#         # Local variables
#         if self.mpi.p_is_io():
#             for iproc in range(p_np_worker):
#                 if self.glist[iproc].ng > 0:
#                     gbuff = np.empty(self.glist[iproc].ng, dtype=np.int32)

#                     for ig in range(self.glist[iproc].ng):
#                         ilon = self.glist[iproc].ilon[ig]
#                         ilat = self.glist[iproc].ilat[ig]
#                         xblk = self.grid.xblk[ilon]
#                         yblk = self.grid.yblk[ilat]
#                         xloc = self.grid.xloc[ilon]
#                         yloc = self.grid.yloc[ilat]

#                         gbuff[ig] = gdata.blk[xblk, yblk].val[xloc, yloc]

#                     if USEMPI:
#                         idest = p_address_worker(iproc)
#                         mpi_send(gbuff, self.glist[iproc].ng, MPI.INT, idest, mpi_tag_data, comm)
#                         del gbuff

#         if self.mpi.p_is_worker():
#             pbuff = [None] * p_np_io

#             for iproc in range(p_np_io):
#                 if self.glist[iproc].ng > 0:
#                     pbuff[iproc] = np.empty(self.glist[iproc].ng, dtype=np.int32)

#                     if USEMPI:
#                         isrc = p_address_io(iproc)
#                         mpi_recv(pbuff[iproc], self.glist[iproc].ng, MPI.INT, isrc, mpi_tag_data, comm, p_err)
#                     else:
#                         pbuff[0] = gbuff
#                         del gbuff

#             for iset in range(1, self.npset + 1):
#                 if self.gweight[iset].val is not None:
#                     ig = np.argmax(self.gweight[iset].val)
#                     iproc = self.address[iset].val[0, ig]
#                     iloc = self.address[iset].val[1, ig]
#                     pdata[iset] = pbuff[iproc][iloc]
#                 else:
#                     pdata[iset] = -9999

#             for iproc in range(p_np_io):
#                 if self.glist[iproc].ng > 0:
#                     del pbuff[iproc]

#             del pbuff

#     def mapping_grid2pset_free_mem(self):
#         # Local variables
#         iproc = None
#         iset = None

#         # Deallocation of grid variables
#         if hasattr(self.grid, 'xblk') and self.grid.xblk is not None:
#             del self.grid.xblk
#         if hasattr(self.grid, 'yblk') and self.grid.yblk is not None:
#             del self.grid.yblk

#         if hasattr(self.grid, 'xloc') and self.grid.xloc is not None:
#             del self.grid.xloc
#         if hasattr(self.grid, 'yloc') and self.grid.yloc is not None:
#             del self.grid.yloc

#         # Deallocation of glist
#         if hasattr(self, 'glist') and self.glist is not None:
#             for iproc in range(len(self.glist)):
#                 if hasattr(self.glist[iproc], 'ilat') and self.glist[iproc].ilat is not None:
#                     del self.glist[iproc].ilat
#                 if hasattr(self.glist[iproc], 'ilon') and self.glist[iproc].ilon is not None:
#                     del self.glist[iproc].ilon
#             del self.glist

#         # Deallocation of address
#         if hasattr(self, 'address') and self.address is not None:
#             for iset in range(1, self.npset + 1):
#                 if hasattr(self.address[iset], 'val') and self.address[iset].val is not None:
#                     del self.address[iset].val
#             del self.address

#         # Deallocation of gweight
#         if hasattr(self, 'gweight') and self.gweight is not None:
#             for iset in range(1, self.npset + 1):
#                 if hasattr(self.gweight[iset], 'val') and self.gweight[iset].val is not None:
#                     del self.gweight[iset].val
#             del self.gweigh
