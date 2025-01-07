# ----------------------------------------------------------------------------
# DESCRIPTION:
#
#    Mapping data types and subroutines from vector data defined on pixelsets
#    to gridded data.
#
#    Notice that:
#    1. A mapping can be built with method mapping%build.
#    2. Overloaded method "map" can map 1D, 2D or 3D vector data to gridded data 
#       by using area weighted scheme. 
#    3. Method "map_split" can split data in a vector according to pixelset type
#       and map data to 3D gridded data. 
#       The dimensions are from [vector] to [type,lon,lat].
# 
# Created by Shupeng Zhang, May 2023
# ----------------------------------------------------------------------------
import numpy as np
from CoLM_DataType import Pointer,DataType
import CoLM_Utils
from CoLM_Grid import Grid_type, GridList

class MappingPset2Grid(object):
    def __init__(self, mpi, nl_colm, gblock):
        self.grid = Grid_type(nl_colm, gblock)
        self.datatype = DataType(gblock)
        self.mpi = mpi
        self.nl_colm = nl_colm
        self.npset = 0
        self.glist = None  # Assuming grid_list_type is a list
        self.address = None
        self.gweight = None

    def build(self, pixelset, fgrid, pixel, mesh, pctpset=None, gfilter=None, missing_value=None, pfilter=None):
        if self.nl_colm['USEMPI']:
            pass

        if self.mpi.p_is_master:
            print(
                f"Making mapping from pixel set to grid: {fgrid.nlat} grids in latitude {fgrid.nlon} grids in longitude")

        # Allocate new arrays
        self.grid.xblk = fgrid.xblk
        self.grid.yblk = fgrid.yblk
        self.grid.xloc = fgrid.xloc
        self.grid.yloc = fgrid.yloc

        self.npset = pixelset.nset

        if self.mpi.p_is_worker:
            afrac = []
            for i in range(pixelset.nset):
                afrac.append(Pointer())
            gfrom = []
            for i in range(pixelset.nset):
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

            list_lat = []
            for i in range(fgrid.nlat):
                list_lat.append(Pointer())
            for iy in range(fgrid.nlat):
                list_lat[iy].val = np.zeros(100, dtype=int)

            ng_lat = np.zeros(fgrid.nlat, dtype=int)

            for iset in range(pixelset.nset):
                ie = pixelset.ielm[iset]
                npxl = pixelset.ipxend[iset] - pixelset.ipxstt[iset] + 1

                afrac[iset].val = np.zeros(npxl)
                gfrom[iset].ilat = np.zeros(npxl, dtype=int)
                gfrom[iset].ilon = np.zeros(npxl, dtype=int)
                gfrom[iset].ng = 0

                for ipxl in range(pixelset.ipxstt[iset] - 1, pixelset.ipxend[iset]):
                    ilat = mesh[ie].ilat[ipxl] - 1
                    ilon = mesh[ie].ilon[ipxl] - 1
                    # if ipxl==0:
                    #     print(ilat, ilon, ys[ilat], yn[ilat], fgrid.yinc, '----do')
                    for iy in range(ys[ilat], yn[ilat] + fgrid.yinc, fgrid.yinc):
                        lat_s = max(fgrid.lat_s[iy], pixel.lat_s[ilat])
                        lat_n = min(fgrid.lat_n[iy], pixel.lat_n[ilat])

                        # if iy ==ys[ilat] and ipxl==0:
                        #     print(lat_s, lat_n, fgrid.lat_s[iy], pixel.lat_s[ilat] ,fgrid.lat_n[iy], pixel.lat_n[ilat],'----iy------')

                        if (lat_n - lat_s) < 1.0e-6:
                            print('continue')
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
                                    ix = (ix % fgrid.nlon) + 1
                                    continue
                            else:
                                if (lon_e + 360.0 - lon_w) < 1.0e-6:
                                    if ix == xe[ilon]:
                                        break
                                    ix = (ix % fgrid.nlon) + 1
                                    continue

                            area = CoLM_Utils.areaquad(lat_s, lat_n, lon_w, lon_e)

                            gfrom[iset].ng, gfrom[iset].ilon, gfrom[
                                iset].ilat, iloc, is_new = CoLM_Utils.insert_into_sorted_list2(ix, iy, gfrom[iset].ng,
                                                                                               gfrom[iset].ilon,
                                                                                               gfrom[iset].ilat)

                            if is_new:
                                if iloc < gfrom[iset].ng:
                                    afrac[iset].val[iloc + 1:gfrom[iset].ng] = afrac[iset].val[
                                                                           iloc:gfrom[iset].ng - 1]
                                afrac[iset].val[iloc] = area
                            else:
                                afrac[iset].val[iloc] += area

                            if gfrom[iset].ng == len(gfrom[iset].ilat):
                                gfrom[iset].ilat = CoLM_Utils.expand_list(gfrom[iset].ilat, 0.2)
                                gfrom[iset].ilon = CoLM_Utils.expand_list(gfrom[iset].ilon, 0.2)
                                afrac[iset].val = CoLM_Utils.expand_list(afrac[iset].val, 0.2)

                            # ix = (ix % fgrid.nlon) + 1

                            ng_lat[iy], list_lat[iy].val, _, _ = CoLM_Utils.insert_into_sorted_list1(ix, ng_lat[iy],
                                                                                                     list_lat[iy].val)

                            if ng_lat[iy] == len(list_lat[iy].val):
                                list_lat[iy] = CoLM_Utils.expand_list(list_lat[iy].val, 0.2)

                            if ix == xe[ilon]:
                                break
                            ix = (ix + 1) % fgrid.nlon

            del ys
            del yn
            del xw
            del xe

            ng_all = np.sum(ng_lat)
            xlist = np.zeros(ng_all, dtype=int)
            ylist = np.zeros(ng_all, dtype=int)

            ig = 0
            for iy in range(fgrid.nlat):
                if ng_lat[iy] > 0:
                    for ix in range(ng_lat[iy]):
                        xlist[ig] = list_lat[iy].val[ix]
                        ylist[ig] = iy
                        ig += 1

            del ng_lat
            for iy in range(fgrid.nlat):
                del list_lat[iy].val
            del list_lat

            if self.nl_colm['USEMPI']:
                pass

            self.glist = []

            for i in range(self.mpi.p_np_io):
                self.glist.append(GridList(ng_all))

            for iproc in range(self.mpi.p_np_io):
                ng = ng_all
                self.glist[iproc].ilat = np.empty(ng, dtype=int)
                self.glist[iproc].ilon = np.empty(ng, dtype=int)

                # Initialize ng to 0
                self.glist[iproc].ng = 0

            for ig in range(ng_all):
                iproc = 0

                self.glist[iproc].ng += 1

                # Update ng for convenience
                ng = self.glist[iproc].ng -1

                self.glist[iproc].ilon[ng] = xlist[ig]
                self.glist[iproc].ilat[ng] = ylist[ig]
            # del xlist
            # del ylist
            if self.nl_colm['USEMPI']:
                pass
                # del ipt
                # del msk

            self.address = []
            for i in range(pixelset.nset):
                self.address.append(Pointer())
            self.olparea = []
            for i in range(pixelset.nset):
                self.olparea.append(Pointer())

            for iset in range(pixelset.nset):
                ng = gfrom[iset].ng
                self.address[iset].val = np.zeros([2, ng],dtype=int)
                self.olparea[iset].val = np.zeros(ng)

                self.olparea[iset].val = afrac[iset].val[:ng]

                if pctpset is not None:
                    self.olparea[iset].val = self.olparea[iset].val * pctpset[iset]

                for ig in range(gfrom[iset].ng):
                    ilon = gfrom[iset].ilon[ig]
                    ilat = gfrom[iset].ilat[ig]
                    xblk = fgrid.xblk[ilon]
                    yblk = fgrid.yblk[ilat]

                    iproc = 0
                    self.address[iset].val[0, ig] = iproc
                    self.address[iset].val[1, ig] = CoLM_Utils.find_in_sorted_list2(
                        ilon, ilat, self.glist[iproc].ng, self.glist[iproc].ilon, self.glist[iproc].ilat)
            del xlist
            del ylist

        # if missing_value is not None:
        #     if self.mpi.p_is_io:
        #         for iproc in range(self.mpi.p_np_worker - 1):
        #             if self.glist[iproc].ng > 0:
        #                 msk = np.zeros(self.glist[iproc].ng)
        #
        #                 for ig in range(self.glist[iproc].ng):
        #                     ilon = self.glist[iproc].ilon[ig]
        #                     ilat = self.glist[iproc].ilat[ig]
        #                     xblk = self.grid.xblk[ilon]
        #                     yblk = self.grid.yblk[ilat]
        #                     xloc = self.grid.xloc[ilon]
        #                     yloc = self.grid.yloc[ilat]
        #
        #                     msk[ig] = gfilter.blk[xblk, yblk].val[xloc, yloc] != missing_value
        #
        #                 if np.any(not msk):
        #
        #                     self.glist[iproc].ng = sum(msk)
        #
        #                     if self.glist(iproc).ng > 0:
        #                         xlist = np.compress(msk, self.glist[iproc].ilon)
        #                         ylist = np.compress(msk, self.glist[iproc].ilat)
        #
        #                     del self.glist[iproc].ilon
        #                     del self.glist[iproc].ilat
        #
        #                     if self.glist(iproc).ng > 0:
        #                         self.glist[iproc].ilon = xlist
        #                         self.glist[iproc].ilat = ylist
        #
        #                     if xlist is not None:
        #                         del xlist
        #                     if ylist is not None:
        #                         del ylist
        #
        #                 del msk
        #     if self.mpi.p_is_worker:
        #         for iset in range(pixelset.nset):
        #             msk = np.zeros(gfrom[iset].ng)
        #
        #             for ig in range(gfrom[iset].ng):
        #                 ilon = gfrom[iset].ilon[ig]
        #                 ilat = gfrom[iset].ilat[ig]
        #                 xblk = fgrid.xblk[ilon]
        #                 yblk = fgrid.yblk[ilat]
        #
        #                 iproc = 0
        #                 msk[ig] = CoLM_Utils.find_in_sorted_list2(ilon, ilat, self.glist[iproc].ng,
        #                                                           self.glist[iproc].ilon, self.glist[iproc].ilat) > 0
        #             if pfilter is not None:
        #                 pfilter[iset] = np.any(msk)
        #
        #             ng0 = gfrom[iset].ng
        #             gfrom[iset].ng = sum(msk)
        #             if np.any(msk) and np.any(not msk):
        #                 ng = gfrom[iset].ng
        #                 gfrom[iset].ilon[: ng] = np.compress(msk, gfrom[iset].ilon[: ng0])
        #                 gfrom[iset].ilat[: ng] = np.compress(msk, gfrom[iset].ilat[: ng0])
        #                 afrac[iset].val[: ng] = np.compress(msk, afrac[iset].val[: ng0])
        #
        #             del msk
        # if self.mpi.p_is_worker:
        #     self.gweight = []
        #     for i in range(pixelset.nset):
        #         self.gweight.append(Pointer())
        #     for iset in range(pixelset.nset):
        #         ng = gfrom[iset].ng
        #         if ng > 0:
        #             self.address[iset].val = np.zeros((2, ng), dtype=int)
        #             self.gweight[iset].val = np.zeros(ng, dtype=int)
        #             if sum(afrac[iset].val[:ng]) < 1.0e-12:
        #                 self.gweight[iset].val[:] = 1.0 / ng
        #             else:
        #                 self.gweight[iset].val[:] = afrac[iset].val[:ng] / sum(afrac[iset].val[:ng])
        #
        #             for ig in range(gfrom[iset].ng):
        #                 ilon = gfrom[iset].ilon[ig]
        #                 ilat = gfrom[iset].ilat[ig]
        #                 xblk = fgrid.xblk[ilon]
        #                 yblk = fgrid.yblk[ilat]
        #                 iproc = 0
        #                 self.address[iset].val[0, ig] = iproc
        #                 self.address[iset].val[1, ig] = CoLM_Utils.find_in_sorted_list2(
        #                     ilon, ilat, self.glist[iproc].ng, self.glist[iproc].ilon, self.glist[iproc].ilat)

            for iset in range(pixelset.nset):
                del afrac[iset].val
                del gfrom[iset].ilon
                del gfrom[iset].ilat

            del afrac
            del gfrom

        #     self.address = []
        #     for i in range(pixelset.nset):
        #         self.address.append(Pointer())
        #     self.gweight = []
        #     for i in range(pixelset.nset):
        #         self.gweight.append(Pointer())
        # 
        #     # Loop over pixelset%nset
        #     for iset in range(pixelset.nset):
        #         ng = gfrom[iset].ng
        #         self.address[iset].val = np.zeros((2, ng), dtype=int)
        #         self.gweight[iset].val = np.zeros(ng, dtype=int)
        # 
        #         self.gweight[iset].val = afrac[iset].val[:ng].copy()
        #         if pctpset is not None:
        #             self.gweight[iset].val *= pctpset[iset]
        # 
        #         for ig in range(gfrom[iset].ng):
        #             ilon = gfrom[iset].ilon[ig]
        #             ilat = gfrom[iset].ilat[ig]
        #             xblk = fgrid.xblk[ilon]
        #             yblk = fgrid.yblk[ilat]
        # 
        #             iproc = 0
        #             if self.nl_colm['USEMPI']:
        #                 pass
        # 
        #             self.address[iset].val[0, ig] = iproc
        #             self.address[iset].val[1, ig] = CoLM_Utils.find_in_sorted_list2(ilon, ilat, self.glist[iproc].ng,
        #                                                                      self.glist[iproc].ilon,
        #                                                                      self.glist[iproc].ilat)
        # 
        #     # Deallocate arrays
        #     del xlist
        #     del ylist
        # 
        #     for iset in range(pixelset.nset):
        #         del afrac[iset].val
        #         del gfrom[iset].ilon
        #         del gfrom[iset].ilat
        # 
        #     del afrac
        #     del gfrom
        # 
        #     if self.nl_colm['USEMPI']:
        #         pass
        # 
        # if self.nl_colm['USEMPI']:
        #     pass

    def map_aweighted_2d(self, gdata, pdata, spval):
        if self.mpi.p_is_io:
            for iproc in range(self.mpi.p_np_worker):
                # print(len(self.glist),'*******************')
                if self.glist[iproc].ng > 0:
                    gbuff = np.zeros(self.glist[iproc].ng)

                    for ig in range(self.glist[iproc].ng):
                        ilon = self.glist[iproc].ilon[ig]
                        ilat = self.glist[iproc].ilat[ig]
                        xblk = self.grid.xblk[ilon]
                        yblk = self.grid.yblk[ilat]
                        xloc = self.grid.xloc[ilon]
                        yloc = self.grid.yloc[ilat]

                        gbuff[ig] = gdata[xblk, yblk].val[xloc, yloc]

        if self.mpi.p_is_worker:
            pbuff = np.zeros(self.mpi.p_np_io)

            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    pbuff[iproc].val = np.zeros(self.glist[iproc].ng)
                    pbuff[0].val = gbuff
                    del gbuff

            for iset in range(self.npset):
                if self.gweight[iset].val is not None:
                    pdata[iset] = 0.0
                    for ig in range(len(self.gweight[iset].val)):
                        iproc = self.address[iset].val[0, ig]
                        iloc = self.address[iset].val[1, ig]
                        pdata[iset] += pbuff[iproc].val[iloc] * self.gweight[iset].val[ig]
                else:
                    pdata[iset] = spval

            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    del pbuff[iproc].val

        return pdata

    # def map_aweighted_3d(self):
    #     pass  # Define map_3d method

    # def map_split(self):
    #     pass  # Define map_split method

    def map(self, pdata, gdata, spv=None, msk=None):
        if self.mpi.p_is_worker:
            pbuff = []
            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    p = Pointer()
                    p.val = np.zeros(self.glist[iproc].ng)
                    pbuff.append(p)
                    if spv is not None:
                        pbuff[iproc].val[:] = spv
                    else:
                        pbuff[iproc].val[:] = 0.0

            for iset in range(self.npset):
                if spv is not None and np.all(pdata[iset] == spv):
                    continue
                if msk is not None and not msk[iset]:
                    continue

                for ig in range(len(self.olparea[iset].val)):
                    iproc = self.address[iset].val[0, ig]
                    iloc = self.address[iset].val[1, ig] - 1

                    if spv is not None:
                        if pbuff[iproc].val[iloc] != spv:
                            pbuff[iproc].val[iloc] += pdata[iset] * self.olparea[iset].val[ig]
                        else:
                            pbuff[iproc].val[iloc] = pdata[iset] * self.olparea[iset].val[ig]
                    else:
                        pbuff[iproc].val[iloc] += pdata[iset] * self.olparea[iset].val[ig]

        if self.mpi.p_is_io:
            if spv is not None:
                self.datatype.flush_block_data(gdata, spv)
            else:
                self.datatype.flush_block_data(gdata, 0.0)

            for iproc in range(self.mpi.p_np_worker):
                if self.glist[iproc].ng > 0:
                    gbuff = np.copy(pbuff[0].val)
                    for ig in range(self.glist[iproc].ng):
                        if spv is not None:
                            if gbuff[ig] != spv:
                                ilon = self.glist[iproc].ilon[ig]
                                ilat = self.glist[iproc].ilat[ig]
                                xblk = self.grid.xblk[ilon]
                                yblk = self.grid.yblk[ilat]
                                xloc = self.grid.xloc[ilon]
                                yloc = self.grid.yloc[ilat]

                                if gdata.blk[xblk, yblk].val[xloc, yloc] != spv:
                                    gdata.blk[xblk, yblk].val[xloc, yloc] += gbuff[ig]
                                else:
                                    gdata.blk[xblk, yblk].val[xloc, yloc] = gbuff[ig]
                        else:
                            ilon = self.glist[iproc].ilon[ig]
                            ilat = self.glist[iproc].ilat[ig]
                            xblk = self.grid.xblk[ilon]
                            yblk = self.grid.yblk[ilat]
                            xloc = self.grid.xloc[ilon]
                            yloc = self.grid.yloc[ilat]

                            gdata.blk[xblk, yblk].val[xloc, yloc] += gbuff[ig]

        if self.mpi.p_is_worker:
            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    del pbuff[iproc].val

            del pbuff

    def map3d(self, pdata, gdata, spv=None, msk=None):
        if self.mpi.p_is_worker:
            pbuff = []
            for iproc in range(self.mpi.p_np_io):
                p = Pointer()
                p.val = np.zeros(self.glist[iproc].ng)
                pbuff.append(p)

            ub1 = pdata.shape[0]

            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    # 模拟分配pbuff中每个元素对应的三维数组，这里假设相应的类和属性已合理定义好来模拟Fortran中的结构
                    pbuff[iproc] = Pointer()

                    if spv is not None:
                        pbuff[iproc].val = np.full((ub1, self.glist[iproc].ng), spv)
                    else:
                        pbuff[iproc].val = np.zeros((ub1, self.glist[iproc].ng))

            for iset in range(self.npset):
                if msk is not None and not msk[iset]:  # 注意Python索引从0开始，所以这里减1
                    continue
                for ig in range(len(self.olparea[iset].val)):  # 同样索引减1
                    iproc = self.address[iset].val[0, ig]
                    iloc = self.address[iset].val[1, ig] - 1
                    for i1 in range(ub1):
                        if spv is not None:
                            if pdata[i1, iset] != spv:
                                if pbuff[iproc].val[i1, iloc] != spv:
                                    pbuff[iproc].val[i1, iloc] += pdata[i1, iset] * \
                                                                      self.olparea[iset].val[ig]
                                else:
                                    pbuff[iproc].val[i1, iloc] = pdata[i1, iset] * \
                                                                     self.olparea[iset].val[ig]
                        else:
                            pbuff[iproc].val[i1, iloc] += pdata[i1, iset] * self.olparea[iset].val[ig]

        # 判断是否是I/O进程（模拟Fortran中的p_is_io，同样需根据实际确定判断逻辑）
        if self.mpi.p_is_io:
            ub1 = pdata.shape[0]
            ub2 = pdata.shape[1]

            if spv is not None:
                self.datatype.flush_block_data(gdata, spv)
            else:
                self.datatype.flush_block_data(gdata, 0.0)

            for iproc in range(self.mpi.p_np_worker):
                if self.glist[iproc].ng > 0:
                    gbuff = pbuff[0].val
                    for ig in range(self.glist[iproc].ng):
                        ilon = self.glist[iproc].ilon[ig]
                        ilat = self.glist[iproc].ilat[ig]
                        xblk = self.grid.xblk[ilon]
                        yblk = self.grid.yblk[ilat]
                        xloc = self.grid.xloc[ilon]
                        yloc = self.grid.yloc[ilat]
                        for i1 in range(ub1):
                                if spv is not None:
                                    if gbuff[i1, ig] != spv:
                                        if gdata.blk[xblk,yblk].val[i1, xloc, yloc] != spv:
                                            gdata.blk[xblk,yblk].val[i1, xloc, yloc] += gbuff[i1, ig]
                                        else:
                                            gdata.blk[xblk,yblk].val[i1, xloc, yloc] = gbuff[i1, ig]
                                else:
                                    gdata.blk[xblk,yblk].val[i1, xloc, yloc] += gbuff[i1, ig]
                    del gbuff

        if self.mpi.p_is_worker:
            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    del pbuff[iproc].val
            del pbuff

    def map4d(self, pdata, gdata, spv=None, msk=None):
        if self.mpi.p_is_worker:
            pbuff = []
            for iproc in range(self.mpi.p_np_io):
                p = Pointer()
                p.val = np.zeros(self.glist[iproc].ng)
                pbuff.append(p)

            ub1 = pdata.shape[0]
            ub2 = pdata.shape[1]

            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    # 模拟分配pbuff中每个元素对应的三维数组，这里假设相应的类和属性已合理定义好来模拟Fortran中的结构
                    pbuff[iproc] = Pointer()

                    if spv is not None:
                        pbuff[iproc].val = np.full((ub1,ub2, self.glist[iproc].ng), spv)
                    else:
                        pbuff[iproc].val = np.zeros((ub1,ub2, self.glist[iproc].ng))

            for iset in range(self.npset):
                if msk is not None and not msk[iset]:  # 注意Python索引从0开始，所以这里减1
                    continue
                for ig in range(len(self.olparea[iset].val)):  # 同样索引减1
                    iproc = self.address[iset].val[0, ig]
                    iloc = self.address[iset].val[1, ig] - 1
                    for i1 in range(ub1):
                        for i2 in range(ub2):
                            if spv is not None:
                                if pdata[i1, i2, iset] != spv:
                                    if pbuff[iproc].val[i1, i2, iloc] != spv:
                                        pbuff[iproc].val[i1, i2, iloc] += pdata[i1, i2, iset] * \
                                                                          self.olparea[iset].val[ig]
                                    else:
                                        pbuff[iproc].val[i1, i2, iloc] = pdata[i1, i2, iset] * \
                                                                         self.olparea[iset].val[ig]
                            else:
                                pbuff[iproc].val[i1, i2, iloc] += pdata[i1, i2, iset] * self.olparea[iset].val[ig]

        # 判断是否是I/O进程（模拟Fortran中的p_is_io，同样需根据实际确定判断逻辑）
        if self.mpi.p_is_io:
            ub1 = pdata.shape[0]
            ub2 = pdata.shape[1]

            if spv is not None:
                self.datatype.flush_block_data(gdata, spv)
            else:
                self.datatype.flush_block_data(gdata, 0.0)

            for iproc in range(self.mpi.p_np_worker):
                if self.glist[iproc].ng > 0:
                    gbuff = pbuff[0].val
                    for ig in range(self.glist[iproc].ng):
                        ilon = self.glist[iproc].ilon[ig]
                        ilat = self.glist[iproc].ilat[ig]
                        xblk = self.grid.xblk[ilon]
                        yblk = self.grid.yblk[ilat]
                        xloc = self.grid.xloc[ilon]
                        yloc = self.grid.yloc[ilat]
                        for i1 in range(ub1):
                            for i2 in range(ub2):
                                if spv is not None:
                                    if gbuff[i1, i2, ig] != spv:
                                        if gdata.blk[xblk,yblk].val[i1, i2, xloc, yloc] != spv:
                                            gdata.blk[xblk,yblk].val[i1, i2, xloc, yloc] += gbuff[i1, i2, ig]
                                        else:
                                            gdata.blk[xblk,yblk].val[i1, i2, xloc, yloc] = gbuff[i1, i2, ig]
                                else:
                                    gdata.blk[xblk,yblk].val[i1, i2, xloc, yloc] += gbuff[i1, i2, ig]
                    del gbuff

        if self.mpi.p_is_worker:
            for iproc in range(self.mpi.p_np_io):
                if self.glist[iproc].ng > 0:
                    del pbuff[iproc].val
            del pbuff
