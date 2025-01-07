# ----------------------------------------------------------------------
#  Aggregation Utilities.
#
# On IO processes, a data daemon is running to provide data
#        at fine resolutions for worker processes.
# On worker processes, request is sent to IO processes and
#        data is returned from IO processes.
#
# ----------------------------------------------------------------------
import numpy as np
import CoLM_Utils


class AggregationRequestData(object):
    def __init__(self, usempi, mpi, mesh, pixel) -> None:
        self.usempi = usempi
        self.mpi = mpi
        self.mesh = mesh
        self.pixel = pixel

    def aggregation_request_data(self, pixelset, iset, grid_in, zip, area=None,
                                 data_r8_2d_in1=None,
                                 data_r8_2d_in2=None,
                                 data_r8_2d_in3=None,
                                 data_r8_2d_in4=None,
                                 data_r8_2d_in5=None,
                                 data_r8_2d_in6=None,
                                 data_r8_3d_in1=None,
                                 n1_r8_3d_in1=None,
                                 lb1_r8_3d_in1=None,
                                 data_r8_3d_in2=None,
                                 n1_r8_3d_in2=None,
                                 lb1_r8_3d_in2=None,
                                 data_i4_2d_in1=None,
                                 data_i4_2d_in2=None,
                                 filledvalue_i4=None):
        ie = pixelset.ielm[iset]
        ipxstt = pixelset.ipxstt[iset]
        ipxend = pixelset.ipxend[iset]

        npxl = ipxend - ipxstt
        # print(ipxend, ipxstt, npxl, '----------------')
        # print(ipxstt,ipxend,npxl,iset,len(pixelset.ipxstt),'---------------')
        area2d = None
        #
        data_r8_2d_out1 = None
        data_r8_2d_out2 = None
        data_r8_2d_out3 = None
        data_r8_2d_out4 = None
        data_r8_2d_out5 = None
        data_r8_2d_out6 = None
        data_r8_3d_out1 = None
        data_r8_3d_out2 = None
        data_i4_2d_out1 = None
        data_i4_2d_out2 = None

        if zip:
            xsorted = np.zeros(npxl)
            ysorted = np.zeros(npxl)
            nx = 0
            ny = 0
            for ipxl in range(ipxstt, ipxend):
                if iset==27:

                    print('**************')
                xgrdthis = grid_in.xgrd[self.mesh[ie].ilon[ipxl]]
                ygrdthis = grid_in.ygrd[self.mesh[ie].ilat[ipxl]]
                iloc, _, nx, xsorted = CoLM_Utils.insert_into_sorted_list(xgrdthis, nx, xsorted)
                iloc, _, ny, ysorted = CoLM_Utils.insert_into_sorted_list(ygrdthis, ny, ysorted)

            xy2d = np.zeros((nx, ny))

            if area is not None:
                area2d = np.zeros((nx, ny))

            for ipxl in range(ipxstt, ipxend ):
                xgrdthis = grid_in.xgrd[self.mesh[ie].ilon[ipxl]]
                ygrdthis = grid_in.ygrd[self.mesh[ie].ilat[ipxl]]
                ix = CoLM_Utils.find_in_sorted_list(xgrdthis, nx, xsorted)
                iy = CoLM_Utils.find_in_sorted_list(ygrdthis, ny, ysorted)
                xy2d[ix, iy] += 1
                if area is not None:
                    area2d[ix, iy] += CoLM_Utils.areaquad(self.pixel.lat_s[self.mesh[ie].ilat[ipxl]],
                                                          self.pixel.lat_n[self.mesh[ie].ilat[ipxl]],
                                                          self.pixel.lon_w[self.mesh[ie].ilon[ipxl]],
                                                          self.pixel.lon_e[self.mesh[ie].ilon[ipxl]])

            totalreq = len(np.where(xy2d > 0)[0])
            # print(totalreq,'--totalreq------')
            xlist = np.zeros(totalreq)
            ylist = np.zeros(totalreq)
            if area is not None:
                area = np.zeros(totalreq)
            ig = 0
            for ix in range(nx):
                for iy in range(ny):
                    if xy2d[ix, iy] > 0:
                        xlist[ig] = xsorted[ix]
                        ylist[ig] = ysorted[iy]
                        if area is not None:
                            area[ig] = area2d[ix, iy]
                        ig += 1
            # 停止对 xsorted, ysorted, xy2d 的引用
            del xsorted, ysorted, xy2d

            # 如果 area 存在，则停止对 area2d 的引用
            if area2d is not None:
                del area2d
        else:
            xlist = np.zeros(npxl, dtype=int)
            ylist = np.zeros(npxl, dtype=int)
            if area2d is not None:
                area = np.zeros(npxl)

            totalreq = npxl
            # print(ipxstt, ipxend,'+++++++++++')
            # print(len(xlist), len(grid_in.xgrd[self.mesh[ie].ilon]), ipxend,'+++++++++++')
            # print(grid_in.ygrd)
            for ipxl in range(ipxstt, ipxend):
                # print(len(xlist), ipxl - ipxstt,len(grid_in.xgrd),len(self.mesh),ie,len(self.mesh[ie].ilon),ipxl,'-------------')

                xlist[ipxl - ipxstt] = grid_in.xgrd[self.mesh[ie].ilon[ipxl]]
                ylist[ipxl - ipxstt] = grid_in.ygrd[self.mesh[ie].ilat[ipxl]]
                if area2d is not None:
                    area[ipxl - ipxstt] = CoLM_Utils.areaquad(
                        self.pixel.lat_s[self.mesh[ie].ilat[ipxl]],
                        self.pixel.lat_n[self.mesh[ie].ilat[ipxl]],
                        self.pixel.lon_w[self.mesh[ie].ilon[ipxl]],
                        self.pixel.lon_e[self.mesh[ie].ilon[ipxl]]
                    )

        if data_r8_2d_in1 is not None:
            data_r8_2d_out1 = np.zeros(totalreq)
        if data_r8_2d_in2 is not None:
            data_r8_2d_out2 = np.zeros(totalreq)
        if data_r8_2d_in3 is not None:
            data_r8_2d_out3 = np.zeros(totalreq)
        if data_r8_2d_in4 is not None:
            data_r8_2d_out4 = np.zeros(totalreq)
        if data_r8_2d_in5 is not None:
            data_r8_2d_out5 = np.zeros(totalreq)
        if data_r8_2d_in6 is not None:
            data_r8_2d_out6 = np.zeros(totalreq)

        if data_r8_3d_in1 is not None and n1_r8_3d_in1 is not None:
            if lb1_r8_3d_in1 is not None:
                lb1 = lb1_r8_3d_in1
            else:
                lbl = 1
            data_r8_3d_out1 = np.zeros((n1_r8_3d_in1 - 1, totalreq))

        if data_r8_3d_in2 is not None and n1_r8_3d_in2 is not None:
            if lb1_r8_3d_in2 is not None:
                lb1 = lb1_r8_3d_in2
            else:
                lbl = 1
            data_r8_3d_out2 = np.zeros((n1_r8_3d_in2 - 1, totalreq))

        if data_i4_2d_in1 is not None:
            data_i4_2d_out1 = np.zeros(totalreq)
            if filledvalue_i4 is not None:
                data_i4_2d_out1[:] = filledvalue_i4

        if data_i4_2d_in2 is not None:
            data_i4_2d_out2 = np.zeros(totalreq)
            if filledvalue_i4 is not None:
                data_i4_2d_out2 = filledvalue_i4

        if self.usempi:
            pass
        else:
            # for x in xlist:
            #     print(x, '---------------')
            # print(xlist[0], xlist[totalreq-1], len(xlist), totalreq, '--------------')
            # return
            # print(totalreq,'*********************')
            for ireq in range(totalreq):
                xblk = int(grid_in.xblk[int(xlist[ireq])])
                yblk = int(grid_in.yblk[int(ylist[ireq])])
                xloc = int(grid_in.xloc[int(xlist[ireq])])
                yloc = int(grid_in.yloc[int(ylist[ireq])])
                # print(xblk, yblk, xloc, yloc, '----------------')

                if data_r8_2d_in1 is not None:
                    data_r8_2d_out1[ireq] = data_r8_2d_in1.blk[xblk][yblk].val[xloc, yloc]

                if data_r8_2d_in2 is not None:
                    data_r8_2d_out2[ireq] = data_r8_2d_in2.blk[xblk][yblk].val[xloc, yloc]

                if data_r8_2d_in3 is not None:
                    data_r8_2d_out3[ireq] = data_r8_2d_in3.blk[xblk][yblk].val[xloc, yloc]

                if data_r8_2d_in4 is not None:
                    data_r8_2d_out4[ireq] = data_r8_2d_in4.blk[xblk][yblk].val[xloc, yloc]

                if data_r8_2d_in5 is not None:
                    data_r8_2d_out5[ireq] = data_r8_2d_in5.blk[xblk][yblk].val[xloc, yloc]

                if data_r8_2d_in6 is not None:
                    data_r8_2d_out6[ireq] = data_r8_2d_in6.blk[xblk][yblk].val[xloc, yloc]

                if data_r8_3d_in1 is not None and n1_r8_3d_in1 is not None:
                    data_r8_3d_out1[:, ireq] = data_r8_3d_in1.blk[xblk][yblk].val[:, xloc, yloc]

                if data_r8_3d_in2 is not None and n1_r8_3d_in2 is not None:
                    data_r8_3d_out2[:, ireq] = data_r8_3d_in2.blk[xblk][yblk].val[:, xloc, yloc]

                if data_i4_2d_in1 is not None:
                    # if data_i4_2d_in1.blk[xblk,yblk].val is None:
                    #     print(ireq,data_i4_2d_out1.shape,data_i4_2d_in1.blk.shape,xblk, yblk, xloc, yloc, '+++++++++++++')
                    data_i4_2d_out1[ireq] = data_i4_2d_in1.blk[xblk,yblk].val[xloc, yloc]

                if data_i4_2d_in2 is not None:
                    data_i4_2d_out2[ireq] = data_i4_2d_in2.blk[xblk][yblk].val[xloc, yloc]

        return area, data_r8_2d_out1, data_r8_2d_out2, data_r8_2d_out3, data_r8_2d_out4, data_r8_2d_out5, \
            data_r8_2d_out6, data_r8_3d_out1, data_r8_3d_out2, data_i4_2d_out1, data_i4_2d_out2
