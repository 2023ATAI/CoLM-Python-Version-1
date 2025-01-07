import copy

import numpy as np


class Pointer:
    def __init__(self):
        # self.val = np.full((row,colm), val, dtype='float')
        self.val = None

    def pointer_int32_2d_free_mem(self):
        if self.val is not None:
            del self.val


class BlockData:
    def __init__(self, row, colm):
        self.blk = []
        for i in range(row):
            temp = []
            for j in range(colm):
                temp.append((Pointer()))
            self.blk.append(temp)
        self.blk = np.array(self.blk)
        # self.blk = np.array([[Pointer()] * colm for i in range(row)])
        # self.blk = np.array([[None] * colm for i in range(row)])

    def block_data_int32_2d_free_mem(self):
        if self.blk is not None:
            del self.blk

class BlockData3:
    def __init__(self, row, colm):
        self.blk = []
        self.lb1 = 0
        self.ub1 = 0
        for i in range(row):
            temp = []
            for j in range(colm):
                temp.append((Pointer()))
            self.blk.append(temp)
        self.blk = np.array(self.blk)

    def block_data_int32_2d_free_mem(self):
        if self.blk is not None:
            del self.blk

class BlockData4:
    def __init__(self, row, colm):
        self.lb1 = 0
        self.ub1 = 0
        self.lb2 = 0
        self.ub2 = 0
        self.blk = []
        for i in range(row):
            temp = []
            for j in range(colm):
                temp.append((Pointer()))
            self.blk.append(temp)
        self.blk = np.array(self.blk)

    def block_data_int32_2d_free_mem(self):
        if self.blk is not None:
            del self.blk


class DataType(object):
    def __init__(self, gblock) -> None:
        self.gblock = gblock

    def allocate_block_data(self, grid):
        gdata = BlockData(self.gblock.nxblk, self.gblock.nyblk)

        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]

            # self.gdata.blk[iblk, jblk] = Pointer(grid.xcnt[iblk], grid.ycnt[jblk])
            # if grid.xcnt[iblk]>0 and grid.ycnt[jblk]>0:
            #     pass
            # print(grid.xcnt[iblk], grid.ycnt[jblk], '--------------')
            # print(grid.xcnt[iblk], grid.ycnt[jblk], '--------shape------')

            # p = Pointer()
            # p.val = np.zeros((grid.xcnt[iblk], grid.ycnt[jblk]), dtype='float')
            gdata.blk[iblk, jblk].val = np.zeros((grid.xcnt[iblk], grid.ycnt[jblk]), dtype='float')

            # self.gdata.blk[iblk, jblk].val = np.zeros((grid.xcnt[iblk], grid.ycnt[jblk]), dtype='float')
        return gdata

    def allocate_block_data3(self, grid, ndim1, lb1=1):
        gdata = BlockData3(self.gblock.nxblk, self.gblock.nyblk)

        if lb1 is not None:
            gdata.lb1 = lb1
        else:
            gdata.lb1 = 1

        gdata.ub1 = gdata.lb1 - 1 + ndim1

        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            gdata.blk[iblk][jblk].val = np.zeros(
                (gdata.ub1 - gdata.lb1 + 1, grid.xcnt[iblk], grid.ycnt[jblk]))

        return gdata


    def allocate_block_data4(self, grid, ndim1, ndim2, lb1=1, lb2 =1):
        # Local variables
        gdata = BlockData4(self.gblock.nxblk, self.gblock.nyblk)

        if lb1 is not None:
            gdata.lb1 = lb1
        else:
            gdata.lb1 = 1

        gdata.ub1 = gdata.lb1-1+ndim1

        if lb2 is not None:
            gdata.lb2 = lb2
        else:
            gdata.lb2 = 1

        gdata.ub2 = gdata.lb2-1+ndim2

        # Allocate block data
        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]

            gdata.blk[iblk, jblk].val = np.zeros((gdata.ub1 - gdata.lb1+1,gdata.ub2 - gdata.lb2+1, grid.xcnt[iblk],grid.ycnt[jblk]))

        return gdata

    def block_data_copy(self, gdata_from, gdata_to, sca=None):
        # print(gdata_from.blk[24, 18].val.shape,gdata_to.blk[24, 18].val.shape,'+++++++++++++')
    # Local variables
        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            if sca is not None:
                gdata_to.blk[iblk, jblk].val = gdata_from.blk[iblk, jblk].val * sca
            else:
                gdata_to.blk[iblk, jblk].val = gdata_from.blk[iblk, jblk].val
        return gdata_to

    def block_data_linear_interp(self, gdata_from1, alp1, gdata_from2, alp2, gdata_to):
        # Local variables
        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            # if gdata_from1.blk[iblk, jblk].val.shape[0]>0:
                # print(iblkme, iblk, jblk,gdata_from1.blk[iblk, jblk].val.shape,'-------1-----')
            # if gdata_from2.blk[iblk, jblk].val.shape[0]>0:
                # print(iblkme, iblk, jblk,gdata_from2.blk[iblk, jblk].val.shape,'--------2----')
            gdata_to.blk[iblk, jblk].val = (
                gdata_from1.blk[iblk, jblk].val * alp1 +
                gdata_from2.blk[iblk, jblk].val * alp2
            )
        return gdata_to

    def allocate_block_data2d(self, grid, ndim1, lb1=None):
        gdata = BlockData(self.gblock.nxblk, self.gblock.nyblk)

        if lb1 is not None:
            gdata.lb1 = lb1
        else:
            gdata.lb1 = 1

        gdata.ub1 = gdata.lb1 - 1 + ndim1

        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            gdata.blk[iblk, jblk].val = np.zeros((grid.xcnt[iblk], grid.ycnt[jblk]))
        return gdata

    def flush_block_data(self, gdata, spval):
        # Local variables
        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            gdata.blk[iblk, jblk].val = np.full((gdata.blk[iblk,jblk].val.shape),spval)

        return gdata

    def block_data_linear_transform(self, gdata, scl=None, dsp=None):
        # Local variables
        if scl is not None:
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]
                gdata.blk[iblk, jblk].val = gdata.blk[iblk, jblk].val.astype(np.float64) * scl

        if dsp is not None:
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]
                gdata.blk[iblk, jblk].val += dsp
        return gdata
