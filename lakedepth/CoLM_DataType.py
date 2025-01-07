import numpy as np

class Pointer:
    def __init__(self):
        self.val = None
     
    def pointer_int32_2d_free_mem(self):
        if self.val is not None:
            del self.val

class BlockData:
    def __init__(self, row, colm):
        self.blk = [[Pointer()] * colm for _ in range(row)]

    def block_data_int32_2d_free_mem(self):
        if self.blk is not None:
            del self.blk

class DataType(object):
    def __init__(self,gblock) -> None:
        self.gblock = gblock
    
    def allocate_block_data (self,grid):
        self.gdata = BlockData(self.gblock.nxblk, self.gblock.nyblk )

        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            self.gdata.blk[iblk-1,jblk-1].val = np.zeros((grid.xcnt[iblk-1], grid.ycnt[jblk-1]))
        return self.gdata
    
    def allocate_block_data2d (self,grid, ndim1, lb1 =None):
        self.gdata = BlockData(self.gblock.nxblk, self.gblock.nyblk )

        if lb1 is not None:
            self.gdata.lb1 = lb1
        else:
            self.gdata.lb1 = 1

        self.gdata.ub1 = self.gdata.lb1-1+ndim1

        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            self.gdata.blk[iblk-1,jblk-1].val = np.zeros((grid.xcnt[iblk-1], grid.ycnt[jblk-1]))
        return self.gdata
    
    def flush_block_data(self, spval):
        # Local variables
        for iblkme in range(self.gblock.nblkme):
            iblk = self.gblock.xblkme[iblkme]
            jblk = self.gblock.yblkme[iblkme]
            self.gdata[iblk-1][jblk-1] = spval
        return self.gdata
    
    def block_data_linear_transform(self, scl=None, dsp=None):
        # Local variables
        if scl is not None:
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]
                self.gdata.blk[iblk - 1][jblk - 1].val *= scl

        if dsp is not None:
            for iblkme in range(self.gblock.nblkme):
                iblk = self.gblock.xblkme[iblkme]
                jblk = self.gblock.yblkme[iblkme]
                self.gdata.blk[iblk - 1][jblk - 1].val += dsp
        return self.gdata
