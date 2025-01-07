from CoLM_DataType import DataType

class CoLM_Vars_2DForcing:
    def __init__(self, mpi, gblock):
        self.mpi = mpi
        self.datatype = DataType(gblock)
        self.forc_xy_pco2m = None
        self.forc_xy_po2m = None
        self.forc_xy_us = None
        self.forc_xy_vs = None
        self.forc_xy_t = None
        self.forc_xy_q = None
        self.forc_xy_prc = None
        self.forc_xy_prl = None
        self.forc_xy_psrf = None
        self.forc_xy_pbot = None
        self.forc_xy_sols = None
        self.forc_xy_soll = None
        self.forc_xy_solsd = None
        self.forc_xy_solld = None
        self.forc_xy_frl = None
        self.forc_xy_hgt_u = None
        self.forc_xy_hgt_t = None
        self.forc_xy_hgt_q = None
        self.forc_xy_rhoair = None
        self.forc_xy_hpbl = None

    def allocate_2D_Forcing(self, grid):
        if self.mpi.p_is_io:
            self.forc_xy_pco2m = self.datatype.allocate_block_data(grid)
            self.forc_xy_po2m = self.datatype.allocate_block_data(grid)
            self.forc_xy_us = self.datatype.allocate_block_data(grid)
            self.forc_xy_vs = self.datatype.allocate_block_data(grid)
            self.forc_xy_t = self.datatype.allocate_block_data(grid)
            self.forc_xy_q = self.datatype.allocate_block_data(grid)
            self.forc_xy_prc = self.datatype.allocate_block_data(grid)
            self.forc_xy_prl = self.datatype.allocate_block_data(grid)
            self.forc_xy_psrf = self.datatype.allocate_block_data(grid)
            self.forc_xy_pbot = self.datatype.allocate_block_data(grid)
            self.forc_xy_sols = self.datatype.allocate_block_data(grid)
            self.forc_xy_soll = self.datatype.allocate_block_data(grid)
            self.forc_xy_solsd = self.datatype.allocate_block_data(grid)
            self.forc_xy_solld = self.datatype.allocate_block_data(grid)
            self.forc_xy_frl = self.datatype.allocate_block_data(grid)
            self.forc_xy_hgt_u = self.datatype.allocate_block_data(grid)
            self.forc_xy_hgt_t = self.datatype.allocate_block_data(grid)
            self.forc_xy_hgt_q = self.datatype.allocate_block_data(grid)
            self.forc_xy_rhoair = self.datatype.allocate_block_data(grid)
            self.forc_xy_hpbl = self.datatype.allocate_block_data(grid)