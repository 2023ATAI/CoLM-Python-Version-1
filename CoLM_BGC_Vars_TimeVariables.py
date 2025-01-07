import CoLM_NetCDFVectorBlk
class CoLM_BGC_Vars_TimeVariables (object):
    def __init__(self, nl_colm, landpatch, nl_soil) -> None:
        self.nl_colm = nl_colm
        self.landpatch = landpatch
        self.tCONC_O2_UNSAT = None
        self.tO2_DECOMP_DEPTH_UNSAT = None
        self.nl_soil = nl_soil

    def READ_BGCTimeVariables(self, file_restart):
        if self.nl_colm['DEF_USE_NITRIF']:
            self.tCONC_O2_UNSAT = CoLM_NetCDFVectorBlk.ncio_read_vector (file_restart, 'tCONC_O2_UNSAT       ',   self.nl_soil, self.landpatch, self.tCONC_O2_UNSAT         )
            self.tO2_DECOMP_DEPTH_UNSAT = CoLM_NetCDFVectorBlk.ncio_read_vector (file_restart, 'tO2_DECOMP_DEPTH_UNSAT',  self.nl_soil, self.landpatch, self.tO2_DECOMP_DEPTH_UNSAT )
