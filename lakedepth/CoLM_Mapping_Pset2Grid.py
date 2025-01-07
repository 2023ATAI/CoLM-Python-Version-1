#----------------------------------------------------------------------------
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
#----------------------------------------------------------------------------
from CoLM_DataType import Pointer


class MappingPset2Grid:
    def __init__(self, grid, npset):
        self.grid = grid
        self.npset = npset
        self.glist = []  # Assuming grid_list_type is a list
        self.address = Pointer()
        self.olparea = Pointer()

    def build(self):
        pass  # Define build method

    def map_2d(self):
        pass  # Define map_2d method

    def map_3d(self):
        pass  # Define map_3d method

    def map_4d(self):
        pass  # Define map_4d method

    def map_split(self):
        pass  # Define map_split method

    def mapping_pset2grid_free_mem(self):
        pass  # Define mapping_pset2grid_free_mem method



class CoLM_Mapping_Pset2Grid(object):
    def __init__(self) -> None:
        pass