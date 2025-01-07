import numpy as np
import CoLM_Utils
from CoLM_DataType import Pointer
from CoLM_NetCDFSerial import NetCDFFile
import CoLM_NetCDFVectorBlk

class element_neighbour_type(object):
    nnb = 0    # number of neighbours
    myarea = 0.0 # area of this element [m^2]
    myelva = 0.0 # elevation of this element [m]

    glbindex = None # neighbour global index

    addr = None

    dist    = None # distance between element centers [m]
    lenbdr  = None # length of boundary line [m]
    area    = None # area of neighbours [m^2]
    elva    = None # elevation of neighbours [m]
    slope   = None # slope (positive) [-]

class neighbour_sendrecv_type(object):
    ndata = 0
    glbindex  = None
    ielement  = None

class CoLM_ElementNeighbour(object):
    def __init__(self, lc_year, nl_colm, mpi, mesh, landelm, landpatch, pixel) -> None:
        self.numelm = mesh.numelm
        self.elementneighbour = None
        recvaddr = None
        sendaddr = None
        """
        Initialize element neighbours based on the land cover data of a given year.

        Parameters:
        lc_year (int): The year of the land cover data used.
        """

        # Local Variables
        neighbour_file = nl_colm['DEF_ElementNeighbour_file']

        # Declare variables
        nnball = None
        idxnball = None
        lenbdall = None
        maxnnb = None

        # Master process reads the neighbour file
        if mpi.p_is_master:
            netcdffile = NetCDFFile(nl_colm['USEMPI'])
            nnball = netcdffile.ncio_read_serial (neighbour_file, 'num_neighbour')
            idxnball = netcdffile.ncio_read_serial (neighbour_file, 'idx_neighbour')
            lenbdall = netcdffile.ncio_read_serial (neighbour_file, 'len_border')

            maxnnb = [idxnball,1]

        # Broadcast maxnnb to all workers

        # Worker processes allocate and initialize arrays
        if mpi.p_is_worker:
            if mesh.numelm > 0:
                eindex = landelm.eindex


            # Assuming nnball, idxnball, and lenbdall are distributed
            icache1[:] = nnball[:]
            icache2[:, :] = idxnball[:, :]
            rcache2[:, :] = lenbdall[:, :]

            for ielm in range(mesh.numelm):
                nnball[ielm] = icache1[eindex[ielm]]
                idxnball[:, ielm] = icache2[:, eindex[ielm]]
                lenbdall[:, ielm] = rcache2[:, eindex[ielm]]

            del icache1, icache2, rcache2

            if mesh.numelm > 0:
                self.elementneighbour = [element_neighbour_type() for _ in range(mesh.numelm)]

                for ielm in range(mesh.numelm):
                    nnb = nnball[ielm]
                    self.elementneighbour[ielm].nnb = nnb

                    if nnb > 0:
                        self.elementneighbour[ielm].glbindex = idxnball[:nnb, ielm]
                        self.elementneighbour[ielm].lenbdr = lenbdall[:nnb, ielm]
                        self.elementneighbour[ielm].addr[:] = -9999

        # Initialize sorted elements and order arrays
        if mpi.p_is_worker:
            if mesh.numelm > 0:
                elm_sorted = eindex
                order = [i for i in range(mesh.numelm)]

                # Sort the elements and get order
                elm_sorted, order = CoLM_Utils.quicksort (mesh.numelm, elm_sorted, order)

                nnbinq = 0
                for ielm in range(mesh.numelm):
                    for inb in range(self.elementneighbour[ielm].nnb):
                        if self.elementneighbour[ielm].glbindex[inb] <= 0:
                            continue  # Skip ocean neighbour

                        iloc = CoLM_Utils.find_in_sorted_list( self.elementneighbour[ielm].glbindex[inb], mesh.numelm, elm_sorted)
                        if iloc > 0:
                            self.elementneighbour[ielm].addr[0, inb] = -1
                            self.elementneighbour[ielm].addr[1, inb] = order[iloc]
            else:
                nnbinq = 0

        if addrelement is not  None:   del addrelement
        if elm_sorted  is not  None:   del elm_sorted 
        if nnball      is not  None:   del nnball     
        if idxnball    is not  None:   del idxnball   
        if lenbdall    is not  None:   del lenbdall   
        if eindex      is not  None:   del eindex     
        if icache1     is not  None:   del icache1    
        if icache2     is not  None:   del icache2    
        if rcache2     is not  None:   del rcache2    
        if order       is not  None:   del order      
        if idxinq      is not  None:   del idxinq     
        if addrinq     is not  None:   del addrinq    
        if mask        is not  None:   del mask       

        # Read topography patches data
        cyear = lc_year
        lndname = f"{nl_colm['DEF_dir_landdata']}/topography/{cyear}/topography_patches.nc"
        topo_patches = CoLM_NetCDFVectorBlk.ncio_read_vector (lndname, 'topography_patches', landpatch, topo_patches)

        # Worker process further processing
        if mpi.p_is_worker > 0:
            for ielm in range(mesh.numelm):
                nnb = self.elementneighbour[ielm].nnb
                if nnb > 0:
                    self.elementneighbour[ielm].dist = np.zeros(nnb, dtype=float)
                    self.elementneighbour[ielm].area = np.zeros(nnb, dtype=float)
                    self.elementneighbour[ielm].elva = np.zeros(nnb, dtype=float)
                    self.elementneighbour[ielm].slope = np.zeros(nnb, dtype=float)

            if mesh.numelm>0:
                rlon_b = np.zeros(mesh.numelm, dtype=float)
                rlat_b = np.zeros(mesh.numelm, dtype=float)
                landelm.get_lonlat_radian(rlon_b, rlat_b)

            rlon_nb = self.allocate_neighbour_data (rlon_nb)
            rlat_nb = self.allocate_neighbour_data (rlat_nb)

            rlon_b = self.retrieve_neighbour_data(rlon_b, rlon_nb)
            rlat_b = self.retrieve_neighbour_data(rlat_b, rlat_nb)

            for ielm in range(mesh.numelm):
                for inb in range(self.elementneighbour[ielm].nnb):
                    if self.elementneighbour[ielm].glbindex[inb] > 0:  # Skip ocean neighbour
                        self.elementneighbour[ielm].dist[inb] = 1.0e3 * CoLM_Utils.arclen(
                            rlat_b[ielm], rlon_b[ielm],
                            rlat_nb[ielm].val[inb], rlon_nb[ielm].val[inb]
                        )

            if mesh.numelm>0:
                area_b = np.zeros(mesh.numelm, dtype=float)
                elva_b = np.empty(mesh.numelm, dtype=float)

                for ielm in range(mesh.numelm):
                    area_b[ielm] = 0
                    for ipxl in range(mesh[ielm].npxl):
                        area_b[ielm] += 1.0e6 * CoLM_Utils.areaquad(
                            pixel.lat_s[mesh[ielm].ilat[ipxl]], pixel.lat_n[mesh[ielm].ilat[ipxl]],
                            pixel.lon_w[mesh[ielm].ilon[ipxl]], pixel.lon_e[mesh[ielm].ilon[ipxl]]
                        )

                    istt = landpatch.elm_patch.substt[ielm]
                    iend = landpatch.elm_patch.subend[ielm]
                    elva_b[ielm] = np.sum(topo_patches[istt:iend] * landpatch.elm_patch.subfrc[istt:iend])

                    self.elementneighbour[ielm].myarea = area_b[ielm]
                    self.elementneighbour[ielm].myelva = elva_b[ielm]

            area_nb = self.allocate_neighbour_data (area_nb)
            area_b = self.retrieve_neighbour_data(area_b, area_nb)

            elva_nb = self.allocate_neighbour_data (elva_nb)
            elva_b = self.retrieve_neighbour_data(elva_b, elva_nb)

            for ielm in range(mesh.numelm):
                for inb in range(self.elementneighbour[ielm].nnb):
                    if self.elementneighbour[ielm].glbindex[inb] > 0:  # Skip ocean neighbour
                        self.elementneighbour[ielm].area[inb] = area_nb[ielm].val[inb]
                        self.elementneighbour[ielm].elva[inb] = elva_nb[ielm].val[inb]
                        self.elementneighbour[ielm].slope[inb] = abs(elva_nb[ielm].val[inb] - elva_b[ielm]) / self.elementneighbour[ielm].dist[inb]

            if rlon_b  is not None: del rlon_b 
            if rlat_b  is not None: del rlat_b 
            if elva_b  is not None: del elva_b 
            if area_b  is not None: del area_b 
            if rlon_nb is not None: del rlon_nb
            if rlat_nb is not None: del rlat_nb
            if area_nb is not None: del area_nb
            if elva_nb is not None: del elva_nb

    def allocate_neighbour_data(self, nbdata):
        """
        Function to allocate memory for neighbour data.
        
        Parameters:
        numelm (int): Number of elements.
        self.elementneighbour (list of dicts): A list where each element is a dictionary representing 
                                        neighbours with key 'nnb' indicating the number of neighbours.
        
        Returns:
        nbdata (list): A list where each element is a NumPy array allocated according to neighbours.
        """
        
        nbdata = []

        if self.numelm > 0:
            nbdata = [Pointer()] * self.numelm
            for ielm in range(self.numelm):
                if self.elementneighbour[ielm].nnb > 0:
                    nbdata[ielm].val = np.zeros(self.elementneighbour[ielm].nnb)

        return nbdata
    
    def retrieve_neighbour_data(self, vec_in, nbdata):
        """
        Function to retrieve neighbour data and update nbdata based on vec_in.
        
        Parameters:
        vec_in (np.ndarray): Input vector of real numbers.
        nbdata (list): List where each element is an object with an attribute `val` that stores neighbour data.
        elementneighbour (list of dicts): List where each element is a dictionary representing neighbours
                                        with keys 'nnb' (number of neighbours) and 'addr' (address information).
        p_is_worker (bool): A flag indicating if the current process is a worker.
        numelm (int): Number of elements.
        
        Returns:
        None. (nbdata is updated in place)
        """

        if self.mpi.p_is_worker:
            for ielm in range(self.numelm):
                for inb in range(self.elementneighbour[ielm].nnb):
                    if self.elementneighbour[ielm].addr[0, inb] == -1:
                        iloc = self.elementneighbour[ielm].addr[1,inb]
                        nbdata[ielm].val[inb] = vec_in[iloc]
        
        return vec_in