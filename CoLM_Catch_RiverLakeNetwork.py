import numpy as np
import CoLM_Utils
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_Grid import Grid_type
from CoLM_DataType import DataType
from CoLM_Mapping_Grid2Pset import MappingGrid2PSet
from CoLM_ElementNeighbour import CoLM_ElementNeighbour

SEND_DATA_DOWN_TO_UP = 1
SEND_DATA_UP_TO_DOWN = 2

class CoLM_Catch_RiverLakeNetwork(object):
    def __init__(self, nl_colm, mpi, mesh, gblock, pixel, landpatch, catch_riverlakenetwork, vti, numelm, spval) -> None:
        # Local Variables
        self.nl_colm = nl_colm
        self.gblock = gblock
        self.pixel = pixel
        self.mpi = mpi
        elementneighbour = CoLM_ElementNeighbour()

        self.addrdown = None
        river_file = nl_colm["DEF_CatchmentMesh_data"]
        use_calc_rivdpt = nl_colm["DEF_USE_EstimatedRiverDepth"]

        numbasin = numelm

        # Placeholder values for data that would be read from files
        # self.lake_id = np.array([])  # Placeholder, replace with actual data read
        # self.riverdown = np.array([])
        # self.riverlen = np.array([])
        # self.riverelv = np.array([])
        # self.basinelv = np.array([])
        # self.riverdpth = np.array([])

        # numbasin = numelm = 0  # Placeholder, replace with actual number of basins
        # bindex = np.array([])

        netfile = NetCDFFile(nl_colm['USEMPI'])

        if mpi.p_is_master:
            # Placeholder: Read data from files
            self.lake_id = netfile.ncio_read_serial(river_file, 'self.lake_id')
            self.riverdown = netfile.ncio_read_serial(river_file, 'basin_downstream')
            self.riverlen = netfile.ncio_read_serial(river_file, 'river_length')
            self.riverelv = netfile.ncio_read_serial(river_file, 'river_elevation')
            self.basinelv = netfile.ncio_read_serial(river_file, 'basin_elevation')

            if not use_calc_rivdpt:
                self.riverdpth = netfile.ncio_read_serial(river_file, 'river_depth')

            self.riverlen *= 1000.0  # Convert from km to m

            nbasin = len(self.riverdown)
            self.to_lake = np.full(nbasin, False, dtype=bool)

            for i in range(nbasin):
                if self.riverdown[i] > 0:
                    self.to_lake[i] = self.lake_id[self.riverdown[i]] > 0

        if use_calc_rivdpt:
            self.calc_riverdepth_from_runoff()

        if mpi.p_is_worker:
            if numbasin > 0:
                bindex = np.zeros(numbasin, dtype=int)
                for ibasin in range(numbasin):
                    bindex[ibasin] = mesh[ibasin].indx

            if numbasin > 0:
                self.lake_id = self.lake_id[bindex]
                self.riverdown = self.riverdown[bindex]
                self.to_lake = self.to_lake[bindex]
                self.riverlen = self.riverlen[bindex]
                self.riverelv = self.riverelv[bindex]
                self.riverdpth = self.riverdpth[bindex]
                self.basinelv = self.basinelv[bindex]

        if mpi.p_is_worker:
            if numbasin > 0:
                basin_sorted = bindex.copy()
                order = np.arange(1, numbasin + 1)

                basin_sorted, order = CoLM_Utils.quicksort(numbasin, basin_sorted, order)

                self.addrdown = np.full(numbasin, -1, dtype=int)

                for ibasin in range(numbasin):
                    if self.riverdown[ibasin] > 0:
                        iloc = CoLM_Utils.find_in_sorted_list1(self.riverdown[ibasin], numbasin, basin_sorted)
                        if iloc > 0:
                            self.addrdown[ibasin] = order[iloc]

        if mpi.p_is_worker:
            if numbasin > 0:
                # Allocate numpy arrays for calculations
                self.lakes = [None] * numbasin
                self.riverarea = np.zeros(numbasin, dtype=float)
                self.riverwth = np.zeros(numbasin, dtype=float)
                self.bedelv = np.zeros(numbasin, dtype=float)
                self.handmin = np.zeros(numbasin, dtype=float)
                self.wtsrfelv = np.zeros(numbasin, dtype=float)
                self.riverlen_ds = np.zeros(numbasin, dtype=float)
                self.wtsrfelv_ds = np.zeros(numbasin, dtype=float)
                self.riverwth_ds = np.zeros(numbasin, dtype=float)
                self.bedelv_ds = np.zeros(numbasin, dtype=float)
                self.outletwth = np.zeros(numbasin, dtype=float)

                for ibasin in range(numbasin):
                    if self.lake_id[ibasin] == 0:
                        self.riverarea[ibasin] = catch_riverlakenetwork.hillslope_network[ibasin].area[0]
                        self.riverwth[ibasin] = self.riverarea[ibasin] / self.riverlen[ibasin]

                        # Modify height above nearest drainage data to consider river depth
                        if catch_riverlakenetwork.hillslope_network[ibasin].nhru > 1:
                            catch_riverlakenetwork.hillslope_network[ibasin].hand[1:] += self.riverdpth[ibasin]

                        self.wtsrfelv[ibasin] = self.riverelv[ibasin]
                        self.bedelv[ibasin] = self.riverelv[ibasin] - self.riverdpth[ibasin]

                    elif self.lake_id[ibasin] > 0:
                        self.wtsrfelv[ibasin] = self.basinelv[ibasin]

                        ps = landpatch.elm_patch.substt[ibasin]
                        pe = landpatch.elm_patch.subend[ibasin]

                        self.bedelv[ibasin] = self.basinelv[ibasin] - np.max(vti.lakedepth[ps:pe])

                        nsublake = pe - ps + 1

                        self.lakes[ibasin].nsub = nsublake
                        self.lakes[ibasin].area0 = np.zeros(nsublake, dtype=float)
                        self.lakes[ibasin].area = np.zeros(nsublake, dtype=float)
                        self.lakes[ibasin].depth0 = np.zeros(nsublake, dtype=float)
                        self.lakes[ibasin].depth = np.zeros(nsublake, dtype=float)
                        
                        for i in range(nsublake):
                            ipatch = i + ps - 1
                            self.lakes[ibasin].area[i] = 0
                            for ipxl in range(landpatch.ipxstt[ipatch], landpatch.ipxend[ipatch] + 1):
                                self.lakes[ibasin].area[i] += (
                                    1.0e6 * CoLM_Utils.areaquad(
                                        self.pixel.lat_s[mesh[ibasin].ilat[ipxl]],
                                        self.pixel.lat_n[mesh[ibasin].ilat[ipxl]],
                                        self.pixel.lon_w[mesh[ibasin].ilat[ipxl]],
                                        self.pixel.lon_e[mesh[ibasin].ilat[ipxl]]
                                    )
                                )

                        # Area data in HRU order
                        self.lakes[ibasin].area0 = self.lakes[ibasin].area

                        self.lakes[ibasin].depth = vti.lakedepth[ps:pe]
                        # Depth data in HRU order
                        self.lakes[ibasin].depth0 = self.lakes[ibasin].depth

                        order = np.arange(1, nsublake + 1)

                        self.lakes[ibasin].depth, order = CoLM_Utils.quicksort(nsublake, self.lakes[ibasin].depth, order)

                        # Area data in depth order
                        self.lakes[ibasin].area = self.lakes[ibasin].area[order - 1]

                        # Adjust to be from deepest to shallowest
                        self.lakes[ibasin].depth = self.lakes[ibasin].depth[nsublake - 1::-1]
                        self.lakes[ibasin].area = self.lakes[ibasin].area[nsublake - 1::-1]

                        self.lakes[ibasin].dep_vol_curve = np.zeros(nsublake, dtype=int)

                        self.lakes[ibasin].dep_vol_curve[0] = 0
                        for i in range(1, nsublake):
                            self.lakes[ibasin].dep_vol_curve[i] = self.lakes[ibasin].dep_vol_curve[i - 1] + np.sum(
                                self.lakes[ibasin].area[:i - 1]) * (
                                self.lakes[ibasin].depth[i - 1] - self.lakes[ibasin].depth[i]
                            )

                        self.riverlen[ibasin] = 0.0

                        del order

                    if self.lake_id[ibasin] <= 0:
                        self.handmin[ibasin] = np.min(catch_riverlakenetwork.hillslope_network[ibasin].hand)

            for ibasin in range(numbasin):
                if self.addrdown[ibasin] > 0:
                    self.riverlen_ds[ibasin] = self.riverlen[self.addrdown[ibasin]]
                    self.wtsrfelv_ds[ibasin] = self.wtsrfelv[self.addrdown[ibasin]]
                    self.riverwth_ds[ibasin] = self.riverwth[self.addrdown[ibasin]]
                    self.bedelv_ds[ibasin] = self.bedelv[self.addrdown[ibasin]]
                else:
                    self.riverlen_ds[ibasin] = spval
                    self.wtsrfelv_ds[ibasin] = spval
                    self.riverwth_ds[ibasin] = spval
                    self.bedelv_ds[ibasin] = spval

            for ibasin in range(numbasin):
                if self.lake_id[ibasin] < 0:
                    self.bedelv[ibasin] = self.wtsrfelv_ds[ibasin] + np.min(catch_riverlakenetwork.hillslope_network[ibasin].hand)

            for ibasin in range(numbasin):
                if self.lake_id[ibasin] == 0:
                    if self.to_lake[ibasin] or self.riverlen[ibasin] <= 0:
                        # River to lake, ocean, inland depression, or out of region
                        self.outletwth[ibasin] = self.riverwth[ibasin]
                    else:
                        # River to river
                        self.outletwth[ibasin] = (self.riverwth[ibasin] + self.riverwth_ds[ibasin]) * 0.5
                elif self.lake_id[ibasin] != 0:
                    if not self.to_lake[ibasin] and self.riverlen[ibasin] != 0:
                        if self.riverlen[ibasin] > 0:
                            # Lake to river
                            self.outletwth[ibasin] = self.riverwth_ds[ibasin]
                        elif self.riverlen[ibasin] == -1:
                            # Lake is inland depression
                            self.outletwth[ibasin] = 0
                    elif self.to_lake[ibasin] or self.riverlen[ibasin] == 0:
                        # Lake to lake or lake catchment to lake or lake to ocean
                        if self.riverlen[ibasin] > 0:
                            # inb = findloc(elementneighbour[ibasin].glbindex, self.riverdown[ibasin], dim=1)
                            inb = np.where(elementneighbour[ibasin].glbindex == self.riverdown[ibasin])[0]
                        else:
                            # inb = findloc(elementneighbour[ibasin].glbindex, -9, dim=1)
                            inb = np.where(elementneighbour[ibasin].glbindex == -9)[0]

                        if inb <= 0:
                            self.outletwth(ibasin) = 0
                        else:
                            self.outletwth(ibasin) = elementneighbour[ibasin].lenbdr[inb]

    def calc_riverdepth_from_runoff(self, landelm, numelm, totalnumelm, elementneighbour, riverdown, elm_data_address, p_itis_worker, p_np_worker, p_is_worker, p_is_master, p_is_io, p_comm_glb, p_root, mpi_tag_mesg, mpi_tag_data, p_err, gblock):
        # Local Variables
        cH_rivdpt = 0.1
        pH_rivdpt = 0.5
        B0_rivdpt = 0.0
        Bmin_rivdpt = 1.0

        file_rnof = f"{self.nl_colm['DEF_dir_runtime']}/runoff_clim.nc"
        
        # Initialize grid and block data structures
        grid_rnof = Grid_type(self.nl_colm, self.gblock)
        grid_rnof.define_from_file(file_rnof, 'lat', 'lon')
        f_rnof = DataType(self.gblock)
        mg2p_rnof = MappingGrid2PSet(self.nl_colm, self.gblock, self.mpi)
        mg2p_rnof.build(grid_rnof, landelm)

        # Initialize MPI-related variables
        if self.mpi.p_is_io:
            f_rnof.allocate_block_data(grid_rnof)
            f_rnof.ncio_read_block(file_rnof, 'ro', grid_rnof)
            for iblkme in range(gblock.nblkme):
                ib = gblock.xblkme[iblkme]
                jb = gblock.yblkme[iblkme]
                for j in range(grid_rnof.ycnt[jb]):
                    for i in range(grid_rnof.xcnt[ib]):
                        f_rnof.blk[ib, jb].val[i, j] = max(f_rnof.blk[ib, jb].val[i, j], 0.0)
        
        if self.mpi.p_is_worker:
            if numelm > 0:
                bsnrnof = np.zeros(numelm)

        mg2p_rnof.map_aweighted(f_rnof, bsnrnof)

        if self.mpi.p_is_worker:
            if numelm > 0:
                bsnrnof /= 24.0 * 3600.0  # from m/day to m/s
                for i in range(numelm):
                    # total runoff in basin, from m/s to m3/s
                    bsnrnof[i] *= elementneighbour[i].myarea

        # MPI operations
        # if self.nl_colm['USEMPI']:
        #     mpi_barrier(p_comm_glb, p_err)

        #     if self.mpi.p_is_worker:
        #         mesg[:] = [p_iam_glb, numelm]
        #         mpi_send(mesg, 2, MPI.INTEGER, p_root, mpi_tag_mesg, p_comm_glb, p_err)
        #         if numelm > 0:
        #             mpi_send(bsnrnof, numelm, MPI.REAL8, p_root, mpi_tag_data, p_comm_glb, p_err)

        #     if self.mpi.p_is_master:
        #         bsnrnof = np.zeros(totalnumelm)
        #         for _ in range(p_np_worker):
        #             mpi_recv(mesg, 2, MPI.INTEGER, MPI.ANY_SOURCE, mpi_tag_mesg, p_comm_glb, p_stat, p_err)
        #             isrc = mesg[0]
        #             ndata = mesg[1]
        #             if ndata > 0:
        #                 rcache = np.zeros(ndata)
        #                 mpi_recv(rcache, ndata, MPI.REAL8, isrc, mpi_tag_data, p_comm_glb, p_stat, p_err)
        #                 bsnrnof[elm_data_address[p_itis_worker[isrc]].val] = rcache
        # else:
        bsnrnof[elm_data_address[0].val] = bsnrnof

        if self.mpi.p_is_master:
            nups_riv = np.zeros(totalnumelm, dtype=int)
            iups_riv = np.zeros(totalnumelm, dtype=int)
            b_up2down = np.zeros(totalnumelm, dtype=int)
            bsndis = np.zeros(totalnumelm)

            nups_riv[:] = 0

            for i in range(totalnumelm):
                j = self.riverdown[i]
                if j > 0:
                    nups_riv[j] += 1

            ithis = 0
            iups_riv[:] = 0
            for i in range(totalnumelm):
                if iups_riv[i] == nups_riv[i]:
                    ithis += 1
                    b_up2down[ithis] = i
                    j = self.riverdown[i]
                    if j > 0:
                        iups_riv[j] += 1
                        while iups_riv[j] == nups_riv[j]:
                            if j < i:
                                ithis += 1
                                b_up2down[ithis] = j
                            j = self.riverdown[j]
                            if j > 0:
                                iups_riv[j] += 1
                            else:
                                break
                else:
                    continue

            bsndis[:] = 0

            for i in range(totalnumelm):
                j = b_up2down[i]
                bsndis[j] += bsnrnof[j]
                if self.riverdown[j] > 0:
                    bsndis[self.riverdown[j]] += bsndis[j]

            self.riverdpth = np.zeros(totalnumelm)
            for i in range(totalnumelm):
                self.riverdpth[i] = max(cH_rivdpt * (bsndis[i] ** pH_rivdpt) + B0_rivdpt, Bmin_rivdpt)

        # Deallocate arrays
        if bsnrnof is not None:
            del bsnrnof
        if bsndis is not None:
            del bsndis
        if nups_riv is not None:
            del nups_riv
        if iups_riv is not None:
            del iups_riv
        if b_up2down is not None:
            del b_up2down

    def river_lake_network_final(self):
        # Local Variables
        ilake = 0

        if self.lake_id  is not None: del self.lake_id  
        if self.riverlen is not None: del self.riverlen 
        if self.riverelv is not None: del self.riverelv 
        if self.riverarea is not None: del self.riverarea
        if self.riverwth is not None: del self.riverwth 
        if self.riverdpth is not None: del self.riverdpth
        if self.basinelv is not None: del self.basinelv 
        if self.bedelv   is not None: del self.bedelv   
        if self.handmin  is not None: del self.handmin  
        if self.wtsrfelv is not None: del self.wtsrfelv 
        if self.riverdown is not None: del self.riverdown
        if self.addrdown is not None: del self.addrdown 
        if self.to_lake  is not None: del self.to_lake  

        if self.riverlen_ds is not None:  del self.riverlen_ds
        if self.wtsrfelv_ds is not None:  del self.wtsrfelv_ds
        if self.riverwth_ds is not None:  del self.riverwth_ds
        if self.bedelv_ds  is not None:  del self.bedelv_ds  
        if self.outletwth  is not None:  del self.outletwth  

        if self.lakes is not None:
            for ilake in range(self.lakes):
                if self.lakes[ilake].area0        is not None: del self.lakes[ilake].area0        
                if self.lakes[ilake].area         is not None: del self.lakes[ilake].area         
                if self.lakes[ilake].depth0       is not None: del self.lakes[ilake].depth0       
                if self.lakes[ilake].depth        is not None: del self.lakes[ilake].depth        
                if self.lakes[ilake].dep_vol_curve is not None: del self.lakes[ilake].dep_vol_curve

            del self.lakes