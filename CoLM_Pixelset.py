# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Pixelset refers to a set of pixels in CoLM.
# 
#    In CoLM, the global/regional area is divided into a hierarchical structure:
#    1. If GRIDBASED or UNSTRUCTURED is defined, it is
#       ELEMENT >>> PATCH
#    2. If CATCHMENT is defined, it is
#       ELEMENT >>> HRU >>> PATCH
#    If Plant FUNCTION Type classification is used, PATCH is further divided into PFT.
#    If Plant Community classification is used,     PATCH is further divided into PC.
#
#    In CoLM, the land surface is first divided into pixels, which are rasterized 
#    points defined by fine-resolution data. Then ELEMENT, PATCH, HRU, PFT, PC 
#    are all consists of pixels, and hence they are all pixelsets.
# 
#    The highest level pixelset in CoLM is ELEMENT, all other pixelsets are subsets 
#    of ELEMENTs. 
#    In a pixelset, pixels are sorted to make pixels in its subsets consecutive.
#    Thus a subset can be represented by starting pixel index and ending pixel index
#    in an ELEMENT. 
#
#                Example of hierarchical pixelsets
#        ************************************************ <-- pixels in an ELEMENT
#        |<------------------- ELEMENT ---------------->| <-- level 1
#        |   subset 1  |       subset 2      | subset 3 | <-- level 2
#        |s11|   s12   | s21 |   s22   | s23 |    s31   | <-- level 3
#
#    "Vector" is a collection of data when each pixelset in a given level is associated
#    with a value, representing its averaged physical, chemical or biological state.
#
#    "Vector" is usually defined on worker process, while its IO is through IO process.
#    To read,  vector is first loaded from files by IO and then scattered from IO to worker.
#    To write, vector is first gathered from worker to IO and then saved to files by IO.
# ------------------------------------------------------------------------------------
import numpy as np
import CoLM_Utils
from typing import Optional, List


class Pixelset_type(object):
    def __init__(self, usempi, mpi, gblock, mesh) -> None:
        self.USEMPI = usempi
        self.mpi = mpi
        self.gblock = gblock
        self.mesh = mesh

        self.nblkgrp = 0

        self.nset = 0
        self.eindex = None
        self.vecgs = Vec_gather_scatter_type()
        self.ipxstt = None
        self.ipxend = None
        self.settyp = None
        self.xblkgrp = None
        self.yblkgrp = None
        self.vlenall = None
        self.xblkall = None
        self.yblkall = None
        self.ielm = None

    def set_vecgs(self):
        if self.USEMPI:
            # CALL mpi_barrier (p_comm_glb, p_err)
            pass

        if self.vecgs.vlen is None:
            self.vecgs.vlen = np.zeros((self.gblock.nxblk, self.gblock.nyblk), dtype=int)
            self.vecgs.vlen[:, :] = 0

        if self.mpi.p_is_worker:
            if self.vecgs.vstt is None:
                self.vecgs.vstt = np.zeros((self.gblock.nxblk, self.gblock.nyblk), dtype=int)
                self.vecgs.vend = np.zeros((self.gblock.nxblk, self.gblock.nyblk), dtype=int)

            self.vecgs.vstt[:, :] = 0
            self.vecgs.vend[:, :] = -1

            ie = 0
            xblk = 0
            yblk = 0
            for iset in range(self.nset):
                # print(iset, len(self.eindex), ie, len(self.mesh.mesh), '------pixelset--------')
                while self.eindex[iset] != self.mesh.mesh[ie].indx:
                    ie = ie + 1

                if self.mesh.mesh[ie].xblk != xblk or self.mesh.mesh[ie].yblk != yblk:
                    xblk = self.mesh.mesh[ie].xblk
                    yblk = self.mesh.mesh[ie].yblk
                    self.vecgs.vstt[xblk, yblk] = iset

                self.vecgs.vend[xblk, yblk] = iset

            self.vecgs.vlen = self.vecgs.vend - self.vecgs.vstt + 1

            if self.USEMPI:
                for jblk in range(self.gblock.nyblk):
                    for iblk in range(self.gblock.nxblk):
                        if self.gblock.pio(iblk, jblk) == self.mpi.p_address_io[self.mpi.p_my_group]:
                            scnt = self.vecgs.vlen(iblk, jblk)
                            # CALL mpi_gather (scnt, 1, MPI_INTEGER, MPI_INULL_P, 1, MPI_INTEGER, p_root, p_comm_group, p_err)

        if self.USEMPI:
            if self.mpi.p_is_io:
                if self.vecgs.vcnt is None:
                    self.vecgs.vcnt = np.zeros((self.mpi.p_np_group - 1, self.gblock.nxblk, self.gblock.nyblk),dtype=int)
                    self.vecgs.vdsp = np.zeros((self.mpi.p_np_group - 1, self.gblock.nxblk, self.gblock.nyblk),dtype=int)

                self.vecgs.vcnt[:, :, :] = 0
                for jblk in range(self.gblock.nyblk):
                    for iblk in range(self.gblock.nxblk):
                        if self.gblock.pio(iblk, jblk) == self.mpi.p_iam_glb:

                            scnt = 0
                            # CALL mpi_gather (scnt, 1, MPI_INTEGER, &
                            #     self.vecgs.vcnt(:,iblk,jblk), 1, MPI_INTEGER, &
                            #     p_root, p_comm_group, p_err)

                            self.vecgs.vdsp[0, iblk, jblk] = 0
                            for iproc in range(self.mpi.p_np_group - 1):
                                self.vecgs.vdsp[iproc, iblk, jblk] = \
                                    self.vecgs.vdsp[iproc - 1, iblk, jblk] + self.vecgs.vcnt[iproc - 1, iblk, jblk]

                            self.vecgs.vlen[iblk, jblk] = sum(self.vecgs.vcnt[:, iblk, jblk])

        if self.mpi.p_is_io or self.mpi.p_is_worker:
            # nonzero = np.zeros((self.gblock.nxblk, self.gblock.nyblk))

            nonzero = self.vecgs.vlen > 0
            if self.USEMPI:
                pass
                # CALL mpi_allreduce (MPI_IN_PLACE, nonzero, self.gblock.nxblk * self.gblock.nyblk, &
                #     MPI_LOGICAL, MPI_LOR, p_comm_group, p_err)

            self.nblkgrp = len(np.where(self.vecgs.vlen > 0)[0])
            if self.xblkgrp is not None:
                del self.xblkgrp
            if self.yblkgrp is not None:
                del self.yblkgrp
            self.xblkgrp = np.zeros(self.nblkgrp, dtype=int)
            self.yblkgrp = np.zeros(self.nblkgrp, dtype=int)

            iblkgrp = 0
            for jblk in range(self.gblock.nyblk):
                for iblk in range(self.gblock.nxblk):
                    if nonzero[iblk, jblk]:
                        self.xblkgrp[iblkgrp] = iblk
                        self.yblkgrp[iblkgrp] = jblk
                        iblkgrp = iblkgrp + 1

            del nonzero

        if self.mpi.p_is_io:

            if self.vlenall is None:
                self.vlenall = np.zeros((self.gblock.nxblk, self.gblock.nyblk))
            self.vlenall = self.vecgs.vlen

            if self.USEMPI:
                # CALL mpi_allreduce (MPI_IN_PLACE, self.vlenall, self.gblock.nxblk * self.gblock.nyblk, &
                #     MPI_INTEGER, MPI_SUM, p_comm_io, p_err)
                pass

            self.nblkall = len(np.where(self.vlenall > 0)[0])

            if self.xblkall is not None:
                del self.xblkall
            if self.yblkall is not None:
                del self.yblkall

            self.xblkall = np.zeros(self.nblkall, dtype='int')
            self.yblkall = np.zeros(self.nblkall, dtype='int')

            iblkall = 0
            for jblk in range(self.gblock.nyblk):
                for iblk in range(self.gblock.nxblk):
                    if self.vlenall[iblk, jblk] > 0:
                        self.xblkall[iblkall] = iblk
                        self.yblkall[iblkall] = jblk
                        iblkall = iblkall + 1

    def pset_pack(self, mask):
        if self.mpi.p_is_worker:
            if self.nset > 0:
                len_mask = len(np.where(mask)[0])
                if len_mask < self.nset:
                    eindex1 = self.eindex[:]
                    ipxstt1 = self.ipxstt[:]
                    ipxend1 = self.ipxend[:]
                    settyp1 = self.settyp[:]
                    ielm1 = self.ielm[:]

                    # del self.eindex
                    # del self.ipxstt
                    # del self.ipxend
                    # del self.settyp
                    # del self.ielm

                    self.nset = len(np.where(mask)[0])
                    if self.nset > 0:
                        self.eindex = [eindex1[i] for i in range(len(eindex1)) if mask[i]]
                        self.ipxstt = [ipxstt1[i] for i in range(len(ipxstt1)) if mask[i]]
                        self.ipxend = [ipxend1[i] for i in range(len(ipxend1)) if mask[i]]
                        self.settyp = [settyp1[i] for i in range(len(settyp1)) if mask[i]]
                        self.ielm = [ielm1[i] for i in range(len(ielm1)) if mask[i]]

                    del eindex1
                    del ipxstt1
                    del ipxend1
                    del settyp1
                    del ielm1

        self.set_vecgs()
        nset_packed = self.nset
        return nset_packed

    def pixelset_free_mem(self):
        if self.eindex is not None:
            del self.eindex
        if self.ipxstt is not None:
            del self.ipxstt
        if self.ipxend is not None:
            del self.ipxend
        if self.settyp is not None:
            del self.settyp
        if self.ielm is not None:
            del self.ielm
        if self.xblkgrp is not None:
            del self.xblkgrp
        if self.yblkgrp is not None:
            del self.yblkgrp
        if self.xblkall is not None:
            del self.xblkall
        if self.yblkall is not None:
            del self.yblkall
        if self.vlenall is not None:
            del self.vlenall

    def get_lonlat_radian(self, rlon, rlat, pi, pixel):
        for iset in range(self.nset):
            ie = self.ielm[iset]
            ipxstt = self.ipxstt[iset] - 1
            ipxend = self.ipxend[iset]
            area = np.zeros(ipxend - ipxstt)
            # allocate(area(ipxstt: ipxend))

            for ipxl in range(ipxstt, ipxend):
                ipxl_area = ipxl - ipxstt
                # print(iset, ie, ipxstt, ipxend, len(area), len(self.mesh.mesh[ie].ilat), ipxl, ipxl_area, self.mesh.mesh[ie].ilon[ipxl], len(pixel.lon_w),'------------')
                # print(ipxl,len(pixel.lat_n), len(self.mesh.mesh[ie]), len(self.mesh.mesh[ie].ilat),'*******')
                # self.mesh.mesh[ie].ilon[ipxl], len(pixel.lon_w)
                area[ipxl_area] = CoLM_Utils.areaquad(
                    pixel.lat_s[self.mesh.mesh[ie].ilat[ipxl]-1],
                    pixel.lat_n[self.mesh.mesh[ie].ilat[ipxl]-1],
                    pixel.lon_w[self.mesh.mesh[ie].ilon[ipxl]-1],
                    pixel.lon_e[self.mesh.mesh[ie].ilon[ipxl]-1] )

            npxl = ipxend - ipxstt
            rlat[iset] = self.get_pixelset_rlat(
                         npxl, self.mesh.mesh[ie].ilat[ipxstt:ipxend], area, pi, pixel)
            rlon[iset] = self.get_pixelset_rlon(
                         npxl, self.mesh.mesh[ie].ilon[ipxstt:ipxend], area, pi, pixel)

            del area
        return rlon, rlat
    
    def get_pixelset_rlat (self, npxl, ilat, area, pi, pixel):
        rlat = 0.0
        for ipxl in range (npxl):
            rlat = rlat + (pixel.lat_s[ilat[ipxl]-1] + pixel.lat_n[ilat[ipxl]-1]) * 0.5 * area[ipxl]
        rlat = rlat / sum(area) * pi/180.0

        return rlat
    
    def get_pixelset_rlon (self, npxl, ilon, area, pi, pixel):
        lon = 0.0
        area_done = 0.0
        for ipxl in range(npxl):
            if pixel.lon_w[ilon[ipxl]-1] > pixel.lon_e[ilon[ipxl]-1]:
                lon0 = (pixel.lon_w[ilon[ipxl]-1] + pixel.lon_e[ilon[ipxl]-1] + 360.0) * 0.5
            else:
                lon0 = (pixel.lon_w[ilon[ipxl]-1] + pixel.lon_e[ilon[ipxl]-1]) * 0.5
        
        
            # normalize_longitude(lon0)
            lon0 = CoLM_Utils.normalize_longitude(lon0)
        
            if lon - lon0 > 180.0:
                lon = lon * area_done + (lon0 + 360.0) * area[ipxl]
            elif lon - lon0 < -180.0:
                lon = lon * area_done + (lon0 - 360.0) * area[ipxl]
            else:
                lon = lon * area_done + lon0 * area[ipxl]

            area_done = area_done + area[ipxl]
            lon = lon / area_done

            lon = CoLM_Utils.normalize_longitude(lon)
        

        rlon = lon * pi / 180.0
        return rlon

class SubsetType:
    def __init__(self, mesh, pixel):
        self.substt = None
        self.subend = None
        self.subfrc = None
        self.mesh = mesh.mesh
        self.pixel = pixel

    def build(self, superset: Pixelset_type, subset: Pixelset_type, use_frac: bool,
              sharedfrac: Optional[List[float]] = None):
        if superset.nset <= 0:
            return

        if self.substt is not None:
            del self.substt
        if self.subend is not None:
            del self.subend

        self.substt = np.zeros(superset.nset)
        self.subend = np.zeros(superset.nset)

        # self.substt[:] = 0
        self.subend[:] = -1

        isuperset = 0
        isubset = 0
        while isubset < subset.nset:
            if (subset.eindex[isubset] == superset.eindex[isuperset] and
                    subset.ipxstt[isubset] >= superset.ipxstt[isuperset] and
                    subset.ipxend[isubset] <= superset.ipxend[isuperset]):
                if self.substt[isuperset] == 0:
                    self.substt[isuperset] = isubset
                self.subend[isuperset] = isubset
                isubset += 1
            else:
                isuperset += 1

        if use_frac:
            if self.subfrc is not None:
                del self.subfrc

            if subset.nset <= 0:
                return

            self.subfrc = np.zeros(subset.nset)
            for isubset in range(subset.nset):
                ielm = subset.ielm[isubset]
                self.subfrc[isubset] = 0
                for ipxl in range(subset.ipxstt[isubset] - 1, subset.ipxend[isubset]):
                    self.subfrc[isubset] += CoLM_Utils.areaquad(self.pixel.lat_s[self.mesh[ielm].ilat[ipxl]-1],
                                                                self.pixel.lat_n[self.mesh[ielm].ilat[ipxl]-1],
                                                                self.pixel.lon_w[self.mesh[ielm].ilon[ipxl]-1],
                                                                self.pixel.lon_e[self.mesh[ielm].ilon[ipxl]-1])
                if sharedfrac is not None:
                    self.subfrc[isubset] *= sharedfrac[isubset]

            for isuperset in range(superset.nset):
                if self.substt[isuperset] != 0:
                    istt = int(self.substt[isuperset])-1
                    iend = int(self.subend[isuperset])
                    total = sum(self.subfrc[istt:iend])
                    if total != 0:
                        self.subfrc[istt:iend] = self.subfrc[istt:iend] / total

    def subset_free_mem(self):
        if self.substt is not None:
            del self.substt
        if self.subend is not None:
            del self.subend
        if self.subfrc is not None:
            del self.subfrc


class Vec_gather_scatter_type(object):
    def __init__(self) -> None:
        self.vlen = None

        # for worker
        self.vstt = None
        self.vend = None

        # for io
        self.vcnt = None
        self.vdsp = None
