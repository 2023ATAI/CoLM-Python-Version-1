# ----------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    Shared pixelset refer to two or more pixelsets sharing the same geographic area.
# 
#    For example, for patch of crops, multiple crops can be planted on a piece of land.
#    When planting these crops, different irrigation schemes may be used. Thus the water 
#    and energy processes have difference in crops and should be modeled independently.
#    By using shared pixelset, crop patch is splitted to two or more shared patches.
#    Each shared patch is assigned with a percentage of area and has its own states.
#
#                Example of shared pixelsets
#        |<------------------- ELEMENT ------------------>| <-- level 1
#        |   subset 1  |       subset 2        | subset 3 | <-- level 2
#                      | subset 2 shared 1 50% |            
#                      | subset 2 shared 2 20% |            <-- subset 2 shares
#                      | subset 2 shared 3 30% |            
#
#
# Created by Shupeng Zhang, May 2023
# ----------------------------------------------------------------------------------------
import numpy as np
from CoLM_AggregationRequestData import AggregationRequestData
import CoLM_Utils


# def __init__(self, mpi, nl_colm, gblock, mesh) -> None:
#     mpi = mpi
#     gblock = gblock
#     mesh = mesh
#     nl_colm = nl_colm

def pixelsetshared_build(nl_colm, mpi, mesh, gblock, pixelset, gshared, datashared, nmaxshared, typfilter, sharedclass,
                         fracin=None):
    fracout = None
    nsetshared = 0
    pctshared = None
    datashared1d = None
    areapixel = None
    rbuff = None

    eindex1 = None
    ipxstt1 = None
    ipxend1 = None
    settyp1 = None
    ielm1 = None

    if nl_colm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)

        # if mpi.p_is_io:
        #     CALL aggregation_data_daemon (gshared, data_r8_3d_in1 = datashared, n1_r8_3d_in1 = nmaxshared)

    if mpi.p_is_worker:
        pctshared = np.zeros((nmaxshared, pixelset.nset))

        for ipset in range(pixelset.nset):
            if np.any(np.array(typfilter) == pixelset.settyp[ipset]):
                ie = pixelset.ielm[ipset]
                ipxstt = pixelset.ipxstt[ipset]
                ipxend = pixelset.ipxend[ipset]

                datashared1d = np.zeros((nmaxshared, ipxend - ipxstt + 1))

                # Aggregation request data
                ard = AggregationRequestData(nl_colm['USEMPI'], mpi, mesh.mesh, gblock.pixel)
                rbuff = ard.aggregation_request_data(pixelset, ipset, gshared, zip=False,
                                                     data_r8_3d_in1=datashared, n1_r8_3d_in1=nmaxshared)

                datashared1d = rbuff

                areapixel = np.zeros(ipxend - ipxstt + 1)
                for ipxl in range(ipxstt, ipxend + 1):
                    areapixel[ipxl] = CoLM_Utils.areaquad(gblock.pixel.lat_s[mesh.mesh[ie].ilat[ipxl]],
                                                          gblock.pixel.lat_n[mesh.mesh[ie].ilat[ipxl]],
                                                          gblock.pixel.lon_w[mesh.mesh[ie].ilon[ipxl]],
                                                          gblock.pixel.lon_e[mesh.mesh[ie].ilon[ipxl]])

                for ishared in range(nmaxshared):
                    pctshared[ishared, ipset] = np.sum(datashared1d[ishared, :] * areapixel)

                if np.any(np.array(pctshared[:, ipset]) > 0):
                    nsetshared += np.count_nonzero(pctshared[:, ipset] > 0)
                    pctshared[:, ipset] /= np.sum(pctshared[:, ipset])

                    # Deallocation of buffers
                    del rbuff
                    del areapixel
                    del datashared1d
            else:
                nsetshared += 1

        if nl_colm['USEMPI']:
            pass
            #  CALL aggregation_worker_done ()

    if mpi.p_is_worker:
        if pixelset.nset > 0:
            eindex1 = np.copy(pixelset.eindex)
            ipxstt1 = np.copy(pixelset.ipxstt)
            ipxend1 = np.copy(pixelset.ipxend)
            settyp1 = np.copy(pixelset.settyp)
            ielm1 = np.copy(pixelset.ielm)

            pixelset.eindex = np.zeros(nsetshared, dtype=int)
            pixelset.ipxstt = np.zeros(nsetshared, dtype=int)
            pixelset.ipxend = np.zeros(nsetshared, dtype=int)
            pixelset.settyp = np.zeros(nsetshared, dtype=int)
            pixelset.ielm = np.zeros(nsetshared, dtype=int)

            fracout = np.zeros(nsetshared, dtype=float)
            sharedclass = np.zeros(nsetshared, dtype=int)

            jpset = 0
            for ipset in range(pixelset.nset):
                if np.any(np.array(typfilter) == settyp1[ipset]):
                    if np.any(np.array(pctshared[:, ipset]) > 0):
                        for ishared in range(nmaxshared):
                            if pctshared[ishared, ipset] > 0:
                                jpset += 1
                                pixelset.eindex[jpset] = eindex1[ipset]
                                pixelset.ipxstt[jpset] = ipxstt1[ipset]
                                pixelset.ipxend[jpset] = ipxend1[ipset]
                                pixelset.settyp[jpset] = settyp1[ipset]
                                pixelset.ielm[jpset] = ielm1[ipset]

                                if fracin is not None:
                                    fracout[jpset] = fracin[ipset] * pctshared[ishared, ipset]
                                else:
                                    fracout[jpset] = pctshared[ishared, ipset]

                                sharedclass[jpset] = ishared
                else:
                    jpset += 1
                    pixelset.eindex[jpset] = eindex1[ipset]
                    pixelset.ipxstt[jpset] = ipxstt1[ipset]
                    pixelset.ipxend[jpset] = ipxend1[ipset]
                    pixelset.settyp[jpset] = settyp1[ipset]
                    pixelset.ielm[jpset] = ielm1[ipset]

                    if fracin is not None:
                        fracout[jpset] = fracin[ipset]
                    else:
                        fracout[jpset] = 0

                    sharedclass[jpset] = 0
            del eindex1
            del ipxstt1
            del ipxend1
            del settyp1
            del ielm1
            del pctshared

            pixelset.nset = nsetshared

    pixelset.set_vecgs()
    return fracout
