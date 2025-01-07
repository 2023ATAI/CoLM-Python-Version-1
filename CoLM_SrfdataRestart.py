# ------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    This module includes subroutines to read/write data of mesh and pixelsets.
# 
# ------------------------------------------------------------------------------------
import os
import sys
import numpy as np
from CoLM_NetCDFSerial import NetCDFFile
from CoLM_NetCDFVectorOneS import CoLM_NetCDFVector
import CoLM_Utils
from CoLM_Mesh import IrregularElement
from CoLM_NetCDFVectorOneP import CoLM_NetCDFVectorP
import CoLM_NetCDFVectorBlk


def release(o):
    if o is not None:
        del o


def pixelset_save_to_file(mpi, co_lm, gblock, mesh, dir_landdata, psetname, pixelset, lc_year):
    cyear = lc_year
    ndsp_worker = None
    nelm_worker = None
    elmindx = None
    npxlall = None
    elmpixels = None

    vectorones = CoLM_NetCDFVector(co_lm, mpi, gblock)

    if co_lm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)
    # endif
    if mpi.p_is_master:
        print('Saving Pixel Sets ' + psetname + ' ...')

    if co_lm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)
    plat_system = sys.platform
    path_folder = os.path.join(dir_landdata, psetname, str(cyear))

    filename = os.path.join(str(path_folder), psetname + '.nc')
    if 'win' in plat_system:
        path_folder = dir_landdata + '\\' + psetname + '\\' + str(cyear)
        filename = path_folder + '\\' + psetname + '.nc'
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    vectorones.ncio_create_file_vector(filename, pixelset)
    vectorones.ncio_define_dimension_vector(filename, pixelset, psetname)

    vectorones.ncio_write_vector(filename, 'eindex', psetname, pixelset, pixelset.eindex,
                                 co_lm['DEF_Srfdata_CompressLevel'])
    vectorones.ncio_write_vector(filename, 'ipxstt', psetname, pixelset, pixelset.ipxstt,
                                 co_lm['DEF_Srfdata_CompressLevel'])
    vectorones.ncio_write_vector(filename, 'ipxend', psetname, pixelset, pixelset.ipxend,
                                 co_lm['DEF_Srfdata_CompressLevel'])
    vectorones.ncio_write_vector(filename, 'settyp', psetname, pixelset, pixelset.settyp,
                                 co_lm['DEF_Srfdata_CompressLevel'])

    if co_lm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)

    if mpi.p_is_master:
        print('SAVE Pixel Sets ' + psetname + ' done.')


def mesh_save_to_file(mpi, co_lm, gblock, mesh, dir_landdata, lc_year):
    # add parameter input for time year
    cyear = lc_year
    net_file = NetCDFFile(mpi)
    nelm_worker = None
    ndsp_worker = None
    elmindx = None
    npxlall = None
    elmpixels = None

    if co_lm['USEMPI']:
        pass
    #   CALL mpi_barrier (p_comm_glb, p_err)

    if mpi.p_is_master:
        print('Saving land elements ...')
        path_mk = os.path.join(dir_landdata, 'mesh', str(cyear))
        if 'win' in sys.platform:
            path_mk = dir_landdata + '\\' + 'mesh' + '\\' + str(cyear)
            if not os.path.exists(path_mk):
                os.mkdir(path_mk)
    if co_lm['USEMPI']:
        pass
        #   CALL mpi_barrier (p_comm_glb, p_err)

    filename = os.path.join(dir_landdata, 'mesh', str(cyear), 'mesh.nc')
    if 'win' in sys.platform:
        filename = dir_landdata + '\\mesh\\' + str(cyear) + '\\mesh.nc'

    for jblk in range(gblock.nyblk):
        for iblk in range(gblock.nxblk):
            if co_lm['USEMPI']:
                # if mpi.p_is_worker:
                #     if gblock.pio[iblk,jblk] == p_address_io[p_my_group]:
                pass
            # endif
            nelm = 0
            totlen = 0
            for ie in range(mesh.numelm):
                if (mesh.mesh[ie].xblk == iblk) and (mesh.mesh[ie].yblk == jblk):
                    nelm = nelm + 1
                    totlen = totlen + mesh.mesh[ie].npxl

            if nelm > 0:
                elmindx = np.zeros(nelm, dtype='int')
                npxlall = np.zeros(nelm, dtype='int')
                elmpixels = np.zeros((2, totlen))

                je = 0
                ndsp = 0
                for ie in range(mesh.numelm):
                    if (mesh.mesh[ie].xblk == iblk) and (mesh.mesh[ie].yblk == jblk):
                        elmindx[je] = mesh.mesh[ie].indx
                        npxlall[je] = mesh.mesh[ie].npxl

                        elmpixels[0, ndsp:ndsp + npxlall[je]] = mesh.mesh[ie].ilon
                        elmpixels[1, ndsp:ndsp + npxlall[je]] = mesh.mesh[ie].ilat
                        ndsp = ndsp + npxlall[je]
                        je = je + 1
            if co_lm['USEMPI']:
                pass
                # CALL mpi_gather (nelm, 1, MPI_INTEGER, &
                #  MPI_INULL_P, 1, MPI_INTEGER, p_root, p_comm_group, p_err)

                # CALL mpi_gatherv (elmindx, nelm, MPI_INTEGER8, &
                #     MPI_INULL_P, MPI_INULL_P, MPI_INULL_P, MPI_INTEGER8, & ! insignificant on workers
                #     p_root, p_comm_group, p_err)

                # CALL mpi_gatherv (npxlall, nelm, MPI_INTEGER, &
                #     MPI_INULL_P, MPI_INULL_P, MPI_INULL_P, MPI_INTEGER, & ! insignificant on workers
                #     p_root, p_comm_group, p_err)

                # ndone = 0
                # DO WHILE (ndone < totlen)
                #     nsend = max(min(totlen-ndone, MesgMaxSize/8), 1)
                #     CALL mpi_send (nsend, 1, &
                #         MPI_INTEGER, p_root, mpi_tag_size, p_comm_group, p_err)
                #     CALL mpi_send (elmpixels(:,ndone+1:ndone+nsend), 2*nsend, &
                #         MPI_INTEGER, p_root, mpi_tag_data, p_comm_group, p_err)
                #     ndone = ndone + nsend

                # if mpi.p_is_io:
                #     if gblock.pio(iblk,jblk) == p_iam_glb:

                #         elen = 0
                #         # CALL mpi_allreduce (MPI_IN_PLACE, elen, 1, MPI_INTEGER, MPI_MAX, p_comm_group, p_err)

                #         nelm_worker = np.zeros([1,p_np_group-1])
                #         nelm_worker[0:]= 0
                #         # CALL mpi_gather (MPI_IN_PLACE, 0, MPI_INTEGER, &
                #         #     nelm_worker, 1, MPI_INTEGER, p_root, p_comm_group, p_err)

                #         nelm = sum(nelm_worker)

                #         ndsp_worker = np.zeros([0,p_np_group-1])
                #         ndsp_worker[0:] = 0
                #         for iworker in range( p_np_group-1):
                #             ndsp_worker[iworker] = ndsp_worker[iworker-1] + nelm_worker[iworker-1]

                #         elmindx = np.zeros(nelm)
                #         # CALL mpi_gatherv (MPI_IN_PLACE, 0, MPI_INTEGER, &
                #         #     elmindx, nelm_worker(0:), ndsp_worker(0:), MPI_INTEGER, &
                #         #     p_root, p_comm_group, p_err)

                #         npxlall = np.zeros (nelm)
                #         # CALL mpi_gatherv (MPI_IN_PLACE, 0, MPI_INTEGER, &
                #         #     npxlall, nelm_worker(0:), ndsp_worker(0:), MPI_INTEGER, &
                #         #     p_root, p_comm_group, p_err)

                #         elmpixels = np.zeros ([2, elen, nelm])
                #         # for iworker in range( p_np_group-1):
                #         #     for ie in range( ndsp_worker(iworker)+1, ndsp_worker(iworker)+nelm_worker(iworker)):
                #                 # CALL mpi_recv (elmpixels(:,:,ie), 2*elen, MPI_INTEGER, &
                #                 # iworker, mpi_tag_data, p_comm_group, p_stat, p_err)

            if mpi.p_is_io:
                if gblock.pio[iblk, jblk] == mpi.p_iam_glb:
                    if nelm > 0:
                        fileblock = gblock.get_filename_block(filename, iblk, jblk)
                        net_file.ncio_create_file(fileblock)

                        net_file.ncio_define_dimension(fileblock, 'element', nelm)
                        net_file.ncio_define_dimension(fileblock, 'ncoor', 2)
                        net_file.ncio_define_dimension(fileblock, 'pixel', totlen)

                        net_file.ncio_write_serial4(fileblock, 'elmindex', elmindx, 'element')
                        net_file.ncio_write_serial4(fileblock, 'elmnpxl', npxlall, 'element')
                        net_file.ncio_write_serial6(fileblock, 'elmpixels', elmpixels,
                                                    'ncoor', 'pixel', compress=1)

            release(elmindx)
            release(npxlall)
            release(elmpixels)

            release(nelm_worker)
            release(ndsp_worker)

            if co_lm['USEMPI']:
                pass
                # CALL mpi_barrier (p_comm_group, p_err)

    if mpi.p_is_master:
        net_file.ncio_create_file(filename)

        net_file.ncio_define_dimension(filename, 'xblk', gblock.nxblk)
        net_file.ncio_define_dimension(filename, 'yblk', gblock.nyblk)
        net_file.ncio_write_serial5('nelm_blk', mesh.nelm_blk, 'xblk', 'yblk')

        net_file.ncio_define_dimension(filename, 'longitude', mesh.gridmesh.nlon)
        net_file.ncio_define_dimension(filename, 'latitude', mesh.gridmesh.nlat)

        lon = np.zeros(mesh.gridmesh.nlon)
        lat = np.zeros(mesh.gridmesh.nlat)

        for i in range(mesh.gridmesh.nlon):
            lon[i] = (mesh.gridmesh.lon_w[i] + mesh.gridmesh.lon_e[i]) * 0.5
            if mesh.gridmesh.lon_w[i] > mesh.gridmesh.lon_e[i]:
                lon[i] = lon[i] + 180.0
                CoLM_Utils.normalize_longitude(lon[i])
        net_file.ncio_write_serial4(filename, 'longitude', lon, 'longitude')

        for i in range(mesh.gridmesh.nlat):
            lat[i] = (mesh.gridmesh.lat_s[i] + mesh.gridmesh.lat_n[i]) * 0.5
        net_file.ncio_write_serial4(filename, 'latitude', lat, 'latitude')

        del lon
        del lat
    if co_lm['USEMPI']:
        pass
        # CALL mpi_barrier (p_comm_glb, p_err)

    if mpi.p_is_master:
        print('SAVE land elements done.')


def mesh_load_from_file(mpi, co_lm, gblock, o_mesh, dir_landdata, lc_year):
    elmindx = None
    npxl = None
    datasize = None
    mesh = []
    if co_lm['USEMPI']:
        pass

    if mpi.p_is_master:
        print('Loading land elements ...')

    # numelm = 0
    pixels2d = None
    pixels = None
    cyear = lc_year
    filename = f"{dir_landdata}/mesh/{cyear}/mesh.nc"
    net_file = NetCDFFile(mpi)
    o_mesh.nelm_blk = net_file.ncio_read_bcast_serial(filename, 'nelm_blk').transpose(1,0)

    if mpi.p_is_io:
        o_mesh.numelm = sum(o_mesh.nelm_blk[gblock.pio == mpi.p_iam_glb])

        if o_mesh.numelm > 0:
            for _ in range(o_mesh.numelm):
                mesh.append(IrregularElement())

            ndsp = 0
            for iblkme in range(gblock.nblkme):
                iblk = gblock.xblkme[iblkme]
                jblk = gblock.yblkme[iblkme]

                nelm = o_mesh.nelm_blk[iblk, jblk]

                if nelm > 0:
                    fileblock = gblock.get_filename_block(filename, iblk, jblk)
                    #jblk, iblk互换，查以下gblock.xblkmen和gblock.yblkme
                    elmindx = net_file.ncio_read_serial(fileblock, 'elmindex')
                    npxl = net_file.ncio_read_serial(fileblock, 'elmnpxl')

                    datasize = net_file.ncio_inquire_varsize(fileblock, 'elmpixels')
                    if datasize is not None and len(datasize) == 3:
                    # if datasize is not None and len(datasize) == 3:
                        pixels2d = net_file.ncio_read_serial(fileblock, 'elmpixels')
                    else:
                        pixels = net_file.ncio_read_serial(fileblock, 'elmpixels')

                    pdsp = 0
                    for ie in range(nelm):
                        mesh[ie + ndsp].indx = elmindx[ie]
                        mesh[ie + ndsp].npxl = npxl[ie]
                        mesh[ie + ndsp].xblk = iblk
                        mesh[ie + ndsp].yblk = jblk

                        mesh[ie + ndsp].ilon = np.zeros(npxl[ie], dtype=int)
                        mesh[ie + ndsp].ilat = np.zeros(npxl[ie], dtype=int)

                        if len(datasize) == 3:
                            mesh[ie + ndsp].ilon = pixels2d[0, 0:npxl[ie], ie]
                            mesh[ie + ndsp].ilat = pixels2d[1, 0:npxl[ie], ie]
                        else:
                            #横纵坐标互换
                            mesh[ie + ndsp].ilon = pixels[pdsp:(pdsp + npxl[ie]), 0]
                            mesh[ie + ndsp].ilat = pixels[pdsp:(pdsp + npxl[ie]), 1]
                            pdsp += npxl[ie]

                    ndsp += nelm

        if elmindx is not None:
            del elmindx
        if npxl is not None:
            del npxl
        if datasize is not None:
            del datasize
        if pixels is not None:
            del pixels
        if pixels2d is not None:
            del pixels2d

    if co_lm['CoLMDEBUG']:
        if mpi.p_is_io:
            print(o_mesh.numelm, ' elements on group ', mpi.p_iam_io)
    if co_lm['USEMPI']:
        pass

    if mpi.p_is_master:
        print("Loading land elements done.")
    o_mesh.mesh = mesh
    return o_mesh


def pixelset_load_from_file(mpi, co_lm, gblock,mesh, dir_landdata, psetname, pixelset, lc_year):
    cyear = lc_year
    net_file = NetCDFFile(mpi)

    vp = CoLM_NetCDFVectorP(co_lm, mpi, gblock)

    if co_lm['USEMPI']:
        pass

    if mpi.p_is_master:
        print(f"Loading Pixel Sets {psetname} ...")
    filename = os.path.join(dir_landdata, psetname, str(cyear), psetname + '.nc')

    if mpi.p_is_io:
        pixelset.nset = 0
        fexists_any = False

        for iblkme in range(gblock.nblkme):
            iblk = gblock.xblkme[iblkme]
            jblk = gblock.yblkme[iblkme]

            if co_lm['VectorInOneFileS'] or co_lm['VectorInOneFileP']:
                blockname = gblock.get_blockname(iblk, jblk)

                nset = vp.ncio_inquire_length_grp(filename, f"{psetname}_{blockname}", 'eindex')
                pixelset.nset += nset
            else:
                fileblock = gblock.get_filename_block(filename, iblk, jblk)
                fexists = os.path.exists(fileblock)
                if fexists:
                    nset = net_file.ncio_inquire_length(fileblock, 'eindex')
                    pixelset.nset += nset
                fexists_any = fexists_any or fexists

        if co_lm['VectorInOneFileS'] or co_lm['VectorInOneFileP']:
            fexists_any = pixelset.nset > 0

        if co_lm['USEMPI']:
            pass

        if not fexists_any:
            print(f"Warning: restart file {filename} not found.")
            # CALL CoLM_stop ()

        if pixelset.nset > 0:
            pixelset.eindex = np.zeros(pixelset.nset, dtype=int)

            ndsp = 0
            for iblkme in range(gblock.nblkme):
                iblk = gblock.xblkme[iblkme]
                jblk = gblock.yblkme[iblkme]

                if co_lm['VectorInOneFileS'] or co_lm['VectorInOneFileP']:
                    blockname = gblock.get_blockname(iblk, jblk)
                    nset = vp.ncio_inquire_length_grp(filename, f"{psetname}_{blockname}", 'eindex')
                    if nset > 0:
                        rbuff = vp.ncio_read_serial_grp_int64_1d(filename, f"{psetname}_{blockname}", 'eindex')
                        pixelset.eindex[ndsp:ndsp + nset] = rbuff
                        ndsp += nset
                else:
                    fileblock = gblock.get_filename_block(filename, iblk, jblk)
                    fexists = os.path.exists(fileblock)
                    if fexists:
                        rbuff = net_file.ncio_read_serial(fileblock, 'eindex')
                        nset = rbuff.size
                        pixelset.eindex[ndsp:ndsp + nset] = rbuff
                        ndsp += nset

    if co_lm['USEMPI']:
        pass

        # if p_is_io:
        #     if pixelset.nset > 0:
        #         iworker = np.empty(pixelset.nset, dtype=np.int32)
        #         msk = np.empty(pixelset.nset, dtype=bool)
        #
        #         ie = 0
        #         je = 0
        #         iblk = mesh[ie].xblk
        #         jblk = mesh[ie].yblk
        #         for iset in range(pixelset.nset):
        #             while pixelset.eindex[iset] != mesh[ie].indx:
        #                 ie += 1
        #                 je += 1
        #                 if mesh[ie].xblk != iblk or mesh[ie].yblk != jblk:
        #                     je = 0
        #                     iblk = mesh[ie].xblk
        #                     jblk = mesh[ie].yblk
        #
        #             nave = nelm_blk[iblk, jblk] // (p_np_group - 1)
        #             nres = nelm_blk[iblk, jblk] % (p_np_group - 1)
        #             left = (nave + 1) * nres
        #             if je <= left:
        #                 iworker[iset] = (je - 1) // (nave + 1) + 1
        #             else:
        #                 iworker[iset] = (je - left - 1) // nave + 1 + nres
        #
        #         for iproc in range(1, p_np_group):
        #             msk = iworker == iproc
        #             nsend = np.sum(msk)
        #             mpi_send(nsend, iproc, mpi_tag_size, comm)
        #
        #             if nsend > 0:
        #                 sbuff = pixelset.eindex[msk]
        #                 mpi_send(sbuff, iproc, mpi_tag_data, comm)
        #
        #     else:
        #         for iproc in range(1, p_np_group):
        #             mpi_send(0, iproc, mpi_tag_size, comm)
        #
        # if not p_is_io:
        #     nrecv = mpi_recv(p_root, mpi_tag_size, comm)
        #     pixelset.nset = nrecv
        #     if nrecv > 0:
        #         pixelset.eindex = mpi_recv(p_root, mpi_tag_data, comm)

    pixelset.set_vecgs()


    pixelset.ipxstt = np.array(CoLM_NetCDFVectorBlk.ncio_read_vector(filename, 'ipxstt', pixelset, pixelset.ipxstt, co_lm['USEMPI'], mpi, gblock),dtype=int)
    pixelset.ipxend = np.array(CoLM_NetCDFVectorBlk.ncio_read_vector(filename, 'ipxend', pixelset, pixelset.ipxend, co_lm['USEMPI'], mpi, gblock),dtype=int)
    pixelset.settyp = np.array(CoLM_NetCDFVectorBlk.ncio_read_vector(filename, 'settyp', pixelset, pixelset.settyp, co_lm['USEMPI'], mpi, gblock),dtype=int)

    if mpi.p_is_worker:
        if pixelset.nset > 0:
            pixelset.ielm = np.zeros(pixelset.nset, dtype=int)
            ie = 0
            for iset in range(pixelset.nset):
                while pixelset.eindex[iset] != mesh.mesh[ie].indx:
                    ie += 1
                pixelset.ielm[iset] = ie
        else:
            print(f"Warning: 0 {psetname} on worker: {mpi.p_iam_glb}")

    numset = pixelset.nset

    if co_lm['CoLMDEBUG']:
        if mpi.p_is_io:
            print(f"{numset} {psetname} on group {mpi.p_iam_io}")

    return pixelset, numset