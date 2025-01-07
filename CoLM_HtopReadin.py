import numpy as np
import CoLM_NetCDFVectorBlk

def HTOP_readin (nl_colm, landpatch, mpi, gblock, dir_landdata, lc_year, srfdata, patchclass, VT, const_LC):
    """
    Read in the canopy tree top height
    """
    cyear = '{:04d}'.format(lc_year)
    landdir = dir_landdata.strip() + '/htop/' + cyear.strip()
    htoplc = None
    if nl_colm['LULC_USGS']:
        pass
    if nl_colm['LULC_IGBP']:
        if nl_colm['SinglePoint']:
            htoplc = srfdata.SITE_htop
        else:
            lndname = landdir.strip() + '/htop_patches.nc'
            htoplc = CoLM_NetCDFVectorBlk.ncio_read_vector(lndname, 'htop_patches', landpatch.landpatch, htoplc,nl_colm['USEMPI'], mpi,gblock)

        if mpi.p_is_worker:
            for npatch in range(landpatch.numpatch):
                m = patchclass[npatch]
                VT.htop[npatch] = const_LC.htop0[m-1]
                VT.hbot[npatch] = const_LC.hbot0[m-1]
                # trees or woody savannas
                if m < 6 or m == 8:
                    # 01/06/2020, yuan: adjust htop reading
                    if htoplc[npatch] > 2.:
                        VT.htop[npatch] = htoplc[npatch]
                        VT.hbot[npatch] = htoplc[npatch] * const_LC.hbot0[m-1] / const_LC.htop0[m-1]
                        VT.hbot[npatch] = max(1., const_LC.hbot[npatch])

        if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
            pass