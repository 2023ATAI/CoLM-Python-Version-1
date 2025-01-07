#-----------------------------------------------------------------------
   # DESCRIPTION:
   #Read in the LAI, the LAI dataset was created by Yuan et al. (2011)
   # ! http://globalchange.bnu.edu.cn
   # !
   # ! Created by Yongjiu Dai, March, 2014
#-----------------------------------------------------------------------
import numpy as np

def LAI_readin (nl_colm, year, time, dir_landdata, srfdata, mpi, landpatch, patchclass, VTV, const_LC):
    if nl_colm['LULC_USGS']:
        # Maximum fractional cover of vegetation [-]
        vegc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    # ! READ in Leaf area index and stem area index
    landdir = dir_landdata.strip() + '/LAI'

    if nl_colm['SinglePoint']:
        if not nl_colm['URBAN_MODEL']:
            iyear = np.where(srfdata.SITE_LAI_year == year)[0]
            if not nl_colm['DEF_LAI_MONTHLY']:
                itime = (time - 1) // 8 + 1

    if nl_colm['LULC_USGS'] or nl_colm['LULC_IGBP']:
        #!TODO-done: need to consider single point for urban model
        if nl_colm['SinglePoint']:
            if not nl_colm['URBAN_MODEL']:
                if  nl_colm['DEF_LAI_MONTHLY']:
                    VTV.tlai = srfdata.SITE_LAI_monthly(time, iyear)
                    VTV.tsai = srfdata.SITE_SAI_monthly(time, iyear)
                else:
                    VTV.tlai = srfdata.SITE_LAI_8day(itime, iyear)
        else:
            pass

        if mpi.p_is_worker:
            if landpatch.numpatch > 0:
                for npatch in range (landpatch.numpatch):
                    m = patchclass[npatch] - 1
                    if nl_colm['URBAN_MODEL']:
                        pass
                    if m == 0:
                        VTV.fveg[npatch] = 0.
                        VTV.tlai[npatch] = 0.
                        VTV.tsai[npatch] = 0.
                        VTV.green[npatch] = 0.
                    else:
                        VTV.fveg[npatch] = const_LC.fveg0[m]  #!fraction of veg. cover
                        if const_LC.fveg0[m] > 0:
                            VTV.tlai[npatch] = VTV.tlai[npatch]/const_LC.fveg0[m]  #!leaf area index
                            if nl_colm['DEF_LAI_MONTHLY']:
                                VTV.tsai[npatch] = VTV.tsai[npatch] / const_LC.fveg0[m] #!stem are index
                            else:
                                VTV.tsai[npatch] = const_LC.sai0[m]
                            VTV.green[npatch] = 1.
                        else:
                            VTV.tlai[npatch] = 0.
                            VTV.tsai[npatch] = 0.
                            VTV.green[npatch] = 0.

        if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
            pass




