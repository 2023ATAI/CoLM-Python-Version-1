import numpy as np
import CoLM_Namelist
import CoLM_TimeManager


# def netsolar(nl_colm, var_global, idate, deltim, dlon, patchtype, forc_sols, forc_soll, forc_solsd, forc_solld, alb, ssun, ssha, ssoi, ssno, lai, sai, rho, tau,fsno):
def netsolar(nl_colm, var_global, ipatch, idate, deltim, dlon, patchtype,
             forc_sols, forc_soll, forc_solsd, forc_solld,
             alb, ssun, ssha, lai, sai, rho, tau, ssoi, ssno, ssno_lyr,
             parsun, parsha, sabvsun, sabvsha, sabg, sabg_soil, sabg_snow,
             fsno, sabg_snow_lyr, sr, solvd, solvi, solnd, solni, srvd, srvi, srnd, srni,
             solvdln, solviln, solndln, solniln, srvdln, srviln, srndln, srniln):
    """
    Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
    software         : Determine net radiation.

    Args:
        ipatch      (INTEGER)  : patch index
        idate       (ndarray)  : model time
        deltim      (float)    : seconds in a time step [second]
        patchtype   (INTEGER)  : land water TYPE (99-sea)
        dlon         (float)   : logitude in radians

        forc_sols    (float)    : atm vis direct beam solar rad onto srf [W/m2]
        forc_soll    (float)    : atm nir direct beam solar rad onto srf [W/m2]
        forc_solsd   (float)    : atm vis diffuse solar rad onto srf [W/m2]
        forc_solld   (float)    : atm nir diffuse solar rad onto srf [W/m2]

        alb          (float)    : averaged albedo [-]
        ssun         (float)    : sunlit canopy absorption for solar radiation [-]
        ssha         (float)    : shaded canopy absorption for solar radiation [-]
        ssoi         (float)    : ground soil absorption [-]
        ssno         (float)    : ground snow absorption [-]

        lai         (float)    : leaf area index
        sai         (float)    : stem area index
        rho         (float)    : leaf reflectance (iw=iband, il=life and dead)
        tau         (float)    : leaf transmittance (iw=iband, il=life and dead)
        fsno        (float)    : snow fractional cover

    Returns:
         parsun          (float)    : PAR absorbed by sunlit vegetation [W/m2]
         parsha          (float)    : PAR absorbed by shaded vegetation [W/m2]
         sabvsun         (float)    : solar absorbed by sunlit vegetation [W/m2]
         sabvsha         (float)    : solar absorbed by shaded vegetation [W/m2]
         sabg            (float)    : solar absorbed by ground  [W/m2]
! 03/06/2020, yuan:
         sabg_soil         (float)    : solar absorbed by ground soil [W/m2]
         sabg_snow         (float)    : solar absorbed by ground snow [W/m2]
         sr                (float)    : total reflected solar radiation (W/m2)
         solvd             (float)    : incident direct beam vis solar radiation (W/m2)
         solvi             (float)    : incident diffuse beam vis solar radiation (W/m2)
         solnd             (float)    : incident direct beam nir solar radiation (W/m2)
         solni             (float)    : incident diffuse beam nir solar radiation (W/m2)
         srvd              (float)    : reflected direct beam vis solar radiation (W/m2)
         srvi              (float)    : reflected diffuse beam vis solar radiation (W/m2)
         srnd              (float)    : reflected direct beam nir solar radiation (W/m2)
         srni              (float)    : reflected diffuse beam nir solar radiation (W/m2)
         solvdln           (float)    : incident direct beam vis solar radiation at local noon(W/m2)
         solviln           (float)    : incident diffuse beam vis solar radiation at local noon(W/m2)
         solndln           (float)    : ncident direct beam nir solar radiation at local noon(W/m2)
         solniln           (float)    : incident diffuse beam nir solar radiation at local noon(W/m2)
         srvdln            (float)    : reflected direct beam vis solar radiation at local noon(W/m2)
         srviln            (float)    : reflected diffuse beam vis solar radiation at local noon(W/m2)
         srndln            (float)    : reflected direct beam nir solar radiation at local noon(W/m2)
         srniln            (float)    : reflected diffuse beam nir solar radiation at local noon(W/m2)
sabg_snow_lyr(maxsnl+1:1)  (float)    : solar absorbed by snow layers [W/m2]
    """
    # Define output variables
    sabvsun = 0.0
    sabvsha = 0.0
    parsun = 0.0
    parsha = 0.0

    sabg = 0.0
    sabg_soil = 0.0
    sabg_snow = 0.0
    sabg_snow_lyr[:] = 0.0
    # -------------------------------------------------------------------------------------------------------------------
    # Sets the solar radiation components to zero when there is negligible vegetation cover,
    # indicating that direct and scattered solar radiation have minimal impact on the surface.
    if lai + sai <= 1.e-6:
        ssun[:, :] = 0.0
        ssha[:, :] = 0.0

    if patchtype == 0:
        # -------------------------------------------------------------------------------------------------------------------
        if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
            pass
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate radiative fluxes onto surface
    if forc_sols + forc_soll + forc_solsd + forc_solld > 0.0:
        if patchtype < 4:  # non-lake and non-ocean
            parsun = forc_sols * ssun[0, 0] + forc_solsd * ssun[1, 0]
            parsha = forc_sols * ssha[0, 0] + forc_solsd * ssha[1, 0]
            sabvsun = forc_sols * ssun[0, 0] + forc_solsd * ssun[1, 0] + forc_soll * ssun[0, 1] + forc_solld * ssun[
                1, 1]
            sabvsha = forc_sols * ssha[0, 0] + forc_solsd * ssha[1, 0] + forc_soll * ssha[0, 1] + forc_solld * ssha[
                1, 1]
            sabvg = forc_sols * (1. - alb[0, 0]) + forc_solsd * (1. - alb[1, 0]) + forc_soll * (
                    1. - alb[0, 1]) + forc_solld * (1. - alb[1, 1])
            sabg = sabvg - sabvsun - sabvsha
            if patchtype == 0:
                if nl_colm['LULC_IGBP_PFT'] or nl_colm['LULC_IGBP_PC']:
                    pass
        else:  # lake or ocean
            sabvg = forc_sols * (1. - alb[0, 0]) + forc_soll * (1. - alb[0, 1]) + forc_solsd * (
                    1. - alb[1, 0]) + forc_solld * (1. - alb[1, 1])
            sabg = sabvg

        # calculate soil and snow solar absorption
        sabg_soil = forc_sols * ssoi[0, 0] + forc_solsd * ssoi[1, 0] + forc_soll * ssoi[0, 1] + forc_solld * ssoi[1, 1]
        sabg_snow = forc_sols * ssno[0, 0] + forc_solsd * ssno[1, 0] + forc_soll * ssno[0, 1] + forc_solld * ssno[1, 1]
        sabg_soil = sabg_soil * (1. - fsno)
        sabg_snow = sabg_snow * fsno
        # balance check and adjustment for soil and snow absorption
        if sabg_soil + sabg_snow - sabg > 1.e-6:  # this could happen when there is adjust to ssun,ssha
            print("MOD_NetSolar.F90: NOTE imbalance in spliting soil and snow surface!")
            print("sabg:", sabg, "sabg_soil:", sabg_soil, "sabg_snow", sabg_snow)
            print("sabg_soil+sabg_snow:", sabg_soil + sabg_snow, "fsno:", fsno)
            sabg_noadj = sabg_soil + sabg_snow
            if sabg_noadj > 0.0:
                sabg_soil = sabg_soil * sabg / sabg_noadj
                sabg_snow = sabg_snow * sabg / sabg_noadj
                ssoi[:, :] = ssoi[:, :] * sabg / sabg_noadj
                ssno[:, :] = ssno[:, :] * sabg / sabg_noadj

        # snow layer absorption calculation and adjustment for SNICAR model
        if nl_colm['DEF_USE_SNICAR']:
            # adjust snow layer absorption due to multiple reflection between ground and canopy
            if sum(ssno[0, 0, :]) > 0.0:
                ssno[0, 0, :] = ssno[0, 0] * ssno[0, 0, :] / sum(ssno[0, 0, :])
            if sum(ssno[0, 1, :]) > 0.0:
                ssno[0, 1, :] = ssno[0, 1] * ssno[0, 1, :] / sum(ssno[0, 1, :])
            if sum(ssno[1, 0, :]) > 0.0:
                ssno[1, 0, :] = ssno[1, 0] * ssno[1, 0, :] / sum(ssno[1, 0, :])
            if sum(ssno[1, 1, :]) > 0.0:
                ssno[1, 1, :] = ssno[1, 1] * ssno[1, 1, :] / sum(ssno[1, 1, :])
            # snow layer absorption
            sabg_snow_lyr[:] = forc_sols * ssno[0, 0, :] + forc_solsd * ssno[0, 1, :] + \
                               forc_soll * ssno[1, 0, :] + forc_solld * ssno[1, 1, :]
            # convert to the whole area producted by snow fractional cover
            sabg_snow_lyr[:] = sabg_snow_lyr[:] * fsno
            # attribute the first layer absorption to soil absorption
            sabg_soil = sabg_soil + sabg_snow_lyr[0]
            sabg_snow = sabg_snow - sabg_snow_lyr[0]
            # make the soil absorption consistent
            sabg_snow_lyr[0] = sabg_soil

    solvd = forc_sols
    solvi = forc_solsd
    solnd = forc_soll
    solni = forc_solld
    srvd = solvd * alb[0, 0]
    srvi = solvi * alb[1, 0]
    srnd = solnd * alb[0, 1]
    srni = solni * alb[1, 1]
    sr = srvd + srvi + srnd + srni

    radpsec = np.pi / 12. / 3600.
    if CoLM_TimeManager.isgreenwich:
        local_secs = idate.sec + round((dlon / radpsec) / deltim) * deltim
        local_secs = local_secs % 86400
    else:
        local_secs = idate.sec

    if local_secs == 86400 / 2:
        solvdln = forc_sols
        solviln = forc_solsd
        solndln = forc_soll
        solniln = forc_solld
        srvdln = solvdln * alb[0, 0]
        srviln = solviln * alb[1, 0]
        srndln = solndln * alb[0, 1]
        srniln = solniln * alb[1, 1]
    else:
        solvdln = var_global.spval
        solviln = var_global.spval
        solndln = var_global.spval
        solniln = var_global.spval
        srvdln = var_global.spval
        srviln = var_global.spval
        srndln = var_global.spval
        srniln = var_global.spval

    return ssun, ssha, ssoi, ssno, ssno_lyr, parsun, parsha, sabvsun, sabvsha, sabg,  sabg_soil, sabg_snow, fsno, sr, solvd, solvi, solnd, solni, srvd, srvi, srnd, srni, solvdln, solviln, solndln, solniln, srvdln, srviln, srndln, srniln, sabg_snow_lyr