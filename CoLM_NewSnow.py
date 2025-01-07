import numpy as np

def newsnow(nl_colm, const_physical, patchtype, maxsnl, deltim, t_grnd, pg_rain, pg_snow, bifall, 
            t_precip, zi_soisno, z_soisno, dz_soisno, t_soisno, 
            wliq_soisno, wice_soisno, fiold, snl, sag, scv, snowdp, fsno, wetwat=None):
    """
    Add new snow nodes.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         : xxxxxxxxxxxxxxxxxxxxxxxxxxxx

    Args:
        patchtype       (int): land water type (0=soil, 1=urban and built-up, 2=wetland, 3=land ice, 4=land water bodies, 99=ocean)
        maxsnl          (int): maximum number of snow layers
        deltim          (float): model time step [second]
        t_grnd          (float): ground surface temperature [k]
        pg_rain         (float): rainfall onto ground including canopy runoff [kg/(m2 s)]
        pg_snow         (float): snowfall onto ground including canopy runoff [kg/(m2 s)]
        bifall          (float): bulk density of newly fallen dry snow [kg/m3]
        t_precip        (float): snowfall/rainfall temperature [kelvin]
        zi_soisno       (ndarray): interface level below a "z" level (m)
        z_soisno        (ndarray): layer depth (m)
        dz_soisno       (ndarray): layer thickness (m)
        t_soisno        (ndarray): soil + snow layer temperature [K]
        wliq_soisno     (ndarray): liquid water (kg/m2)
        wice_soisno     (ndarray): ice lens (kg/m2)
        fiold           (ndarray): fraction of ice relative to the total water
        snl             (int): number of snow layers
        sag             (float): non dimensional snow age [-]
        scv             (float): snow mass (kg/m2)
        snowdp          (float): snow depth (m)
        fsno            (float): fraction of soil covered by snow [-]

    Returns:
        None
    """
    newnode = 0

    dz_snowf = pg_snow / bifall
    snowdp = snowdp + dz_snowf * deltim
    scv = scv + pg_snow * deltim

    if patchtype == 2 and t_grnd > const_physical.tfrz:
        if wetwat is not None and nl_colm['DEF_USE_VariablySaturatedFlow']:
            wetwat += scv
        scv = 0.
        snowdp = 0.
        sag = 0.
        fsno = 0.

    zi_soisno[5] = 0.

    if snl == 0 and pg_snow > 0.0 and snowdp >= 0.01:
        snl = -1
        newnode = 1
        dz_soisno[4] = snowdp
        z_soisno[4] = -0.5 * dz_soisno[4]
        zi_soisno[4] = -dz_soisno[4]

        sag = 0.
        t_soisno[4] = min(const_physical.tfrz, t_precip)
        wice_soisno[4] = scv
        wliq_soisno[4] = 0.
        fiold[4] = 1.
        fsno = min(1., np.tanh(0.1 * pg_snow * deltim))

    if snl < 0 and newnode == 0:
        lb = snl + 1

        wice_soisno[-maxsnl+lb-1] = wice_soisno[-maxsnl+lb-1] + deltim * pg_snow
        dz_soisno[-maxsnl+lb-1] = dz_soisno[-maxsnl+lb-1] + dz_snowf * deltim
        z_soisno[-maxsnl+lb-1] = zi_soisno[-maxsnl+lb] - 0.5 * dz_soisno[-maxsnl+lb-1]
        zi_soisno[-maxsnl+lb - 1] = zi_soisno[-maxsnl+lb] - dz_soisno[-maxsnl+lb-1]

        fsno = 1. - (1. - np.tanh(0.1 * pg_snow * deltim)) * (1. - fsno)
        fsno = min(1., fsno)
    return zi_soisno, z_soisno, dz_soisno, t_soisno, wliq_soisno, wice_soisno, fiold, snl, sag, scv, snowdp, fsno, wetwat