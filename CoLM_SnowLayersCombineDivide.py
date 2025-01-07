# from CoLM_Const_Physical import CoLM_Const_Physical
import numpy as np

def combo(var_global, dz_soisno, wliq_soisno, wice_soisno, t, dz2, wliq2, wice2, t2):
    cpice = var_global.cpice
    cpliq = var_global.cpliq
    hfus  = var_global.hfus
    tfrz = var_global.tfrz
    # Combine the properties of two layers into one
    dzc = dz_soisno + dz2
    wicec = wice_soisno + wice2
    wliqc = wliq_soisno + wliq2
    h = (cpice * wice_soisno + cpliq * wliq_soisno) * (t - tfrz) + hfus * wliq_soisno
    h2 = (cpice * wice2 + cpliq * wliq2) * (t2 - tfrz) + hfus * wliq2
    hc = h + h2

    if hc < 0.0:
        tc = tfrz + hc / (cpice * wicec + cpliq * wliqc)
    elif hc <= hfus * wliqc:
        tc = tfrz
    else:
        tc = tfrz + (hc - hfus * wliqc) / (cpice * wicec + cpliq * wliqc)

    dz_soisno = dzc
    wice_soisno = wicec
    wliq_soisno = wliqc
    t = tc
    return dz_soisno, wliq_soisno, wice_soisno, t
    
def snowcompaction(const_physical, lb, deltim, imelt, fiold, t_soisno, wliq_soisno, wice_soisno, forc_us, forc_vs, dz_soisno):

    # This is the main subroutine to execute the calculation of thermal processes and surface fluxes.
    # Original author: Yongjiu Dai, 09/15/1999; 08/30/2002
    # Four of metamorphisms of changing snow characteristics are implemented,
    # i.e., destructive, overburden, melt and wind drift. The treatments of the destructive compaction
    # was from SNTHERM.89 and SNTHERM.99 (1991, 1999). The contribution due to
    # melt metamorphism is simply taken as a ratio of snow ice fraction after
    # the melting versus before the melting. The treatments of the overburden compaction and the drifting compaction
    # were borrowed from CLM5.0 which based on Vionnet et al. (2012) and van Kampenhout et al (2017).
    #
    #
    # REVISIONS:
    # Hua Yuan, 12/2019: added initial codes for PFT and Plant Community (PC)
    #                    vegetation classification processes
    # Nan Wei,  01/2021: added variables passing of plant hydraulics and precipitation sensible heat
    #                    with canopy and ground for PFT and Plant Community (PC)

    """
    Determine fraction of foliage covered by water and fraction of foliage that is dry and transpiring.
    Original author  : Qinghliang Li, 17/02/2024;
    Supervise author : Jinlong Zhu,   xx/xx/xxxx;
    software         : xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    Args:
        ipatch (INTEGER):  patch index [-]
        lb (INTEGER): lower bound of array [-]
        patchtype (INTEGER):! land water TYPE (0=soil, 1=urban or built-up, 2=wetland, 3=glacier/ice sheet, 4=land water bodies)


        deltim (float): model time step [second]
        trsmx0 (float): max transpiration for moist soil+100% veg.  [mm/s]
        zlnd (float)  : roughness length for soil [m]
        zsno (float)  : roughness length for snow [m]
        deltim (float): drag coefficient for soil under canopy [-]
        trsmx0 (float): maximum dew
        zlnd (float)  : tuning factor to turn first layer T into surface T
        zsno (float)  : Crank Nicholson factor between 0 and 1

        ##! soil physical parameters
        vf_quartz [nl_soil](array):  volumetric fraction of quartz within mineral soil
        vf_gravels[nl_soil](array):  volumetric fraction of gravels
        vf_om     [nl_soil](array):  volumetric fraction of organic matter
        vf_sand   [nl_soil](array):  volumetric fraction of sand
        wf_gravels[nl_soil](array):  gravimetric fraction of gravels
        wf_sand   [nl_soil](array):  gravimetric fraction of sand
        csol      [nl_soil](array):  heat capacity of soil solids [J/(m3 K)]
        porsl     [nl_soil](array):  soil porosity [-]
        psi0      [nl_soil](array):  soil water suction, negative potential [mm]

        theta_r  [nl_soil](array):  VG parameters for soil water retention curve
        alpha_vgm[nl_soil](array):  VG parameters for soil water retention curve
        n_vgm    [nl_soil](array):  VG parameters for soil water retention curve
        L_vgm    [nl_soil](array):  VG parameters for soil water retention curve
        sc_vgm   [nl_soil](array):  VG parameters for soil water retention curve
        fc_vgm   [nl_soil](array):  VG parameters for soil water retention curve

        k_solids  [nl_soil](array):  thermal conductivity of minerals soil [W/m-K]
        dkdry     [nl_soil](array):  thermal conductivity of dry soil [W/m-K]
        dksatu    [nl_soil](array):  thermal conductivity of saturated unfrozen soil [W/m-K]
        dksatf    [nl_soil](array):  thermal conductivity of saturated frozen soil [W/m-K]
        hksati    [nl_soil](array):  hydraulic conductivity at saturation [mm h2o/s]
        BA_alpha  [nl_soil](array):  alpha in Balland and Arp(2005) thermal conductivity scheme
        BA_beta   [nl_soil](array):  beta in Balland and Arp(2005) thermal conductivity scheme

        
        ##!  vegetation parameters
        lai (float):  adjusted leaf area index for seasonal variation [-]
        sai (float):  stem area index  [-]
        htop (float):  canopy crown top height [m]
        hbot (float):  canopy crown bottom height [m]
        sqrtdi (float):  inverse sqrt of leaf dimension [m**-0.5]
        rootfr[nl_soil](array):root fraction

        effcon (float):  quantum efficiency of RuBP regeneration (mol CO2/mol quanta)
        vmax25 (float):  maximum carboxylation rate at 25 C at canopy top
        kmax_sun (float): 
        kmax_sha (float): 
        kmax_xyl (float): 
        kmax_root (float): 
        psi50_sun (float): water potential at 50% loss of sunlit leaf tissue conductance (mmH2O)
        psi50_sha (float): water potential at 50% loss of shaded leaf tissue conductance (mmH2O)
        psi50_xyl (float): water potential at 50% loss of xylem tissue conductance (mmH2O)
        psi50_root (float): ! water potential at 50% loss of root tissue conductance (mmH2O)
        ck (float):  shape-fitting parameter for vulnerability curve (-)
        slti (float):  slope of low temperature inhibition function      [s3]
        hlti (float):  1/2 point of low temperature inhibition function  [s4]
        shti (float):  slope of high temperature inhibition function     [s1]
        hhti (float):  1/2 point of high temperature inhibition function [s2]
        trda (float):  temperature coefficient in gs-a model             [s5]
        trdm (float):  temperature coefficient in gs-a model             [s6]
        trop (float):  temperature coefficient in gs-a model
        gradm (float):  conductance-photosynthesis slope parameter
        binter (float):  conductance-photosynthesis intercept
        extkn (float):  coefficient of leaf nitrogen allocation


        ##! atmospherical variables and observational height
        forc_hgt_u (float):  observational height of wind [m]
        forc_hgt_t (float):  observational height of temperature [m]
        forc_hgt_q (float):  observational height of humidity [m]
        forc_u (float):  wind component in eastward direction [m/s]
        forc_v (float):  wind component in northward direction [m/s]
        forc_t (float):  temperature at agcm reference height [kelvin]
        forc_q (float):  specific humidity at agcm reference height [kg/kg]
        forc_rhoair (float):  density air [kg/m3]
        forc_psrf (float): atmosphere pressure at the surface [pa]
        forc_pco2m (float):  CO2 concentration in atmos. (pascals)
        forc_po2m (float): O2 concentration in atmos. (pascals)
        forc_hpbl (float): atmospheric boundary layer height [m]
        pg_rain (float):  rainfall onto ground including canopy runoff [kg/(m2 s)]
        pg_snow (float):  snowfall onto ground including canopy runoff [kg/(m2 s)]
        t_precip (float):  snowfall/rainfall temperature [kelvin]
        qintr_rain (float):  rainfall interception [(mm h2o/s)]
        qintr_snow (float):  snowfall interception [(mm h2o/s)]

        ##!  radiative fluxes
        coszen (float):  cosine of the solar zenith angle
        parsun (float):  photosynthetic active radiation by sunlit leaves (W m-2)
        parsha (float):  photosynthetic active radiation by shaded leaves (W m-2)
        sabvsun (float):  solar radiation absorbed by vegetation [W/m2]
        sabvsha (float):  solar radiation absorbed by vegetation [W/m2]
        sabg (float):  solar radiation absorbed by ground [W/m2]
        frl (float):  atmospheric infrared (longwave) radiation [W/m2]
        extkb (float):  (k, g(mu)/mu) direct solar extinction coefficient
        extkd (float):  diffuse and scattered diffuse PAR extinction coefficient
        thermk (float):  canopy gap fraction for tir radiation

        ##! state variable (1)
        fsno (float):  fraction of ground covered by snow
        sigf (float):  fraction of veg cover, excluding snow-covered veg [-]
        dz_soisno[lb:nl_soil] (array):  layer thickiness [m]
        z_soisno [lb:nl_soil] (array):  node depth [m]
        zi_soisno[lb-1:nl_soil]  ! interface depth [m]


    Returns:
        fwet (float): fraction of foliage covered by water [-]
        fdry (float): fraction of foliage that is green and dry [-]
    """
    # Local variables
    c1 = 2.777e-7
    c2 = 23.0e-3
    c3 = 2.777e-6
    c4 = 0.04
    c5 = 2.0
    c6 = 5.15e-7
    c7 = 4.0
    dm = 100.0
    eta0 = 9.e5

    burden = 0.0
    zpseudo = 0.0
    mobile = True

    for j in range(lb, -1, -1):
        wx = wice_soisno[j] + wliq_soisno[j]
        void = 1.0 - (wice_soisno[j] / const_physical.denice + wliq_soisno[j] / const_physical.denh2o) / dz_soisno[j]

        # Disallow compaction for water saturated node and lower ice lens node.
        if void <= 0.001 or wice_soisno[j] <= 0.1:
            burden += wx

            # saturated node is immobile
            # This is only needed IF wind_dependent_snow_density is true, but it's
            # simplest just to update mobile always
            mobile = False

            continue

        bi = wice_soisno[j] / dz_soisno[j]
        fi = wice_soisno[j] / wx
        td = const_physical.tfrz - t_soisno[j]

        dexpf = np.exp(-c4 * td)

        # Compaction due to destructive metamorphism
        ddz1 = -c3 * dexpf
        if bi > dm:
            ddz1 *= np.exp(-46.0e-3 * (bi - dm))

        # Liquid water term
        if wliq_soisno[j] > 0.01 * dz_soisno[j]:
            ddz1 *= c5

        # Compaction due to overburden
        f1 = 1.0 / (1.0 + 60.0 * wliq_soisno[j] / (const_physical.denh2o * dz_soisno[j]))
        f2 = 4.0  # currently fixed to maximum value, holds in absence of angular grains
        eta = f1 * f2 * (bi / 450.0) * np.exp(0.1 * td + c2 * bi) * 7.62237e6
        ddz2 = -(burden + wx / 2.0) / eta

        # Compaction occurring during melt
        if imelt[j] == 1:
            ddz3 = -1.0 / deltim * max(0.0, (fiold[j] - fi) / fiold[j])
        else:
            ddz3 = 0.0

        # Compaction occurring due to wind drift
        forc_wind = np.sqrt(forc_us**2 + forc_vs**2)
        zpseudo, mobile, ddz4= winddriftcompaction(bi, forc_wind, dz_soisno[j], zpseudo, mobile, ddz4)

        # Time rate of fractional change in dz (units of s-1)
        pdzdtc = ddz1 + ddz2 + ddz3 + ddz4

        # The change in dz_soisno due to compaction
        dz_soisno[j] *= (1.0 + pdzdtc * deltim)
        dz_soisno[j] = max(dz_soisno[j], (wice_soisno[j] / const_physical.denice + wliq_soisno[j] / const_physical.denh2o))

        # Pressure of overlying snow
        burden += wx
        
    return dz_soisno

def winddriftcompaction(bi, forc_wind, dz, zpseudo, mobile, compaction_rate):
    # --------------------------------------------------------------------------------------------------------------
    # Compute wind drift compaction for a single column and level.
    # ! Also updates zpseudo and mobile for this column. However, zpseudo remains unchanged
    # ! IF mobile is already false or becomes false within this SUBROUTINE.
    # !
    # ! The structure of the updates done here for zpseudo and mobile requires that this
    # ! SUBROUTINE be called first for the top layer of snow, THEN for the 2nd layer down,
    # ! etc. - and finally for the bottom layer. Before beginning the loops over layers,
    # ! mobile should be initialized to .true. and zpseudo should be initialized to 0.
    # --------------------------------------------------------------------------------------------------------------
    rho_min = 50.0    # wind drift compaction / minimum density [kg/m3]
    rho_max = 350.0   # wind drift compaction / maximum density [kg/m3]
    drift_gs = 0.35e-3 # wind drift compaction / grain size (fixed value for now)
    drift_sph = 1.0    # wind drift compaction / sphericity
    tau_ref = 48.0 * 3600.0  # wind drift compaction / reference time [s]

    if mobile:
        Frho = 1.25 - 0.0042 * (max(rho_min, bi) - rho_min)
        # assuming dendricity = 0, sphericity = 1, grain size = 0.35 mm Non-dendritic snow
        MO = 0.34 * (-0.583 * drift_gs - 0.833 * drift_sph + 0.833) + 0.66 * Frho
        SI = -2.868 * np.exp(-0.085 * forc_wind) + 1.0 + MO

        if SI > 0.0:
            SI = min(SI, 3.25)
            # Increase zpseudo (wind drift / pseudo depth) to the middle of
            # the pseudo-node for the sake of the following calculation
            zpseudo = zpseudo + 0.5 * dz * (3.25 - SI)
            gamma_drift = SI * np.exp(-zpseudo / 0.1)
            tau_inverse = gamma_drift / tau_ref
            compaction_rate = -max(0.0, rho_max - bi) * tau_inverse
            # Further increase zpseudo to the bottom of the pseudo-node for
            # the sake of calculations done on the underlying layer (i.e.,
            # the next time through the j loop).
            zpseudo = zpseudo + 0.5 * dz * (3.25 - SI)
        else:  # SI <= 0
            mobile = False
            compaction_rate = 0.0
    else:  # not mobile
        compaction_rate = 0.0
    return zpseudo, mobile, compaction_rate

#=======================================================================
# snowlayerscombine function
# Original author: Yongjiu Dai, September 15, 1999
#
# Checks for elements which are below prescribed minimum for thickness or mass.
# If snow element thickness or mass is less than a prescribed minimum,
# it is combined with neighboring element to be best combine with,
# and executes the combination of mass and energy in clm_combo.f90
#=======================================================================

def snowlayerscombine(var_global, lb, snl, z_soisno, dz_soisno, zi_soisno, wliq_soisno, wice_soisno, t_soisno, scv, snowdp):
    # Local variables
    dzmin = np.array([0.010, 0.015, 0.025, 0.055, 0.115])
    burden = 0.0
    zwice = 0.0
    zwliq = 0.0

    # Check the mass of ice lens of snow, when the total less than a small value,
    # combine it with the underlying neighbor
    msn_old = snl
    for j in range(msn_old + 1, 1, -1):
        if wice_soisno[j] <= 0.1:
            wliq_soisno[j + 1] += wliq_soisno[j]
            wice_soisno[j + 1] += wice_soisno[j]

            # Shift all elements above this down one
            if j > snl + 1 and snl < -1:
                for i in range(j, snl + 3, -1):
                    t_soisno[i] = t_soisno[i - 1]
                    wliq_soisno[i] = wliq_soisno[i - 1]
                    wice_soisno[i] = wice_soisno[i - 1]
                    dz_soisno[i] = dz_soisno[i - 1]

            snl += 1
            # print('one snow layer is gone')

    if snl == 0:
        scv = 0.0
        snowdp = 0.0
        # write(6,*) 'all snow has gone'
        return wliq_soisno, wice_soisno, t_soisno, dz_soisno, zi_soisno, snowdp, scv, snl
    else:
        scv = 0.0
        snowdp = 0.0
        zwice = 0.0
        zwliq = 0.0
        for j in range(snl + 1, 1, -1):
            scv += wice_soisno[j] + wliq_soisno[j]
            snowdp += dz_soisno[j]
            zwice += wice_soisno[j]
            zwliq += wliq_soisno[j]

    # Check the snow depth
    if snowdp < 0.01:  # all snow gone
        snl = 0
        scv = zwice
        if scv <= 0.0:
            snowdp = 0.0

        # The liquid water assumed ponding on soil surface
        wliq_soisno[0] += zwliq
        # write(6,'(17h all snow is gone)')
        return wliq_soisno, wice_soisno, t_soisno, dz_soisno, zi_soisno, snowdp, scv, snl

    else:  # snow layers combined
        # Two or more layers
        if snl < -1:
            msn_old = snl
            mssi = 1
            for i in range(msn_old + 1, 1, -1):
                # If top node is removed, combine with bottom neighbor
                if dz_soisno[i] < dzmin[mssi - 1]:
                    if i == snl + 1:
                        neighbor = i + 1
                    elif i == 0:
                        neighbor = i - 1
                    else:
                        neighbor = i + 1
                        if (dz_soisno[i - 1] + dz_soisno[i]) < (dz_soisno[i + 1] + dz_soisno[i]):
                            neighbor = i - 1

                    # Node l and j are combined and stored as node j
                    if neighbor > i:
                        j = neighbor
                        l = i
                    else:
                        j = i
                        l = neighbor

                    dz_soisno[j], wliq_soisno[j], wice_soisno[j], t_soisno[j] = combo(var_global,
                        dz_soisno[j], wliq_soisno[j], wice_soisno[j], t_soisno[j],
                        dz_soisno[l], wliq_soisno[l], wice_soisno[l], t_soisno[l]
                    )

                    # Now shift all elements above this down one
                    if j - 1 > snl + 1:
                        for k in range(j - 1, snl + 3, -1):
                            t_soisno[k] = t_soisno[k - 1]
                            wice_soisno[k] = wice_soisno[k - 1]
                            wliq_soisno[k] = wliq_soisno[k - 1]
                            dz_soisno[k] = dz_soisno[k - 1]

                    snl += 1

                    # write(6,'(7h Nodes ,i4,4h and,i4,14h combined into,i4)') l,j,j
                    if snl >= -1:
                        break

                else:
                    mssi += 1

        # Reset the node depth and the depth of layer interface
        zi_soisno[0] = 0.0
        for k in range(0, snl, -1):
            z_soisno[k] = zi_soisno[k] - 0.5 * dz_soisno[k]
            zi_soisno[k - 1] = zi_soisno[k] - dz_soisno[k]

    return wliq_soisno, wice_soisno, t_soisno, dz_soisno, zi_soisno, snowdp, scv, snl
    
    
def snowlayersdivide(var_global, lb, snl, z_soisno, dz_soisno, zi_soisno, wliq_soisno, wice_soisno, t_soisno):
    # =======================================================================
    # Original author : Yongjiu Dai, September 15, 1999
    #
    # subdivides snow layer when its thickness exceed the prescribed maximum
    # =======================================================================

    # -------------------------- Dummy argument -----------------------------

    # Local variables
    drr = 0.0       # thickness of the combined [m]
    dzsno = np.zeros(5)  # Snow layer thickness [m]
    swice = np.zeros(5)  # Partial volume of ice [m3/m3]
    swliq = np.zeros(5)  # Partial volume of liquid water [m3/m3]
    tsno = np.zeros(5)   # Nodel temperature [K]

    zwice = 0.0
    zwliq = 0.0
    propor = 0.0

    # -----------------------------------------------------------------------

    msno = abs(snl)
    for k in range(msno):
        dzsno[k] = dz_soisno[k + snl]
        swice[k] = wice_soisno[k + snl]
        swliq[k] = wliq_soisno[k + snl]
        tsno[k] = t_soisno[k + snl]

    if msno == 1:
        if dzsno[0] > 0.03:
            msno = 2
            # Specified a new snow layer
            dzsno[0] = dzsno[0] / 2
            swice[0] = swice[0] / 2
            swliq[0] = swliq[0] / 2

            dzsno[1] = dzsno[0]
            swice[1] = swice[0]
            swliq[1] = swliq[0]
            tsno[1] = tsno[0]
            # write(6,*)'Subdivided Top Node into two layer (1/2)'

    if msno > 1:
        if dzsno[0] > 0.02:
            drr = dzsno[0] - 0.02
            propor = drr / dzsno[0]
            zwice = propor * swice[0]
            zwliq = propor * swliq[0]

            propor = 0.02 / dzsno[0]
            swice[0] = propor * swice[0]
            swliq[0] = propor * swliq[0]
            dzsno[0] = 0.02

            dzsno[1], swliq[1], swice[1], tsno[1] = combo(var_global, dzsno[1], swliq[1], swice[1], tsno[1], drr, zwliq, zwice, tsno[0])

            # write(6,*) 'Subdivided Top Node 20 mm combined into underlying neighbor'

            if msno <= 2 and dzsno[1] > 0.07:
                # subdivided a new layer
                msno = 3
                dzsno[1] = dzsno[1] / 2
                swice[1] = swice[1] / 2
                swliq[1] = swliq[1] / 2

                dzsno[2] = dzsno[1]
                swice[2] = swice[1]
                swliq[2] = swliq[1]
                tsno[2] = tsno[1]

    if msno > 2:
        if dzsno[1] > 0.05:
            drr = dzsno[1] - 0.05
            propor = drr / dzsno[1]
            zwice = propor * swice[1]
            zwliq = propor * swliq[1]

            propor = 0.05 / dzsno[1]
            swice[1] = propor * swice[1]
            swliq[1] = propor * swliq[1]
            dzsno[1] = 0.05

            dzsno[2], swliq[2], swice[2], tsno[2] = combo(var_global, dzsno[2], swliq[2], swice[2], tsno[2], drr, zwliq, zwice, tsno[1])

            # write(6,*)'Subdivided 50 mm from the subsurface layer and combined into underlying neighbor'

            if msno <= 3 and dzsno[2] > 0.18:
                # subdivided a new layer
                msno = 4
                dzsno[2] = dzsno[2] / 2
                swice[2] = swice[2] / 2
                swliq[2] = swliq[2] / 2

                dzsno[3] = dzsno[2]
                swice[3] = swice[2]
                swliq[3] = swliq[2]
                tsno[3] = tsno[2]

    if msno > 3:
        if dzsno[2] > 0.11:
            drr = dzsno[2] - 0.11
            propor = drr / dzsno[2]
            zwice = propor * swice[2]
            zwliq = propor * swliq[2]

            propor = 0.11 / dzsno[2]
            swice[2] = propor * swice[2]
            swliq[2] = propor * swliq[2]
            dzsno[2] = 0.11

            dzsno[3], swliq[3], swice[3], tsno[3] = combo(dzsno[2], swliq[2], swice[2], tsno[2], dzsno[3], swliq[3], swice[3], tsno[3], drr, zwliq, zwice, tsno[2])

            # write(6,*)'Subdivided 110 mm from the third Node and combined into underlying neighbor'

            if msno <= 4 and dzsno[3] > 0.41:
                # subdivided a new layer
                msno = 5
                dzsno[3] = dzsno[3] / 2
                swice[3] = swice[3] / 2
                swliq[3] = swliq[3] / 2

                dzsno[4] = dzsno[3]
                swice[4] = swice[3]
                swliq[4] = swliq[3]
                tsno[4] = tsno[3]

    if msno > 4:
        if dzsno[3] > 0.23:
            drr = dzsno[3] - 0.23
            propor = drr / dzsno[3]
            zwice = propor * swice[3]
            zwliq = propor * swliq[3]

            propor = 0.23 / dzsno[3]
            swice[3] = propor * swice[3]
            swliq[3] = propor * swliq[3]
            dzsno[3] = 0.23

            dzsno[4], swliq[4], swice[4], tsno[4] = combo(var_global, dzsno[4], swliq[4], swice[4], tsno[4], drr, zwliq, zwice, tsno[3])

            # write(6,*)'Subdivided 230 mm from the fourth Node and combined into underlying neighbor'

    snl = -msno

    for k in range(snl + 1,1):
        dz_soisno[k] = dzsno[k - snl]
        wice_soisno[k] = swice[k - snl]
        wliq_soisno[k] = swliq[k - snl]
        t_soisno[k] = tsno[k - snl]

    zi_soisno[0] = 0.0
    for k in range(0, snl, -1):
        z_soisno[k] = zi_soisno[k] - 0.5 * dz_soisno[k]
        zi_soisno[k - 1] = zi_soisno[k] - dz_soisno[k]
        
    return snl, wice_soisno, wliq_soisno, t_soisno, dz_soisno, z_soisno, zi_soisno

def snowlayerscombine_snicar (lb,snl, 
              z_soisno,dz_soisno,zi_soisno,wliq_soisno,wice_soisno,t_soisno,scv,snowdp,
              mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi,
              mss_dst1 , mss_dst2 , mss_dst3 , mss_dst4 ):
    msn_old = snl
    dzmin = [0.010, 0.015, 0.025, 0.055, 0.115]
    for j in range(msn_old + 1, 1):
        if wice_soisno[j] <= 0.1:
            wliq_soisno[j + 1] += wliq_soisno[j]
            wice_soisno[j + 1] += wice_soisno[j]

            # Aerosol Fluxes (January 07, 2023)
            if j < 0:  # 01/11/2023, yuan: add j < 0
                mss_bcphi[j + 1] += mss_bcphi[j]
                mss_bcpho[j + 1] += mss_bcpho[j]
                mss_ocphi[j + 1] += mss_ocphi[j]
                mss_ocpho[j + 1] += mss_ocpho[j]
                mss_dst1[j + 1] += mss_dst1[j]
                mss_dst2[j + 1] += mss_dst2[j]
                mss_dst3[j + 1] += mss_dst3[j]
                mss_dst4[j + 1] += mss_dst4[j]

            # shift all elements above this down one.
            if j > snl + 1 and snl < -1:
                for i in range(j, snl + 1, -1):
                    t_soisno[i] = t_soisno[i - 1]
                    wliq_soisno[i] = wliq_soisno[i - 1]
                    wice_soisno[i] = wice_soisno[i - 1]
                    dz_soisno[i] = dz_soisno[i - 1]

                    # Aerosol Fluxes (January 07, 2023)
                    mss_bcphi[i] = mss_bcphi[i - 1]
                    mss_bcpho[i] = mss_bcpho[i - 1]
                    mss_ocphi[i] = mss_ocphi[i - 1]
                    mss_ocpho[i] = mss_ocpho[i - 1]
                    mss_dst1[i] = mss_dst1[i - 1]
                    mss_dst2[i] = mss_dst2[i - 1]
                    mss_dst3[i] = mss_dst3[i - 1]
                    mss_dst4[i] = mss_dst4[i - 1]

            snl += 1
            # write(6,*) 'one snow layer is gone'

    if snl == 0:
        scv = 0.0
        snowdp = 0.0

        # Aerosol Fluxes (January 07, 2023)
        mss_bcphi[:] = 0.0
        mss_bcpho[:] = 0.0
        mss_ocphi[:] = 0.0
        mss_ocpho[:] = 0.0
        mss_dst1[:] = 0.0
        mss_dst2[:] = 0.0
        mss_dst3[:] = 0.0
        mss_dst4[:] = 0.0

        # write(6,*) 'all snow has gone'
        return
    else:
        scv = 0.0
        snowdp = 0.0
        zwice = 0.0
        zwliq = 0.0
        for j in range(snl + 1, 1):
            scv += wice_soisno[j] + wliq_soisno[j]
            snowdp += dz_soisno[j]
            zwliq += wliq_soisno[j]
            zwice += wliq_soisno[j]

    if snowdp < 0.01:
        snl = 0
        scv = zwice
        if scv <= 0.0:
            snowdp = 0.0
        mss_bcphi[:] = 0.0
        mss_bcpho[:] = 0.0
        mss_ocphi[:] = 0.0
        mss_ocpho[:] = 0.0
        mss_dst1[:] = 0.0
        mss_dst2[:] = 0.0
        mss_dst3[:] = 0.0
        mss_dst4[:] = 0.0

        # the liquid water assumed ponding on soil surface
        wliq_soisno[0] += zwliq
        # write(6,'(17h all snow is gone)')
        return
    else:
        if snl < -1:
            msn_old = snl
            mssi = 1
            for i in range(msn_old + 1, 1):  # Fortran DO i = msn_old+1, 0 translates to Python range(msn_old + 1, 1)

                if dz_soisno[i] < dzmin[mssi]:
                    if i == snl + 1:
                        neibor = i + 1
                    elif i == 0:
                        neibor = i - 1
                    else:
                        neibor = i + 1
                        if (dz_soisno[i - 1] + dz_soisno[i]) < (dz_soisno[i + 1] + dz_soisno[i]):
                            neibor = i - 1

                    if neibor > i:
                        j = neibor
                        l = i
                    else:
                        j = i
                        l = neibor

                    dz_soisno[j], wliq_soisno[j], wice_soisno[j], t_soisno[j] = combo(
                        dz_soisno[j], wliq_soisno[j], wice_soisno[j], t_soisno[j],
                        dz_soisno[l], wliq_soisno[l], wice_soisno[l], t_soisno[l]
                    )

                    # Aerosol Fluxes (January 07, 2023)
                    mss_bcphi[j] += mss_bcphi[l]
                    mss_bcpho[j] += mss_bcpho[l]
                    mss_ocphi[j] += mss_ocphi[l]
                    mss_ocpho[j] += mss_ocpho[l]
                    mss_dst1[j] += mss_dst1[l]
                    mss_dst2[j] += mss_dst2[l]
                    mss_dst3[j] += mss_dst3[l]
                    mss_dst4[j] += mss_dst4[l]

                    # Now shift all elements above this down one.
                    if j - 1 > snl + 1:
                        for k in range(j - 1, snl + 1, -1):
                            t_soisno[k] = t_soisno[k - 1]
                            wice_soisno[k] = wice_soisno[k - 1]
                            wliq_soisno[k] = wliq_soisno[k - 1]
                            dz_soisno[k] = dz_soisno[k - 1]

                            # Aerosol Fluxes (January 07, 2023)
                            mss_bcphi[k] = mss_bcphi[k - 1]
                            mss_bcpho[k] = mss_bcpho[k - 1]
                            mss_ocphi[k] = mss_ocphi[k - 1]
                            mss_ocpho[k] = mss_ocpho[k - 1]
                            mss_dst1[k] = mss_dst1[k - 1]
                            mss_dst2[k] = mss_dst2[k - 1]
                            mss_dst3[k] = mss_dst3[k - 1]
                            mss_dst4[k] = mss_dst4[k - 1]

                    snl += 1

                    if snl >= -1:
                        break

                else:
                    mssi += 1

    zi_soisno[0] = 0.0

    # Iterate over the range and update z_soisno and zi_soisno
    for k in range(0, snl, -1):  # range in Python is exclusive of the end, so snl+1 becomes snl+2
        z_soisno[k] = zi_soisno[k] - 0.5 * dz_soisno[k]
        zi_soisno[k - 1] = zi_soisno[k] - dz_soisno[k]
    return wice_soisno, wliq_soisno, t_soisno, dz_soisno, z_soisno, zi_soisno, snowdp, scv, snl,mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi, mss_dst1 , mss_dst2 , mss_dst3 , mss_dst4

def snowlayersdivide_snicar(var_global, lb,snl,z_soisno,dz_soisno,zi_soisno,
                                       wliq_soisno,wice_soisno,t_soisno,
                                       mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi,
                                       mss_dst1 , mss_dst2 , mss_dst3 , mss_dst4):
    
    dzsno = np.zeros(5)
    swice = np.zeros(5)
    swliq = np.zeros(5)
    tsno = np.zeros(5)
    mss_aerosol = np.zeros(lb)

    msno = abs(snl)

    for k in range(0, msno):
        dzsno[k] = dz_soisno[k + snl]
        swice[k] = wice_soisno[k + snl]
        swliq[k] = wliq_soisno[k + snl]
        tsno[k] = t_soisno[k + snl]

        # Aerosol Fluxes (January 07, 2023)
        mss_aerosol[k, 0] = mss_bcphi[k + snl]
        mss_aerosol[k, 1] = mss_bcpho[k + snl]
        mss_aerosol[k, 2] = mss_ocphi[k + snl]
        mss_aerosol[k, 3] = mss_ocpho[k + snl]
        mss_aerosol[k, 4] = mss_dst1[k + snl]
        mss_aerosol[k, 5] = mss_dst2[k + snl]
        mss_aerosol[k, 6] = mss_dst3[k + snl]
        mss_aerosol[k, 7] = mss_dst4[k + snl]

    if msno == 1:
        if dzsno[0] > 0.03:
            msno = 2
            # Specified a new snow layer
            dzsno[0] /= 2
            swice[0] /= 2
            swliq[0] /= 2

            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[0, :] /= 2

            dzsno[1] = dzsno[0]
            swice[1] = swice[0]
            swliq[1] = swliq[0]

            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[1, :] = mss_aerosol[0, :]

            tsno[1] = tsno[0]

            # print('Subdivided Top Node into two layer (1/2)')
    if msno > 1:
        if dzsno[0] > 0.02:
            drr = dzsno[0] - 0.02
            propor = drr / dzsno[0]
            zwice = propor * swice[0]
            zwliq = propor * swliq[0]
            
            # Aerosol Fluxes (January 07, 2023)
            z_mss_aerosol = propor * mss_aerosol[0, :]

            propor = 0.02 / dzsno[0]
            swice[0] = propor * swice[0]
            swliq[0] = propor * swliq[0]
            
            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[0, :] = propor * mss_aerosol[0, :]

            dzsno[0] = 0.02

            # Assuming `combo` is a function defined elsewhere
            dzsno[1], swliq[1], swice[1], tsno[1] = combo(var_global, dzsno[1], swliq[1], swice[1], tsno[1], drr, zwliq, zwice, tsno[0])

            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[1, :] = z_mss_aerosol + mss_aerosol[1, :]

            # Print statement commented out
            # print('Subdivided Top Node 20 mm combined into underlying neighbor')

            if msno <= 2 and dzsno[1] > 0.07:
                # Subdivided a new layer
                msno = 3
                dzsno[1] /= 2
                swice[1] /= 2
                swliq[1] /= 2
                
                # Aerosol Fluxes (January 07, 2023)
                mss_aerosol[2, :] /= 2

                dzsno[2] = dzsno[1]
                swice[2] = swice[1]
                swliq[2] = swliq[1]
                
                # Aerosol Fluxes (January 07, 2023)
                mss_aerosol[2, :] = mss_aerosol[1, :]

                tsno[2] = tsno[1]

    if msno > 2:
        if dzsno[1] > 0.05:
            drr = dzsno[1] - 0.05
            propor = drr / dzsno[1]
            zwice = propor * swice[1]
            zwliq = propor * swliq[1]
            
            # Aerosol Fluxes (January 07, 2023)
            z_mss_aerosol = propor * mss_aerosol[1, :]

            propor = 0.05 / dzsno[1]
            swice[1] = propor * swice[1]
            swliq[1] = propor * swliq[1]
            
            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[1, :] = propor * mss_aerosol[1, :]

            dzsno[1] = 0.05

            # Assuming `combo` is a function defined elsewhere
            dzsno[2], swliq[2], swice[2], tsno[2] = combo(var_global, dzsno[2], swliq[2], swice[2], tsno[2], drr, zwliq, zwice, tsno[1])

            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[2, :] = z_mss_aerosol + mss_aerosol[2, :]

            # Print statement commented out
            # print('Subdivided 50 mm from the subsurface layer and combined into underlying neighbor')

            if msno <= 3 and dzsno[2] > 0.18:
                # Subdivided a new layer
                msno = 4
                dzsno[2] /= 2
                swice[2] /= 2
                swliq[2] /= 2
                
                # Aerosol Fluxes (January 07, 2023)
                mss_aerosol[3, :] /= 2

                dzsno[3] = dzsno[2]
                swice[3] = swice[2]
                swliq[3] = swliq[2]
                
                # Aerosol Fluxes (January 07, 2023)
                mss_aerosol[3, :] = mss_aerosol[2, :]

                tsno[3] = tsno[2]
    if msno > 3:
        if dzsno[2] > 0.11:
            drr = dzsno[2] - 0.11
            propor = drr / dzsno[2]
            zwice = propor * swice[2]
            zwliq = propor * swliq[2]
            
            # Aerosol Fluxes (January 07, 2023)
            z_mss_aerosol = propor * mss_aerosol[2, :]

            propor = 0.11 / dzsno[2]
            swice[2] = propor * swice[2]
            swliq[2] = propor * swliq[2]
            
            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[2, :] = propor * mss_aerosol[2, :]

            dzsno[2] = 0.11

            # Assuming `combo` is a function defined elsewhere
            dzsno[3], swliq[3], swice[3], tsno[3] = combo(var_global, dzsno[3], swliq[3], swice[3], tsno[3], drr, zwliq, zwice, tsno[2])

            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[3, :] = z_mss_aerosol + mss_aerosol[3, :]

            # Print statement commented out
            # print('Subdivided 110 mm from the third node and combined into underlying neighbor')

            if msno <= 4 and dzsno[3] > 0.41:
                # Subdivided a new layer
                msno = 5
                dzsno[3] /= 2
                swice[3] /= 2
                swliq[3] /= 2
                
                # Aerosol Fluxes (January 07, 2023)
                mss_aerosol[3, :] /= 2

                dzsno[4] = dzsno[3]
                swice[4] = swice[3]
                swliq[4] = swliq[3]
                
                # Aerosol Fluxes (January 07, 2023)
                mss_aerosol[4, :] = mss_aerosol[3, :]

                tsno[4] = tsno[3]
    
    if msno > 4:
        if dzsno[3] > 0.23:
            drr = dzsno[3] - 0.23
            propor = drr / dzsno[3]
            zwice = propor * swice[3]
            zwliq = propor * swliq[3]
            
            # Aerosol Fluxes (January 07, 2023)
            z_mss_aerosol = propor * mss_aerosol[3, :]

            propor = 0.23 / dzsno[3]
            swice[3] = propor * swice[3]
            swliq[3] = propor * swliq[3]
            
            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[3, :] = propor * mss_aerosol[3, :]

            dzsno[3] = 0.23

            # Assuming `combo` is a function defined elsewhere
            dzsno[4], swliq[4], swice[4], tsno[4] = combo(var_global, dzsno[4], swliq[4], swice[4], tsno[4], drr, zwliq, zwice, tsno[4])

            # Aerosol Fluxes (January 07, 2023)
            mss_aerosol[4, :] = z_mss_aerosol + mss_aerosol[4, :]

            # Print statement commented out
            # print('Subdivided 230 mm from the fourth node and combined into underlying neighbor')

    # Update snl value
    snl = -msno

    # Copy and update arrays
    for k in range(snl + 1, 1):
        dz_soisno[k] = dzsno[k - snl]
        wice_soisno[k] = swice[k - snl]
        wliq_soisno[k] = swliq[k - snl]
        
        # Aerosol Fluxes (January 07, 2023)
        mss_bcphi[k] = mss_aerosol[k - snl, 0]
        mss_bcpho[k] = mss_aerosol[k - snl, 1]
        mss_ocphi[k] = mss_aerosol[k - snl, 2]
        mss_ocpho[k] = mss_aerosol[k - snl, 3]
        mss_dst1[k] = mss_aerosol[k - snl, 4]
        mss_dst2[k] = mss_aerosol[k - snl, 5]
        mss_dst3[k] = mss_aerosol[k - snl, 6]
        mss_dst4[k] = mss_aerosol[k - snl, 7]

        t_soisno[k] = tsno[k - snl]

    # Initialize zi_soisno and compute z_soisno
    zi_soisno[0] = 0.0
    for k in range(0, snl+1, -1):
        z_soisno[k] = zi_soisno[k] - 0.5 * dz_soisno[k]
        zi_soisno[k - 1] = zi_soisno[k] - dz_soisno[k]
    return wice_soisno, wliq_soisno, t_soisno, dz_soisno, z_soisno, zi_soisno, snl













