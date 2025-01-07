import numpy as np
from CoLM_Utils import tridia

def plc(x, psi50, ck):
    """
    DESCRIPTION:
    Return value of vulnerability curve at x

    ARGUMENTS:
    x : float
        Water potential input
    psi50 : float
        Water potential at 50% loss of sunlit leaf tissue conductance (mmH2O)
    ck : float
        Shape-fitting parameter for vulnerability curve (-)

    RETURNS:
    plc : float
        Attenuated conductance [0:1] 0=no flow
    """
    # LOCAL VARIABLES
    tmp = max(-(x / psi50) ** ck, -500.0)
    plc = 2.0 ** tmp
    if plc < 0.00001:
        plc = 1.0e-5
    return plc


def d1plc(x, psi50, ck):
    """
    DESCRIPTION:
    Return 1st derivative of vulnerability curve at x

    ARGUMENTS:
    x : float
        Water potential input
    psi50 : float
        Water potential at 50% loss of tissue conductance (mmH2O)
    ck : float
        Parameter ck

    RETURNS:
    d1plc : float
        First derivative of plc curve at x
    """
    tmp = max(-(x / psi50) ** ck, -500.0)
    return ck * np.log(2.0) * (2.0 ** tmp) * tmp / x

def getqflx_qflx2gs_twoleaf(gb_mol, gs_mol_sun, gs_mol_sha, qflx_sun, qflx_sha, qsatl, qaf, qg, qm,
                            rhoair, psrf, laisun, laisha, sai, fwet, tl, rss, raw, rd, co_lm):
    if qflx_sun > 0 or qflx_sha > 0:
        tprcor = 44.6 * 273.16 * psrf / 1.013e5
        cf = tprcor / tl * 1.e6  # gb->gbmol conversion factor

        delta = 0.0
        if qsatl - qaf > 0:
            delta = 1.0

        caw = 1.0 / raw
        if qg < qaf:
            cgw = 1.0 / rd
        else:
            if co_lm['DEF_RSS_SCHEME'] == 4:
                cgw = rss / rd
            else:
                cgw = 1.0 / (rd + rss)

        cwet = (1.0 - delta * (1.0 - fwet)) * (laisun + laisha + sai) * gb_mol / cf
        cqi_wet = caw + cgw + cwet
        cqi_leaf = caw * (qsatl - qm) + cgw * (qsatl - qg)

        # Solve equations:
        # A1 * csunw_dry + B1 * cfshaw_dry = C1
        # A2 * csunw_dry + B2 * cfshaw_dry = C2

        A1 = cqi_leaf - qflx_sun / rhoair
        B1 = -qflx_sun / rhoair
        C1 = qflx_sun * cqi_wet / rhoair
        A2 = -qflx_sha / rhoair
        B2 = cqi_leaf - qflx_sha / rhoair
        C2 = qflx_sha * cqi_wet / rhoair

        csunw_dry = (B1 * C2 - B2 * C1) / (B1 * A2 - B2 * A1)
        cshaw_dry = (A1 * C2 - A2 * C1) / (A1 * B2 - B1 * A2)

        if qflx_sun > 0:
            gs_mol_sun = 1.0 / ((1.0 - fwet) * delta * laisun / csunw_dry / cf - 1.0 / gb_mol)
        if qflx_sha > 0:
            gs_mol_sha = 1.0 / ((1.0 - fwet) * delta * laisha / cshaw_dry / cf - 1.0 / gb_mol)

    return gs_mol_sun, gs_mol_sha, qflx_sun, qflx_sha
def spacAF_twoleaf(x, nvegwcs, nl_soil, qflx_sun, qflx_sha, laisun, laisha, sai, htop, qeroot, dqeroot,
                   kmax_sun, kmax_sha, kmax_xyl, kmax_root, psi50_sun, psi50_sha, psi50_xyl, psi50_root, ck):
    # Initialize local variables
    #Returns invA, the inverse matrix relating delta(vegwp) to f
   # !   d(vegwp)=invA*f
   # !   evaluated at vegwp(p)
   # ! The methodology is currently hardcoded for linear algebra assuming the
   # ! number of vegetation segments is four. Thus the matrix A and it's inverse
   # ! invA are both 4x4 matrices. A more general method could be done using for
   # ! example a LINPACK linear algebra solver.

    f = np.zeros(nvegwcs)
    dx = np.zeros(nvegwcs)
    tol_lai = 1.e-7  # Minimum lai WHERE transpiration is calculated
    leafsun = 1
    leafsha = 2
    xyl = 3
    root = 4

    grav1 = htop * 1000.0

    # Compute conductance attenuation for each segment
    fsto1 = plc(x[leafsun - 1], psi50_sun, ck)
    fsto2 = plc(x[leafsha - 1], psi50_sha, ck)
    fx = plc(x[xyl - 1], psi50_xyl, ck)
    fr = plc(x[root - 1], psi50_root, ck)

    # Compute the first derivatives of conductance attenuation for each segment
    dfsto1 = d1plc(x[leafsun - 1], psi50_sun, ck)
    dfsto2 = d1plc(x[leafsha - 1], psi50_sha, ck)
    dfx = d1plc(x[xyl - 1], psi50_xyl, ck)
    dfr = d1plc(x[root - 1], psi50_root, ck)

    # Compute matrix elements
    A11 = - laisun * kmax_sun * fx - qflx_sun * dfsto1
    A13 = laisun * kmax_sun * dfx * (x[xyl - 1] - x[leafsun - 1]) + laisun * kmax_sun * fx
    A22 = - laisha * kmax_sha * fx - qflx_sha * dfsto2
    A23 = laisha * kmax_sha * dfx * (x[xyl - 1] - x[leafsha - 1]) + laisha * kmax_sha * fx
    A31 = laisun * kmax_sun * fx
    A32 = laisha * kmax_sha * fx
    A33 = - laisun * kmax_sun * dfx * (x[xyl - 1] - x[leafsun - 1]) - laisun * kmax_sun * fx \
          - laisha * kmax_sha * dfx * (x[xyl - 1] - x[leafsha - 1]) - laisha * kmax_sha * fx \
          - sai * kmax_xyl / htop * fr
    A34 = sai * kmax_xyl / htop * dfr * (x[root - 1] - x[xyl - 1] - grav1) + sai * kmax_xyl / htop * fr
    A43 = sai * kmax_xyl / htop * fr
    A44 = - sai * kmax_xyl / htop * fr - sai * kmax_xyl / htop * dfr * (x[root - 1] - x[xyl - 1] - grav1) + dqeroot

    # Compute flux divergence across each plant segment
    f[leafsun - 1] = qflx_sun * fsto1 - laisun * kmax_sun * fx * (x[xyl - 1] - x[leafsun - 1])
    f[leafsha - 1] = qflx_sha * fsto2 - laisha * kmax_sha * fx * (x[xyl - 1] - x[leafsha - 1])
    f[xyl - 1] = laisun * kmax_sun * fx * (x[xyl - 1] - x[leafsun - 1]) + laisha * kmax_sha * fx * (
                x[xyl - 1] - x[leafsha - 1]) \
                 - sai * kmax_xyl / htop * fr * (x[root - 1] - x[xyl - 1] - grav1)
    f[root - 1] = sai * kmax_xyl / htop * fr * (x[root - 1] - x[xyl - 1] - grav1) - qeroot

    if qflx_sha > 0:
        determ = A44 * A22 * A33 * A11 - A44 * A22 * A31 * A13 - A44 * A32 * A23 * A11 - A43 * A11 * A22 * A34

        if determ != 0:
            dx[leafsun - 1] = ((A22 * A33 * A44 - A22 * A34 * A43 - A23 * A32 * A44) * f[
                leafsun - 1] + A13 * A32 * A44 * f[leafsha - 1] \
                               - A13 * A22 * A44 * f[xyl - 1] + A13 * A22 * A34 * f[root - 1]) / determ
            dx[leafsha - 1] = (A23 * A31 * A44 * f[leafsun - 1] + (
                        A11 * A33 * A44 - A11 * A34 * A43 - A13 * A31 * A44) * f[leafsha - 1] \
                               - A11 * A23 * A44 * f[xyl - 1] + A11 * A23 * A34 * f[root - 1]) / determ
            dx[xyl - 1] = (-A22 * A31 * A44 * f[leafsun - 1] - A11 * A32 * A44 * f[leafsha - 1] \
                           + A11 * A22 * A44 * f[xyl - 1] - A11 * A22 * A34 * f[root - 1]) / determ
            dx[root - 1] = (A22 * A31 * A43 * f[leafsun - 1] + A11 * A32 * A43 * f[leafsha - 1] \
                            - A11 * A22 * A43 * f[xyl - 1] + (A11 * A22 * A33 - A11 * A23 * A32 - A13 * A22 * A31) * f[
                                root - 1]) / determ
        else:
            dx = np.zeros(nvegwcs)
    else:
        A33 = - laisun * kmax_sun * dfx * (
                    x[xyl - 1] - x[leafsun - 1]) - laisun * kmax_sun * fx - sai * kmax_xyl / htop * fr
        f[xyl - 1] = laisun * kmax_sun * fx * (x[xyl - 1] - x[leafsun - 1]) - sai * kmax_xyl / htop * fr * (
                    x[root - 1] - x[xyl - 1] - grav1)
        determ = A11 * A33 * A44 - A34 * A11 * A43 - A13 * A31 * A44
        if determ != 0:
            dx[leafsun - 1] = (- A13 * A44 * f[xyl - 1] + A13 * A34 * f[root - 1] + (A33 * A44 - A34 * A43) * f[
                leafsun - 1]) / determ
            dx[xyl - 1] = (A11 * A44 * f[xyl - 1] - A11 * A34 * f[root - 1] - A31 * A44 * f[leafsun - 1]) / determ
            dx[root - 1] = (- A11 * A43 * f[xyl - 1] + (A11 * A33 - A13 * A31) * f[root - 1] + A31 * A43 * f[
                leafsun - 1]) / determ

            dx[leafsha - 1] = x[leafsun - 1] - x[leafsha - 1] + dx[leafsun - 1]
        else:
            dx = np.zeros(nvegwcs)

    return dx

def getrootqflx_x2qe(nl_soil, smp, x_root_top, z_soisno, krad, kax):
    amx_hr = np.zeros(nl_soil - 1, dtype=np.float64)  # "a" left off diagonal of tridiagonal matrix
    bmx_hr = np.zeros(nl_soil - 1, dtype=np.float64)  # "b" diagonal column for tridiagonal matrix
    cmx_hr = np.zeros(nl_soil - 1, dtype=np.float64)  # "c" right off diagonal tridiagonal matrix
    rmx_hr = np.zeros(nl_soil - 1, dtype=np.float64)  # "r" forcing term of tridiagonal matrix
    drmx_hr = np.zeros(nl_soil - 1, dtype=np.float64)  # "dr" forcing term of tridiagonal matrix for d/dxroot(1)
    x = np.zeros(nl_soil - 1, dtype=np.float64)  # root water potential from layer 2 to nl_soil
    dx = np.zeros(nl_soil - 1,
                  dtype=np.float64)  # derivative of root water potential from layer 2 to nl_soil (dxroot(:)/dxroot(1))
    xroot = np.zeros(nl_soil, dtype=np.float64)  # root water potential from layer 2 to nl_soil
    zmm = np.zeros(nl_soil, dtype=np.float64)  # layer depth [mm]
    qeroot_nl = np.zeros(nl_soil, dtype=np.float64)  # root water potential from layer 2 to nl_soil

    zmm = [z * 1000.0 for z in z_soisno]  # Convert depth to mm
    xroot[0] = x_root_top + zmm[0]

    # For the 2nd soil layer
    j = 2
    den1 = zmm[1] - zmm[0]
    den2 = zmm[2] - zmm[1]
    amx_hr[j - 1] = 0
    bmx_hr[j - 1] = kax[j - 1] / den1 + kax[j] / den2 + krad[j]
    cmx_hr[j - 1] = -kax[j] / den2
    rmx_hr[j - 1] = krad[j] * smp[j] + kax[j - 1] - kax[j] + kax[j - 1] / den1 * xroot[0]
    drmx_hr[j - 1] = kax[j - 1] / den1

    # For the middle soil layers
    for j in range(2, nl_soil):
        den1 = zmm[j] - zmm[j - 1]
        den2 = zmm[j + 1] - zmm[j]
        amx_hr[j - 1] = -kax[j - 1] / den1
        bmx_hr[j - 1] = kax[j - 1] / den1 + kax[j] / den2 + krad[j]
        cmx_hr[j - 1] = -kax[j] / den2
        rmx_hr[j - 1] = krad[j] * smp[j] + kax[j - 1] - kax[j]
        drmx_hr[j - 1] = 0.0  # equivalent to 0._r8 in Fortran

    # For the bottom soil layer
    j = nl_soil
    den_AHR = zmm[j] - zmm[j - 1]
    amx_hr[j - 1] = -kax[j - 1] / den_AHR
    bmx_hr[j - 1] = kax[j - 1] / den_AHR + krad[j]
    cmx_hr[j - 1] = 0
    rmx_hr[j - 1] = krad[j] * smp[j] + kax[j - 1]
    drmx_hr[j - 1] = 0.0  # equivalent to 0._r8 in Fortran

    # Solve for root pressure potential using tridiagonal matrix solver
    x = tridia(nl_soil - 1, amx_hr, bmx_hr, cmx_hr, rmx_hr)

    for j in range(2, nl_soil + 1):
        xroot[j] = x[j - 2]

    # Solve the tridiagonal system of equations dx(:)/dxroot(1) = A^-1 * dr
    dx = tridia(nl_soil - 1, amx_hr, bmx_hr, cmx_hr, drmx_hr)

    # Calculate dxroot2
    dxroot2 = dx[0]

    # Calculate the water flux
    j = 0
    den2 = zmm[j + 1] - zmm[j]
    qeroot = krad[j] * (smp[j] - xroot[j]) + (xroot[j + 1] - xroot[j]) * kax[j] / den2 - kax[j]

    # Calculate dqeroot/dx_root_top
    dqeroot = - krad[j] + (dxroot2 - 1) * kax[j] / den2

    # Calculate qeroot_nl for all soil layers
    qeroot_nl = [krad[j] * (smp[j] - xroot[j]) for j in range(nl_soil)]

    return qeroot, dqeroot

def getqflx_gs2qflx_twoleaf(co_lm, gb_mol, gs_mol_sun, gs_mol_sha, qsatl, qaf,
                             rhoair, psrf, laisun, laisha, sai, fwet, tl, rss, raw, rd, qg, qm,
                             rstfacsun=None, rstfacsha=None):
    tprcor = 44.6 * 273.16 * psrf / 1.013e5
    cf = tprcor / tl * 1.e6  # gb->gbmol conversion factor

    delta = 0.0
    if qsatl - qaf > 0.0:
        delta = 1.0

    caw = 1.0 / raw
    if qg < qaf:
        cgw = 1.0 / rd
    else:
        if co_lm['DEF_RSS_SCHEME == 4']:
            cgw = rss / rd
        else:
            cgw = 1.0 / (rd + rss)

    cfw = (1.0 - delta * (1.0 - fwet)) * (laisun + laisha + sai) * gb_mol / cf + \
          (1.0 - fwet) * delta * (laisun / (1.0 / gb_mol + 1.0 / gs_mol_sun) / cf +
                                  laisha / (1.0 / gb_mol + 1.0 / gs_mol_sha) / cf)
    wtsqi = 1.0 / (caw + cgw + cfw)

    wtaq0 = caw * wtsqi
    wtgq0 = cgw * wtsqi
    wtlq0 = cfw * wtsqi

    qflx_sun = rhoair * (1.0 - fwet) * delta * \
               laisun / (1.0 / gb_mol + 1.0 / gs_mol_sun) / cf * \
               ((wtaq0 + wtgq0) * qsatl - wtaq0 * qm - wtgq0 * qg)
    if rstfacsun is not None:
        if rstfacsun <= 1.e-2:
            qflx_sun = 0.0

    qflx_sha = rhoair * (1.0 - fwet) * delta * \
               laisha / (1.0 / gb_mol + 1.0 / gs_mol_sha) / cf * \
               ((wtaq0 + wtgq0) * qsatl - wtaq0 * qm - wtgq0 * qg)
    if rstfacsha is not None:
        if rstfacsha <= 1.e-2:
            qflx_sha = 0.0

    return gs_mol_sun, gs_mol_sha, qflx_sun, qflx_sha

def getrootqflx_qe2x(nl_soil, smp, z_soisno, krad, kax, qeroot):
    zmm = [z * 1000.0 for z in z_soisno]  # layer depth in mm

    amx_hr = [0.0] * nl_soil
    bmx_hr = [0.0] * nl_soil
    cmx_hr = [0.0] * nl_soil
    rmx_hr = [0.0] * nl_soil
    x = [0.0] * nl_soil

    # For the top soil layer
    den2 = zmm[1] - zmm[0]
    amx_hr[0] = 0
    bmx_hr[0] = kax[0] / den2 + krad[0]
    cmx_hr[0] = -kax[0] / den2
    rmx_hr[0] = krad[0] * smp[0] - qeroot - kax[0]

    # For the middle soil layers
    for j in range(1, nl_soil - 1):
        den1 = zmm[j] - zmm[j - 1]
        den2 = zmm[j + 1] - zmm[j]
        amx_hr[j] = -kax[j - 1] / den1
        bmx_hr[j] = kax[j - 1] / den1 + kax[j] / den2 + krad[j]
        cmx_hr[j] = -kax[j] / den2
        rmx_hr[j] = krad[j] * smp[j] + kax[j - 1] - kax[j]

    # For the bottom soil layer
    j = nl_soil - 1
    den_AHR = zmm[j] - zmm[j - 1]
    amx_hr[j] = -kax[j - 1] / den_AHR
    bmx_hr[j] = kax[j - 1] / den_AHR + krad[j]
    cmx_hr[j] = 0
    rmx_hr[j] = krad[j] * smp[j] + kax[j - 1]

    # Solve for root pressure potential using tridiagonal matrix solver
    x = tridia(nl_soil, amx_hr, bmx_hr, cmx_hr, rmx_hr)

    xroot = x[:]
    x_root_top = xroot[0] - zmm[0]

    return xroot, x_root_top
def calcstress_twoleaf(co_lm, x, nvegwcs, rstfacsun, rstfacsha, etrsun, etrsha, rootflux,
                       gb_mol, gs0sun, gs0sha, qsatl, qaf, qg, qm, rhoair,
                       psrf, fwet, laisun, laisha, sai, htop, tl, kmax_sun,
                       kmax_sha, kmax_xyl, kmax_root, psi50_sun, psi50_sha,
                       psi50_xyl, psi50_root, ck, nl_soil, z_soi, rss, raw, rd, smp,
                       k_soil_root, k_ax_root, gssun, gssha):

    A = [[0.0] * nvegwcs for _ in range(nvegwcs)]  # matrix relating d(vegwp) and f: d(vegwp) = A * f
    f = [0.0] * nvegwcs  # flux divergence (mm/s)
    dx = [0.0] * nvegwcs  # change in vegwp from one iter to the next [mm]
    xroot = [0.0] * nl_soil  # local gs_mol copies
    tol_lai = 1.0e-7  # minimum lai WHERE transpiration is calc'd
    leafsun, leafsha, xyl, root = 1, 2, 3, 4
    itmax = 50  # EXIT newton's method IF iters > itmax
    toldx = 1.0e-9  # tolerances for a satisfactory solution
    tolf = 1.0e-6  # tolerance for a satisfactory solution
    tolf_leafxyl = 1.0e-16  # tolerance for a satisfactory solution
    tolf_root = 1.0e-14  # tolerance for a satisfactory solution

    # temporary flag for night time vegwp(sun) > 0

    gssun = gs0sun
    gssha = gs0sha
    gssun, gssha, qflx_sun, qflx_sha = getqflx_gs2qflx_twoleaf(co_lm, gb_mol,  qsatl, qaf,
                            rhoair, psrf, laisun, laisha, sai, fwet, tl, rss, raw, rd, qg, qm)
    x_root_top = x[root]

    if qflx_sun > 0 or qflx_sha > 0:
        qeroot, dqeroot = getrootqflx_x2qe(nl_soil, smp, x_root_top, z_soi, k_soil_root, k_ax_root)

        spacAF_twoleaf(x, nvegwcs, dx, nl_soil, qflx_sun, qflx_sha, laisun, laisha, sai, htop,
                       qeroot, dqeroot, kmax_sun, kmax_sha, kmax_xyl, kmax_root,
                       psi50_sun, psi50_sha, psi50_xyl, psi50_root, ck)

        if max(abs(dx)) > 200000.0:
            maxscale = min(max(abs(dx)), max(abs(x))) / 2
            dx = maxscale * dx / max(abs(dx))

        x += dx

        if x[xyl] > x[root]:
            x[xyl] = x[root]
        if x[leafsun] > x[xyl]:
            x[leafsun] = x[xyl]
        if x[leafsha] > x[xyl]:
            x[leafsha] = x[xyl]

        etrsun = qflx_sun * plc(x[leafsun], psi50_sun, ck)
        etrsha = qflx_sha * plc(x[leafsha], psi50_sha, ck)

        getqflx_qflx2gs_twoleaf(gb_mol, gssun, gssha, etrsun, etrsha, qsatl, qaf,
                                rhoair, psrf, laisun, laisha, sai, fwet, tl, rss, raw, rd, qg, qm)

        tprcor = 44.6 * 273.16 * psrf / 1.013e5

        rstfacsun = max(gssun / gs0sun, 1.0e-2)
        rstfacsha = max(gssha / gs0sha, 1.0e-2)
        qeroot = etrsun + etrsha
        xroot, x_root_top = getrootqflx_qe2x(nl_soil, smp, z_soi, k_soil_root, k_ax_root, qeroot)
        x[root] = x_root_top
        rootflux = k_soil_root * (smp - xroot)

    else:
        if x[xyl] > x[root]:
            x[xyl] = x[root]
        if x[leafsun] > x[xyl]:
            x[leafsun] = x[xyl]
        if x[leafsha] > x[xyl]:
            x[leafsha] = x[xyl]

        etrsun = 0.0
        etrsha = 0.0
        rstfacsun = max(plc(x[leafsun], psi50_sun, ck), 1.0e-2)
        rstfacsha = max(plc(x[leafsha], psi50_sha, ck), 1.0e-2)
        gssun = gs0sun * rstfacsun
        gssha = gs0sha * rstfacsha
        rootflux = 0.0

    soilflux = sum(rootflux)

    return rstfacsun, rstfacsha, etrsun, etrsha, rootflux, gssun,  gssha

def PlantHydraulicStress_twoleaf (nl_soil, nvegwcs, z_soi, dz_soi, rootfr, psrf, qsatl, qaf, tl, rb, rss, ra, rd,
                                 rstfacsun, rstfacsha, cintsun, cintsha, laisun, laisha, rhoair, fwet, sai,
                                 kmax_sun, kmax_sha, kmax_xyl, kmax_root, psi50_sun, psi50_sha, psi50_xyl,
                                 psi50_root, htop, ck, smp, hk, hksati, vegwp, etrsun, etrsha, rootflux, qg,
                                 qm, gs0sun, gs0sha, k_soil_root, k_ax_root, gssun, gssha):
    iterationtotal = 6
    c3 = 1.0  # c3 vegetation
    tprcor = 1.0  # coefficient for unit transfer
    # Initialize arrays
    fs = np.zeros(nl_soil)
    rai = np.zeros(nl_soil)
    # Set parameters
    croot_lateral_length = 0.25
    c_to_b = 2.0
    rpi = 3.14159265358979
    root_type = 4
    toldb = 1e-2
    K_axs = 2.0e-1

    # Temporary input parameters
    froot_carbon = 288.392056287006
    root_radius = 2.9e-4
    root_density = 310000.0
    froot_leaf = 1.5
    krmax = 3.981071705534969e-009

# ----------------calculate root-soil interface conductance-----------------

    for j in range(nl_soil):
        # Calculate conversion from conductivity to conductance
        root_biomass_density = c_to_b * froot_carbon * rootfr[j] / dz_soi[j]
        # Ensure minimum root biomass (using 1gC/m2)
        root_biomass_density = max(c_to_b * 1.0, root_biomass_density)

        # Root length density: m root per m3 soil
        root_cross_sec_area = rpi * root_radius ** 2
        root_length_density = root_biomass_density / (root_density * root_cross_sec_area)

        # Root-area index (RAI)
        rai[j] = (sai + laisun + laisha) * froot_leaf * rootfr[j]

        # Fix coarse root_average_length to specified length
        croot_average_length = croot_lateral_length

        # Calculate r_soil using Gardner/spa equation (Bonan, GMD, 2014)
        r_soil = np.sqrt(1.0 / (rpi * root_length_density))

        # Length scale approach
        soil_conductance = min(hksati[j], hk[j]) / (1.0e3 * r_soil)

        # Use vegetation plc function to adjust root conductance
        fs[j] = plc(max(smp[j], -1.0), psi50_root, ck)

        # krmax is root conductance per area per length
        root_conductance = (fs[j] * rai[j] * krmax) / (croot_average_length + z_soi[j])
        soil_conductance = max(soil_conductance, 1.0e-16)
        root_conductance = max(root_conductance, 1.0e-16)

        # Sum resistances in soil and root
        rs_resis = 1.0 / soil_conductance + 1.0 / root_conductance

        # Conductance is inverse resistance
        # Explicitly set conductance to zero for top soil layer
        if rai[j] * rootfr[j] > 0.0:
            k_soil_root[j] = 1.0 / rs_resis
        else:
            k_soil_root[j] = 0.0
        k_ax_root[j] = (rootfr[j] / (dz_soi[j] * 1000)) * K_axs * 0.6
    # ----------------------------------------------------------------------------------------------------------------
    tprcor = 44.6 * 273.16 * psrf / 1.013e5
    cf = tprcor / tl * 1.0e6  # gb->gbmol conversion factor

    # One side leaf boundary layer conductance for water vapor [=1/(2*rb)]
    # ATTENTION: rb in CLM is for one side leaf, but for SiB2 rb for
    # 2-side leaf, so the gbh2o shold be " 0.5/rb * tprcor/tl "
    gb_mol = 1.0 / rb * cf  # resistance to conductance (s/m -> umol/m**2/s)

    x = vegwp[:nvegwcs]

    calcstress_twoleaf(x, nvegwcs, rstfacsun, rstfacsha, etrsun, etrsha, rootflux,
                       gb_mol, gs0sun, gs0sha, qsatl, qaf, qg, qm, rhoair,
                       psrf, fwet, laisun, laisha, sai, htop, tl, kmax_sun,
                       kmax_sha, kmax_xyl, kmax_root, psi50_sun, psi50_sha,
                       psi50_xyl, psi50_root, ck, nl_soil, z_soi, rss, ra, rd,
                       smp, k_soil_root, k_ax_root, gssun, gssha)

    vegwp[:nvegwcs] = x

    return

def getvegwp_twoleaf(x, nvegwcs, nl_soil, z_soi, gb_mol, gs_mol_sun, gs_mol_sha,
                     qsatl, qaf, qg, qm, rhoair, psrf, fwet, laisun, laisha, htop, sai, tl, rss,
                     raw, rd, smp, k_soil_root, k_ax_root, kmax_xyl, kmax_root, rstfacsun, rstfacsha,
                     psi50_sun, psi50_sha, psi50_xyl, psi50_root, ck, rootflux):
    # ----------------------------------------------------------------------------------------------------------------
    tol_lai = 1.e-7
    leafsun = 1
    leafsha = 2
    xyl = 3
    root = 4

    grav1 = 1000.0 * htop
    grav2 = 1000.0 * z_soi[:nl_soil]

    # Compute transpiration demand
    havegs = True
    gs_mol_sun, gs_mol_sha, etrsun, etrsha = getqflx_gs2qflx_twoleaf(gb_mol, qsatl, qaf, rhoair,
                            psrf, laisun, laisha, sai, fwet, tl, rss, raw, rd, qg, qm,
                            rstfacsun, rstfacsha)

    # Calculate root water potential
    qeroot = etrsun + etrsha
    xroot,x_root_top = getrootqflx_qe2x(nl_soil, smp, z_soi, k_soil_root, k_ax_root, qeroot)

    # Calculate xylem water potential
    fr = plc(x[root], psi50_root, ck)
    x[xyl] = x[root] - grav1 - (etrsun + etrsha) / (fr * kmax_root / htop * sai)

    # Calculate sun/sha leaf water potential
    fx = plc(x[xyl], psi50_xyl, ck)
    x[leafsha] = x[xyl] - (etrsha / (fx * kmax_xyl * laisha))
    x[leafsun] = x[xyl] - (etrsun / (fx * kmax_xyl * laisun))

    # Calculate soil flux
    rootflux = [k_soil_root[j] * (smp[j] - xroot[j]) for j in range(nl_soil)]
    soilflux = sum(rootflux)

    return x, gs_mol_sun, gs_mol_sha, etrsun, etrsha, rootflux