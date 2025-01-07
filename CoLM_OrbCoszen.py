import math

def orb_coszen(calday,dlon,dlat):
    """
    Original author  : Qinghliang Li, 17/02/2024; Jinlong Zhu,   17/02/2024;
    software         : return the cosine of the solar zenith angle. Assumes 365.0 days/year.
                    Compute earth/orbit parameters using formula suggested by Duane Thresher.
                     Use formulas from Berger, Andre 1978: Long-Term Variations of Daily Insolation
                      and Quaternary Climatic Changes. J. of the Atmo. Sci. 35:2362-2367.

    Args:
            Code     (type)           Standard name                              Units

        |-- calday   (float)       : Julian cal day                           [1.xx to 365.xx]
        |-- dlat     (float)       : Centered latitude                        [radians]
        |-- dlon     (float)       : Centered longitude                       [radians]

    Returns:
            cosz     (float)       :the cosine of the solar zenith angle      [-]

    """
    # Constants
    dayspy = 365.0  # Days per year
    ve = 80.5  # Calday of the vernal equinox assumes Jan 1 = calday 1
    eccen = 1.672393084E-2  # Eccentricity
    obliqr = 0.409214646  # Earth's obliquity in radians
    lambm0 = -3.2625366E-2  # Mean long of perihelion at the vernal equinox (radians)
    mvelpp = 4.92251015  # Moving vernal equinox longitude of perihelion plus pi (radians)

    # Local variables

    pi = 4.0*math.atan(1.0)  # Value of pi
    lambm = lambm0 + (calday - ve) * 2 * pi / dayspy  # Mean longitude of perihelion
    lmm = lambm - mvelpp  # Intermediate argument involving lambm
    sinl = math.sin(lmm)  # Sine of lmm
    lamb = lambm + eccen * (2 * sinl + eccen * (1.25 * math.sin(2 * lmm)
                                                + eccen * ((13.0 / 12.0) * math.sin(3 * lmm) - 0.25 * sinl)))
    invrho = (1. + eccen * math.cos(lamb - mvelpp)) / (1. - eccen * eccen)  # Inverse normalized sun/earth distance
    declin = math.asin(math.sin(obliqr) * math.sin(lamb))  # Solar declination ).arcsin
    eccf = invrho * invrho  # Earth-sun distance factor (i.e., (1/r)**2)

    # Compute cosine of solar zenith angle
    orb_coszen = math.sin(dlat) * math.sin(declin) - math.cos(dlat) * math.cos(declin) * math.cos(calday * 2.0 * pi + dlon)
    return orb_coszen

