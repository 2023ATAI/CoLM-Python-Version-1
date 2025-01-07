#-----------------------------------------------------------------------
   # DESCRIPTION:
   #provides leaf and stem area parameters
#-----------------------------------------------------------------------

def LAI_empirical(nl_colm,ivt,nl_soil,rootfr,t):
    if nl_colm['LULC_USGS']:
        # Maximum fractional cover of vegetation [-]
        vegc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        # Maximum leaf area index, the numbers are based on the data of
        # "worldwide histrorical estimates of leaf area index, 1932-2000" :
        # http://www.daac.ornl.gov/global_vegetation/HistoricalLai/data"
        xla = [1.50, 3.29, 4.18, 3.50, 2.50, 3.60, 2.02, 1.53,
               2.00, 0.85, 4.43, 4.42, 4.56, 3.95, 4.50, 0.00,
               4.00, 3.63, 0.00, 0.64, 1.60, 1.00, 0.00, 0.00]

        # Minimum leaf area index
        xla0 = [1.00, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 0.50,
                0.50, 0.30, 0.50, 0.50, 4.00, 4.00, 4.00, 0.00,
                3.00, 3.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

        # Stem area index [-]
        sai0 = [0.20, 0.20, 0.30, 0.30, 0.50, 0.50, 1.00, 0.50,
                1.00, 0.50, 2.00, 2.00, 2.00, 2.00, 2.00, 0.00,
                2.00, 2.00, 0.00, 0.10, 0.10, 0.10, 0.00, 0.00]
    elif nl_colm['SIB2_CLASSIFICATION']:
        # Maximum fractional cover of vegetation [-]
        vegc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        # Maximum leaf area index
        xla = [4.8, 3.9, 5.6, 5.5, 4.6, 1.7, 1.3, 2.1, 3.6, 0.0, 0.0]

        # Minimum leaf area index
        xla0 = [4.0, 0.6, 0.5, 5.0, 0.5, 0.3, 0.6, 0.4, 0.2, 0.0, 0.0]

        # Stem area index [-]
        sai0 = [1.6, 1.8, 1.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]

    elif nl_colm['BATS_CLASSIFICATION']:
        # Maximum fractional cover of vegetation [-]
        vegc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        # Maximum leaf area index
        xla = [5.1, 1.6, 4.8, 4.8, 4.8, 5.4, 4.8, 0.0, 3.6, 4.8,
               0.6, 0.0, 4.8, 0.0, 0.0, 4.8, 4.8, 4.8, 4.8]

        # Minimum leaf area index
        xla0 = [0.425, 0.4, 4.0, 0.8, 0.8, 4.5, 0.4, 0.0, 0.3, 0.4,
                0.05, 0.0, 0.4, 0.0, 0.0, 4.0, 0.8, 2.4, 2.4]

        # Stem area index [-]
        sai0 = [0.425, 3.2, 1.6, 1.6, 1.6, 1.8, 1.6, 0.0, 0.3, 0.4,
                0.2, 0.0, 1.6, 0.0, 0.0, 1.6, 1.6, 1.6, 1.6]
    else:
        #!#elif(defined LULC_IGBP)
        # Maximum fractional cover of vegetation [-]
        vegc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]

        # Maximum leaf area index
        xla = [4.8, 5.4, 4.8, 4.8, 4.7, 4.7, 1.6, 4.7, 4.8, 1.7,
               4.6, 4.9, 3.8, 4.8, 0.0, 0.06, 0.0]

        # Minimum leaf area index
        xla0 = [4.0, 4.5, 0.8, 0.8, 2.2, 1.6, 0.15, 1.8, 0.9, 0.4,
                0.4, 0.4, 0.9, 2.0, 0.0, 0.006, 0.0]

        # Stem area index [-]
        sai0 = [1.6, 1.8, 1.6, 1.6, 1.5, 1.5, 0.45, 1.4, 1.6, 3.1,
                1.6, 0.4, 1.1, 1.3, 0.0, 0.14, 0.0]
    roota = 0.0
    jrt = 1
    for j in range(nl_soil):
        roota += rootfr[j]
        if roota > 0.9:
            jrt = j
            break
    # Adjust leaf area index for seasonal variation
    f = max(0.0, 1.0 - 0.0016 * max(298.0 - t[jrt - 1], 0.0) ** 2)
    lai = xla[ivt - 1] + (xla0[ivt - 1] - xla[ivt - 1]) * (1.0 - f)

    # Sum leaf area index and stem area index
    sai = sai0[ivt - 1]

    # Fractional vegetation cover
    fveg = vegc[ivt - 1]

    green = 0.0
    if fveg > 0.0:
        green = 1.0

    return lai,sai,fveg,green

