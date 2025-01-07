import numpy as np

PONDMIN  = 1.e-4 
nmanning_hslp = 0.3

# Define the function in Python
def hillslope_flow(dt, mpi, numelm, grav, landhru, catch_riverlakenetwork, hydro_vtv, rsur, elm_patch):
    #catch_riverlakenetwork.lake_id: catch_riverlakenetwork
    #elm_patch landpatch
    if not mpi.p_is_worker:
        numbasin = numelm

        for ibasin in range(numbasin):
            hs = landhru.basin_hru.substt[ibasin]
            he = landhru.basin_hru.subend[ibasin]

            if catch_riverlakenetwork.lake_id[ibasin] > 0:
                hydro_vtv.veloc_hru[hs:he+1] = 0
                hydro_vtv.momen_hru[hs:he+1] = 0
                continue  # skip lakes
            else:
                for i in range(hs, he+1):
                    # momentum is less or equal than the momentum at last time step.
                    hydro_vtv.momen_hru[i] = min(hydro_vtv.wdsrf_hru_prev[i], hydro_vtv.wdsrf_hru[i]) * hydro_vtv.veloc_hru[i]

            hillslope = catch_riverlakenetwork.hillslope_network[ibasin]

            nhru = hillslope.nhru

            wdsrf_h = np.zeros(nhru)
            veloc_h = np.zeros(nhru)
            momen_h = np.zeros(nhru)

            sum_hflux_h = np.zeros(nhru)
            sum_mflux_h = np.zeros(nhru)
            sum_zgrad_h = np.zeros(nhru)

            xsurf_h = np.zeros(nhru)

            for i in range(nhru):
                idx = hillslope.ihru[i]  # Convert to zero-based index
                wdsrf_h[i] = hydro_vtv.wdsrf_hru[idx]
                momen_h[i] = hydro_vtv.momen_hru[idx]
                if wdsrf_h[i] > 0:
                    veloc_h[i] = momen_h[i] / wdsrf_h[i]
                else:
                    veloc_h[i] = 0

            dt_res = dt
            while dt_res > 0:
                for i in range(nhru):
                    sum_hflux_h[i] = 0
                    sum_mflux_h[i] = 0
                    sum_zgrad_h[i] = 0

                dt_this = dt_res

                for i in range(nhru):
                    j = hillslope.inext[i]  # Convert to zero-based index

                    if j < 0:
                        continue  # lowest HRUs

                    # dry HRU
                    if wdsrf_h[i] < PONDMIN and wdsrf_h[j] < PONDMIN:
                        continue

                    # reconstruction of height of water near interface
                    hand_fc = max(hillslope.hand[i], hillslope.hand[j])
                    wdsrf_up = max(0, hillslope.hand[i] + wdsrf_h[i] - hand_fc)
                    wdsrf_dn = max(0, hillslope.hand[j] + wdsrf_h[j] - hand_fc)

                    # velocity at hydrounit downstream face
                    veloc_fc = 0.5 * (veloc_h[i] + veloc_h[j]) + np.sqrt(grav * wdsrf_up) - np.sqrt(grav * wdsrf_dn)

                    # depth of water at downstream face
                    wdsrf_fc = 1 / grav * (0.5 * (np.sqrt(grav * wdsrf_up) + np.sqrt(grav * wdsrf_dn)) + 0.25 * (veloc_h[i] - veloc_h[j]))**2.0

                    if wdsrf_up > 0:
                        vwave_up = min(veloc_h[i] - np.sqrt(grav * wdsrf_up), veloc_fc - np.sqrt(grav * wdsrf_fc))
                    else:
                        vwave_up = veloc_h[j] - 2.0 * np.sqrt(grav * wdsrf_dn)

                    if wdsrf_dn > 0:
                        vwave_dn = max(veloc_h[j] + np.sqrt(grav * wdsrf_dn), veloc_fc + np.sqrt(grav * wdsrf_fc))
                    else:
                        vwave_dn = veloc_h[i] + 2.0 * np.sqrt(grav * wdsrf_up)

                    hflux_up = veloc_h[i] * wdsrf_up
                    hflux_dn = veloc_h[j] * wdsrf_dn
                    mflux_up = veloc_h[i]**2 * wdsrf_up + 0.5 * grav * wdsrf_up**2
                    mflux_dn = veloc_h[j]**2 * wdsrf_dn + 0.5 * grav * wdsrf_dn**2

                    if vwave_up >= 0:
                        hflux_fc = hillslope.flen[i] * hflux_up
                        mflux_fc = hillslope.flen[i] * mflux_up
                    elif vwave_dn <= 0:
                        hflux_fc = hillslope.flen[i] * hflux_dn
                        mflux_fc = hillslope.flen[i] * mflux_dn
                    else:
                        hflux_fc = hillslope.flen[i] * (vwave_dn * hflux_up - vwave_up * hflux_dn + vwave_up * vwave_dn * (wdsrf_dn - wdsrf_up)) / (vwave_dn - vwave_up)
                        mflux_fc = hillslope.flen[i] * (vwave_dn * mflux_up - vwave_up * mflux_dn + vwave_up * vwave_dn * (hflux_dn - hflux_up)) / (vwave_dn - vwave_up)

                    sum_hflux_h[i] += hflux_fc
                    sum_hflux_h[j] -= hflux_fc

                    sum_mflux_h[i] += mflux_fc
                    sum_mflux_h[j] -= mflux_fc

                    sum_zgrad_h[i] += hillslope.flen[i] * 0.5 * grav * wdsrf_up**2
                    sum_zgrad_h[j] -= hillslope.flen[i] * 0.5 * grav * wdsrf_dn**2

                for i in range(nhru):
                    # constraint 1: CFL condition
                    if hillslope.inext[i] > 0:
                        if veloc_h[i] != 0 or wdsrf_h[i] > 0:
                            dt_this = min(dt_this, hillslope.plen[i] / (abs(veloc_h[i]) + np.sqrt(grav * wdsrf_h[i])) * 0.8)

                    # constraint 2: Avoid negative values of water
                    xsurf_h[i] = sum_hflux_h[i] / hillslope.area[i]
                    if xsurf_h[i] > 0:
                        dt_this = min(dt_this, wdsrf_h[i] / xsurf_h[i])

                    # constraint 3: Avoid change of flow direction
                    if abs(veloc_h[i]) > 0.1 and veloc_h[i] * (sum_mflux_h[i] - sum_zgrad_h[i]) > 0:
                        dt_this = min(dt_this, abs(momen_h[i] * hillslope.area[i] / (sum_mflux_h[i] - sum_zgrad_h[i])))

                for i in range(nhru):
                    wdsrf_h[i] = max(0, wdsrf_h[i] - xsurf_h[i] * dt_this)

                    if wdsrf_h[i] < PONDMIN:
                        momen_h[i] = 0
                    else:
                        friction = grav * nmanning_hslp**2 * abs(momen_h[i]) / wdsrf_h[i]**(7.0 / 3.0)
                        momen_h[i] = (momen_h[i] - (sum_mflux_h[i] - sum_zgrad_h[i]) / hillslope.area[i] * dt_this) / (1.0 + friction / hillslope.area[i] * dt_this)

                        if hillslope.inext[i] <= 0:
                            momen_h[i] = min(momen_h[i], 0.)

                        if all(hillslope.inext != i):
                            momen_h[i] = max(momen_h[i], 0.)

                if hillslope.indx[0] == 0:
                    srfbsn = np.min(hillslope.hand + wdsrf_h)
                    if srfbsn < wdsrf_h[0]:
                        mask = np.zeros(hillslope.nhru, dtype=bool)
                        dvol = (wdsrf_h[0] - srfbsn) * hillslope.area[0]
                        momen_h[0] = srfbsn / wdsrf_h[0] * momen_h[0]
                        wdsrf_h[0] = srfbsn

                        while dvol > 0:
                            mask = hillslope.hand + wdsrf_h > srfbsn
                            nexta = np.sum(hillslope.area[~mask])
                            if np.any(mask):
                                nextl = np.min((hillslope.hand + wdsrf_h)[mask])
                                nextv = nexta * (nextl - srfbsn)
                                if dvol > nextv:
                                    ddep = nextl - srfbsn
                                    dvol -= nextv
                                else:
                                    ddep = dvol / nexta
                                    dvol = 0
                            else:
                                ddep = dvol / nexta
                                dvol = 0

                            srfbsn += ddep

                            # Update wdsrf_h where mask is False
                            wdsrf_h[~mask] += ddep
                        del mask

                for i in range(hillslope.nhru):
                    if wdsrf_h[i] < PONDMIN:
                        veloc_h[i] = 0
                    else:
                        veloc_h[i] = momen_h[i] / wdsrf_h[i]

                    hydro_vtv.wdsrf_hru_ta[hillslope.ihru[i]] += wdsrf_h[i] * dt_this
                    hydro_vtv.momen_hru_ta[hillslope.ihru[i]] += momen_h[i] * dt_this

                if hillslope.indx[0] == 0:
                    ps = elm_patch.substt[ibasin]
                    pe = elm_patch.subend[ibasin]
                    rsur[ps:pe] -= np.sum(sum_hflux_h[0]) * dt_this / np.sum(hillslope.area) * 1.0e3

                dt_res = dt_res - dt_this
            for i in range(nhru):
                hydro_vtv.wdsrf_hru(hillslope.ihr[i]) = wdsrf_h[i]
                hydro_vtv.veloc_hru(hillslope.ihru[i]) = veloc_h[i]

            del wdsrf_h
            del veloc_h
            del momen_h

            del sum_hflux_h
            del sum_mflux_h
            del sum_zgrad_h

            del xsurf_h
        hydro_vtv.wdsrf_hru_prev[:] = hydro_vtv.wdsrf_hru[:]



