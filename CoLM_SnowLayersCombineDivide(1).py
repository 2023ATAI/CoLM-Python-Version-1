import numpy as np
import CoLM_SoilThermalParameters

def winddriftcompaction(bi, forc_wind, dz, zpseudo, mobile):
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


def snowcompaction (lb, deltim, ssi, wimp, pg_rain, qseva, qsdew, qsubl, qfros, dz_soisno, wice_soisno, wliq_soisno):
    # --------------------------------------------------------------------------------------------------------------
    # Four of metamorphisms of changing snow characteristics are implemented,
    # ! i.e., destructive, overburden, melt and wind drift. The treatments of the destructive compaction
    # ! was from SNTHERM.89 and SNTHERM.99 (1991, 1999). The contribution due to
    # ! melt metamorphism is simply taken as a ratio of snow ice fraction after
    # ! the melting versus before the melting. The treatments of the overburden comaction and the drifing compaction
    # ! were borrowed from CLM5.0 which based on Vionnet et al. (2012) and van Kampenhout et al (2017).
    # --------------------------------------------------------------------------------------------------------------

    # Constants
    c1 = 2.777e-7  # [m2/(kg s)]
    c2 = 23.0e-3  # [m3/kg]
    c3 = 2.777e-6  # [1/s]
    c4 = 0.04  # [1/K]
    c5 = 2.0
    c6 = 5.15e-7
    c7 = 4.0
    dm = 100.0  # Upper Limit on Destructive Metamorphism Compaction [kg/m3]
    eta0 = 9.e5  # The Viscosity Coefficient Eta0 [kg-s/m2]

    #Begin calculation - note that the following column loops are only invoked IF lb < 0
    burden = 0.0
    zpseudo = 0.0
    mobile = True

    for j in range(lb, -1, -1):
        wx = wice_soisno[j] + wliq_soisno[j]
        void = 1.0 - (wice_soisno[j] / denice + wliq_soisno[j] / denh2o) / dz_soisno[j]

    # Disallow compaction for water saturated node and lower ice lens node.
        if void <= 0.001 or wice_soisno[j] <= 0.1:
            burden += wx
            mobile = False
            continue

        bi = wice_soisno[j] / dz_soisno[j]  # bi is partial density of ice [kg/m3]
        fi = wice_soisno[j] / wx  # fi is fraction of ice relative to the total water content at current time step
        td = tfrz - t_soisno[j]  # td is t_soisno - tfrz [K]

        dexpf = np.exp(-c4 * td)  # expf=exp(-c4*(273.15-t_soisno)).

        # Compaction due to destructive metamorphism
        ddz1 = -c3 * dexpf
        if bi > dm:
            ddz1 *= np.exp(-46.0e-3 * (bi - dm))

        # Liquid water term
        if wliq_soisno[j] > 0.01 * dz_soisno[j]:
            ddz1 *= c5

        # Compaction due to overburden
        f1 = 1.0 / (1.0 + 60.0 * wliq_soisno[j] / (denh2o * dz_soisno[j]))
        f2 = 4.0  # currently fixed to maximum value, holds in absence of angular grains
        eta = f1 * f2 * (bi / 450.0) * np.exp(0.1 * td + c2 * bi) * 7.62237e6
        ddz2 = -(burden + wx / 2.0) / eta

        # Compaction occurring during melt
        if imelt[j] == 1:
            ddz3 = -1.0 / deltim * max(0.0, (fiold[j] - fi) / fiold[j])
        else:
            ddz3 = 0.0

            # Compaction occurring due to wind drift
        forc_wind = np.sqrt(forc_us ** 2 + forc_vs ** 2)
        ddz4 = winddriftcompaction(bi, forc_wind, dz_soisno[j], zpseudo, mobile)

        # Time rate of fractional change in dz (units of s-1)
        pdzdtc = ddz1 + ddz2 + ddz3 + ddz4

        # The change in dz_soisno due to compaction
        dz_soisno[j] *= (1.0 + pdzdtc * deltim)
        dz_soisno[j] = max(dz_soisno[j], (wice_soisno[j] / denice + wliq_soisno[j] / denh2o))

        # Pressure of overlying snow
        burden += wx



