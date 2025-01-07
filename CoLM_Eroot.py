import numpy as np
from CoLM_Hydro_SoilFunction import soil_psi_from_vliq
def eroot(co_lm, nl_soil, trsmx0, porsl, psi0, rootfr, dz_soisno, t_soisno, wliq_soisno, tfrz,
          bsw=None, theta_r=None, alpha_vgm=None, n_vgm=None, L_vgm=None, sc_vgm=None, fc_vgm=None):
    roota = 1e-10  # must be non-zero to begin
    rresis = np.zeros(nl_soil)
    rootr = np.zeros(nl_soil)

    for i in range(nl_soil):
        if np.all(t_soisno[i] > tfrz) and np.all(porsl[i] >= 1e-6):
            smpmax = -1.5e5
            s_node = max(wliq_soisno[i] / (1000. * dz_soisno[i] * porsl[i]), 0.001) #divide by zero
            s_node = min(1., s_node)
            if co_lm['Campbell_SOIL_MODEL']:
                smp_node = max(smpmax, psi0[i] * s_node ** (-bsw[i]))
            if co_lm['vanGenuchten_Mualem_SOIL_MODEL']:
                smp_node = soil_psi_from_vliq(co_lm, s_node * (porsl[i] - theta_r[i]) + theta_r[i],
                                              porsl[i], theta_r[i], psi0[i],
                                              5, [alpha_vgm[i], n_vgm[i], L_vgm[i], sc_vgm[i], fc_vgm[i]])
                smp_node = max(smpmax, smp_node)
            rresis[i] = (1. - smp_node / smpmax) / (1. - psi0[i] / smpmax)
            rootr[i] = rootfr[i] * rresis[i]
            roota += rootr[i]
        else:
            rootr[i] = 0.

    # Normalize root resistances to get layer contribution to ET
    rootr /= roota

    # Determine maximum possible transpiration rate
    etrc = trsmx0 * roota
    rstfac = roota

    return rootr, etrc, rstfac

