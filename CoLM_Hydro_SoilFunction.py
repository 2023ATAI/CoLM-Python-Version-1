import numpy as np

minsmp = -1.0e8


def get_derived_parameters_vGM  (self, psi_s ,alpha_vgm, n_vgm):
    """
    Original author  : Qinghliang Li, 17/02/2024; Jinlong Zhu,   17/02/2024;
    software         : Provide snow cover fraction
    """
    m_vgm = 1.0 - 1.0 / n_vgm
    sc_vgm = (1.0 + (- alpha_vgm * psi_s) ** n_vgm) ** (-m_vgm)
    fc_vgm = 1.0 - (1.0 - sc_vgm ** (1.0 / m_vgm)) ** m_vgm

    return sc_vgm, fc_vgm


def soil_vliq_from_psi (nl_colm,psi,porsl, vl_r, psi_s, nprm, prms):
    soil_vliq_from_psi1 = 0
    if psi >= psi_s:
        soil_vliq_from_psi1  = porsl
        return soil_vliq_from_psi1

    if nl_colm['Campbell_SOIL_MODEL']:
        soil_vliq_from_psi1 = porsl * (psi / psi_s) ** (-1.0 / prms[0])

    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        m_vgm = 1.0 - 1.0 / prms[1]
        esat = (1.0 + (psi * (-prms[0])) ** prms[1]) ** (-m_vgm) / prms[3]
        soil_vliq_from_psi1 = (porsl - vl_r) * esat + vl_r

    return soil_vliq_from_psi1

def soil_hk_from_psi(nl_colm,psi, psi_s, hksat, nprm, prms):

    if psi >= psi_s:
        soil_hk_from_psi = hksat

    if nl_colm['Campbell_SOIL_MODEL']:
        soil_hk_from_psi = hksat * (psi / psi_s) ** (- 3.0 / prms[0] - 2.0)

    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        m_vgm = 1.0 - 1.0 / prms[1]
        esat = (1.0 + (- prms[0] * psi) ** prms[1]) ** (-m_vgm) / prms[3]
        soil_hk_from_psi = hksat * esat ** prms[2] * (
                    (1.0 - (1.0 - (esat * prms[3]) ** (1.0 / m_vgm)) ** m_vgm) / prms[4]) ** 2.0
    return soil_hk_from_psi

def soil_psi_from_vliq (nl_colm,vliq, porsl, vl_r, psi_s, nprm, prms):
    soil_psi_from_vliq1 = 0.0
    if vliq >= porsl:
        soil_psi_from_vliq1 = psi_s
        return  soil_psi_from_vliq1
    elif vliq <= max(vl_r, 1.0e-8):
        soil_psi_from_vliq1 = minsmp
        return soil_psi_from_vliq1

    if nl_colm['Campbell_SOIL_MODEL']:
        soil_psi_from_vliq1 = psi_s * (vliq / porsl) ** (-prms[0])

    if nl_colm['vanGenuchten_Mualem_SOIL_MODEL']:
        m_vgm = 1.0 - 1.0 / prms[1]#divide by zero
        esat = (vliq - vl_r) / (porsl - vl_r)
        soil_psi_from_vliq1 = -((esat * prms[3]) ** (-1.0 / m_vgm) - 1.0) ** (1.0 / prms[1]) / prms[0]
        #divide by zero two
    soil_psi_from_vliq1 = max(soil_psi_from_vliq1, minsmp)

    return soil_psi_from_vliq1