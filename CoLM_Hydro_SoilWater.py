import copy

import numpy as np
import CoLM_Hydro_SoilFunction
import CoLM_UserDefFun

class Hydro_SoilWater(object):
    def __init__(self,nl_colm) -> None:
       #   boundary condition:
       # ! 1: fixed pressure head
       # ! 2: rainfall condition with a ponding layer on top of groud surface
       # !    and a flux such as rainfall into the ponding layer
       # ! 3: fixed flux
       # ! 4: drainage condition with aquifers below soil columns
        self.bc_fix_head  = 1
        self.bc_rainfall  = 2
        self.bc_fix_flux  = 3
        self.bc_drainage  = 4
       # formula of effective hydraulic conductivity between levels
       #    ! Please refer to Dai et al. (2019) for definitions
        self.type_upstream_mean = 1
        self.type_weighted_geometric_mean  = 2

        self.effective_hk_type = self.type_weighted_geometric_mean
        self.max_iters_richards  = 10
        self.tol_richards  = 1.e-7
        self.nl_colm = nl_colm
        self.is_solvable = False

        if self.nl_colm['CoLMDEBUG']:
            self.count_implicit  = 0
            self.count_explicit  = 0
            self.count_wet2dry   = 0
    def secant_method_iteration(self, fval, fval_k1, x_i, x_k1, x_l, x_r):
        """
        Performs a single iteration of the secant method for root finding.
        
        Parameters:
            fval (float): Current function value.
            fval_k1 (float): Previous function value (will be updated).
            x_i (float): Current estimate of the root (will be updated).
            x_k1 (float): Previous estimate of the root (will be updated).
            x_l (float): Lower bound of the interval (will be updated).
            x_r (float): Upper bound of the interval (will be updated).
            alp (float): Damping factor (default is 0.9).
        """
        alp=0.9
        if fval > 0.0:
            x_r = x_i
        else:
            x_l = x_i

        fval_k2 = fval_k1
        fval_k1 = fval

        x_k2 = x_k1
        x_k1 = x_i

        if fval_k1 == fval_k2:
            x_i = (x_l + x_r) * 0.5
        else:
            x_i = (fval_k1 * x_k2 - fval_k2 * x_k1) / (fval_k1 - fval_k2)
            x_i = max(x_i, x_l * alp + x_r * (1.0 - alp))
            x_i = min(x_i, x_l * (1.0 - alp) + x_r * alp)

        return fval_k1, x_i, x_k1, x_l, x_r

    def check_and_update_level(self, dz, vl_s, vl_r, psi_s, hksat, nprm, prms,
                            is_sat, has_wf, has_wt,
                            wf, vl, wt, psi, hk,
                            is_update_psi_hk, tol_v):
        """
        Check and update the level properties based on saturation, water flux,
        and other parameters.

        Parameters:
        dz : float
            The depth of the soil layer.
        vl_s : float
            Saturated soil moisture.
        vl_r : float
            Residual soil moisture.
        psi_s : float
            Soil water potential for saturation.
        hksat : float
            Saturated hydraulic conductivity.
        nprm : int
            Number of parameters for the functions.
        prms : numpy.ndarray
            Array of parameters for the functions.
        is_sat : bool
            Indicator for soil saturation.
        has_wf : bool
            Indicator for water flux.
        has_wt : bool
            Indicator for water tension.
        wf : float
            Water flux.
        vl : float
            Soil moisture.
        wt : float
            Water tension.
        psi : float
            Soil water potential.
        hk : float
            Hydraulic conductivity.
        is_update_psi_hk : bool
            Indicator for updating psi and hk.
        tol_v : float
            Tolerance value for soil moisture.
        """
        if not is_sat:
            if has_wf:
                wf = min(max(wf, 0.0), dz)
            else:
                wf = 0

            if has_wt:
                wt = min(max(wt, 0.0), dz)
            else:
                wt = 0

            if has_wf and has_wt:
                if wf + wt > dz:
                    alpha = wf / (wf + wt)
                    wf = dz * alpha
                    wt = dz * (1.0 - alpha)
            vl = min(vl, vl_s)
            vl = max(vl, tol_v)

            if is_update_psi_hk:
                # print(vl, vl_s, vl_r, psi_s, nprm, prms,'----check----')
                psi = CoLM_Hydro_SoilFunction.soil_psi_from_vliq(self.nl_colm, vl, vl_s, vl_r, psi_s, nprm, prms)
                hk = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi, psi_s, hksat, nprm, prms)
        else:
            vl = vl_s
            psi = psi_s
            hk = hksat
        
        return wf, vl, wt, psi, hk

    def soilwater_aquifer_exchange(self, nlev, exwater, sp_zi, is_permeable, porsl, vl_r, psi_s, hksat, 
                               nprm, prms, porsl_wa, ss_dp, ss_vliq, zwt, wa):
        """
        Performs the soil water and aquifer exchange process.

        Parameters:
            nlev (int): Number of levels.
            exwater (float): Total water exchange [mm].
            sp_zi (list): Soil parameter interfaces of level [mm].
            is_permeable (list): Permeability status of levels.
            porsl (list): Soil porosity [mm^3/mm^3].
            vl_r (list): Residual soil moisture [mm^3/mm^3].
            psi_s (list): Saturated capillary potential [mm].
            hksat (list): Saturated hydraulic conductivity [mm/s].
            nprm (int): Number of parameters included in soil function.
            prms (list): Parameters included in soil function.
            porsl_wa (float): Soil porosity in aquifer [mm^3/mm^3].
            ss_dp (float): Depth of ponding water [mm].
            ss_vliq (list): Volume content of liquid water [mm^3/mm^3].
            zwt (float): Location of water table [mm].
            wa (float): Water in aquifer [mm, negative].

        Returns:
            Updated values of ss_dp, ss_vliq, zwt, wa, izwt.
        """
        temp = sp_zi[1:nlev+1] - sp_zi[:nlev]
        sp_dz = copy.copy(temp)

        # tolerances
        tol_z = self.tol_richards / np.sqrt(nlev*1.0) * 0.5 * 1800.0
        tol_v = tol_z / max(sp_dz)

        # water table location
        izwt = CoLM_UserDefFun.findloc_ud(zwt >= sp_zi, back=True)
        reswater = exwater
        # print(izwt, nlev,zwt, exwater, reswater,tol_z,tol_v,sp_dz,'----reswater----')

        if reswater > 0.0:
            # remove water from aquifer
            while reswater > 0.0:
                if izwt + 1 <= nlev:
                    if is_permeable[izwt]:
                        zwtp = self.get_zwt_from_wa(porsl[izwt], vl_r[izwt], psi_s[izwt], hksat[izwt],
                                                nprm, prms[:, izwt], tol_v, tol_z, -reswater, zwt)

                        if zwtp < sp_zi[izwt+1]:
                            ss_vliq[izwt] = (ss_vliq[izwt] * (zwt - sp_zi[izwt]) + porsl[izwt] * (zwtp - zwt) - reswater) / (zwtp - sp_zi[izwt])
                            reswater = 0.0
                            zwt = zwtp
                        else:
                            psi = psi_s[izwt] - (zwtp - 0.5 * (sp_zi[izwt+1] + zwt))
                            vliq = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm, psi, porsl[izwt], vl_r[izwt], psi_s[izwt], nprm, prms[:, izwt])
                            if reswater > (porsl[izwt] - vliq) * (sp_zi[izwt+1] - zwt):
                                ss_vliq[izwt] = (ss_vliq[izwt] * (zwt - sp_zi[izwt]) + vliq * (sp_zi[izwt+1] - zwt)) / sp_dz[izwt]
                                reswater = reswater - (porsl[izwt] - vliq) * (sp_zi[izwt+1] - zwt)
                            else:
                                ss_vliq[izwt] = (ss_vliq[izwt] * (zwt - sp_zi[izwt]) + porsl[izwt] * (sp_zi[izwt+1] - zwt) - reswater) / sp_dz[izwt]
                                reswater = 0.0

                            zwt = sp_zi[izwt+1]
                            izwt += 1
                    else:
                        zwt = sp_zi[izwt+1]
                        izwt += 1
                else:
                    zwt = self.get_zwt_from_wa(porsl_wa, vl_r[nlev - 1], psi_s[nlev - 1], hksat[nlev - 1], 
                                            nprm, prms[:, nlev - 1], tol_v, tol_z, wa - reswater, sp_zi[nlev])
                    wa = wa - reswater
                    reswater = 0.0
        elif reswater < 0.0:
            # increase water in aquifer
            while reswater < 0.0:
                if izwt + 1 > nlev:
                    if wa <= reswater:
                        wa = wa - reswater
                        reswater = 0.0
                    else:
                        reswater = reswater - wa
                        wa = 0.0
                        izwt = nlev - 1
                        zwt = sp_zi[nlev]
                elif izwt >= 0:
                    if is_permeable[izwt]:
                        air = (porsl[izwt] - ss_vliq[izwt]) * (zwt - sp_zi[izwt])
                        if air > -reswater:
                            ss_vliq[izwt] = ss_vliq[izwt] - reswater / (zwt - sp_zi[izwt])
                            reswater = 0.0
                        else:
                            ss_vliq[izwt] = porsl[izwt]
                            reswater = reswater + air
                            izwt -= 1
                            zwt = sp_zi[izwt+1]
                    else:
                        izwt -= 1
                        zwt = sp_zi[izwt+1]
                else:
                    ss_dp = ss_dp - reswater
                    reswater = 0.0
                    izwt = 0

        return ss_dp, ss_vliq, zwt, wa, izwt

    def initialize_sublevel_structure(self, lb, ub, dz, zbtm,
                                   vl_s, vl_r, psi_s, hksat, nprm, prms,
                                   ubc_typ, ubc_val, lbc_typ, lbc_val,
                                   is_sat, has_wf, has_wt,
                                   wf, vl, wt, dp, psi, hk,
                                   tol_v, tol_z):

        # Initialize the `is_sat` array based on the tolerance values
        for ilev in range(lb-1, ub):
            is_sat[ilev] = (abs(vl[ilev] - vl_s[ilev]) < tol_v) or \
                        (abs(wf[ilev] + wt[ilev] - dz[ilev]) < tol_z)

        # Handle boundary conditions for the upper boundary
        if ubc_typ == self.bc_fix_head:
            if ubc_val < psi_s[lb-1]:
                if is_sat[lb-1]:
                    is_sat[lb-1] = False
                    wf[lb-1] = 0
                    vl[lb-1] = vl_s[lb-1]
                    wt[lb-1] = 0.9 * dz[lb-1]
                elif wf[lb-1] >= tol_z:
                    vl[lb-1] = (wf[lb-1] * vl_s[lb-1] + vl[lb-1] * (dz[lb-1] - wf[lb-1] - wt[lb-1])) / (dz[lb-1] - wt[lb-1])
                    wf[lb-1] = 0

        # Handle boundary conditions for the lower boundary
        if lbc_typ == self.bc_fix_head:
            if lbc_val < psi_s[ub-1]:
                if is_sat[ub-1]:
                    is_sat[ub-1] = False
                    wf[ub-1] = 0.9 * dz[ub-1]
                    vl[ub-1] = vl_s[ub-1]
                    wt[ub-1] = 0
                elif wt[ub-1] >= tol_z:
                    vl[ub-1] = (wt[ub-1] * vl_s[ub-1] + vl[ub-1] * (dz[ub-1] - wf[ub-1] - wt[ub-1])) / (dz[ub-1] - wf[ub-1])
                    wt[ub-1] = 0

        # Update the state for each level
        for ilev in range(lb-1, ub):
            if is_sat[ilev]:
                wf[ilev] = 0
                wt[ilev] = dz[ilev]
                vl[ilev] = vl_s[ilev]
                # has_wf[ilev] = True
            else:
                if ilev > lb-1:
                    if is_sat[ilev - 1]:
                        has_wf[ilev] = True
                    else:
                        has_wf[ilev] = (wf[ilev] >= tol_z) or (wt[ilev - 1] >= tol_z)

                    if has_wf[ilev]:
                        if (wf[ilev] < tol_z) and (psi_s[ilev] < psi_s[ilev - 1]):
                            wf[ilev] = 0.1 * (dz[ilev] - wt[ilev])
                else:
                    if ubc_typ == self.bc_rainfall:
                        has_wf[lb-1] = (dp >= tol_z) or (wf[lb-1] >= tol_z)
                        if has_wf[lb-1] and (wf[lb-1] < tol_z):
                            wf[lb-1] = 0.01 * (dz[lb-1] - wt[lb-1])
                    elif ubc_typ == self.bc_fix_head:
                        has_wf[lb-1] = (ubc_val > psi_s[lb-1]) or (wf[lb-1] >= tol_z)
                        if has_wf[lb-1] and (wf[lb-1] < tol_z):
                            wf[lb-1] = 0.01 * (dz[lb-1] - wt[lb-1])
                    elif ubc_typ == self.bc_fix_flux:
                        has_wf[lb-1] = wf[lb-1] >= tol_z

                if ilev < ub-1:
                    if is_sat[ilev + 1]:
                        has_wt[ilev] = True
                    else:
                        has_wt[ilev] = (wt[ilev] >= tol_z) or (wf[ilev + 1] >= tol_z)

                    if has_wt[ilev]:
                        if (wt[ilev] < tol_z) and (psi_s[ilev] < psi_s[ilev + 1]):
                            wt[ilev] = 0.1 * (dz[ilev] - wf[ilev])
                else:
                    if lbc_typ == self.bc_drainage:
                        has_wt[ub-1] = (wt[ub-1] >= tol_z)
                    elif lbc_typ == self.bc_fix_head:
                        has_wt[ub-1] = (lbc_val > psi_s[lb]) or (wt[ub-1] >= tol_z)
                        if has_wt[ub-1] and (wt[ub-1] < tol_z):
                            wt[ub-1] = 0.01 * (dz[ub-1] - wf[ub-1])
                    elif lbc_typ == self.bc_fix_flux:
                        has_wt[ub-1] = (wt[ub-1] >= tol_z)

            # print(dz[ilev], vl_s[ilev], vl_r[ilev], psi_s[ilev], hksat[ilev],
            #                     nprm, prms[:, ilev], is_sat[ilev], has_wf[ilev], has_wt[ilev],
            #                     wf[ilev], vl[ilev], wt[ilev], psi[ilev], hk[ilev], True, tol_v)
            # Call the `check_and_update_level` function
            wf[ilev], vl[ilev], wt[ilev], psi[ilev], hk[ilev]= self.check_and_update_level(dz[ilev], vl_s[ilev], vl_r[ilev], psi_s[ilev], hksat[ilev],
                                nprm, prms[:, ilev], is_sat[ilev], has_wf[ilev], has_wt[ilev],
                                wf[ilev], vl[ilev], wt[ilev], psi[ilev], hk[ilev], True, tol_v)
        # print(psi, hk, '-=-=-=-=')
            
        return is_sat, has_wf, has_wt, wf, vl, wt, dp, psi, hk 

    def flux_inside_hm_soil(self, psi_s, hksat, nprm, prms, dz, psi_u, psi_l, hk_u, hk_l):
        """
        This function calculates the soil flux inside a hydrological model.

        Parameters:
        psi_s (float): Soil potential.
        hksat (float): Saturated hydraulic conductivity.
        nprm (int): Number of parameters.
        prms (list of float): List of parameters.
        dz (float): Depth increment.
        psi_u (float): Upper soil potential.
        psi_l (float): Lower soil potential.
        hk_u (float): Upper hydraulic conductivity.
        hk_l (float): Lower hydraulic conductivity.
        effective_hk_type (str): Type of effective hydraulic conductivity (self.type_upstream_mean or self.type_weighted_geometric_mean).

        Returns:
        float: The soil flux inside the hydrological model.
        """
        # print(psi_l, psi_u, dz, '-=-=-=-=-=-=')
        grad_psi = 1.0 - (psi_l - psi_u) / dz

        if self.effective_hk_type == self.type_upstream_mean:
            if grad_psi < 0:
                return hk_l * grad_psi
            else:
                return hk_u * grad_psi

        elif self.effective_hk_type == self.type_weighted_geometric_mean:
            # For Campbell Soil Model
            r0=  0.0
            if self.nl_colm['Campbell_SOIL_MODEL']:
                r0 = 1.0 / (3.0 / prms[0] + 2.0)
            # For vanGenuchten Mualem Soil Model
            if self.nl_colm['vanGenuchten_Mualem_SOIL_MODEL' ]:
                r0 = 1.0 / (prms[2] * (prms[1] - 1.0) + prms[1] * 2.0)

            if grad_psi < 0:
                rr = r0
                hk_m = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_l - dz, psi_s, hksat, nprm, prms)
                return hk_u**rr * hk_m**(1.0 - rr) * grad_psi
            elif grad_psi == 0:
                return 0
            elif 0 < grad_psi < 1:
                rr = max(1.0 + r0 * psi_l / dz, 1.0 - r0)
                return hk_u**rr * hk_l**(1.0 - rr) * grad_psi
            elif grad_psi == 1:
                return hk_u
            elif grad_psi > 1:
                rr = r0
                return hk_u + (psi_u - psi_l) / dz * hk_u**(1.0 - rr) * hk_l**rr

        return 0

    def water_balance(self, lb, ub, dz, dt, is_sat, vl_s, q, ubc_typ, ubc_val, lbc_typ, lbc_val,
                   wf, vl, wt, dp, waquifer, wf_m1, vl_m1, wt_m1, dp_m1, waquifer_m1, tol=None):
        # Constants for boundary conditions
        # self.bc_rainfall = 1  # Example value, update as needed
        # self.bc_drainage = 2  # Example value, update as needed

        # Initialize output variables
        blc = np.zeros(ub + 1 - lb + 2)
        
        # Calculate the water balance
        if ubc_typ == self.bc_rainfall:
            dmss = max(dp, 0.0) - max(dp_m1, 0.0)
            qsum = ubc_val - q[lb - 1]
            blc[lb - 1] = dmss - qsum * dt

        ilev = lb - 1
        for jlev in range(lb - 1, ub):
            dmss = (vl_s[jlev] - vl_m1[jlev]) * (wf[jlev] - wf_m1[jlev])
            dmss += (vl_s[jlev] - vl_m1[jlev]) * (wt[jlev] - wt_m1[jlev])
            dmss += (dz[jlev] - wt[jlev] - wf[jlev]) * (vl[jlev] - vl_m1[jlev])

            qsum = q[jlev] - q[jlev+ 1]

            if not is_sat[jlev]:
                ilev = jlev + 1
                if ubc_typ != self.bc_rainfall and blc[lb - 1] != 0:
                    blc[ilev] += blc[lb - 1]
                    blc[lb - 1] = 0

            blc[ilev] += dmss - qsum * dt

        if lbc_typ == self.bc_drainage:
            if waquifer == 0 and q[ub] >= 0:
                blc[ilev] -= waquifer_m1 + q[ub] * dt
            else:
                blc[ub + 1] = waquifer - waquifer_m1 - q[ub] * dt
                if ubc_typ != self.bc_rainfall and blc[lb - 1] != 0:
                    blc[ub + 1] += blc[lb - 1]
                    blc[lb - 1] = 0

        if self.is_solvable is not None:
            if tol is not None:
                self.is_solvable = (ubc_typ == self.bc_rainfall) or (blc[lb - 1] < tol)
            else:
                self.is_solvable = (ubc_typ == self.bc_rainfall) or (blc[lb - 1] == 0)

        return blc, self.is_solvable

    def flux_at_unsaturated_interface(self, nprm, psi_s_u, hksat_u, prms_u, dz_u, psi_u, hk_u,
                                  psi_s_l, hksat_l, prms_l, dz_l, psi_l, hk_l, 
                                  tol_q, tol_p):
        """
        This function calculates the flux at an unsaturated interface between two soil layers.

        Parameters:
        nprm (int): Number of parameters.
        psi_s_u (float): Soil potential for upper layer.
        hksat_u (float): Saturated hydraulic conductivity for upper layer.
        prms_u (list of float): List of parameters for upper layer.
        dz_u (float): Depth increment for upper layer.
        psi_u (float): Soil potential at the upper layer.
        hk_u (float): Hydraulic conductivity at the upper layer.
        psi_s_l (float): Soil potential for lower layer.
        hksat_l (float): Saturated hydraulic conductivity for lower layer.
        prms_l (list of float): List of parameters for lower layer.
        dz_l (float): Depth increment for lower layer.
        psi_l (float): Soil potential at the lower layer.
        hk_l (float): Hydraulic conductivity at the lower layer.
        tol_q (float): Tolerance for flux calculation.
        tol_p (float): Tolerance for potential calculation.

        Returns:
        tuple: (flux_u, flux_l) Flux values for the upper and lower layers.
        """
        flux_u = 0.0
        flux_l = 0.0
        psi_i_r = max(psi_u + dz_u, psi_l - dz_l)
        psi_i_l = min(psi_u + dz_u, psi_l - dz_l)

        psi_s_min = min(psi_s_u, psi_s_l)

        if psi_i_r > psi_s_min:
            hk_i_u = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_s_min, psi_s_u, hksat_u, nprm, prms_u)
            hk_i_l = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_s_min, psi_s_l, hksat_l, nprm, prms_l)

            flux_u = self.flux_inside_hm_soil(psi_s_u, hksat_u, nprm, prms_u, dz_u, psi_u, psi_s_min, hk_u, hk_i_u)
            flux_l = self.flux_inside_hm_soil(psi_s_l, hksat_l, nprm, prms_l, dz_l, psi_s_min, psi_l, hk_i_l, hk_l)

            if flux_u >= flux_l:
                return flux_u, flux_l
            else:
                psi_i_r = psi_s_min

        psi_i = (dz_l * psi_u + dz_u * psi_l) / (dz_u + dz_l)
        if psi_i < psi_i_l or psi_i > psi_i_r:
            psi_i = (psi_i_r + psi_i_l) / 2.0

        iter = 0
        while iter < 50:
            hk_i_u = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s_u, hksat_u, nprm, prms_u)
            hk_i_l = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s_l, hksat_l, nprm, prms_l)

            flux_u = self.flux_inside_hm_soil(psi_s_u, hksat_u, nprm, prms_u, dz_u, psi_u, psi_i, hk_u, hk_i_u)
            flux_l = self.flux_inside_hm_soil(psi_s_l, hksat_l, nprm, prms_l, dz_l, psi_i, psi_l, hk_i_l, hk_l)

            fval = flux_l - flux_u

            if abs(fval) < tol_q or psi_i_r - psi_i_l < tol_p:
                break
            else:
                if iter == 0:
                    if fval < 0:
                        psi_i_l = psi_i
                    else:
                        psi_i_r = psi_i

                    psi_i_k1 = psi_i
                    fval_k1 = fval

                    psi_i = (psi_i_r + psi_i_l) / 2.0
                else:
                    fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r = self.secant_method_iteration(fval, fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r)

            iter += 1
        if self.nl_colm['CoLMDEBUG']:
            if iter == 50:
                print('Warning: flux_at_unsaturated_interface: not converged.')

        return flux_u, flux_l

    def flux_both_transitive_interface(self, ilev_us_u, ilev_us_l, dz, psi_s, hksat, nprm, prms, dz_us_u, psi_us_u, hk_us_u,
                                       dz_us_l, psi_us_l, hk_us_l, qlc, tol_q, tol_z, tol_p):
        # Local variables
        q_us_u = 0
        nlev_sat = ilev_us_l - ilev_us_u - 1
        if psi_s[ilev_us_u] <= psi_s[ilev_us_u + 1] or dz_us_u < tol_z:
            psi_i = max(psi_s[ilev_us_u], psi_s[ilev_us_u + 1])
            # Assume these are functions to be implemented later
            q_us_l, qlc = self.flux_btm_transitive_interface(psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l], dz_us_l,
                                          psi_us_l, hk_us_l, nlev_sat, dz[ilev_us_u + 1:ilev_us_l - 1],
                                          psi_s[ilev_us_u + 1:ilev_us_l - 1], hksat[ilev_us_u + 1:ilev_us_l - 1], psi_i,
                                          tol_q=tol_q, tol_z=tol_z, tol_p=tol_p)
            if dz_us_u < tol_z:
                q_us_u = qlc[ilev_us_u + 1]
            else:
                # Assume this is a function to be implemented later
                q_us_u = self.flux_inside_hm_soil(psi_s[ilev_us_u], hksat[ilev_us_u], nprm, prms[:, ilev_us_u], dz_us_u,
                                             psi_us_u, psi_s[ilev_us_u], hk_us_u, hksat[ilev_us_u])
            return q_us_u, q_us_l, qlc[ilev_us_u:ilev_us_l-1]
        if psi_s[ilev_us_l] <= psi_s[ilev_us_l - 1] or dz_us_l < tol_z:
            psi_i = max(psi_s[ilev_us_l - 1], psi_s[ilev_us_l])
            # Assume these are functions to be implemented later
            q_us_u, qlc = self.flux_top_transitive_interface(psi_s[ilev_us_u], hksat[ilev_us_u], nprm, prms[:, ilev_us_u], dz_us_u,
                                          psi_us_u, hk_us_u, nlev_sat, dz[ilev_us_u + 1:ilev_us_l - 1],
                                          psi_s[ilev_us_u + 1:ilev_us_l - 1], hksat[ilev_us_u + 1:ilev_us_l - 1], psi_i,
                                          tol_q=tol_q, tol_z=tol_z, tol_p=tol_p)
            if dz_us_l < tol_z:
                q_us_l = qlc[ilev_us_l - 1]
            else:
                # Assume this is a function to be implemented later
                q_us_l = self.flux_inside_hm_soil(psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l], dz_us_l,
                                             psi_s[ilev_us_l], psi_us_l, hksat[ilev_us_l], hk_us_l)
            return q_us_u, q_us_l, qlc[ilev_us_u:ilev_us_l-1]
        psi_i_l = psi_s[ilev_us_l - 1]
        # Assume these are functions to be implemented later
        q_us_u, qlc = self.flux_top_transitive_interface(psi_s[ilev_us_u], hksat[ilev_us_u], nprm, prms[:, ilev_us_u], dz_us_u, psi_us_u,
                                      hk_us_u, nlev_sat, dz[ilev_us_u + 1:ilev_us_l - 1],
                                      psi_s[ilev_us_u + 1:ilev_us_l - 1], hksat[ilev_us_u + 1:ilev_us_l - 1], psi_i_l,
                                      tol_q=tol_q / 2, tol_z=tol_z, tol_p=tol_p)
        # Assume this is a function to be implemented later
        hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i_l, psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l])
        q_us_l = self.flux_inside_hm_soil(psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l], dz_us_l, psi_i_l,
                                     psi_us_l, hk_i, hk_us_l)
        if qlc[ilev_us_l - 1] <= q_us_l:
            return q_us_u, q_us_l, qlc[ilev_us_u:ilev_us_l-1]
        psi_i_r = psi_s[ilev_us_l]
        # Assume these are functions to be implemented later
        q_us_u, qlc = self.flux_top_transitive_interface(psi_s[ilev_us_u], hksat[ilev_us_u], nprm, prms[:, ilev_us_u], dz_us_u, psi_us_u,
                                      hk_us_u, nlev_sat, dz[ilev_us_u + 1:ilev_us_l - 1],
                                      psi_s[ilev_us_u + 1:ilev_us_l - 1], hksat[ilev_us_u + 1:ilev_us_l - 1], psi_i_r,
                                      tol_q=tol_q / 2, tol_z=tol_z, tol_p=tol_p)
        # Assume this is a function to be implemented later
        hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i_r, psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l])
        q_us_l = self.flux_inside_hm_soil(psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l], dz_us_l, psi_i_r,
                                     psi_us_l, hk_i, hk_us_l)
        if qlc[ilev_us_l - 1] >= q_us_l:
            return q_us_u, q_us_l, qlc[ilev_us_u:ilev_us_l-1]
        psi_i_k1 = psi_i_r
        fval_k1 = q_us_l - qlc[ilev_us_l - 1]
        psi_i = (psi_i_r + psi_i_l) / 2
        iter = 0
        while iter < 50:
            # Assume these are functions to be implemented later
            q_us_u, qlc = self.flux_top_transitive_interface(psi_s[ilev_us_u], hksat[ilev_us_u], nprm, prms[:, ilev_us_u], dz_us_u,
                                          psi_us_u, hk_us_u, nlev_sat, dz[ilev_us_u + 1:ilev_us_l - 1],
                                          psi_s[ilev_us_u + 1:ilev_us_l - 1], hksat[ilev_us_u + 1:ilev_us_l - 1], psi_i,
                                          tol_q=tol_q / 2.0, tol_z=tol_z, tol_p=tol_p)
            # Assume this is a function to be implemented later
            hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l])
            q_us_l = self.flux_inside_hm_soil(psi_s[ilev_us_l], hksat[ilev_us_l], nprm, prms[:, ilev_us_l], dz_us_l, psi_i,
                                         psi_us_l, hk_i, hk_us_l)
            fval = q_us_l - qlc[ilev_us_l - 1]
            if abs(fval) < tol_q or psi_i_r - psi_i_l < tol_p:
                break
            else:
                # Assume this is a function to be implemented later
                fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r = self.secant_method_iteration(fval, fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r)
            iter += 1
        if iter == 50:
            print('Warning : flux_both_transitive_interface: not converged.')
        return q_us_u, q_us_l, qlc[ilev_us_u:ilev_us_l-1]


    def flux_sat_zone_fixed_bc(self, nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_btm, qlc, flux_top=None, flux_btm=None):
        # Local variables
        psi = np.zeros(nlev_sat)
        psi[0] = psi_top
        psi[nlev_sat-1] = psi_btm
        spr = np.zeros(nlev_sat)

        if flux_top is not None and flux_btm is not None:
            if flux_top >= flux_btm:
                qlc[:] = flux_btm
                return qlc

        for ilev in range(nlev_sat):
            if ilev < nlev_sat-1:
                psi[ilev] = max(psi_sat[ilev], psi_sat[ilev+1])
            qlc[ilev] = -hk_sat[ilev] * ((psi[ilev] - psi[ilev - 1]) / dz_sat[ilev] - 1)
            spr[ilev] = ilev
        
        ilev_u = nlev_sat
        ilev_l = ilev_u

        # print(ilev_l, nlev_sat, len(qlc),'************')

        while True:
            if ilev_l < nlev_sat-1:
                ilev = np.where(spr == spr[ilev_l+1])[0][-1]  # Equivalent to findloc with BACK option in Fortran
                while qlc[ilev_u] >= qlc[ilev]:
                    ilev_l = ilev
                    qlc[ilev_u:ilev_l+1] = - (psi[ilev_l] - psi[ilev_u - 1] -
                                                np.sum(dz_sat[ilev_u - 1:ilev_l])) / \
                                            np.sum(dz_sat[ilev_u :ilev_l+1] / hk_sat[ilev_u:ilev_l+1])
                    spr[ilev_u:ilev_l+1] = ilev_u
                    if ilev_l < nlev_sat-1:
                        spr[ilev_l:] -= 1
                        ilev = np.where(spr == spr[ilev_l])[0][-1]
                    else:
                        break

            if ilev_l == nlev_sat and flux_btm is not None:
                if qlc[ilev_l - 1] > flux_btm:
                    qlc[ilev_u - 1:ilev_l] = flux_btm

            if ilev_u > 1:
                ilev_u -= 1
                ilev_l = ilev_u
            else:
                if flux_top is not None:
                    for ilev in range(1, nlev_sat + 1):
                        if flux_top > qlc[ilev - 1]:
                            qlc[ilev - 1] = flux_top
                        else:
                            break
                break
        return qlc

    def flux_btm_transitive_interface(self,
        psi_s_l, hksat_l, nprm, prms_l,
        dz_us, psi_us, hk_us,
        nlev_sat, dz_sat, psi_sat, hk_sat, psi_top,
        tol_q, tol_z, tol_p, flux_top=None):
        # Initialize variables
        q_us_l = 0.0
        qlc = np.zeros(nlev_sat)  # assuming qlc is provided or initialized elsewhere

        if dz_us < tol_z:
            psi_i = max(psi_sat[nlev_sat - 1], psi_s_l)
            if flux_top is not None:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc, flux_top=flux_top)
            else:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc)
            q_us_l = qlc[nlev_sat - 1]
            return q_us_l, qlc

        if psi_sat[nlev_sat - 1] >= psi_s_l:
            psi_i = psi_s_l
            hk_i = hksat_l
            q_us_l = self.flux_inside_hm_soil(psi_s_l, hksat_l, nprm, prms_l, dz_us, psi_i, psi_us, hk_i, hk_us)

            psi_i = psi_sat[nlev_sat - 1]
            if flux_top is not None:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc, flux_top=flux_top)
            else:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc)
            return q_us_l, qlc

        psi_i = psi_sat[nlev_sat - 1]

        if flux_top is not None:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc, flux_top=flux_top)
        else:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc)
        
        hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s_l, hksat_l, nprm, prms_l)
        q_us_l = self.flux_inside_hm_soil(psi_s_l, hksat_l, nprm, prms_l, dz_us, psi_i, psi_us, hk_i, hk_us)

        if qlc[nlev_sat - 1] <= q_us_l:
            return q_us_l, qlc
        else:
            psi_i_l = psi_sat[nlev_sat - 1]

        psi_i = psi_s_l
        if flux_top is not None:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc, flux_top=flux_top)
        else:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc)

        hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s_l, hksat_l, nprm, prms_l)
        q_us_l = self.flux_inside_hm_soil(psi_s_l, hksat_l, nprm, prms_l, dz_us, psi_i, psi_us, hk_i, hk_us)

        if qlc[nlev_sat - 1] >= q_us_l:
            return q_us_l, qlc
        else:
            psi_i_r = psi_s_l

        psi_i_k1 = psi_i_r
        fval_k1 = q_us_l - qlc[nlev_sat - 1]

        psi_i = (psi_i_r + psi_i_l) / 2.0
        iter = 0
        while iter < 50:
            if flux_top is not None:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc, flux_top=flux_top)
            else:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_top, psi_i, qlc)

            hk_i = CoLM_Hydro_SoilFunction. soil_hk_from_psi(self.nl_colm, psi_i, psi_s_l, hksat_l, nprm, prms_l)
            q_us_l = self.flux_inside_hm_soil(psi_s_l, hksat_l, nprm, prms_l, dz_us, psi_i, psi_us, hk_i, hk_us)

            fval = q_us_l - qlc[nlev_sat - 1]

            if abs(fval) < tol_q or (psi_i_r - psi_i_l < tol_p):
                break
            else:
                fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r = self.secant_method_iteration(fval, fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r)
            
            iter += 1

        if iter == 50:
            print('Warning: flux_btm_transitive_interface: not converged.')

        return q_us_l, qlc
    
    def flux_top_transitive_interface(self,
        psi_s_u, hksat_u, nprm, prms_u,
        dz_us, psi_us, hk_us,
        nlev_sat, dz_sat, psi_sat, hk_sat, psi_btm,
        tol_q, tol_z, tol_p, flux_btm=None):
        # Initialize variables
        q_us_up = 0.0
        qlc = np.zeros(nlev_sat)  # Initialize qlc array
        psi_i_l = None
        psi_i_r = None

        if dz_us < tol_z:
            psi_i = max(psi_s_u, psi_sat[0])
            if flux_btm is not None:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc, flux_btm=flux_btm)
            else:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc)
            q_us_up = qlc[0]
            return q_us_up, qlc

        if psi_s_u <= psi_sat[0]:
            # The case psi_s_u < psi_sat(1) does not exist in principle.
            psi_i = psi_s_u
            hk_i = hksat_u
            q_us_up = self.flux_inside_hm_soil(psi_s_u, hksat_u, nprm, prms_u, dz_us, psi_us, psi_i, hk_us, hk_i)
            psi_i = psi_sat[0]
            if flux_btm is not None:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc, flux_btm=flux_btm)
            else:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc)
            return q_us_up, qlc

        psi_i = psi_sat[0]
        hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s_u, hksat_u, nprm, prms_u)
        q_us_up = self.flux_inside_hm_soil(psi_s_u, hksat_u, nprm, prms_u, dz_us, psi_us, psi_i, hk_us, hk_i)

        if flux_btm is not None:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc, flux_btm=flux_btm)
        else:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc)

        if q_us_up <= qlc[0]:
            return q_us_up, qlc
        else:
            psi_i_l = psi_sat[0]

        psi_i = psi_s_u
        hk_i = hksat_u
        q_us_up = self.flux_inside_hm_soil(psi_s_u, hksat_u, nprm, prms_u, dz_us, psi_us, psi_i, hk_us, hk_i)

        if flux_btm is not None:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc, flux_btm=flux_btm)
        else:
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc)

        if q_us_up >= qlc[0]:
            return q_us_up, qlc
        else:
            psi_i_r = psi_s_u

        psi_i_k1 = psi_i_r
        fval_k1 = qlc[0] - q_us_up

        psi_i = (psi_i_r + psi_i_l) / 2.0
        iter = 0
        while iter < 50:
            hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s_u, hksat_u, nprm, prms_u)
            q_us_up = self.flux_inside_hm_soil(psi_s_u, hksat_u, nprm, prms_u, dz_us, psi_us, psi_i, hk_us, hk_i)

            if flux_btm is not None:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc, flux_btm=flux_btm)
            else:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_i, psi_btm, qlc)

            fval = qlc[0] - q_us_up

            if abs(fval) < tol_q or (psi_i_r - psi_i_l) < tol_p:
                break
            else:
                fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r = self.secant_method_iteration(fval, fval_k1, psi_i, psi_i_k1, psi_i_l, psi_i_r)

            iter += 1

        if self.nl_colm['CoLMDEBUG']:
            if iter == 50:
                print('Warning: flux_top_transitive_interface: not converged.')

        return q_us_up, qlc

    def flux_sat_zone_all(self, lb, ub, i_stt, i_end, dz, sp_zc, sp_zi,
                      vl_s, psi_s, hksat, nprm, prms,
                      ubc_typ, ubc_val, lbc_typ, lbc_val,
                      is_sat, has_wf, has_wt, is_update_sublevel,
                      wf, vl, wt, wdsrf, infl_max, zwt, psi_us, hk_us,
                      qq, qq_wt, qq_wf, tol_q, tol_z, tol_p):

        # Local variables
        i_stt = i_stt-1
        i_end=  i_end - 1
        is_trans = False
        top_at_ground = (i_stt == lb-1) and is_sat[i_stt]
        top_at_interface = (not top_at_ground) and (wt[i_stt] < tol_z)
        top_inside_level = not (top_at_ground or top_at_interface)

        btm_at_bottom = (i_end == ub-1) and is_sat[i_end]
        btm_at_interface = (not btm_at_bottom) and (wf[i_end] < tol_z)
        btm_inside_level = not (btm_at_bottom or btm_at_interface)

        if top_at_interface:
            i_s = i_stt + 1
        else:
            i_s = i_stt

        if btm_at_interface:
            i_e = i_end - 1
        else:
            i_e = i_end

        nlev_sat = i_e - i_s + 1

        qlc = np.zeros(nlev_sat)

        dz_sat = copy.copy(dz[i_s:i_e+1])
        psi_sat = copy.copy(psi_s[i_s:i_e+1])
        hk_sat = copy.copy(hksat[i_s:i_e+1])

        if top_inside_level:
            dz_sat[i_s-i_s] = wt[i_stt]
        if btm_inside_level:
            dz_sat[i_e-i_s] = wf[i_end]

        if not top_at_ground:
            dz_us_top = (dz[i_stt] - wt[i_stt] - wf[i_stt]) * (sp_zi[i_stt+1] - sp_zc[i_stt]) / dz[i_stt]

        if not btm_at_bottom:
            dz_us_btm = (dz[i_end] - wt[i_end] - wf[i_end]) * (sp_zc[i_end] - sp_zi[i_end]) / dz[i_end]

        # Case 1
        if top_at_ground and btm_at_bottom:
            if (ubc_typ == self.bc_fix_head) and (lbc_typ == self.bc_fix_head):
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ubc_val, lbc_val, qlc)
            if (ubc_typ == self.bc_rainfall) and (lbc_typ == self.bc_fix_head):
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, wdsrf, lbc_val, qlc)
                if (wdsrf < tol_z) and (qlc[lb] > infl_max):
                    ptop = psi_s[lb]
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ptop, lbc_val, qlc, flux_top=infl_max)
            if (ubc_typ == self.bc_fix_flux) and (lbc_typ == self.bc_fix_head):
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[lb], lbc_val, qlc, flux_top=ubc_val)
            if (ubc_typ == self.bc_fix_head) and (lbc_typ == self.bc_fix_flux):
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ubc_val, psi_s[ub-1], qlc, flux_btm=lbc_val)
            if (ubc_typ == self.bc_rainfall) and (lbc_typ == self.bc_fix_flux):
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, wdsrf, psi_s[ub-1], qlc, flux_btm=lbc_val)
                if (wdsrf < tol_z) and (qlc[lb] > infl_max):
                    ptop = psi_s[lb]
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ptop, psi_s[ub-1], qlc, flux_top=infl_max, flux_btm=lbc_val)
            if (ubc_typ == self.bc_fix_flux) and (lbc_typ == self.bc_fix_flux):
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[lb], psi_s[ub-1], qlc, flux_top=ubc_val, flux_btm=lbc_val)
            if (ubc_typ == self.bc_fix_head) and (lbc_typ == self.bc_drainage):
                if zwt > sp_zi[ub]:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ubc_val, psi_s[ub-1], qlc)
                else:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ubc_val, psi_s[ub-1], qlc, flux_btm=0.0)
            if (ubc_typ == self.bc_rainfall) and (lbc_typ == self.bc_drainage):
                if zwt > sp_zi[ub]:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, wdsrf, psi_s[ub-1], qlc)
                else:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, wdsrf, psi_s[ub-1], qlc, flux_btm=0.0)
                if (wdsrf < tol_z) and (qlc[lb] > infl_max):
                    ptop = psi_s[lb]
                    if zwt > sp_zi[ub]:
                        qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ptop, psi_s[ub-1], qlc, flux_top=infl_max)
                    else:
                        qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ptop, psi_s[ub-1], qlc, flux_top=infl_max, flux_btm=0.0)
            if (ubc_typ == self.bc_fix_flux) and (lbc_typ == self.bc_drainage):
                if zwt > sp_zi[ub]:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[lb], psi_s[ub-1], qlc, flux_top=ubc_val)
                else:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[lb], psi_s[ub-1], qlc, flux_top=ubc_val, flux_btm=0.0)

        # Case 2
        if top_at_ground and btm_at_interface:
            if ubc_typ == self.bc_fix_head:
                qq_wf[i_end] = self.flux_btm_transitive_interface(psi_s[i_end], hksat[i_end], nprm, prms[:,i_end],
                                            dz_us_btm, psi_us[i_end], hk_us[i_end],
                                            nlev_sat, dz_sat, psi_sat, hk_sat, ubc_val,
                                            qq_wf[i_end], qlc, tol_q, tol_z, tol_p)
            if ubc_typ == self.bc_fix_flux:
                qq_wf[i_end] = self.flux_btm_transitive_interface(psi_s[i_end], hksat[i_end], nprm, prms[:,i_end],
                                            dz_us_btm, psi_us[i_end], hk_us[i_end],
                                            nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[lb],
                                            qq_wf[i_end], qlc, tol_q, tol_z, tol_p, flux_top=ubc_val)
            if ubc_typ == self.bc_rainfall:
                qq_wf[i_end] = self.flux_btm_transitive_interface(psi_s[i_end], hksat[i_end], nprm, prms[:,i_end],
                                            dz_us_btm, psi_us[i_end], hk_us[i_end],
                                            nlev_sat, dz_sat, psi_sat, hk_sat, wdsrf,
                                            qq_wf[i_end], qlc, tol_q, tol_z, tol_p)
                if (wdsrf < tol_z) and (qlc[lb] > infl_max):
                    ptop = psi_s[lb]
                    qq_wf[i_end] = self.flux_btm_transitive_interface(psi_s[i_end], hksat[i_end], nprm, prms[:,i_end],
                                                dz_us_btm, psi_us[i_end], hk_us[i_end],
                                                nlev_sat, dz_sat, psi_sat, hk_sat, ptop,
                                                qq_wf[i_end], qlc, tol_q, tol_z, tol_p, flux_top=infl_max)

        # Case 3
        if top_at_ground and btm_inside_level:
            if ubc_typ == self.bc_fix_head:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ubc_val, psi_s[i_end], qlc)
            if ubc_typ == self.bc_fix_flux:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[lb], psi_s[i_end], qlc, flux_top=ubc_val)
            if ubc_typ == self.bc_rainfall:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, wdsrf, psi_s[i_end], qlc)
                if (wdsrf < tol_z) and (qlc[lb] > infl_max):
                    ptop = psi_s[lb]
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, ptop, psi_s[i_end], qlc, flux_top=infl_max)

        # Case 4
        if top_at_interface and btm_at_bottom:
            if lbc_typ == self.bc_fix_head:
                qq_wf[i_stt], qlc = self.flux_top_transitive_interface(psi_s[i_stt], hksat[i_stt], nprm, prms[:,i_stt],
                                            dz_us_top, psi_us[i_stt], hk_us[i_stt],
                                            nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[ub-1],
                                            tol_q, tol_z, tol_p)
            if lbc_typ == self.bc_fix_flux:
                qq_wf[i_stt], qlc = self.flux_top_transitive_interface(psi_s[i_stt], hksat[i_stt], nprm, prms[:,i_stt],
                                            dz_us_top, psi_us[i_stt], hk_us[i_stt],
                                            nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[ub-1],
                                            tol_q, tol_z, tol_p, flux_btm=lbc_val)
            if lbc_typ == self.bc_drainage:
                if zwt > sp_zi[ub]:
                    qq_wf[i_stt], qlc = self.flux_top_transitive_interface(psi_s[i_stt], hksat[i_stt], nprm, prms[:,i_stt],
                                                dz_us_top, psi_us[i_stt], hk_us[i_stt],
                                                nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[ub-1],
                                                tol_q, tol_z, tol_p)
                else:
                    qq_wf[i_stt], qlc = self.flux_top_transitive_interface(psi_s[i_stt], hksat[i_stt], nprm, prms[:,i_stt],
                                                dz_us_top, psi_us[i_stt], hk_us[i_stt],
                                                nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[ub-1],
                                                tol_q, tol_z, tol_p, flux_btm=0.0)

        # Case 5
        if top_at_interface and btm_at_interface:
            qq_wt[i_stt], qq_wf[i_end], qlc = self.flux_both_transitive_interface(
            i_stt, i_end, dz[i_stt: i_end], 
            psi_s[i_stt: i_end], hksat[i_stt: i_end], nprm, prms[:, i_stt: i_end], 
            dz_us_top, psi_us[i_stt], hk_us[i_stt], 
            dz_us_btm, psi_us[i_end], hk_us[i_end], 
            qlc,
            tol_q, tol_z, tol_p)

        # Case 6
        if top_at_interface and btm_inside_level:
            qq_wf[i_stt], qlc = self.flux_top_transitive_interface(psi_s[i_end], hksat[i_end], nprm, prms[:, i_stt],
                                                              dz_us_btm, psi_us[i_stt], hk_us[i_stt],
                                                              nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_end],
                                                              qq_wf[i_stt], qlc, tol_q, tol_z, tol_p)

        # Case 7
        if top_inside_level and btm_at_bottom:
            if lbc_typ==self.bc_fix_head:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_stt], psi_s[i_end], qlc)
            elif lbc_typ==self.bc_fix_head:
                qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_stt], psi_s[i_end], qlc)
            elif lbc_typ == self.bc_drainage:
                if zwt>sp_zi[ub]:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_stt], psi_s[i_end],
                                                      qlc)
                else:
                    qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_stt], psi_s[i_end],
                                                      qlc)

        # Case 8
        if top_at_interface and btm_inside_level:
            q_us_up, qlc = self.flux_top_transitive_interface(psi_s[i_stt], hksat[i_stt], nprm, prms[:,i_stt],
                                        dz_us_top, psi_us[i_stt], hk_us[i_stt],
                                        nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_end],
                                        qq_wf[i_stt], qlc, tol_q, tol_z, tol_p)

        # Case 9
        if top_inside_level and btm_inside_level:
            # Assume flux_sat_zone_fixed_bc is a function to be implemented later
            qlc = self.flux_sat_zone_fixed_bc(nlev_sat, dz_sat, psi_sat, hk_sat, psi_s[i_stt], psi_s[i_end], qlc)

        if top_inside_level:
            if dz_us_top < tol_z:
                qq_wt[i_stt] = qq[i_stt]
            else:
                # Assume flux_inside_hm_soil is a function to be implemented later
                qq_wt[i_stt] = self.flux_inside_hm_soil(psi_s[i_stt], hksat[i_stt], nprm, prms[:,i_stt], dz_us_top, psi_us[i_stt], psi_s[i_stt], hk_us[i_stt], hksat[i_stt])

        if top_at_interface:
            if dz_us_top < tol_z:
                qq_wf[i_stt] = qq_wt[i_stt]

        if top_at_ground:
            if ubc_typ == self.bc_fix_head:
                qq[lb - 1] = qlc[lb - 1]
                is_trans = False
            elif ubc_typ == self.bc_fix_flux:
                qq[lb - 1] = ubc_val  # min(qlc(i_stt), ubc_val)
                is_trans = qlc[lb - 1] > ubc_val
            elif ubc_typ == self.bc_rainfall:
                if wdsrf < tol_z:
                    qq[lb - 1 ] = min(qlc[lb - 1], infl_max)
                    is_trans = qlc[lb - 1] > infl_max
                else:
                    qq[lb - 1 ] = qlc[lb - 1]
                    is_trans = False

            if is_update_sublevel:
                if is_trans and is_sat[lb - 1]:
                    is_sat[lb - 1] = False
                    has_wf[lb - 1] = False
                    has_wt[lb - 1] = True

                    wt[lb - 1] = dz[lb - 1]
                    vl[lb - 1] = vl_s[lb - 1]
                    wf[lb - 1] = 0

                    qq_wt[lb - 1] = qq[lb - 1]

        for iface in range(i_stt, i_end):
            if top_at_interface and iface == i_stt:
                qupper = qq_wt[i_stt]
            else:
                qupper = qlc[iface-i_stt]

            if btm_at_interface and iface == i_end - 1:
                qlower = qq_wf[i_end]
            else:
                # print(iface, i_stt, len(qlc),iface + 1-i_stt, '---qlower----')
                qlower = qlc[iface + 1-i_stt]

            if qlower - qupper >= tol_z:
                if (psi_s[iface] < psi_s[iface + 1] or
                        (psi_s[iface] == psi_s[iface + 1] and is_sat[iface + 1]) or
                        (top_at_interface and iface == i_stt)):
                    qq[iface+1] = qupper

                    if is_update_sublevel and is_sat[iface + 1]:
                        is_sat[iface + 1] = False
                        has_wf[iface + 1] = False
                        has_wt[iface + 1] = True

                        wt[iface + 1] = dz[iface + 1]
                        vl[iface + 1] = vl_s[iface + 1]
                        wf[iface + 1] = 0

                        qq_wf[iface + 1] = qq[iface+1]
                        qq_wt[iface + 1] = qq[iface+1]

                        if top_at_interface and iface == i_stt:
                            has_wt[iface] = False
                elif (psi_s[iface] > psi_s[iface + 1] or
                      (psi_s[iface] == psi_s[iface + 1] and not is_sat[iface + 1]) or
                      (btm_at_interface and iface == i_end - 1)):
                    qq[iface+1] = qlower

                    if is_update_sublevel and is_sat[iface]:
                        is_sat[iface] = False
                        has_wt[iface] = False
                        has_wf[iface] = True

                        wf[iface] = dz[iface]
                        vl[iface] = vl_s[iface]
                        wt[iface] = 0

                        qq_wf[iface] = qq[iface+1]
                        qq_wt[iface] = qq[iface+1]

                        if btm_at_interface and iface == i_end - 1:
                            has_wf[iface + 1] = False
            elif qupper - qlower >= tol_z:
                if top_at_interface and iface == i_stt:
                    qq[iface+1] = qlower
                if btm_at_interface and iface == i_end - 1:
                    qq[iface+1] = qupper
            else:
                qq[iface+1] = (qupper + qlower) * 0.5

        if btm_at_bottom:
            # print(ub, i_stt,len(qlc),len(qq))
            qq[ub] = qlc[ub - 1-i_stt]

        if btm_at_interface:
            if dz_us_btm < tol_z:
                qq_wt[i_end] = qq_wf[i_end]

        if btm_inside_level:
            if dz_us_btm < tol_z:
                qq_wf[i_end] = qq[i_end+1]
            else:
                # Assume flux_inside_hm_soil is a function to be implemented later
                qq_wf[i_end] = self.flux_inside_hm_soil(psi_s[i_end], hksat[i_end], nprm, prms[:,i_end], dz_us_btm, psi_s[i_end], psi_us[i_end], hksat[i_end], hk_us[i_end])

        del qlc
        del dz_sat
        del psi_sat
        del hk_sat
        return is_sat, has_wf, has_wt, wf, vl, wt, qq, qq_wt, qq_wf

    def find_unsat_lev_lower(self, is_sat, lb, ub, ilev):
        """
        This function finds the first unsatisfied level starting from ilev within the bounds lb and ub.

        Parameters:
        is_sat (list of bool): List of boolean values indicating whether each level is satisfied.
        lb (int): Lower bound of the range.
        ub (int): Upper bound of the range.
        ilev (int): Starting level to check.

        Returns:
        int: The first unsatisfied level starting from ilev within the bounds lb and ub.
        """
        find_unsat_lev_lower = ilev
        while find_unsat_lev_lower <= ub:
            if is_sat[find_unsat_lev_lower-1]:
                find_unsat_lev_lower += 1
            else:
                break
        return find_unsat_lev_lower

    def flux_all(self, lb, ub, dz, sp_zc, sp_zi, vl_s, psi_s, hksat, nprm, prms, 
                ubc_typ, ubc_val, lbc_typ, lbc_val, lev_update, is_update_sublevel, 
                is_sat, has_wf, has_wt, infl_max, wf, vl, wt, dp, zwt, 
                psi_us, hk_us, qq, qq_wf, qq_wt, tol_q, tol_z, tol_p):

        # Local variables
        ilev_u = lb - 1
        ilev_l = self.find_unsat_lev_lower(is_sat, lb, ub, ilev_u + 1)
        # print(ilev_u, ilev_l, '-=-=-=-==')
        while True:
            if lev_update[ilev_u] or lev_update[ilev_l]:
                if ilev_l == lb:
                    # CASE 1: water flux on top
                    dz_this = (dz[lb-1] - wt[lb-1] - wf[lb-1]) * (sp_zc[lb-1] - sp_zi[lb - 1]) / dz[lb-1]
                    if ubc_typ == self.bc_fix_head:
                        if has_wf[lb - 1]:
                            qq[lb - 1] = -hksat[lb-1] * ((psi_s[lb-1] - ubc_val) / wf[lb-1] - 1)
                        else:
                            hk_top = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, ubc_val, psi_s[lb-1], hksat[lb-1], nprm, prms[:, lb-1])
                            qq[lb - 1] = self.flux_inside_hm_soil(
                                psi_s[lb-1], hksat[lb-1], nprm, prms[:, lb-1],
                                dz_this, ubc_val, psi_us[lb-1], hk_top, hk_us[lb-1])
                    elif ubc_typ == self.bc_rainfall:
                        if has_wf[lb-1]:
                            if wf[lb-1] < tol_z:
                                qq[lb - 1] = infl_max
                            else:
                                qq[lb - 1] = -hksat[lb-1] * ((psi_s[lb-1] - dp) / wf[lb-1] - 1)
                                qq[lb - 1] = min(qq[lb - 1], infl_max)
                        else:
                            qq[lb - 1] = infl_max
                            if is_update_sublevel and infl_max > hksat[lb-1]:
                                qtest = self.flux_inside_hm_soil(
                                    psi_s[lb-1], hksat[lb-1], nprm, prms[:, lb-1],
                                    dz_this, psi_s[lb-1], psi_us[lb-1], hksat[lb-1], hk_us[lb-1])
                                if qq[lb - 1] > qtest:
                                    has_wf[lb-1] = True
                                    wf[lb-1] = 0.0
                    elif ubc_typ == self.bc_fix_flux:
                        qq[lb - 1] = ubc_val
                        if is_update_sublevel and not has_wf[lb-1] and ubc_val > hksat[lb-1]:
                            qtest = self.flux_inside_hm_soil(
                                psi_s[lb-1], hksat[lb-1], nprm, prms[:, lb-1],
                                dz_this, psi_s[lb-1], psi_us[lb-1], hksat[lb-1], hk_us[lb-1])
                            if qq[lb - 1] > qtest:
                                has_wf[lb-1] = True
                                wf[lb-1] = 0.0
                    if has_wf[lb-1] and dz_this >= tol_z:
                        qq_wf[lb-1] = self.flux_inside_hm_soil(
                            psi_s[lb-1], hksat[lb-1], nprm, prms[:, lb-1],
                            dz_this, psi_s[lb-1], psi_us[lb-1], hksat[lb-1], hk_us[lb-1])
                    else:
                        qq_wf[lb-1] = qq[lb] if has_wf[lb-1] else qq[lb - 1]
                elif ilev_u == ub:
                    # CASE 2: water flux at bottom
                    dz_this = (dz[ub-1] - wf[ub-1] - wt[ub-1]) * (sp_zi[ub] - sp_zc[ub-1]) / dz[ub-1]
                    if lbc_typ == self.bc_fix_head:
                        if has_wt[ub-1]:
                            qq[ub] = -hksat[ub-1] * ((lbc_val - psi_s[ub-1]) / wt[ub-1] - 1)
                        else:
                            hk_btm = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, lbc_val, psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1])
                            qq[ub] = self.flux_inside_hm_soil(
                                psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1],
                                dz_this, psi_us[ub-1], lbc_val, hk_us[ub-1], hk_btm)
                    elif lbc_typ == self.bc_drainage:
                        if has_wt[ub-1]:
                            qq[ub] = hksat[ub-1] if zwt > sp_zi[ub] else 0
                        else:
                            if zwt > sp_zi[ub]:
                                pbtm = psi_s[ub-1] + sp_zi[ub] - zwt
                                hk_btm = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, pbtm, psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1])
                                qq[ub] = self.flux_inside_hm_soil(
                                    psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1],
                                    dz_this, psi_us[ub-1], pbtm, hk_us[ub-1], hk_btm)
                            else:
                                qq[ub] = self.flux_inside_hm_soil(
                                    psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1],
                                    dz_this, psi_us[ub-1], psi_s[ub-1], hk_us[ub-1], hksat[ub-1])
                                if is_update_sublevel and qq[ub-1] > 0:
                                    has_wt[ub-1] = True
                                    wt[ub-1] = 0
                                    qq[ub] = 0
                    elif lbc_typ == self.bc_fix_flux:
                        qq[ub] = lbc_val
                        if is_update_sublevel and not has_wt[ub-1] and lbc_val < hksat[ub-1]:
                            qtest = self.flux_inside_hm_soil(
                                psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1],
                                dz_this, psi_us[ub-1], psi_s[ub-1], hk_us[ub-1], hksat[ub-1])
                            if qtest > lbc_val:
                                has_wt[ub-1] = True
                                wt[ub-1] = 0
                    if has_wt[ub-1] and dz_this >= tol_z:
                        qq_wt[ub-1] = self.flux_inside_hm_soil(
                            psi_s[ub-1], hksat[ub-1], nprm, prms[:, ub-1],
                            dz_this, psi_us[ub-1], psi_s[ub-1], hk_us[ub-1], hksat[ub-1])
                    else:
                        qq_wt[ub-1] = qq[ub-1] if has_wt[ub-1] else qq[ub]
                else:
                    # CASE 3: inside soil column
                    has_sat_zone = False
                    if ilev_u == lb - 1 or ilev_l == ub + 1:
                        has_sat_zone = True
                    elif has_wf[ilev_l-1]:
                        has_sat_zone = True
                        if ilev_l == ilev_u + 1 and wf[ilev_l-1] < tol_z and wt[ilev_u-1] < tol_z:
                            has_sat_zone = False
                    if has_sat_zone:
                        # CASE 3(1): inside soil column, saturated zone
                        is_sat, has_wf, has_wt, wf, vl, wt, qq, qq_wt, qq_wf = self.flux_sat_zone_all(
                            lb, ub, max(ilev_u, lb), min(ilev_l, ub),
                            dz, sp_zc, sp_zi, vl_s, psi_s, hksat, nprm, prms,
                            ubc_typ, ubc_val, lbc_typ, lbc_val,
                            is_sat, has_wf, has_wt, is_update_sublevel,
                            wf, vl, wt, dp, infl_max, zwt, psi_us, hk_us,
                            qq, qq_wt, qq_wf, tol_q, tol_z, tol_p)
                        # print(qq_wt, '------qqwt6-----')
                    else:
                        # CASE 3(2): inside soil column, unsaturated zone

                        dz_upp = (dz[ilev_u-1] - wf[ilev_u-1]) * (sp_zi[ilev_u] - sp_zc[ilev_u-1]) / dz[ilev_u-1]
                        dz_low = (dz[ilev_l-1] - wt[ilev_l-1]) * (sp_zc[ilev_l-1] - sp_zi[ilev_u]) / dz[ilev_l-1]
                        # print(dz_low, dz_upp, tol_z, (dz_upp >= tol_z) and (dz_low >= tol_z),(dz_upp >= tol_z) and (dz_low < tol_z),(dz_upp < tol_z) and (dz_low >= tol_z),(dz_upp < tol_z) and (dz_low < tol_z),'----********-----')
                        # if dz_low==0.0:
                        #     print(ilev_l, ilev_u,dz[ilev_l-1],wt[ilev_l-1],sp_zc[ilev_l-1],sp_zi[ilev_u],'----======-----')
                        if (dz_upp >= tol_z) and (dz_low >= tol_z):
                            qq_wt[ilev_u-1], qq_wf[ilev_l-1] = self.flux_at_unsaturated_interface(
                                nprm, psi_s[ilev_u-1], hksat[ilev_u-1], prms[:, ilev_u-1], dz_upp, psi_us[ilev_u-1], hk_us[ilev_u-1],
                                psi_s[ilev_l-1], hksat[ilev_l-1], prms[:, ilev_l-1], dz_low, psi_us[ilev_l-1], hk_us[ilev_l-1],
                                tol_q, tol_p)
                            # print(qq_wt[ilev_u-1], qq_wf[ilev_l-1], hksat[ilev_u-1], prms[:, ilev_u-1], dz_upp, psi_us(ilev_u), hk_us[ilev_u-1],
                            #     psi_s[ilev_l-1], hksat[ilev_l-1], prms[:, ilev_l-1], dz_low, psi_us[ilev_l-1], hk_us[ilev_l-1], '------qqwt4-----')

                            if abs(qq_wt[ilev_u-1] - qq_wf[ilev_l-1]) < tol_q:
                                qq[ilev_u] = (qq_wt[ilev_u-1] + qq_wf[ilev_l-1]) * 0.5
                                # print(ilev_u ,ilev_l,qq_wt[ilev_u-1],qq_wf[ilev_l-1], qq[ilev_u],'-=-=-qq=-=-=-=')
                                qq_wt[ilev_u-1] = qq[ilev_u]
                                qq_wf[ilev_l-1] = qq[ilev_u]
                                # print(ilev_u, qq_wt[ilev_u-1], '------qqwt7-----')
                                if is_update_sublevel:
                                    has_wt[ilev_u-1] = False
                                    has_wf[ilev_l-1] = False
                            elif qq_wt[ilev_u-1] > qq_wf[ilev_l-1]:
                                if is_update_sublevel:
                                    has_wt[ilev_u-1] = True
                                    wt[ilev_u-1] = 0
                                    has_wf[ilev_l-1] = True
                                    wf[ilev_l-1] = 0
                                if has_wt[ilev_u-1] and has_wf[ilev_l-1]:
                                    if psi_s[ilev_u-1] >= psi_s[ilev_l-1]:
                                        qq[ilev_u] = qq_wt[ilev_u-1]
                                    else:
                                        qq[ilev_u] = qq_wf[ilev_l-1]
                                else:
                                    qq[ilev_u] = (qq_wt[ilev_u-1] + qq_wf[ilev_l-1]) * 0.5
                        elif (dz_upp >= tol_z) and (dz_low < tol_z):
                            psi_i = min(psi_s[ilev_u-1], psi_s[ilev_l-1])
                            hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s[ilev_u-1], hksat[ilev_u-1], nprm, prms[:, ilev_u-1])
                            qq[ilev_u] = self.flux_inside_hm_soil(
                                psi_s[ilev_u-1], hksat[ilev_u-1], nprm, prms[:, ilev_u-1],
                                dz_upp, psi_us[ilev_u-1], psi_i, hk_us[ilev_u-1], hk_i
                            )
                            qq_wt[ilev_u-1] = qq[ilev_u]
                            qq_wf[ilev_l-1] = qq[ilev_u]
                            qq_wt[ilev_l-1] = qq[ilev_u]
                        elif (dz_upp < tol_z) and (dz_low >= tol_z):
                            psi_i = min(psi_s[ilev_u-1], psi_s[ilev_l-1])
                            hk_i = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_i, psi_s[ilev_l-1], hksat[ilev_l-1], nprm, prms[:, ilev_l-1])
                            qq[ilev_u] = self.flux_inside_hm_soil(
                                psi_s[ilev_l-1], hksat[ilev_l-1], nprm, prms[:, ilev_l-1],
                                dz_low, psi_i, psi_us[ilev_l-1], hk_i, hk_us[ilev_l-1]
                            )
                            qq_wf[ilev_u-1] = qq[ilev_u]
                            qq_wt[ilev_u-1] = qq[ilev_u]
                            qq_wf[ilev_l-1] = qq[ilev_u]
                        elif (dz_upp < tol_z) and (dz_low < tol_z):
                            # This CASE does not exist in principle.
                            qq[ilev_u] = min(hksat[ilev_u-1], hksat[ilev_l-1])
                            qq_wf[ilev_u-1] = qq[ilev_u]
                            qq_wt[ilev_u-1] = qq[ilev_u]
                            qq_wf[ilev_l-1] = qq[ilev_u]
                            qq_wt[ilev_l-1] = qq[ilev_u]
            if ilev_l == ub+1:
                break
            else:
                ilev_u = ilev_l
                ilev_l = self.find_unsat_lev_lower(is_sat, lb, ub, ilev_u + 1)
        return is_sat, has_wf, has_wt, wf, vl, wt, dp, zwt, qq, qq_wf, qq_wt

    def use_explicit_form(self,
        lb, ub, dt, dz, sp_zc, sp_zi,
        vl_s, vl_r, psi_s, hksat, nprm, prms,
        vl_s_wa,
        ubc_typ, ubc_val, lbc_typ, lbc_val,
        q, q_wf, q_wt, wf, vl, wt, dp, waquifer, zwt,
        wf_m1, vl_m1, wt_m1, dp_m1, waquifer_m1,
        tol_q, tol_z, tol_v):
        # Initialize local variables
        ilev = 0
        air_m1, wa_m1, dwat, dwat_s = 0.0, 0.0, 0.0, 0.0
        alp, zwf_this, zwt_this, vl_wa = 0.0, 0.0, 0.0, 0.0
        dmss, mblc = 0.0, 0.0

        # Handle depleted condition: decrease outflux from top down
        if ubc_typ == self.bc_rainfall:
            if dp_m1 < - (ubc_val - q[lb-1]) * dt:
                q[lb-1] = dp_m1 / dt + ubc_val

        for ilev in range(lb-1, ub):
            dwat = (q[ilev] - q[ilev+1]) * dt
            wa_m1 = (wt_m1[ilev] + wf_m1[ilev]) * vl_s[ilev] + (dz[ilev] - wt_m1[ilev] - wf_m1[ilev]) * vl_m1[ilev]
            if dwat <= -wa_m1:
                q[ilev+1] = q[ilev] + wa_m1 / dt

        if lbc_typ == self.bc_fix_flux and q[ub] < lbc_val:
            q[ub] = lbc_val
            for ilev in range(ub-1, lb - 2, -1):
                dwat = (q[ilev] - q[ilev+1]) * dt
                wa_m1 = (wt_m1[ilev] + wf_m1[ilev]) * vl_s[ilev] + (dz[ilev] - wt_m1[ilev] - wf_m1[ilev]) * vl_m1[ilev]
                if dwat <= -wa_m1:
                    q[ilev] = q[ilev+1] - wa_m1 / dt

        # Handle overfilled condition: increase influx from bottom up
        if lbc_typ == self.bc_drainage:
            if q[ub] * dt > -waquifer_m1:
                q[ub] = -waquifer_m1 / dt

        for ilev in range(ub-1, lb-2, -1):
            dwat = (q[ilev] - q[ilev+1]) * dt
            air_m1 = (vl_s[ilev] - vl_m1[ilev]) * (dz[ilev] - wt_m1[ilev] - wf_m1[ilev])
            if dwat >= air_m1:
                q[ilev] = q[ilev+1] + air_m1 / dt

        if lbc_typ == self.bc_fix_flux and q[lb-1] < ubc_val:
            q[lb-1] = ubc_val
            for ilev in range(lb-1, ub):
                dwat = (q[ilev] - q[ilev+1]) * dt
                air_m1 = (vl_s[ilev] - vl_m1[ilev]) * (dz[ilev] - wt_m1[ilev] - wf_m1[ilev])
                if dwat >= air_m1:
                    q[ilev+1] = q[ilev] - air_m1 / dt

        # Update prognostic variables: dp, wf, vl, wt, zwt
        if ubc_typ == self.bc_rainfall:
            dp = max(0.0, dp_m1 + (ubc_val - q[lb-1]) * dt)

        for ilev in range(lb-1, ub):
            dwat = (q[ilev] - q[ilev+1]) * dt
            wt[ilev] = 0.0
            wf[ilev] = 0.0
            vl[ilev] = ((wt_m1[ilev] + wf_m1[ilev]) * vl_s[ilev] + (dz[ilev] - wt_m1[ilev] - wf_m1[ilev]) * vl_m1[ilev] + dwat) / dz[ilev]

        if lbc_typ == self.bc_drainage:
            waquifer = waquifer_m1 + q[ub] * dt
            zwt = self.get_zwt_from_wa(
                vl_s_wa, vl_r[ub-1], psi_s[ub-1], hksat[ub-1], nprm, prms[:,ub-1], tol_v, tol_z,
                waquifer, sp_zi[ub]
            )
        return q, wf, vl, wt, dp, waquifer, zwt

    def var_perturb_rainfall(self, blc_srf, dp):
        """
        Perturbs the rainfall value based on the surface block condition.

        Parameters:
        - blc_srf: float, surface block condition
        - dp: float, current rainfall value

        Returns:
        - dp_p: float, perturbed rainfall value
        - delta: float, perturbation amount
        - is_act: bool, indicates if perturbation is active
        """
        # Constants
        wstep = 1.0e-1

        # Initialize outputs
        delta = 0.0
        is_act = False

        # Determine the perturbation delta
        if blc_srf > 0:
            if dp > 0:
                delta = - min(wstep, dp * 0.5)
        elif blc_srf < 0:
            delta = wstep

        # Calculate perturbed dp and perturbation activity flag
        dp_p = dp + delta
        is_act = (delta != 0)

        return dp_p, delta, is_act

    def richards_solver(self, lb, ub, dt, sp_zc, sp_zi, vl_s, vl_r, psi_s, hksat, nprm, prms,
                        vl_s_wa, ubc_typ, ubc_val, lbc_typ, lbc_val, ss_dp, waquifer, 
                        ss_vl, ss_wt, ss_q, tol_q, tol_z, tol_v, tol_p):

        # Local variables
        zwt = 0.0
        dlt = 0.0
        sp_dz = np.zeros(ub + 1 -lb)
        ss_wf = np.zeros(ub + 1 -lb)

        is_sat = np.zeros(ub + 1-lb, dtype=bool)
        has_wf = np.zeros(ub + 1-lb, dtype=bool)
        has_wt = np.zeros(ub + 1-lb, dtype=bool)

        infl_max = 0.0

        psi = np.zeros(ub + 1-lb)
        hk = np.zeros(ub-lb + 1)

        psi_pb = np.zeros(ub-lb + 1)
        hk_pb = np.zeros(ub-lb + 1)

        q_this = np.zeros(ub-lb+2)
        q_wf = np.zeros(ub-lb + 1)
        q_wt = np.zeros(ub-lb + 1)

        dp_m1 = 0.0
        wf_m1 = np.zeros(ub-lb + 1)
        vl_m1 = np.zeros(ub-lb + 1)
        wt_m1 = np.zeros(ub-lb + 1)
        waquifer_m1 = 0.0

        q_0 = np.zeros(ub-lb + 2)
        q_wf_0 = np.zeros(ub-lb + 1)
        q_wt_0 = np.zeros(ub-lb + 1)

        dp_pb = 0.0
        vl_pb = np.zeros(ub-lb + 1)
        wf_pb = np.zeros(ub-lb + 1)
        wt_pb = np.zeros(ub-lb + 1)
        zwt_pb = 0.0
        waquifer_pb = 0.0

        q_pb = np.zeros(ub-lb + 1+1)
        q_wf_pb = np.zeros(ub-lb + 1)
        q_wt_pb = np.zeros(ub-lb + 1)

        blc = np.zeros(ub+1 -lb + 1+1)
        self.is_solvable = False
        lev_update = np.ones(ub-lb + 1+2, dtype=bool)
        blc_pb = np.zeros(ub-lb + 1+2)
        vact = np.zeros(ub-lb + 1+2, dtype=bool)
        jsbl = np.zeros(ub-lb + 1, dtype=int)

        dr_dv = np.zeros((ub-lb + 1+2, ub-lb + 1+2))
        dv = np.zeros(ub-lb + 1+2)

        f2_norm = np.zeros(self.max_iters_richards)
        for ilev in range(lb-1, ub):
            sp_dz[ilev] = sp_zi[ilev+1]-sp_zi[ilev]

        dt_explicit = dt / self.max_iters_richards
        ss_q = 0
        dt_done = 0

        while dt_done < dt:
            dt_this = dt - dt_done

            wf_m1[:] = ss_wf[:]
            vl_m1[:] = ss_vl[:]
            wt_m1[:] = ss_wt[:]

            wsum_m1 = np.sum(ss_vl * (sp_dz - ss_wt)) + np.sum(ss_wt * vl_s)
            if ubc_typ == self.bc_rainfall:
                wsum_m1 += ss_dp
            if lbc_typ == self.bc_drainage:
                wsum_m1 += waquifer

            if ubc_typ == self.bc_rainfall:
                dp_m1 = max(ss_dp, 0.0)
                if dp_m1 < tol_z:
                    dp_m1 = 0.0
                infl_max = dp_m1 / dt_this + ubc_val

            if lbc_typ == self.bc_drainage:
                waquifer_m1 = waquifer
                zwt = self.get_zwt_from_wa(vl_s_wa, vl_r[ub-1], psi_s[ub-1], hksat[ub-1],
                                    nprm, prms[:, ub-1], tol_v, tol_z, waquifer, sp_zi[ub])

            iter = 0
            while True:
                iter += 1
                # print(has_wf, '----haswf11----')
                # if iter==2:
                # print( ss_wt[6], psi[0], hk[0],'-=-=-1=-=-=-=')
                is_sat, has_wf, has_wt, ss_wf, ss_vl, ss_wt, ss_dp, psi, hk = self.initialize_sublevel_structure(lb, ub, sp_dz, sp_zi[ub], vl_s, vl_r,
                                            psi_s, hksat, nprm, prms, ubc_typ, ubc_val, 
                                            lbc_typ, lbc_val, is_sat, has_wf, has_wt, 
                                            ss_wf, ss_vl, ss_wt, ss_dp, psi, hk, 
                                            tol_v, tol_z)
                # print(ss_wt[6], psi[0], hk[0], '-=-=-2=-=-=-=')
                # print(has_wf, '----haswf22----')

                lev_update[:] = True
                # print ( iter, q_wt,'--flux_all 1---')
                # print(q_this, '---qthis1-----')
                is_sat, has_wf, has_wt, ss_wf, ss_vl, ss_wt, ss_dp, zwt, q_this, q_wf, q_wt = self.flux_all(lb, ub, sp_dz, sp_zc, sp_zi, vl_s, psi_s, hksat, nprm, prms,
                        ubc_typ, ubc_val, lbc_typ, lbc_val, lev_update, True, is_sat, 
                        has_wf, has_wt, infl_max, ss_wf, ss_vl, ss_wt, ss_dp, zwt, 
                        psi, hk, q_this, q_wf, q_wt, tol_q, tol_z, tol_p)
                # print(ss_wt[6], psi[0], hk[0], q_wt,'-=-=-3=-=-=-=')
                # print(q_this, '---qthis2-----')

                # print(blc, '----blc1-----')


                blc, self.is_solvable = self.water_balance(lb, ub, sp_dz, dt_this, is_sat, vl_s, q_this,
                            ubc_typ, ubc_val, lbc_typ, lbc_val, ss_wf, ss_vl, 
                            ss_wt, ss_dp, waquifer, wf_m1, vl_m1, wt_m1, dp_m1, 
                            waquifer_m1, self.tol_richards * dt_this)
                # print(blc, '----blc2-----')


                if iter == 1:
                    q_0 = q_this[:]
                    q_wf_0 = q_wf[:]
                    q_wt_0 = q_wt[:]

                wet2dry = False
                if ubc_typ == self.bc_rainfall:
                    if dp_m1 > 0.0 and q_0[lb-1] >= infl_max:
                        wet2dry = True

                f2_norm[iter-1] = np.sqrt(np.sum(blc ** 2))
                # print(iter, f2_norm, blc, '-----f2_norm2-----')
                # print (iter, len(f2_norm), f2_norm[iter-1],self.tol_richards,dt_this,dt_explicit,self.max_iters_richards,self.is_solvable,wet2dry,'---break----')
                if (f2_norm[iter-1] < self.tol_richards * dt_this or dt_this < dt_explicit
                    or iter >= self.max_iters_richards or not self.is_solvable or wet2dry):
                    
                    if dt_this < dt_explicit or iter >= self.max_iters_richards or not self.is_solvable or wet2dry:
                        dt_this = min(dt_this, dt_explicit)
                        q_this[:] = q_0[:]

                        q_this, ss_wf, ss_vl, ss_wt, ss_dp, waquifer, zwt = self.use_explicit_form(lb, ub, dt_this, sp_dz, sp_zc, sp_zi, vl_s, 
                                        vl_r, psi_s, hksat, nprm, prms, vl_s_wa, 
                                        ubc_typ, ubc_val, lbc_typ, lbc_val, q_this, 
                                        q_wf_0, q_wt_0, ss_wf, ss_vl, ss_wt, 
                                        ss_dp, waquifer, zwt, wf_m1, vl_m1, wt_m1, 
                                        dp_m1, waquifer_m1, tol_q, tol_z, tol_v)

                    dt_done += dt_this

                    # Debugging and count updates
                    if self.nl_colm['CoLMDEBUG']:
                        if f2_norm[iter-1] < self.tol_richards * dt_this:
                            self.count_implicit += 1
                        else:
                            self.count_explicit += 1
                            if wet2dry:
                                self.count_wet2dry += 1

                    break

                dr_dv[:] = 0.0
                vact[:] = False

                if ubc_typ == self.bc_rainfall:
                    dp_pb, dlt, vact[lb-1] = self.var_perturb_rainfall(blc[lb-1], ss_dp)
                    if vact[lb-1]:
                        q_pb[:] = q_this[:]
                        q_wf_pb[:] = q_wf[:]
                        q_wt_pb[:] = q_wt[:]
                        lev_update[:] = False
                        lev_update[lb-1] = True
                        is_sat, has_wf, has_wt, ss_wf, ss_vl, ss_wt, dp_pb, zwt, q_pb, q_wf_pb, q_wt_pb = self.flux_all(lb, ub, sp_dz, sp_zc, sp_zi, vl_s, psi_s, hksat, nprm,
                                prms, ubc_typ, ubc_val, lbc_typ, lbc_val, lev_update, 
                                False, is_sat, has_wf, has_wt, infl_max, ss_wf, 
                                ss_vl, ss_wt, dp_pb, zwt, psi, hk, q_pb, q_wf_pb, 
                                q_wt_pb, tol_q, tol_z, tol_p)
                        blc_pb, self.is_solvable = self.water_balance(lb, ub, sp_dz, dt_this, is_sat, vl_s, q_pb,
                                  ubc_typ, ubc_val, lbc_typ, lbc_val, ss_wf, 
                                  ss_vl, ss_wt, ss_dp, waquifer, wf_m1, vl_m1, 
                                  wt_m1, dp_pb, waquifer_pb, self.tol_richards * dt_this)
                        dr_dv[:,lb-1] = (blc_pb - blc) / dlt

                for ilev in range(lb-1, ub):
                    if not is_sat[ilev]:
                        wf_pb = ss_wf.copy()
                        vl_pb = ss_vl.copy()
                        wt_pb = ss_wt.copy()
                        psi_pb = psi.copy()
                        hk_pb = hk.copy()

                        # 假设var_perturb_level函数已经正确定义，这里只是占位调用
                        jsbl[ilev], wf_pb[ilev], vl_pb[ilev], wt_pb[ilev], dlt, psi_pb[ilev], hk_pb[ilev], vact[ilev+1] = self.var_perturb_level(jsbl[ilev], blc[ilev + 1], sp_dz[ilev], sp_zc[ilev], sp_zi[ilev+1],
                                          vl_s[ilev], vl_r[ilev], psi_s[ilev], hksat[ilev],
                                          nprm, prms[:, ilev],
                                          is_sat[ilev], has_wf[ilev], has_wt[ilev],
                                          q_this[ilev], q_this[ilev+1], q_wf[ilev], q_wt[ilev],
                                          wf_pb[ilev], vl_pb[ilev], wt_pb[ilev], dlt,
                                          psi_pb[ilev], hk_pb[ilev], vact[ilev+1],
                                          tol_v)

                        if vact[ilev+ 1]:
                            q_pb = q_this.copy()
                            q_wf_pb = q_wf.copy()
                            q_wt_pb = q_wt.copy()

                            lev_update[:] = False
                            lev_update[ilev + 1] = True

                            # 假设flux_all函数已经正确定义，这里只是占位调用
                            # print(iter, ilev, vact[ilev+ 1], ubc_typ,self.bc_rainfall,'--flux_all 3---')
                            is_sat, has_wf, has_wt, wf_pb, vl_pb, wt_pb, ss_dp, zwt, q_pb, q_wf_pb, q_wt_pb = self.flux_all(lb, ub, sp_dz, sp_zc, sp_zi,
                                     vl_s, psi_s, hksat, nprm, prms,
                                     ubc_typ, ubc_val, lbc_typ, lbc_val,
                                     lev_update, False,
                                     is_sat, has_wf, has_wt, infl_max,
                                     wf_pb, vl_pb, wt_pb, ss_dp, zwt, psi_pb, hk_pb,
                                     q_pb, q_wf_pb, q_wt_pb,
                                     tol_q, tol_z, tol_p)

                            # 假设water_balance函数已经正确定义，这里只是占位调用
                            blc_pb, self.is_solvable = self.water_balance(lb, ub, sp_dz, dt_this, is_sat, vl_s, q_pb,
                                          ubc_typ, ubc_val, lbc_typ, lbc_val,
                                          wf_pb, vl_pb, wt_pb, ss_dp, waquifer,
                                          wf_m1, vl_m1, wt_m1, dp_m1, waquifer_m1)
#all 3
                            dr_dv[:, ilev+ 1] = (blc_pb - blc) / dlt

                if lbc_typ == self.bc_drainage:
                    # 假设var_perturb_drainage函数已经正确定义，这里只是占位调用
                    zwt_pb, dlt, vact[ub ] = self.var_perturb_drainage(sp_zi[ub], blc[ub ], zwt)

                    if vact[ub]:
                        q_pb = q_this.copy()
                        q_wf_pb = q_wf.copy()
                        q_wt_pb = q_wt.copy()

                        waquifer_pb = - (zwt_pb - sp_zi[ub]) * (vl_s_wa - CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm, psi_s[ub-1] + (sp_zi[ub] - zwt_pb) * 0.5,
                                                                                   vl_s_wa, vl_r[ub-1], psi_s[ub-1], nprm, prms[:, ub-1]))

                        lev_update[:] = False
                        lev_update[ub] = True

                        # 假设flux_all函数已经正确定义，这里只是占位调用
                        # print(iter, ub, vact[ub], lbc_typ,self.bc_drainage,'--flux_all 4---')
                        is_sat, has_wf, has_wt, ss_wf, ss_vl, ss_wt, ss_dp, zwt_pb, q_pb, q_wf_pb, q_wt_pb = self.flux_all(lb, ub, sp_dz, sp_zc, sp_zi,
                                 vl_s, psi_s, hksat, nprm, prms,
                                 ubc_typ, ubc_val, lbc_typ, lbc_val,
                                 lev_update, False,
                                 is_sat, has_wf, has_wt, infl_max,
                                 ss_wf, ss_vl, ss_wt, ss_dp, zwt_pb, psi, hk,
                                 q_pb, q_wf_pb, q_wt_pb,
                                 tol_q, tol_z, tol_p)

                        # 假设water_balance函数已经正确定义，这里只是占位调用
                        blc_pb, self.is_solvable = self.water_balance(lb, ub, sp_dz, dt_this, is_sat, vl_s, q_pb,
                                      ubc_typ, ubc_val, lbc_typ, lbc_val,
                                      ss_wf, ss_vl, ss_wt, ss_dp, waquifer_pb,
                                      wf_m1, vl_m1, wt_m1, dp_m1, waquifer_m1,
                                      blc_pb)

                        dr_dv[:, ub +1] = (blc_pb - blc) / dlt

                for ilev in range(lb - 1, ub + 2):
                    vact[ilev] = vact[ilev] and (abs(dr_dv[ilev, ilev]) > tol_q)

                    # 假设solve_least_squares_problem函数已经正确定义，这里只是占位调用
                # print(dv, '----dv1-----')
                dv = self.solve_least_squares_problem(ub - lb + 3, dr_dv, vact, blc)
                # print(dv, '----dv2-----')

                if vact[lb - 1]:
                    ss_dp = ss_dp - dv[lb - 1]
                    ss_dp = max(ss_dp, 0)
                # print(vact, '---vact----')
                for ilev in range(lb-1, ub):
                    if vact[ilev+1]:
                        if jsbl[ilev] == 1:
                            if (ss_wf[ilev] == sp_dz[ilev]) and (dv[ilev+1] > 0):
                                ss_wf[ilev] = ss_wf[ilev] - min(dv[ilev+1], sp_dz[ilev])

                                psi[ilev] = psi_s[ilev] + (1 - q_this[ilev] / hksat[ilev]) * min(dv[ilev+1], sp_dz[ilev]) * (
                                        sp_zc[ilev] - sp_zi[ilev - 1]) / sp_dz[ilev]
                                ss_vl[ilev] = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(psi[ilev],
                                                                 vl_s[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                                hk[ilev] = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi[ilev],
                                                            psi_s[ilev], hksat[ilev], nprm, prms[:, ilev])
                            else:
                                ss_wf[ilev] = ss_wf[ilev] - dv[ilev+1]
                                ss_wf[ilev] = max(ss_wf[ilev], 0)
                                ss_wf[ilev] = min(ss_wf[ilev], sp_dz[ilev] - ss_wt[ilev])

                        if jsbl[ilev] == 2:
                            # print(ss_vl[ilev], dv[ilev],tol_v,vl_s[ilev],'--jsbl2---')
                            ss_vl[ilev] = ss_vl[ilev] - dv[ilev+1]
                            ss_vl[ilev] = max(ss_vl[ilev], tol_v)
                            ss_vl[ilev] = min(ss_vl[ilev], vl_s[ilev])

                        if jsbl[ilev] == 3:
                            if (ss_wt[ilev] == sp_dz[ilev]) and (dv[ilev+1] > 0):
                                ss_wt[ilev] = ss_wt[ilev] - min(dv[ilev+1], sp_dz[ilev])

                                psi[ilev] = psi_s[ilev] - (1 - q_this[ilev - 1] / hksat[ilev]) * min(dv[ilev+1],
                                                                                                     sp_dz[ilev]) * (
                                                    sp_zi[ilev] - sp_zc[ilev]) / sp_dz[ilev]
                                ss_vl[ilev] = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(psi[ilev],
                                                                 vl_s[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                                hk[ilev] = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi[ilev],
                                                            psi_s[ilev], hksat[ilev], nprm, prms[:, ilev])
                            else:
                                ss_wt[ilev] = ss_wt[ilev] - dv[ilev+1]
                                ss_wt[ilev] = max(ss_wt[ilev], 0)
                                ss_wt[ilev] = min(ss_wt[ilev], sp_dz[ilev] - ss_wf[ilev])
                        # print(ss_wt[6], psi[0], hk[0], sp_dz[ilev],
                        #                        vl_s[ilev], vl_r[ilev], psi_s[ilev], hksat[ilev],
                        #                        nprm, prms[:, ilev],
                        #                        is_sat[ilev], has_wf[ilev], has_wt[ilev],
                        #                        ss_wf[ilev], ss_vl[ilev], ss_wt[ilev], psi[ilev], hk[ilev],
                        #                        jsbl[ilev] == 2, tol_v,'-=-=-3=-=-=-=')
                        # 假设check_and_update_level函数已经正确定义，这里只是占位调用
                        ss_wf[ilev], ss_vl[ilev], ss_wt[ilev], psi[ilev], hk[ilev] = self.check_and_update_level(sp_dz[ilev],
                                               vl_s[ilev], vl_r[ilev], psi_s[ilev], hksat[ilev],
                                               nprm, prms[:, ilev],
                                               is_sat[ilev], has_wf[ilev], has_wt[ilev],
                                               ss_wf[ilev], ss_vl[ilev], ss_wt[ilev], psi[ilev], hk[ilev],
                                               jsbl[ilev] == 2, tol_v)
                        # print(ilev, ss_wt[6], psi[0], hk[0], jsbl[ilev] ,is_sat[ilev],jsbl[ilev] == 2,'-=-=-4=-=-=-=')

                if vact[ub + 1]:
                    zwt = zwt - dv[ub + 1]
                    zwt = max(zwt, sp_zi[ub])
                    waquifer = - (zwt - sp_zi[ub]) * (vl_s_wa -
                                                      CoLM_Hydro_SoilFunction.soil_vliq_from_psi(psi_s[ub] + (sp_zi[ub] - zwt) * 0.5,
                                                                         vl_s_wa, vl_r[ub], psi_s[ub], nprm, prms[:, ub]))

            ss_q = ss_q + q_this * dt_this

            wsum = sum(ss_vl * (sp_dz - ss_wt - ss_wf)) + sum((ss_wt + ss_wf) * vl_s)
            if ubc_typ == self.bc_rainfall:
                wsum = wsum + ss_dp
            if lbc_typ == self.bc_drainage:
                wsum = wsum + waquifer

            werr = wsum - (wsum_m1 + ubc_val * dt_this - lbc_val * dt_this)

        ss_q /= dt

        # 模拟Fortran中的循环结构，这里假设lb和ub是整数，代表循环范围的下限和上限
        for ilev in range(lb-1, ub):
            if abs(sp_dz[ilev] - ss_wt[ilev]) > tol_z:
                ss_vl[ilev] = (ss_wf[ilev] * vl_s[ilev] + (sp_dz[ilev] - ss_wf[ilev] - ss_wt[ilev]) * ss_vl[ilev]) / (
                        sp_dz[ilev] - ss_wt[ilev])

        return ss_dp, waquifer, ss_vl,ss_wt,ss_q

    def var_perturb_level(self, jsbl, blc, dz, zc, zi, vl_s, vl_r, psi_s, hksat, nprm, prms,
                          is_sat, has_wf, has_wt, qin, qout, q_wf, q_wt,
                          wf_p, vl_p, wt_p, delta, psi_p, hk_p, is_act, tol_v):
        """
        模拟Fortran中的var_perturb_level子例程的功能

        参数:
        jsbl (int): 整型变量，对应Fortran中的jsbl，作为输出参数
        blc (float): 输入的实数，对应Fortran中的blc
        dz (float): 输入的实数，对应Fortran中的dz
        zc (float): 输入的实数，对应Fortran中的zc
        zi (float): 输入的实数，对应Fortran中的zi
        vl_s (float): 输入的实数，对应Fortran中的vl_s
        vl_r (float): 输入的实数，对应Fortran中的vl_r
        psi_s (float): 输入的实数，对应Fortran中的psi_s
        hksat (float): 输入的实数，对应Fortran中的hksat
        nprm (int): 整型变量，对应Fortran中的nprm
        prms (np.ndarray): 一维数组，对应Fortran中的prms，形状为 (nprm,)
        is_sat (bool): 布尔变量，对应Fortran中的is_sat
        has_wf (bool): 布尔变量，对应Fortran中的has_wf
        has_wt (bool): 布尔变量，对应Fortran中的has_wt
        qin (float): 输入的实数，对应Fortran中的qin
        qout (float): 输入的实数，对应Fortran中的qout
        q_wf (float): 输入的实数，对应Fortran中的q_wf
        q_wt (float): 输入的实数，对应Fortran中的q_wt
        wf_p (float): 输入输出的实数，对应Fortran中的wf_p
        vl_p (float): 输入输出的实数，对应Fortran中的vl_p
        wt_p (float): 输入输出的实数，对应Fortran中的wt_p
        delta (float): 输出的实数，对应Fortran中的delta
        psi_p (float): 输入输出的实数，对应Fortran中的psi_p
        hk_p (float): 输入输出的实数，对应Fortran中的hk_p
        is_act (bool): 输出的布尔变量，对应Fortran中的is_act
        tol_v (float): 输入的实数，对应Fortran中的tol_v

        返回:
        tuple: 包含更新后的jsbl (int), delta (float), is_act (bool)的元组
        """
        vstep = 1.0e-6
        wstep = 1.0e-1

        jsbl = 2

        if has_wt:
            # print(blc, q_wt, qout, (blc < 0) and (q_wt > qout), blc < 0,q_wt > qout,'-----has_wt-----')
            if (wt_p == dz) or ((blc >= 0) and (q_wt < qout) and (wt_p > 0)):
                # reduce water table
                jsbl = 3
                delta = - min(wstep, wt_p * 0.1)

                if wt_p == dz:
                    psi_p = psi_s - (1 - qin / hksat) * (-delta) * (zi - zc) / dz
                    vl_p = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm, psi_p, vl_s, vl_r, psi_s, nprm, prms)
                    hk_p = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, psi_p, psi_s, hksat, nprm, prms)

                wt_p = wt_p + delta

            elif (blc < 0) and (q_wt > qout):
                # increase water table
                jsbl = 3
                delta = min(wstep, (dz - wf_p - wt_p) * 0.1)
                wt_p = wt_p + delta

        if (jsbl == 2) and has_wf:
            if (wf_p == dz) or ((blc >= 0) and (qin < q_wf) and (wf_p > 0)):
                # reduce wetting front
                jsbl = 1
                delta = - min(wstep, wf_p * 0.1)
                if wf_p == dz:
                    psi_p = psi_s + (1 - qout / hksat) * (-delta) * (dz - (zi - zc)) / dz
                    vl_p = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm,psi_p, vl_s, vl_r, psi_s, nprm, prms)
                    hk_p = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self, psi_p, psi_s, hksat, nprm, prms)

                wf_p = wf_p + delta

            elif (blc < 0) and (qin > q_wf):
                # increase wetting front
                jsbl = 1
                delta = min(wstep, (dz - wf_p - wt_p) * 0.1)
                wf_p = wf_p + delta

        if jsbl == 2:
            if ((blc > 0) and (vl_p > vl_r + tol_v)) or (vl_p >= vl_s):
                # reduce water content
                delta = - min(vstep, (vl_p - vl_r - tol_v) * 0.5)
            elif (blc <= 0 and vl_p > vl_r + tol_v ) or vl_p >= vl_s:
                    # increase water content
                delta = + min(vstep, (vl_s - vl_p) * 0.5)
            else:
                delta = 0

            vl_p = vl_p + delta

        is_act = (delta != 0)

        if is_act:
            # print(jsbl, jsbl == 2, has_wt, '----ischeck-----')
            wf_p, vl_p, wt_p, psi_p, hk_p = self.check_and_update_level(dz, vl_s, vl_r, psi_s, hksat, nprm, prms,
                                   is_sat, has_wf, has_wt, wf_p, vl_p, wt_p, psi_p, hk_p,
                                   jsbl == 2, tol_v)

        return jsbl, wf_p, vl_p, wt_p, delta, psi_p, hk_p, is_act
    def solve_least_squares_problem(self, ndim, dr_dv, lact, rhs):
        """
        模拟Fortran中的solve_least_squares_problem子例程的功能

        参数:
        ndim (int): 问题的维度，对应Fortran中的ndim
        dr_dv (numpy.ndarray): 二维数组，对应Fortran中的dr_dv，形状为 (ndim, ndim)
        lact (numpy.ndarray): 一维布尔数组，对应Fortran中的lact，形状为 (ndim,)
        rhs (numpy.ndarray): 一维数组，对应Fortran中的rhs，形状为 (ndim,)

        返回:
        numpy.ndarray: 一维数组dv，对应Fortran中的dv，形状为 (ndim,)
        """
        Amatrix = dr_dv.copy()
        res = rhs.copy()
        dv = np.zeros(ndim)

        for i in range(ndim):
            if lact[i]:
                for j in range(i + 1, ndim):
                    if Amatrix[j, i] != 0:
                        if abs(Amatrix[j, i]) > abs(Amatrix[i, i]):
                            tau = Amatrix[i, i] / Amatrix[j, i]
                            s = 1 / np.sqrt(1 + tau ** 2)
                            c = s * tau
                        else:
                            tau = Amatrix[j, i] / Amatrix[i, i]
                            c = 1 / np.sqrt(1 + tau ** 2)
                            s = c * tau

                        Amatrix[i, i] = c * Amatrix[i, i] + s * Amatrix[j, i]
                        Amatrix[j, i] = 0

                        for k in range(i + 1, ndim):
                            if lact[k]:
                                tmp = c * Amatrix[i, k] + s * Amatrix[j, k]
                                Amatrix[j, k] = - s * Amatrix[i, k] + c * Amatrix[j, k]
                                Amatrix[i, k] = tmp

                        tmp = c * res[i] + s * res[j]
                        res[j] = - s * res[i] + c * res[j]
                        res[i] = tmp

        for i in range(ndim - 1, -1, -1):
            if lact[i]:
                dv[i] = res[i]

                for k in range(i + 1, ndim):
                    if lact[k]:
                        dv[i] = dv[i] - Amatrix[i, k] * dv[k]

                dv[i] = dv[i] / Amatrix[i, i]

        return dv

    def var_perturb_drainage(self, zmin, blc_btm, zwt):
        """
        模拟Fortran中的var_perturb_drainage子例程的功能

        参数:
        zmin (float): 输入的实数，对应Fortran中的zmin
        blc_btm (float): 输入的实数，对应Fortran中的blc_btm
        zwt (float): 输入的实数，对应Fortran中的zwt

        返回:
        tuple: 包含zwt_p (float), delta (float), is_act (bool)的元组，分别对应Fortran中的zwt_p、delta和is_act
        """
        wstep = 1.0e-1

        delta = 0
        if blc_btm > 0:
            delta = wstep
        elif blc_btm < 0:
            delta = - min(max((zwt - zmin) * 0.5, 0.0), wstep)

        zwt_p = zwt + delta
        is_act = (delta != 0)

        return zwt_p, delta, is_act

    def get_zwt_from_wa(self, vl_s, vl_r, psi_s, hksat, nprm, prms, tol_v, tol_z, wa, zmin):
        if wa >= 0:
            zwt = zmin
            vl = vl_s
            return zwt

        zwt = zmin + (-wa) / vl_s * 2.0
        psi = psi_s - (zwt - zmin) * 0.5
        vl = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm, psi, vl_s, vl_r, psi_s, nprm, prms)

        while wa <= -(zwt - zmin) * (vl_s - vl):
            zwt = zmin + (zwt - zmin) * 2 + 0.1
            psi = psi_s - (zwt - zmin) * 0.5
            vl = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm, psi, vl_s, vl_r, psi_s, nprm, prms)

        zwt_r = zwt
        zwt_l = zmin

        zwt_k1 = zwt_l
        fval_k1 = wa

        zwt = (zwt_l + zwt_r) / 2.0
        iter = 0

        while iter < 50:
            psi = psi_s - (zwt - zmin) * 0.5
            vl = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(self.nl_colm, psi, vl_s, vl_r, psi_s, nprm, prms)
            fval = wa + (zwt - zmin) * (vl_s - vl)
            # print(iter, fval,tol_v, zwt_r, zwt_l, tol_z,'---exit------')
            # print(wa,zwt, zmin, vl_s, vl,'-------')
            # print(psi,  vl_s, vl_r, psi_s, nprm, prms, '------vl------')
            # print(fval, fval_k1, zwt, zwt_k1, zwt_l, zwt_r,'------zwt--------')
            # print('---------------------------')

            if abs(fval) < tol_v or (zwt_r - zwt_l) < tol_z:
                break
            else:
                fval_k1, zwt, zwt_k1, zwt_l, zwt_r = self.secant_method_iteration(fval, fval_k1, zwt, zwt_k1, zwt_l, zwt_r)

            iter += 1

        if self.nl_colm['CoLMDEBUG']:
            if iter == 50:
                print('Warning: get_zwt_from_wa: not converged.')

        return zwt

    def get_water_equilibrium_state(self, zwtmm, nlev, wliq, smp, hk, wa, sp_zc, sp_zi, porsl, vl_r, psi_s, hksat, nprm, prms):
        # wliq = np.zeros(nlev)
        # smp = np.zeros(nlev)
        # hk = np.zeros(nlev)
        vliq = np.zeros(nlev)
        # water table location
        izwt = CoLM_UserDefFun.findloc_ud(zwtmm >= sp_zi, back=True)

        if izwt <= nlev:
            psi_zwt = psi_s[izwt]
        else:
            psi_zwt = psi_s[nlev-1]

        for ilev in range(nlev):
            if ilev < izwt:
                smp[ilev] = psi_zwt - (zwtmm - sp_zc[ilev])
                vliq[ilev] = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(smp[ilev], porsl[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                wliq[ilev] = vliq[ilev] * (sp_zi[ilev + 1] - sp_zi[ilev])
                hk[ilev] = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, smp[ilev], psi_s[ilev], hksat[ilev], nprm, prms[:, ilev])
            elif ilev == izwt:
                smp_up = psi_zwt - (zwtmm - sp_zi[ilev ]) * (sp_zi[ilev+1] - sp_zc[ilev]) / (
                            sp_zi[ilev+1] - sp_zi[ilev ])
                vliq_up = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(smp_up, porsl[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                wliq[ilev] = vliq_up * (zwtmm - sp_zi[ilev]) + porsl[ilev] * (sp_zi[ilev+1] - zwtmm)
                vliq[ilev] = wliq[ilev] / (sp_zi[ilev+1] - sp_zi[ilev])
                smp[ilev] = CoLM_Hydro_SoilFunction.soil_psi_from_vliq(vliq[ilev], porsl[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                hk[ilev] = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm, smp[ilev], psi_s[ilev], hksat[ilev], nprm, prms[:, ilev])
            else:
                wliq[ilev] = porsl[ilev] * (sp_zi[ilev+1] - sp_zi[ilev])
                smp[ilev] = psi_s[ilev]
                hk[ilev] = hksat[ilev]

        if izwt == nlev:
            psi = psi_zwt - (zwtmm - sp_zi[nlev ]) * 0.5
            vl = CoLM_Hydro_SoilFunction.soil_vliq_from_psi(psi, porsl[nlev - 1], vl_r[nlev - 1], psi_s[nlev - 1], nprm, prms[:, nlev - 1])
            wa = -(zwtmm - sp_zi[nlev ]) * (porsl[nlev ] - vl)
        else:
            wa = 0.0

        return wliq, smp, hk, wa

    def soil_water_vertical_movement(self, nlev, dt, sp_zc, sp_zi, is_permeable, porsl, vl_r, psi_s, hksat, nprm, prms, porsl_wa,
                                    qgtop, etr, rootr, rootflux, rsubst, qinfl, ss_dp, zwt, wa, ss_vliq, smp,    hk  ,tolerance):
        """
        Main subroutine to execute the calculation of soil water movement.
        """
        # tfrz = 273.15  # Assuming tfrz is the freezing point in Kelvin
        smp = np.zeros(nlev)
        hk = np.zeros(nlev)

        sp_dz = np.zeros(nlev)
        etroot = np.zeros(nlev)
        ss_wt = np.zeros(nlev)
        ss_q = np.zeros(nlev+1)

        temp = np.zeros(nlev)
        sp_dz = sp_zi[1:nlev+1] - sp_zi[:nlev]

        dp_m1 = ss_dp
        deficit = 0.

        # tolerances
        tol_q = tolerance / nlev / dt / 2.0
        tol_z = tol_q * dt
        tol_v = tol_z / np.max(sp_dz)
        tol_p = 1.0e-14

        # water table location
        izwt = CoLM_UserDefFun.findloc_ud(zwt>=sp_zi, back=True)

        if self.nl_colm['CoLMDEBUG']:
            # total water mass
            w_sum_before = ss_dp
            for ilev in range(nlev):
                if is_permeable[ilev]:
                    if ilev <= izwt-1:
                        w_sum_before += ss_vliq[ilev] * sp_dz[ilev]
                    elif ilev == izwt:
                        w_sum_before += ss_vliq[izwt ] * (zwt - sp_zi[izwt ])
                        w_sum_before += porsl[izwt ] * (sp_zi[izwt+1 ] - zwt)
                    else:
                        w_sum_before += porsl[ilev] * sp_dz[ilev]
            w_sum_before += wa

        # Transpiration
        if not self.nl_colm['DEF_USE_PLANTHYDRAULICS']:
            sumroot = np.sum(rootr[is_permeable & (rootr > 0)])
            etroot[:] = 0.
            if sumroot > 0.:
                for i in range(len(rootr)):
                    if is_permeable[i]:
                        etroot[i] = etr * max(rootr[i], 0) / sumroot
                deficit = 0.
            else:
                deficit = etr * dt
        else:
            deficit = 0.
            etroot[:] = rootflux
        # print(ss_vliq,izwt,is_permeable,'----ss_vliq1----')
        for ilev in range(izwt):
            if is_permeable[ilev]:
                ss_vliq[ilev] = (ss_vliq[ilev] * sp_dz[ilev] - etroot[ilev] *dt - deficit) / sp_dz[ilev]
                # print(ss_vliq[ilev], ss_vliq[ilev] > porsl[ilev],'-----1-----')
                if ss_vliq[ilev] < 0:
                    deficit = -ss_vliq[ilev] * sp_dz[ilev]
                    ss_vliq[ilev] = 0
                elif ss_vliq[ilev] > porsl[ilev]:
                    deficit = -(ss_vliq[ilev] - porsl[ilev]) * sp_dz[ilev]
                    ss_vliq[ilev] = porsl[ilev]
                else:
                    deficit = 0.
            else:
                deficit += etroot[ilev] * dt
        # print(ss_vliq, '----ss_vliq2----')

        for ilev in range(izwt, nlev):
            deficit += etroot[ilev] * dt

        # Exchange water with aquifer

        wexchange = rsubst * dt + deficit
        # print(izwt, zwt, '----izwt3----')
        # print(nlev, wexchange, sp_zi, is_permeable, porsl, vl_r, psi_s, hksat, nprm, prms, porsl_wa,
        #                         ss_dp, ss_vliq, zwt, wa)
        ss_dp, ss_vliq, zwt, wa, izwt = self.soilwater_aquifer_exchange(nlev, wexchange, sp_zi, is_permeable, porsl, vl_r, psi_s, hksat, nprm, prms, porsl_wa, 
                                ss_dp, ss_vliq, zwt, wa)
        # print(ss_vliq, '----ss_vliq3----')
        # print(izwt, zwt, '----izwt4----')

        # Water table location
        ss_wt[:] = 0.0
        if 1 <= izwt+1 <= nlev:
            ss_wt[izwt] = sp_zi[izwt+1] - zwt
        for ilev in range(izwt+1, nlev):
            ss_wt[ilev] = sp_dz[ilev]

        # Handle impermeable levels and call Richards solver
        ub = nlev
        while ub >= 1:
            is_break = False
            while not is_permeable[ub-1]:
                ss_q[ub - 2:ub] = 0.0
                if ub > 1:
                    ub -= 1
                else:
                    is_break = True
                    break
            if is_break:
                break
            
            lb = ub
            while lb > 1:
                if is_permeable[lb - 2]:
                    lb -= 1
                else:
                    break

            ubc_typ_sub = self.bc_rainfall if lb == 1 else self.bc_fix_flux
            ubc_val_sub = qgtop if lb == 1 else 0
            lbc_typ_sub = self.bc_drainage if ub == nlev and izwt + 1 > nlev else self.bc_fix_flux
            lbc_val_sub = 0

            ss_dp, wa, ss_vliq[lb-1:ub],ss_wt[lb-1:ub],ss_q[:ub+1] = self.richards_solver(lb, ub, dt, sp_zc[lb-1:ub], sp_zi, porsl[lb-1:ub], vl_r[lb-1:ub], psi_s[lb-1:ub],
                            hksat[lb-1:ub], nprm, prms[:, lb-1:ub], porsl_wa, ubc_typ_sub, ubc_val_sub, lbc_typ_sub, lbc_val_sub, 
                            ss_dp, wa, ss_vliq[lb-1:ub], ss_wt[lb-1:ub], ss_q[lb-1:ub+1], tol_q, tol_z, tol_v, tol_p)

            ub = lb - 1
        # print(ss_vliq, '----ss_vliq4----')

        if not is_permeable[0]:
            ss_dp = max(ss_dp + qgtop * dt, 0.0)

        if wa >= 0:
            for ilev in range(nlev-1, -1, -1):
                is_sat = not is_permeable[ilev] or (ss_vliq[ilev] > porsl[ilev] - tol_v) or (ss_wt[ilev] > sp_dz[ilev] - tol_z)
                if not is_sat:
                    zwt = sp_zi[ilev+1] - ss_wt[ilev]
                    break
            if is_sat:
                zwt = 0.0
        else:
            if is_permeable[nlev-1]:
                zwt = self.get_zwt_from_wa(porsl_wa, vl_r[nlev-1], psi_s[nlev-1], hksat[nlev-1], nprm, prms[:, nlev-1], tol_v, tol_z, wa, sp_zi[nlev])

        izwt = CoLM_UserDefFun.findloc_ud(zwt >= sp_zi, back=True)

        for ilev in range(izwt - 1, -1, -1):
            if is_permeable[ilev]:
                # print(ilev, ss_vliq[ilev], sp_dz[ilev],ss_wt[ilev],porsl[ilev],'----ss_vliq----')
                ss_vliq[ilev] = (ss_vliq[ilev] * (sp_dz[ilev] - ss_wt[ilev]) + porsl[ilev] * ss_wt[ilev]) / sp_dz[ilev]

        qinfl = qgtop - (ss_dp - dp_m1) / dt
        # print(ss_dp, nlev, izwt, zwt,'---after-----')

        if self.nl_colm['CoLMDEBUG']:
            w_sum_after = ss_dp
            for ilev in range(nlev):
                if is_permeable[ilev]:
                    if ilev <= izwt - 1:
                        w_sum_after += ss_vliq[ilev] * sp_dz[ilev]
                        # print(w_sum_after, '-----if w_sum_after-----')
                    elif ilev == izwt:
                        w_sum_after += ss_vliq[izwt] * (zwt - sp_zi[izwt])
                        w_sum_after += porsl[izwt] * (sp_zi[izwt+1] - zwt)
                        # print(w_sum_after, '-----ifelse w_sum_after-----')
                    else:
                        w_sum_after += porsl[ilev] * sp_dz[ilev]
                        # print(w_sum_after, '-----else w_sum_after-----')
            w_sum_after += wa

            wblc = w_sum_after - (w_sum_before + (qgtop - etr - rsubst) * dt)
            # print(wblc,   w_sum_after,w_sum_before,qgtop,etr,rsubst,dt,wa, '----wblc----')
            if abs(wblc) > tolerance:
                print('soil_water_vertical_movement balance error:', wblc)
                # print(w_sum_after, w_sum_before, qgtop, etr, rsubst, is_permeable[0], ss_dp)

        for ilev in range(nlev):
            if ilev < izwt:
                smp[ilev] = CoLM_Hydro_SoilFunction.soil_psi_from_vliq(self.nl_colm,ss_vliq[ilev], porsl[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                hk[ilev] = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm,smp[ilev], psi_s[ilev], hksat[ilev], nprm, prms[:, ilev])
            elif ilev == izwt:
                vliq = (ss_vliq[izwt] * (zwt - sp_zi[izwt]) + porsl[izwt] * (sp_zi[izwt+1] - zwt)) / (sp_zi[izwt+1] - sp_zi[izwt])
                smp[ilev] = CoLM_Hydro_SoilFunction.soil_psi_from_vliq(self.nl_colm,vliq, porsl[ilev], vl_r[ilev], psi_s[ilev], nprm, prms[:, ilev])
                hk[ilev] = CoLM_Hydro_SoilFunction.soil_hk_from_psi(self.nl_colm,smp[ilev], psi_s[ilev], hksat[ilev], nprm, prms[:, ilev])
            else:
                smp[ilev] = psi_s[ilev]
                hk[ilev] = hksat[ilev]
    
        return qinfl, ss_dp, zwt, wa, ss_vliq, smp, hk

    def print_VSF_iteration_stat_info(self, mpi):
        count_implicit_accum = 0
        count_explicit_accum = 0
        count_wet2dry_accum = 0
        if self.nl_colm['CoLMDEBUG']:
            if mpi.p_is_worker:

                if mpi.p_iam_worker == 0:
                    count_implicit_accum += self.count_implicit
                    count_explicit_accum += self.count_explicit
                    count_wet2dry_accum += self.count_wet2dry

            if mpi.p_is_master:
                # iwork = 0  # Placeholder for the actual worker rank/address

                # self.count_implicit = comm_glb.recv(source=iwork, tag=0)
                # self.count_explicit = comm_glb.recv(source=iwork, tag=1)
                # self.count_wet2dry = comm_glb.recv(source=iwork, tag=2)
                # count_implicit_accum = comm_glb.recv(source=iwork, tag=3)
                # count_explicit_accum = comm_glb.recv(source=iwork, tag=4)
                # count_wet2dry_accum = comm_glb.recv(source=iwork, tag=5)

                print(f'\nVSF scheme this step: {self.count_implicit:13} (implicit) {self.count_explicit:13} (explicit) {self.count_wet2dry:13} (wet2dry)')
                print(f'VSF scheme all steps: {count_implicit_accum:13} (implicit) {count_explicit_accum:13} (explicit) {count_wet2dry_accum:13} (wet2dry)')

            self.count_implicit = 0
            self.count_explicit = 0
            self.count_wet2dry = 0


