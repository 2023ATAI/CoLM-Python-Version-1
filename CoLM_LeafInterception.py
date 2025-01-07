import numpy as np

class LeafInterception:
    """
    Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
    software         : For calculating vegetation canopy preciptation interception.
! This MODULE is the coupler for the colm and CaMa-Flood model.
!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------
   !* :SUBROUTINE:"LEAF_interception_CoLM2014"   : interception and drainage of precipitation schemes based on colm2014 version
   !* :SUBROUTINE:"LEAF_interception_CoLM202x"   : interception and drainage of precipitation schemes besed on new colm version (under development)
   !* :SUBROUTINE:"LEAF_interception_CLM4"       : interception and drainage of precipitation schemes modified from CLM4
   !* :SUBROUTINE:"LEAF_interception_CLM5"       : interception and drainage of precipitation schemes modified from CLM5
   !* :SUBROUTINE:"LEAF_interception_NOAHMP"     : interception and drainage of precipitation schemes modified from Noah-MP
   !* :SUBROUTINE:"LEAF_interception_MATSIRO"    : interception and drainage of precipitation schemes modified from MATSIRO 2021 version
   !* :SUBROUTINE:"LEAF_interception_VIC"        : interception and drainage of precipitation schemes modified from VIC
   !* :SUBROUTINE:"LEAF_interception_JULES"      : interception and drainage of precipitation schemes modified from JULES
   !* :SUBROUTINE:"LEAF_interception_pftwrap"    : wapper for pft land use classification
   !* :SUBROUTINE:"LEAF_interception_pcwrap"     : wapper for pc land use classification
    """
    def __init__(self,nl_colm,CP):
        self.CICE = 2.094e06  #!specific heat capacity of ice (j/m3/k)
        self.bp = 20
        self.HFUS = 0.3336e06  #!latent heat of fusion (j/kg)
        self.CWAT = 4.188e06   #!specific heat capacity of water (j/m3/k)
        self.pcoefs = np.array([[20.0, 0.206e-8], [0.0001, 0.9999]], dtype=np.float64)
        self.pcoefs = np.array([[20.0, 0.206e-8], [0.0001, 0.9999]], dtype=np.float64)
        self.nl_colm = nl_colm
        self.CP = CP

    def LEAF_interception_CoLM2014(self, tfrz, deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,
                                          prc_rain,prc_snow,prl_rain,prl_snow,
                                          ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,qintr,qintr_rain,qintr_snow):
        """
        Original author  : Qinghliang Li,Jinlong Zhu, 17/02/2024;
        software         : For calculating vegetation canopy preciptation interception.
      ! Calculation of  interception and drainage of precipitation
      ! the treatment are based on Sellers et al. (1996)

       !References:
       !-------------------
      !---Dai, Y., Zeng, X., Dickinson, R.E., Baker, I., Bonan, G.B., BosiloVICh, M.G., Denning, A.S.,
      !   Dirmeyer, P.A., Houser, P.R., Niu, G. and Oleson, K.W., 2003.
      !   The common land model. Bulletin of the American Meteorological Society, 84(8), pp.1013-1024.

      !---Lawrence, D.M., Thornton, P.E., Oleson, K.W. and Bonan, G.B., 2007.
      !   The partitioning of evapotranspiration into transpiration, soil evaporation,
      !   and canopy evaporation in a GCM: Impacts on land–atmosphere interaction. Journal of Hydrometeorology, 8(4), pp.862-880.

      !---Oleson, K., Dai, Y., Bonan, B., BosiloVIChm, M., Dickinson, R., Dirmeyer, P., Hoffman,
      !   F., Houser, P., Levis, S., Niu, G.Y. and Thornton, P., 2004.
      !   Technical description of the community land model (CLM).

      !---Sellers, P.J., Randall, D.A., Collatz, G.J., Berry, J.A., Field, C.B., Dazlich, D.A., Zhang, C.,
      !   Collelo, G.D. and Bounoua, L., 1996. A revised land surface parameterization (SiB2) for atmospheric GCMs.
      !   Part I: Model formulation. Journal of climate, 9(4), pp.676-705.

      !---Sellers, P.J., Tucker, C.J., Collatz, G.J., Los, S.O., Justice, C.O., Dazlich, D.A. and Randall, D.A., 1996.
      !   A revised land surface parameterization (SiB2) for atmospheric GCMs. Part II:
      !   The generation of global fields of terrestrial biophysical parameters from satellite data.
      !   Journal of climate, 9(4), pp.706-737.
   !ANCILLARY FUNCTIONS AND SUBROUTINES
        """
        qflx_irrig_sprinkler = 0.


        if lai+sai > 1e-6:
            lsai = lai + sai
            vegt = lsai
            satcap = dewmx * vegt
            # print(prc_rain,prc_snow,prl_rain,prl_snow,qflx_irrig_sprinkler,deltim,'=====')

            p0 = (prc_rain + prc_snow + prl_rain + prl_snow + qflx_irrig_sprinkler) * deltim
            ppc = (prc_rain + prc_snow) * deltim
            ppl = (prl_rain + prl_snow + qflx_irrig_sprinkler) * deltim

            w = ldew + p0
            xsc_snow = 0.
            xsc_rain = 0.
            if tleaf > self.CP.tfrz:
                xsc_rain = max(0., ldew - satcap)
            else:
                xsc_snow = max(0., ldew - satcap)
            ldew -= xsc_rain + xsc_snow

            ap = self.pcoefs[1, 0]  # Python中索引从0开始
            cp = self.pcoefs[1, 1]

            if p0 > 1.e-8:
                # print(p0,'----p0if-----')
                ap = ppc / p0 * self.pcoefs[0, 0] + ppl / p0 * self.pcoefs[1, 0]
                cp = ppc / p0 * self.pcoefs[0, 1] + ppl / p0 * self.pcoefs[1, 1]

                # ----------------------------------------------------------------------
                # proportional saturated area (xs) and leaf drainage(tex)
                # ----------------------------------------------------------------------
                chiv = chil
                if abs(chiv) <= 0.01:
                    chiv = 0.01
                aa1 = 0.5 - 0.633 * chiv - 0.33 * chiv * chiv
                bb1 = 0.877 * (1. - 2. * aa1)
                exrain = aa1 + bb1
                # coefficient of interception
                # set fraction of potential interception to max 0.25 (Lawrence et al. 2007)
                # assume alpha_rain = alpha_snow
                alpha_rain = 0.25
                fpi = alpha_rain * (1. - np.exp(-exrain * lsai))
                tti_rain = (prc_rain + prl_rain + qflx_irrig_sprinkler) * deltim * (1. - fpi)
                tti_snow = (prc_snow + prl_snow) * deltim * (1. - fpi)

                xs = 1.
                if p0 * fpi > 1.e-9:
                    arg = (satcap - ldew) / (p0 * fpi * ap) - cp / ap
                    if arg > 1.e-9:
                        xs = -1. / self.bp * np.log(arg)
                        xs = min(xs, 1.)
                        xs = max(xs, 0.)
                # assume no fall down of the intercepted snowfall in a time step
                # drainage
                tex_rain = (prc_rain + prl_rain + qflx_irrig_sprinkler) * deltim * fpi * (
                            ap / self.bp * (1. - np.exp(-self.bp * xs)) + cp * xs) - (satcap - ldew) * xs
                tex_rain = max(tex_rain, 0.)
                tex_snow = 0.

                if self.nl_colm['CoLMDEBUG']:
                    if tex_rain + tex_snow + tti_rain + tti_snow - p0 > 1.e-10:
                        print('tex_ + tti_ > p0 in interception code :')
            else:
                # print(p0, '----p0else-----')
                # print('------else------')
                #! all intercepted by canopy leves for very small precipitation
                tti_rain = 0.
                tti_snow = 0.
                tex_rain = 0.
                tex_snow = 0.

            # !----------------------------------------------------------------------
            # !   total throughfall(thru) and store augmentation
            # !----------------------------------------------------------------------
            thru_rain = tti_rain + tex_rain
            thru_snow = tti_snow + tex_snow
            pinf = p0 - (thru_rain + thru_snow)
            ldew = ldew + pinf

            pg_rain = (xsc_rain + thru_rain) / deltim
            pg_snow = (xsc_snow + thru_snow) / deltim
            qintr = pinf / deltim

            qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
            qintr_snow = prc_snow + prl_snow - thru_snow / deltim

            # 如果定义了CoLMDEBUG（这里假设通过一个布尔变量来模拟这种条件判断）
            if self.nl_colm['CoLMDEBUG']:
                w = w - ldew - (pg_rain + pg_snow) * deltim
                if abs(w) > 1.e-6:
                    print('something wrong in interception code : ')
                    print(w, ldew, (pg_rain + pg_snow) * deltim, satcap)
                    # 这里假设没有直接等同于Fortran中CALL abort的操作，可能需要根据具体情况处理异常或错误退出
                    raise Exception("Interception code error detected.")
        else:
            # 07/15/2023, yuan: #bug found for ldew value reset.
            # NOTE: this bug should exist in other interception schemes @Zhongwang.
            if ldew > 0.:
                if tleaf > tfrz:
                    pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew / deltim
                    pg_snow = prc_snow + prl_snow
                else:
                    pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
                    pg_snow = prc_snow + prl_snow + ldew / deltim
            else:
                pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
                pg_snow = prc_snow + prl_snow

            ldew = 0.
            qintr = 0.
            qintr_rain = 0.
            qintr_snow = 0.
        return ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain, pg_snow, qintr, qintr_rain, qintr_snow

    def LEAF_interception_wrap(self, tfrz, deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,
                                                            prc_rain,prc_snow,prl_rain,prl_snow,
                                                         ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,
                                                pg_snow,qintr,qintr_rain,qintr_snow):
        """
       !wrapper for calculation of canopy interception using USGS or IGBP land cover classification
       !ANCILLARY FUNCTIONS AND SUBROUTINES
        """
        DEF_Interception_scheme = int(self.nl_colm['DEF_Interception_scheme'])  # Replace with actual value

        # Call the appropriate function based on DEF_Interception_scheme
        if DEF_Interception_scheme == 1:
            ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain, pg_snow, qintr, qintr_rain, qintr_snow = self.LEAF_interception_CoLM2014(tfrz, deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,
                                             prc_rain,prc_snow,prl_rain,prl_snow,
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,
                                             pg_snow,qintr,qintr_rain,qintr_snow)
            # LEAF_interception_CoLM2014(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
            #                         prc_rain, prc_snow, prl_rain, prl_snow,
            #                         ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
            #                         pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 2:
        #     LEAF_interception_CLM4(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                         prc_rain, prc_snow, prl_rain, prl_snow,
        #                         ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                         pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 3:
        #     LEAF_interception_CLM5(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                         prc_rain, prc_snow, prl_rain, prl_snow,
        #                         ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                         pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 4:
        #     LEAF_interception_NoahMP(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                             prc_rain, prc_snow, prl_rain, prl_snow,
        #                             ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                             pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 5:
        #     LEAF_interception_matsiro(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                             prc_rain, prc_snow, prl_rain, prl_snow,
        #                             ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                             pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 6:
        #     LEAF_interception_vic(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                         prc_rain, prc_snow, prl_rain, prl_snow,
        #                         ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                         pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 7:
        #     LEAF_interception_JULES(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                             prc_rain, prc_snow, prl_rain, prl_snow,
        #                             ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                             pg_snow, qintr, qintr_rain, qintr_snow)
        # elif DEF_Interception_scheme == 8:
        #     LEAF_interception_colm202x(deltim, dewmx, forc_us, forc_vs, chil, sigf, lai, sai, tair, tleaf,
        #                             prc_rain, prc_snow, prl_rain, prl_snow,
        #                             ldew, ldew_rain, ldew_snow, z0m, hu, pg_rain,
        #                             pg_snow, qintr, qintr_rain, qintr_snow)

        return tleaf, ldew,ldew_rain,ldew_snow,pg_rain, pg_snow,qintr,qintr_rain,qintr_snow