import numpy as np

class CoLM_SnowFraction(object):
    def __init__(self) -> None:
        self.m = 1.0

    def snowfraction (self, lai,sai,z0m,zlnd,scv,snowdp):
        """
        Original author  : Qinghliang Li, 17/02/2024; Jinlong Zhu,   17/02/2024;
        software         : Provide snow cover fraction
        """
        m = 1.0
        if lai + sai > 1e-6:
            # Fraction of vegetation buried (covered) by snow
            wt = 0.1 * snowdp / z0m
            wt = wt / (1. + wt)

            # Fraction of vegetation cover free of snow
            sigf = 1. - wt
        else:
            wt = 0.
            sigf = 0.

        # Fraction of soil covered by snow
        fsno = 0.0
        if snowdp > 0.:
            fmelt = (scv / snowdp / 100.) ** self.m
            fsno = np.tanh(snowdp / (2.5 * zlnd * fmelt))

        return wt,sigf,fsno

