import numpy as np
import math

class CoLM_Const_Physical(object):
    def __init__(self) -> None:
        # Module constants
        self.denice = 917.  # Density of ice [kg/m3]
        self.denh2o = 1000.  # Density of liquid water [kg/m3]
        self.cpliq = 4188.  # Specific heat of water [J/kg-K]
        self.cpice = 2117.27  # Specific heat of ice [J/kg-K]
        self.cpair = 1004.64  # Specific heat of dry air [J/kg/K]
        self.hfus = 0.3336e6  # Latent heat of fusion for ice [J/kg]
        self.hvap = 2.5104e6  # Latent heat of evaporation for water [J/kg]
        self.hsub = 2.8440e6  # Latent heat of sublimation [J/kg]
        self.tkair = 0.023  # Thermal conductivity of air [W/m/K]
        self.tkice = 2.290  # Thermal conductivity of ice [W/m/K]
        self.tkwat = 0.6  # Thermal conductivity of water [W/m/K]
        self.tfrz = 273.16  # Freezing temperature [K]
        self.rgas = 287.04  # Gas constant for dry air [J/kg/K]
        self.roverg = 4.71047e4  # rw/g = (8.3144/0.018)/(9.80616)*1000. mm/K
        self.rwat = 461.296  # Gas constant for water vapor [J/(kg K)]
        self.grav = 9.80616  # Gravity constant [m/s2]
        self.vonkar = 0.4  # von Karman constant [-]
        self.stefnc = 5.67e-8  # Stefan-Boltzmann constant  [W/m2/K4]