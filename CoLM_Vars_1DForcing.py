import numpy as np

class CoLM_Vars_1DForcing(object):
    def __init__(self, nl_colm, mpi, numpatch, numelm) -> None:
        self.nl_colm = nl_colm
        self.mpi = mpi
        self.numpatch = numpatch
        self.numelm  = numelm
        # Allocate arrays with the size of numelm
        self.forc_pco2m = None   # CO2 concentration in atmos. (pascals)
        self.forc_po2m = None    # O2 concentration in atmos. (pascals)
        self.forc_us = None      # wind in eastward direction [m/s]
        self.forc_vs = None      # wind in northward direction [m/s]
        self.forc_t = None       # temperature at reference height [kelvin]
        self.forc_q = None       # specific humidity at reference height [kg/kg]
        self.forc_prc = None     # convective precipitation [mm/s]
        self.forc_prl = None     # large scale precipitation [mm/s]
        self.forc_rain = None    # rain [mm/s]
        self.forc_snow = None    # snow [mm/s]
        self.forc_psrf = None    # atmospheric pressure at the surface [pa]
        self.forc_pbot = None    # atm bottom level pressure (or reference height) (pa)
        self.forc_sols = None    # atm vis direct beam solar rad onto srf [W/m2]
        self.forc_soll = None    # atm nir direct beam solar rad onto srf [W/m2]
        self.forc_solsd = None   # atm vis diffuse solar rad onto srf [W/m2]
        self.forc_solld = None   # atm nir diffuse solar rad onto srf [W/m2]
        self.forc_frl = None     # atmospheric infrared (longwave) radiation [W/m2]
        self.forc_hgt_u = None   # observational height of wind [m]
        self.forc_hgt_t = None   # observational height of temperature [m]
        self.forc_hgt_q = None   # observational height of humidity [m]
        self.forc_rhoair = None  # air density [kg/m3]
        self.forc_ozone = None   # air density [kg/m3]

        # For Forcing_Downscaling
        self.forc_topo = None    # topography [m]
        self.forc_th = None      # potential temperature [K]

        self.forc_topo_elm = None  # atmospheric surface height [m]
        self.forc_t_elm = None     # atmospheric temperature [Kelvin]
        self.forc_th_elm = None    # atmospheric potential temperature [Kelvin]
        self.forc_q_elm = None     # atmospheric specific humidity [kg/kg]
        self.forc_pbot_elm = None  # atmospheric pressure [Pa]
        self.forc_rho_elm = None   # atmospheric density [kg/m**3]
        self.forc_prc_elm = None   # convective precipitation in grid [mm/s]
        self.forc_prl_elm = None   # large-scale precipitation in grid [mm/s]
        self.forc_lwrad_elm = None # grid downward longwave [W/m**2]
        self.forc_hgt_elm = None   # atmospheric reference height [m]

        self.forc_hpbl = None      # atmospheric boundary layer height [m]
        self.forc_aerdep = None  # atmospheric aerosol deposition data [kg/m/s]

    def allocate_1D_Forcing(self):
        if self.mpi.p_is_worker:
            if self.numpatch > 0:
                self.forc_pco2m = np.zeros(self.numpatch)     # CO2 concentration in atmos. (pascals)
                self.forc_po2m = np.zeros(self.numpatch)      # O2 concentration in atmos. (pascals)
                self.forc_us = np.zeros(self.numpatch)        # wind in eastward direction [m/s]
                self.forc_vs = np.zeros(self.numpatch)        # wind in northward direction [m/s]
                self.forc_t = np.zeros(self.numpatch)         # temperature at reference height [kelvin]
                self.forc_q = np.zeros(self.numpatch)         # specific humidity at reference height [kg/kg]
                self.forc_prc = np.zeros(self.numpatch)       # convective precipitation [mm/s]
                self.forc_prl = np.zeros(self.numpatch)       # large scale precipitation [mm/s]
                self.forc_rain = np.zeros(self.numpatch)      # rain [mm/s]
                self.forc_snow = np.zeros(self.numpatch)      # snow [mm/s]
                self.forc_psrf = np.zeros(self.numpatch)      # atmospheric pressure at the surface [pa]
                self.forc_pbot = np.zeros(self.numpatch)      # atm bottom level pressure (or reference height) (pa)
                self.forc_sols = np.zeros(self.numpatch)      # atm vis direct beam solar rad onto srf [W/m2]
                self.forc_soll = np.zeros(self.numpatch)      # atm nir direct beam solar rad onto srf [W/m2]
                self.forc_solsd = np.zeros(self.numpatch)     # atm vis diffuse solar rad onto srf [W/m2]
                self.forc_solld = np.zeros(self.numpatch)     # atm nir diffuse solar rad onto srf [W/m2]
                self.forc_frl = np.zeros(self.numpatch)       # atmospheric infrared (longwave) radiation [W/m2]
                self.forc_swrad = np.zeros(self.numpatch)     # atmospheric shortwave radiation [W/m2]
                self.forc_hgt_u = np.zeros(self.numpatch)     # observational height of wind [m]
                self.forc_hgt_t = np.zeros(self.numpatch)     # observational height of temperature [m]
                self.forc_hgt_q = np.zeros(self.numpatch)     # observational height of humidity [m]
                self.forc_rhoair = np.zeros(self.numpatch)    # air density [kg/m3]
                self.forc_ozone = np.zeros(self.numpatch)     # air density [kg/m3]
                self.forc_hpbl = np.zeros(self.numpatch)      # atmospheric boundary layer height [m]

                if self.nl_colm['DEF_USE_Forcing_Downscaling']:
                    self.forc_topo = np.zeros(self.numpatch)
                    self.forc_th = np.zeros(self.numpatch)

                # if self.nl_colm['DEF_Aerosol_Readin']:
                self.forc_aerdep = np.zeros((14, self.numpatch))  # atmospheric aerosol deposition data [kg/m/s]

                if self.nl_colm['DEF_USE_Forcing_Downscaling']:
                    if self.numelm > 0:
                        self.forc_topo_elm = None  # atmospheric surface height [m]
                        self.forc_t_elm = None     # atmospheric temperature [Kelvin]
                        self.forc_th_elm = None    # atmospheric potential temperature [Kelvin]
                        self.forc_q_elm = None     # atmospheric specific humidity [kg/kg]
                        self.forc_pbot_elm = None  # atmospheric pressure [Pa]
                        self.forc_rho_elm = None   # atmospheric density [kg/m**3]
                        self.forc_prc_elm = None   # convective precipitation in grid [mm/s]
                        self.forc_prl_elm = None   # large-scale precipitation in grid [mm/s]
                        self.forc_lwrad_elm = None # grid downward longwave [W/m**2]
                        self.forc_hgt_elm = None   # atmospheric reference height [m]
                        
                        
    def deallocate_1D_Forcing (self):
        if self.mpi.p_is_worker:
            if self.numpatch > 0:
                del self.forc_pco2m   # CO2 concentration in atmos. (pascals
                del self.forc_po2m    # O2 concentration in atmos. (pascals
                del self.forc_us      # wind in eastward direction [m/s]
                del self.forc_vs      # wind in northward direction [m/s]
                del self.forc_t       # temperature at reference height [kelvin]
                del self.forc_q       # specific humidity at reference height [kg/kg]
                del self.forc_prc     # convective precipitation [mm/s]
                del self.forc_prl     # large scale precipitation [mm/s]
                del self.forc_rain    # rain [mm/s]
                del self.forc_snow    # snow [mm/s]
                del self.forc_psrf    # atmospheric pressure at the surface [pa]
                del self.forc_pbot    # atm bottom level pressure (or reference height (pa
                del self.forc_sols    # atm vis direct beam solar rad onto srf [W/m2]
                del self.forc_soll    # atm nir direct beam solar rad onto srf [W/m2]
                del self.forc_solsd   # atm vis diffuse solar rad onto srf [W/m2]
                del self.forc_solld   # atm nir diffuse solar rad onto srf [W/m2]
                del self.forc_frl     # atmospheric infrared (longwave radiation [W/m2]
                del self.forc_hgt_u   # observational height of wind [m]
                del self.forc_hgt_t   # observational height of temperature [m]
                del self.forc_hgt_q   # observational height of humidity [m]
                del self.forc_rhoair  # air density [kg/m3]
                del self.forc_ozone   # Ozone partial pressure [mol/mol]

            if self.nl_colm['DEF_USE_Forcing_Downscaling']:
                del self.forc_topo
                del self.forc_th
            del self.forc_hpbl
                
            
            