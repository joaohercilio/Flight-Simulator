import numpy as np
import ussa1976

_alt_values = None
_rho_values = None
_g_values = None

def initialize():
    global _alt_values, _rho_values, _g_values
    atmosphere = ussa1976.compute()
    _alt_values = atmosphere["z"].values
    _rho_values = atmosphere["rho"].values
    _g_values = ussa1976.core.compute_gravity(_alt_values)

def getDensity(altitude):
    return np.interp(altitude, _alt_values, _rho_values)

def getGravity(altitude):
    return np.interp(altitude, _alt_values, _g_values)
