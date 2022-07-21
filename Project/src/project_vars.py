import numpy as np
from typing import NamedTuple

from Project.src.project_utils import compute_ballistic_coeff


class EntryVehicleParams(NamedTuple):
    mass: float
    omega_planet: float
    ballistic_coeff: float
    radius_planet: float
    gravity_planet: float
    lift_drag_ratio: float
    scale_height: float
    density_planet: float
    atmosphere_altitude_planet: float


def create_STS_13_params():
    # STS-13 Params
    rE = 6378*1000  # m

    A_ref = 250  # m^2
    Cd = 0.78
    phi = np.radians(56)
    l_d_ratio = 1.07
    rho_0 = 1.225  # kg/m^3
    rho_init = 1.874e-07 * rho_0
    g_0 = 9.806
    H = 7.2*1000  # meters
    mass = 92000  # kg
    beta = compute_ballistic_coeff(mass, Cd, A_ref)
    h_atm = 125*1000.0 # meters
    w_earth = 7.2921159e-5

    return EntryVehicleParams(mass, w_earth, beta, rE, g_0, l_d_ratio, H, rho_0, h_atm)


STS_13_params = create_STS_13_params()
