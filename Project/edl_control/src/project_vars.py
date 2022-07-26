import numpy as np
from typing import NamedTuple

from Project.src.project_utils import compute_ballistic_coeff, inch_to_meter, normalize_angle


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
    desired_position: np.ndarray
    initial_position: np.ndarray


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
    p_des = np.radians([242.163116-360, 34.930885])  # (long, lat)
    p_init = np.radians([150-360, 0])

    return EntryVehicleParams(mass, w_earth, beta, rE, g_0, l_d_ratio, H, rho_0, h_atm, p_des, p_init)

def create_APOLLO_4_params():
    # STS-13 Params
    rE = 6378*1000  # m

    ref_radius = inch_to_meter(183.5)

    A_ref = np.pi * (3.91/2)**2  # m^2
    Cd = 1.2
    l_d_ratio = .38
    rho_0 = 1.225  # kg/m^3
    rho_init = 1.874e-07 * rho_0
    g_0 = 9.806
    H = 7.2*1000  # meters
    mass = 5395.3659  # kg
    beta = compute_ballistic_coeff(mass, Cd, A_ref)
    h_atm = 121920 # meters
    w_earth = 7.2921159e-5
    p_des = np.degrees(normalize_angle(np.radians([-157.976, 32.465])))  # (long, lat)
    p_init = np.radians([155.637, 23.398])

    return EntryVehicleParams(mass, w_earth, beta, rE, g_0, l_d_ratio, H, rho_0, h_atm, p_des, p_init)


STS_13_params = create_STS_13_params()
APOLLO_4_params = create_APOLLO_4_params()