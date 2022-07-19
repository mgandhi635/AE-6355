import numpy as np
import scipy as sp
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

nd = np.ndarray
sin = np.sin
cos = np.cos
tan = np.tan


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


def compute_gravity_from_altitude(altitude: float, params: EntryVehicleParams):
    """Gravity computation"""
    return params.gravity_planet * \
           (params.radius_planet ** 2) / (params.radius_planet + altitude) ** 2


def compute_density_from_altitude(altitude: float, params: EntryVehicleParams):
    """
    Altitude and scale height are in meters
    :param altitude:
    :param params:
    :return:
    """
    return params.density_planet * np.exp(-altitude / params.scale_height)


def non_planar_eom(t: nd, state: nd, control: nd, params: EntryVehicleParams):
    """
    The state vector is speed, flight path angle, altitude, heading, longitude, latitude
    The control vector is bank angle, thrust force, thrust angle from flight path
    :param t:
    :param state:
    :param control:
    :param params:
    :return:
    """
    V = state[0]
    gamma = state[1]
    h = state[2]  # The altitude is in meters
    heading = state[3]
    longitude = state[4]
    latitude = state[5]

    T = control[0]
    epsilon = control[1]
    sigma = control[2]

    r = h + params.radius_planet

    g = compute_gravity_from_altitude(h, params)
    rho = compute_density_from_altitude(h, params)

    dv1 = T * cos(epsilon) / params.mass
    dv2 = -rho * V ** 2 / (2 * params.ballistic_coeff)
    dv3 = -g * sin(gamma)
    dv4 = params.omega_planet ** 2 * r * cos(latitude) * \
          (sin(gamma) * cos(latitude) - cos(gamma) * sin(latitude) * sin(heading))

    dvdt = dv1 + dv2 + dv3 + dv4

    dg1 = T * sin(epsilon) / (V * params.mass) * cos(sigma)
    dg2 = V * cos(gamma) / r
    dg3 = rho * V / (2 * params.ballistic_coeff) * params.lift_drag_ratio * cos(sigma)
    dg4 = -g * cos(gamma) / V
    dg5 = 2 * params.omega_planet * cos(latitude) * cos(heading)
    dg6 = params.omega_planet ** 2 * r / V * cos(latitude)
    dg7 = cos(gamma) * cos(latitude) + sin(gamma) * sin(latitude) * sin(heading)

    dgdt = dg1 + dg2 + dg3 + dg4 + dg5 + dg6 * dg7

    dp1 = T * sin(epsilon) * sin(sigma) / (V * params.mass * cos(gamma))
    dp2 = rho * V / (2 * params.ballistic_coeff) * params.lift_drag_ratio * sin(sigma) / cos(gamma)
    dp3 = -V * cos(gamma) * cos(heading) * tan(latitude) / r
    dp4 = 2 * params.omega_planet * (tan(gamma) * cos(latitude) * sin(heading) - sin(latitude))
    dp5 = -params.omega_planet ** 2 * r / (V * cos(gamma)) * sin(latitude) * cos(latitude) * cos(heading)

    dpdt = dp1 + dp2 + dp3 + dp4 + dp5

    dhdt = V * sin(gamma)

    dthetadt = V * cos(gamma) * cos(heading) / (r * cos(latitude))

    dphidt = V * cos(gamma) * sin(heading) / r

    state_dot = np.array([dvdt, dgdt, dhdt, dpdt, dthetadt, dphidt])

    return state_dot


def altitude_zero_event(t, state, *args):
    h = state[2]  # The altitude is in meters
    return h


altitude_zero_event.terminal = True
altitude_zero_event.direction = -1


def altitude_exoatmosphere_event(t: float, state: nd, params: EntryVehicleParams):
    x = state[2] - params.atmosphere_altitude_planet  # The altitude is in meters
    return x


