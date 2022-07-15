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


class EntryVehicleParams(NamedTuple):
    mass: float
    omega_planet: float
    ballistic_coeff: float
    radius_planet: float
    gravity_planet: float
    lift_drag_ratio: float


def compute_gravity_from_altitude(altitude: float, params: EntryVehicleParams):
    """Gravity computation"""
    return params.gravity_planet


def compute_density_from_altitude(altitude: float, params: EntryVehicleParams):
    return 0


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
    dg6 = params.omega_planet**2 * r / V * cos(latitude)
    dg7 = cos(gamma)*cos(latitude) + sin(gamma)*sin(latitude)*sin(heading)

    dgdt = dg1 + dg2 + dg3 + dg4 + dg5 + dg6*dg7
