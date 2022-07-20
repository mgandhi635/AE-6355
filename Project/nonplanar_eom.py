import numpy as np
import scipy as sp
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt

from project_vars import EntryVehicleParams
from project_utils import compute_ballistic_coeff
from environments import (compute_density_from_altitude, compute_gravity_from_altitude)

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

nd = np.ndarray
sin = np.sin
cos = np.cos
tan = np.tan


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


if __name__ == "__main__":
    # get a set of parameters
    from project_vars import STS_13_params
    from project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma)
    from scipy.integrate import solve_ivp

    rE = 6378  # km
    muE = 3.986e5  # km^3 / s^2
    v_c = np.sqrt(muE / rE) * 1000  # m/s

    V_0 = 7456  # m/s
    gamma_0 = equilibrium_glide_gamma(V_0, v_c, STS_13_params.lift_drag_ratio, STS_13_params.scale_height, STS_13_params.radius_planet)
    rho_0 = STS_13_params.density_planet  # kg/m^3
    rho_init = 1.874e-07 * rho_0
    h_0 = altitude_from_exponential_atmosphere_density(rho_init, rho_0, STS_13_params.scale_height)
    psi_0 = 0
    theta_0 = 0
    phi_0 = 0
    x_0 = np.array([V_0, gamma_0, h_0, psi_0, theta_0, phi_0])
    sigma = np.radians(56)
    u_0 = [0,0,sigma]

    t_span = (0, 1.5e3)
    t_eval = np.linspace(t_span[0], t_span[-1], int(1e5))

    # Solve numerically first
    sol = solve_ivp(non_planar_eom, t_span, x_0,
                    t_eval=t_eval, args=(u_0, STS_13_params), events=altitude_zero_event)

    state = sol.y
    plt.figure(0)
    plt.xlabel(r'Velocity ($\frac{km}{s}$)')
    plt.ylabel(r'Altitude ($km$)')
    plt.plot(state[0] / 1000, state[2] / 1000, label='Planar EOM', c='b', linestyle='--', linewidth=2, alpha=0.7)
    plt.show()

