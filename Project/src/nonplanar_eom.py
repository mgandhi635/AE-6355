import numpy as np
import scipy as sp
from typing import NamedTuple
from control import lqr

import matplotlib as mpl
import matplotlib.pyplot as plt

from Project.src.project_vars import EntryVehicleParams
from Project.src.environments import (compute_density_from_altitude, compute_gravity_from_altitude,
                                      compute_density_std_atmosphere)
from Project.src.project_utils import compute_desired_heading, normalize_angle

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

nd = np.ndarray
sin = np.sin
cos = np.cos
tan = np.tan


def compute_control(x, u_0, x_traj, K, params: EntryVehicleParams):
    # Scale the trajectory
    x_scale = x.copy()
    x_scale[0] = x_scale[0] / 1000
    x_scale[2] = x_scale[2] / 1000
    x_traj_scale = x_traj.copy()
    x_traj_scale[0] = x_traj_scale[0] / 1000
    x_traj_scale[2] = x_traj_scale[2] / 1000

    # Find the closest trajectory point
    idx = np.argmin(np.sum((x_traj_scale.T - x_scale) ** 2, 1))
    x_linearize = x_traj[:, idx]
    K_i = K[idx, :, :]

    dx = x - x_linearize

    # Compute the heading separately
    curr_pos = x[4:]
    des_pos = params.desired_position
    heading_des = compute_desired_heading(curr_pos, des_pos)

    heading_delta = normalize_angle(x[3] - heading_des)
    dx[3] = heading_delta
    # print(np.degrees(heading_des))

    u = u_0.copy()
    u[2:] = u_0[2:] - K_i @ dx
    # print(K_i @ dx)

    u[2] = np.clip(u[2], -np.pi / 2, np.pi / 2)
    u[3] = np.clip(u[3], 0, params.lift_drag_ratio)

    return u


def non_planar_eom(t: nd, state: nd, control: nd, params: EntryVehicleParams,
                   exp_atmosphere: bool = True,
                   control_gain = None, x_traj = None):
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
    ld_ratio = control[3]

    if control_gain is not None:
        control_data = compute_control(state, control, x_traj , control_gain, params)
        T = control_data[0]
        epsilon = control_data[1]
        sigma = control_data[2]
        ld_ratio = control_data[3]

    r = h + params.radius_planet

    g = compute_gravity_from_altitude(h, params)
    if exp_atmosphere:
        rho = compute_density_from_altitude(h, params)
    else:
        rho = compute_density_std_atmosphere(h)

    dv1 = T * cos(epsilon) / params.mass
    dv2 = -rho * V ** 2 / (2 * params.ballistic_coeff)
    dv3 = -g * sin(gamma)
    dv4 = params.omega_planet ** 2 * r * cos(latitude) * \
          (sin(gamma) * cos(latitude) - cos(gamma) * sin(latitude) * sin(heading))

    dvdt = dv1 + dv2 + dv3 + dv4

    dg1 = T * sin(epsilon) / (V * params.mass) * cos(sigma)
    dg2 = V * cos(gamma) / r
    dg3 = rho * V / (2 * params.ballistic_coeff) * ld_ratio * cos(sigma)
    dg4 = -g * cos(gamma) / V
    dg5 = 2 * params.omega_planet * cos(latitude) * cos(heading)
    dg6 = params.omega_planet ** 2 * r / V * cos(latitude)
    dg7 = cos(gamma) * cos(latitude) + sin(gamma) * sin(latitude) * sin(heading)

    dgdt = dg1 + dg2 + dg3 + dg4 + dg5 + dg6 * dg7

    dp1 = T * sin(epsilon) * sin(sigma) / (V * params.mass * cos(gamma))
    dp2 = rho * V / (2 * params.ballistic_coeff) * ld_ratio * sin(sigma) / cos(gamma)
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


def nonplanar_eom_jacobians(t, state: nd, control: nd, params: EntryVehicleParams, exp_atmosphere: bool = True):
    """
    {, {0},
   }
    :param t:
    :param state:
    :param control:
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
    ldratio = control[3]

    R = params.radius_planet
    omega = params.omega_planet
    beta = params.ballistic_coeff
    m = params.mass

    g = compute_gravity_from_altitude(h, params)
    if exp_atmosphere:
        rho = compute_density_from_altitude(h, params)
    else:
        rho = compute_density_std_atmosphere(h)

    f1 = np.array([[-1 * V * beta ** (-1) * rho],
                   [(-1 * g * cos(gamma) + (h + R) * omega ** 2 * cos(latitude) * (
                           cos(gamma) * cos(latitude) + sin(gamma) * sin(latitude) * sin(heading)))],
                   [-1 * (h + R) * omega ** 2 * cos(gamma) * cos(latitude) * cos(heading) * sin(latitude)],
                   [omega ** 2 * cos(latitude) * (
                           cos(latitude) * sin(gamma) + -1 * cos(gamma) * sin(latitude) * sin(heading))],
                   [0],
                   [((h + R) * omega ** 2 * cos(latitude) * (
                           -1 * sin(gamma) * sin(latitude) + -1 * cos(gamma) * cos(latitude) *
                           sin(heading)) + -1 * (h + R) * omega ** 2 * sin(latitude) * (
                             cos(latitude) * sin(gamma) + -1 * cos(gamma) * sin(latitude) * sin(heading)))]])

    f2 = np.array( [ [( ( ( h + R ) )**( -1 ) * cos( gamma ) + ( g * ( V
                        )**( -2 ) * cos( gamma ) + ( 1/2 * ldratio * ( beta )**( -1 ) * rho *
                        cos( sigma ) + ( -1 * ( m )**( -1 ) * T * ( V )**( -2 ) * cos( sigma
                        ) * sin( epsilon ) + -1 * ( h + R ) * ( V )**( -2 ) * ( omega )**(
                        2 ) * cos( latitude) * ( cos( gamma ) * cos( latitude) + sin( gamma ) *
                        sin( latitude) * sin( heading ) ) ) ) ) ),] ,
                     [( g * ( V )**( -1 )
                        * sin( gamma ) + ( -1 * ( ( h + R ) )**( -1 ) * V * sin( gamma ) + (
                        h + R ) * ( V )**( -1 ) * ( omega )**( 2 ) * cos( latitude) * ( -1 *
                        cos( latitude) * sin( gamma ) + cos( gamma ) * sin( latitude) * sin(
                        heading ) ) ) ),] ,
                     [( ( h + R ) * ( V )**( -1 ) * ( omega )**( 2
                        ) * cos( latitude) * cos( heading ) * sin( gamma ) * sin( latitude) + -2 *
                        omega * cos( latitude ) * sin( heading ) ),] ,
                     [( -1 * ( ( h + R )
                        )**( -2 ) * V * cos( gamma ) + ( V )**( -1 ) * ( omega )**( 2 ) *
                        cos( latitude) * ( cos( gamma ) * cos( latitude) + sin( gamma ) * sin(
                        latitude) * sin( heading ) ) ),] ,
                     [0,] ,
                     [( -2 * omega *
                        cos( heading ) * sin( latitude) + ( ( h + R ) * ( V )**( -1 ) * ( omega
                        )**( 2 ) * cos( latitude) * ( -1 * cos( gamma ) * sin( latitude) + cos(
                        latitude) * sin( gamma ) * sin( heading ) ) + -1 * ( h + R ) * ( V )**( -1
                        ) * ( omega )**( 2 ) * sin( latitude) * ( cos( gamma ) * cos( latitude)
                        + sin( gamma ) * sin( latitude) * sin( heading ) ) ) ),]] )


    f4 = np.array( [ [( 1/2 * ldratio * ( beta )**( -1 ) * rho * 1/cos(
                            gamma ) * sin( sigma ) + ( -1 * ( m )**( -1 ) * T * ( V )**( -2 ) * 
                            1/cos( gamma ) * sin( epsilon ) * sin( sigma ) + ( ( h + R ) * ( V 
                            )**( -2 ) * ( omega )**( 2 ) * cos( latitude ) * cos( heading ) * 1/cos( 
                            gamma ) * sin( latitude ) + -1 * ( ( h + R ) )**( -1 ) * cos( gamma ) * 
                            cos( heading ) * tan( latitude ) ) ) )],
                 [( 2 * omega * cos( 
                        latitude ) * ( 1/cos( gamma ) )**( 2 ) * sin( heading ) + ( 1/2 * ldratio * 
                        V * ( beta )**( -1 ) * rho * 1/cos( gamma ) * sin( sigma ) * tan( 
                        gamma ) + ( ( m )**( -1 ) * T * ( V )**( -1 ) * 1/cos( gamma ) * sin( 
                        epsilon ) * sin( sigma ) * tan( gamma ) + ( -1 * ( h + R ) * ( V 
                        )**( -1 ) * ( omega )**( 2 ) * cos( latitude ) * cos( heading ) * 1/cos( 
                        gamma ) * sin( latitude ) * tan( gamma ) + ( ( h + R ) )**( -1 ) * V * 
                        cos( heading ) * sin( gamma ) * tan( latitude ) ) ) ) )] ,
                 [( ( h + 
                        R ) * ( V )**( -1 ) * ( omega )**( 2 ) * cos( latitude ) * 1/cos( gamma 
                        ) * sin( latitude ) * sin( heading ) + ( 2 * omega * cos( latitude ) * cos( 
                        heading ) * tan( gamma ) + ( ( h + R ) )**( -1 ) * V * cos( gamma ) * 
                        sin( heading ) * tan( latitude ) ) ),],
                 [( -1 * ( V )**( -1 ) * ( 
                        omega )**( 2 ) * cos( latitude ) * cos( heading ) * 1/cos( gamma ) * sin( 
                        latitude ) + ( ( h + R ) )**( -2 ) * V * cos( gamma ) * cos( heading ) * 
                        tan( latitude ) ),] ,
                 [0,] ,
                 [( -1 * ( h + R ) * ( V )**( 
                        -1 ) * ( omega )**( 2 ) * ( cos( latitude ) )**( 2 ) * cos( heading ) * 
                        1/cos( gamma ) + ( -1 * ( ( h + R ) )**( -1 ) * V * cos( gamma ) * 
                        cos( heading ) * ( 1/cos( latitude ) )**( 2 ) + ( ( h + R ) * ( V )**( -1 ) 
                        * ( omega )**( 2 ) * cos( heading ) * 1/cos( gamma ) * ( sin( latitude ) 
                        )**( 2 ) + 2 * omega * ( -1 * cos( latitude ) + -1 * sin( latitude ) * 
                        sin( heading ) * tan( gamma ) ) ) ) ),] ] )

    f3 = np.array( [( [sin( gamma ),] ),
                 ( [V * cos( gamma ),] ),
                 ( [0,] ),
                 ( [0,] ),
                 ( [0,] ),
                 ( [0,] ),] )

    f5 = np.array( [( [( ( h + R ) )**( -1 ) * cos( gamma ) * cos( heading ) * 1/cos( latitude ),] ),
                 ( [-1 * ( ( h + R ) )**( -1 ) * V * cos( heading ) * 1/cos( latitude ) * sin( gamma ),] ),
                 ( [-1 * ( ( h + R ) )**( -1 ) * V * cos( gamma ) * 1/cos( latitude ) * sin( heading ),] ),
                 ( [-1 * ( ( h + R ) )**( -2 ) * V * cos( gamma ) * cos( heading ) * 1/cos( latitude ),] ),
                 ( [0,] ),
                 ( [( ( h + R ) )**( -1 ) * V * cos( gamma ) * cos( heading ) * 1/cos( latitude ) * tan( latitude ),] ),] )
    
    f6 = np.array( [( [( ( h + R ) )**( -1 ) * cos( gamma ) * sin( heading ),]),
                 ( [-1 * ( ( h + R ) )**( -1 ) * V * sin( gamma ) * sin( heading ),] ),
                 ( [( ( h + R ) )**( -1 ) * V * cos( gamma ) * cos( heading ),]),
                 ( [-1 * ( ( h + R ) )**( -2 ) * V * cos( gamma ) * sin( heading ),] ),
                 ( [0,] ),
                 ( [0,] ),] )


    F = np.hstack((f1, f2,f3,f4,f5,f6)).T

    g1 = np.array( [( [( m )**( -1 ) * cos( epsilon ),] ),
                    ( [-1 * ( m )**( -1 ) * T * sin( epsilon ),] ),
                    ( [0,] ),
                    ( [0,] )])

    g2 = np.array( [( [( m )**( -1 ) * ( V )**( -1 ) * cos( sigma ) * sin( epsilon ),] ),
                 ( [( m )**( -1 ) * T * ( V )**( -1 ) * cos( epsilon ) * cos( sigma ),] ),
                 ( [( -1/2 * ldratio * V * ( beta )**( -1 ) * rho * sin( sigma ) + -1 * ( m )**( -1 ) * T * ( V )**( -1 ) * sin( epsilon ) * sin( sigma ) ),] ) ,
                   [rho*V*cos(sigma)/ (2 * beta)]])

    g3 = np.zeros((4, 1))

    g4 = np.array( [( [( m )**( -1 ) * ( V )**( -1 ) * 1/cos( gamma ) * sin( epsilon ) * sin( sigma ),] ),
                 ( [( m )**( -1 ) * T * ( V )**( -1 ) * cos( epsilon ) * 1/cos( gamma ) * sin( sigma ),] ),
                 ( [( 1/2 * ldratio * V * ( beta )**( -1 ) * rho * cos( sigma ) * 1/cos( gamma ) + ( m )**( -1 ) * T * ( V )**( -1 ) * cos( sigma ) * 1/cos( gamma ) * sin( epsilon ) ),] ),
                    [rho*V*sin(sigma)/(cos(gamma)*2*beta)]] )

    g56 = np.zeros((4,2))

    G = np.hstack((g1, g2, g3, g4, g56)).T

    # Only control roll angle
    G = G[:,2:]
    return F, G


def finite_difference(F, x, u, Fx, Fu):
    """Function that computes the jacobians of f via finite differencing.
    State derivative method.
    Inputs:
        F: function handle that returns the derivative of x
        x: Nx x N ndarray where N can be the time horizon points (DDP) or
                            N is the collocation points (PSOC)
        u: Nu x N ndarray
    Outputs:
            Fx: Nx x Nx x N ndarray of state derivatives. Each slice is arranged
                [dF1_dx1 , dF1_dx2 , ... , dF1_dxNx]
                [dF2_dx1 , dF2_dx2 , ... , dF2_dxNx]
                [  ...   ,                  ...    ]
                [dFNx_dx1 , DFNx_dx2, ... , dFNx_dxNx]
            Fu: Nx x Nu x N ndarray of control derivatives. Each slice is arranged
                [dF1_du1 , dF1_du2 , ... , dF1_duNu]
                [dF2_du1 , dF2_du2 , ... , dF2_duNu]
                [  ...   ,                  ...    ]
                [dFNx_du1 , DFNx_du2, ... , dFNx_duNu]
    """
    eps = np.sqrt(7./3 - 4./3 - 1)  # Finite difference step size
    F0 = F(x, u)  # Initial point
    Nx = np.size(x, 0)  # State dimension
    Nu = np.size(u, 0)
    # N = np.size(x, 1)  # Number of points
    # Fx = np.zeros([Nx, Nx, N])  # Preallocate A matrix and B matrix
    # Fu = np.zeros([Nx, Nu, N])
    for i in range(Nx):
        x_shift = x.copy()
        x_shift[i] += eps
        F_shift = F(x_shift, u)
        Fx[:, i] = (F_shift - F0) / eps

    for i in range(Nu):
        u_shift = u.copy()
        u_shift[i] += eps
        F_shift = F(x, u_shift)
        Fu[:, i] = (F_shift - F0) / eps
    return Fx, Fu[:,2:]


if __name__ == "__main__":
    from Project.src.project_vars import APOLLO_4_params
    from Project.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma,
                                           normalize_angle, inch_to_meter)

    V_0 = inch_to_meter(36333 * 12)  # m/s
    gamma_0 = np.radians(-7.350)
    h_0 = 100*1000
    psi_0 = np.radians(90 - 66.481)

    pos_final = np.degrees(normalize_angle(np.radians([-157.976, 32.465])))  # (long, lat)
    pos_initial = [155.637, 23.398]

    theta_0 = np.radians(pos_initial[0])
    phi_0 = np.radians(pos_initial[1])
    x_0 = np.array([V_0, gamma_0, h_0, psi_0, theta_0, phi_0])

    sigma = np.radians(56)
    u_0 = np.array([0, 0, sigma, APOLLO_4_params.lift_drag_ratio])

    Fx_al, Fu_al = nonplanar_eom_jacobians(0, x_0, u_0, APOLLO_4_params)

    # Get a wrapper
    def wrapper_f(x,u):
        return non_planar_eom(0, x, u, APOLLO_4_params, True)

    Fx_fd = np.zeros((6, 6))
    Fu_fd = np.zeros((6, 4))

    Fx_fd, Fu_fd = finite_difference(wrapper_f, x_0, u_0, Fx_fd, Fu_fd)

    print(np.linalg.norm(Fx_fd - Fx_al))
    print(np.linalg.norm(Fu_fd - Fu_al))

    print(Fu_fd)
    print(Fu_al)
