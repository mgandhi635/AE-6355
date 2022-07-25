import numpy as np


def compute_ballistic_coeff(mass, Cd, A):
    return mass / (Cd * A)


def altitude_from_exponential_atmosphere_density(rho, rho_0, H):
    density_ratio = rho / rho_0
    return -np.log(density_ratio) * H


def equilibrium_glide_gamma(v, v_c, l_d_ratio, H, rp):
    """
    Compute flight path angle as a function of velocity for the equilibrium glide case. Assumes constant L/D.
    :param v: (m/s)
    :param v_c: (m/s)
    :param l_d_ratio:
    :param H_m: (m)
    :param rp: (m)
    :return:
    """
    return (v_c / v) ** 2 * (1 / l_d_ratio) * -2 * H / rp


def normalize_angle(angle):
    angle -= np.ceil(angle / (2*np.pi) - 0.5) * (2*np.pi)
    return angle

def inch_to_meter(inch):
    return 0.0254*inch


def compute_desired_heading(cur_position, des_position):
    pos_vector = des_position - cur_position
    heading_des = np.arctan2(pos_vector[1], pos_vector[0])
    return normalize_angle(heading_des)


# p1 = np.array([10,2])
# p2 = np.array([0,0])
#
# print(np.degrees(compute_desired_heading(p2, p1)))
