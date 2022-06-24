import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True


def exponential_atmosphere(h, H, rho_0):
    """
    Compute density according to exponential atmosphere
    :param h: current altitude in km
    :param H: Scale height in km
    :param rho_0: Density at sea level in kg / m^3
    :return: rho: density at current altitude in kg / m^3
    """
    return rho_0 * np.exp(-h / H)


def gravity_variation(h, g_0, rp):
    """
    Compute gravitational constant using inverse square law
    :param h: altitude in km
    :param g_0: acceleration of gravity at sea level in m/s^2
    :param rp: planet radius in km
    :return: g: acceleration of gravity at altitude in m/s^2
    """
    return g_0 * (rp**2) / (rp + h)**2


def ballistic_eom(t, x, rp, H, B, g_0, rho_0):
    """
    Numerical equations of motion for purely ballistic entry, considering a constant scale height and exponential
    atmosphere model.
    :param x: State [velocity (m/s^2), flight_path_angle (radians), altitude (km)]
    :param rp: planet radius in km
    :param H: Scale height in km
    :param B: Ballistic coefficient in kg/m^2
    :param g_0: gravity at sea level m/s^2
    :param rho_0: density at sea level kg/m^3
    :return: x_dot: State derivative
    """

    V = x[0]
    gamma = x[1]
    h = x[2]

    # Convert altitude to km
    h_km = h / 1000

    # Compute the current density
    rho = exponential_atmosphere(h_km, H, rho_0)

    # Compute current acceleration of gravity
    g = gravity_variation(h_km, g_0, rp)
    # g= g_0

    V_dot = -rho * V ** 2 / (2 * B) - g * np.sin(gamma)
    gamma_dot = 1. / V * (V ** 2 * np.cos(gamma) / (rp*1000 + h) - g * np.cos(gamma) )
    h_dot = V * np.sin(gamma)

    x_dot = np.array([V_dot, gamma_dot, h_dot])
    return x_dot


def altitude_zero_event(t, x, *args):
    return x[2]


altitude_zero_event.terminal = True
altitude_zero_event.direction = -1


def compute_constant_C(rho_0, H, Beta, gamma):
    H_m = H * 1000
    return rho_0 * H_m / (2 * Beta * np.sin(gamma))


def allen_eggers_velocity(h, rho_0, H, B, gamma, V_0):
    H_m = H*1000
    C = compute_constant_C(rho_0, H, B, gamma)
    return V_0 * np.exp (C * np.exp (-h / H_m))


def compute_max_accel_numerical(x, rp, H, B, g_0, rho_0):
    x_dot = ballistic_eom(None, x, rp, H, B, g_0, rho_0)
    v_dot = x_dot[0]

    max_idx = np.argmax(np.abs(x_dot[0]))
    n_max = v_dot[max_idx] / g_0
    h_n_max = x[2, max_idx]
    v_n_max = x[0, max_idx]
    return n_max, h_n_max, v_n_max


def compute_max_accel_allen_eggers(rho_0, H, B, gamma, V_0):
    H_m = H*1000
    C = compute_constant_C(rho_0, H, B, gamma)
    h_n_max = H_m * np.log(-2*C)
    v_n_max = allen_eggers_velocity(h_n_max, rho_0, H, B, gamma, V_0)
    v_n_max_check = V_0 * np.exp(-0.5)
    # print(v_n_max_check - v_n_max)
    n_max = V_0 ** 2 * np.sin(gamma) / (2 * np.e * 9.806 * H_m)
    return n_max, h_n_max, v_n_max

line_styles = {'nm6': 'b-',
               'nm10': 'g-',
               'nm50': 'r-',
               'ae6': 'b--',
               'ae10': 'g--',
               'ae50': 'r--'}

line_labels = {'nm': 'DOP853',
               'ae': 'AE  '}


def plot_problem_1(state, gamma, fig_num=0, prefix='nm'):

    style_key = prefix+str(np.abs(gamma))
    style = line_styles[style_key]
    label = line_labels[prefix]

    plt.figure(fig_num)
    plt.xlabel(r'Velocity ($\frac{km}{s}$)')
    plt.ylabel(r'Altitude ($km$)')
    plt.plot(state[0]/1000, state[2]/1000, style, label=label+ r': $\gamma_{atm}=$'+f'{gamma}'+r'$^\circ$')
    plt.legend()

    plt.figure(fig_num+1)
    plt.xlabel(r'Velocity ($\frac{km}{s}$)')
    plt.ylabel(r'$\gamma$ ($^\circ$)')
    plt.plot(state[0]/1000, np.degrees(state[1]), style, label=label+ r': $\gamma_{atm}=$'+f'{gamma}'+r'$^\circ$')
    plt.legend()


def general_cross_range(ld_ratio, phi, rp):
    t1 = np.pi**2 / 48 * ld_ratio**2 * np.sin(2*phi)
    t2 = np.pi**2 * (1 + 4 / (ld_ratio * np.cos(phi))**2)
    return t1*(1 - 6 / t2) * rp


def sphere_volume(radius):
    return 4 / 3 * np.pi * radius ** 3


def compute_ballistic_coeff(mass, Cd, A):
    return mass / (Cd * A)


def allen_eggers_acceleration(h, V_atm, gamma, H, rho_0, Beta):
    H_m = H * 1000
    C = compute_constant_C(rho_0, H, Beta, gamma)
    t1 = np.exp(-h / H_m)
    t2 = np.exp(2 * C * t1)
    t3 = -C * V_atm**2 * np.sin(gamma) / H_m
    return t1 * t2 * t3


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


def equilibrium_glide_acceleration(v, v_c, l_d_ratio):
    t1 = np.sqrt((1 / l_d_ratio)**2 + 1)
    t2 = (1 - (v / v_c)**2)
    return -1 * t1 * t2


def equilibrium_glide_peak_acceleration(l_d_ratio):
    return -np.sqrt((1 / l_d_ratio)**2 + 1)


def skipping_entry_gamma(v, v_atm, l_d_ratio, gamma_atm):
    return gamma_atm - (l_d_ratio * np.log(v / v_atm))


def skipping_entry_density(gamma, gamma_atm, H_m, Beta, l_d_ratio):
    t1 = np.cos(gamma) - np.cos(gamma_atm)
    t2 = 2*Beta / (l_d_ratio*H_m)
    return t1*t2


def skipping_entry_acceleration(v, rho, Beta):
    return -rho * v **2 / (2 * Beta)


def skipping_entry_peak_acceleration(v_atm, gamma_atm, l_d_ratio, g_0, H_m):
    return v_atm**2 * gamma_atm**2 * np.exp(2 * gamma_atm / l_d_ratio) / (2 * g_0 * H_m) * (1 + (1 / l_d_ratio)**2) ** (1/2)