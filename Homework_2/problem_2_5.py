import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from util import (planar_eom, altitude_zero_event, altitude_from_exponential_atmosphere_density,
                  equilibrium_glide_gamma, compute_ballistic_coeff, lifting_entry_velocity)

"""
7.7 km/s and -.0022 rad
"""
rE = 6378  # km
muE = 3.986e5  # km^3 / s^2
v_c = np.sqrt(muE / rE) * 1000  # m/s

A_ref = 250  # m^2
Cd = 0.78
l_d_ratio = 1.07
phi = np.radians(56)
rho_0 = 1.225  # kg/m^3
rho_init = 1.874e-07 * rho_0
g_0 = 9.806
H = 7.2  # km
mass = 92000  # kg

Beta = compute_ballistic_coeff(mass, Cd, A_ref)

h_init = altitude_from_exponential_atmosphere_density(rho_init, rho_0, H)  # m
v_init = 7456  # m/s
gamma_init = equilibrium_glide_gamma(v_init, v_c, l_d_ratio*np.cos(phi), H, rE)
print(h_init)
print(v_init)
print(np.degrees(gamma_init))

t_span = (0, 1.5e3)
t_eval = np.linspace(t_span[0], t_span[-1], int(1e5))

# Solve numerically first
x_initial = [v_init, gamma_init, h_init]
sol = solve_ivp(planar_eom, t_span, x_initial,
                t_eval=t_eval, args=(Beta, rho_0, g_0, rE, H, l_d_ratio, phi), events=altitude_zero_event, max_step=0.1)

state = sol.y

""" STS data
100 7456 1.874e-07
200 7432 2.417e-06
250 7465 7.620e-06
300 7394 1.947e-05
350 7290 2.837e-05
400 7173 3.400e-05
450 7047 3.800e-05
500 6913 4.283e-05
550 6771 4.793e-05
600 6611 5.929e-05
650 6429 6.800e-05
700 6210 8.088e-05
750 5948 1.039e-04
800 5674 1.359e-04
870 5221 2.031e-04
900 4992 2.317e-04
950 4569 3.464e-04
1000 4084 4.703e-04
1050 3598 6.191e-04
1100 3097 8.320e-04
1200 2223 2.227e-03
1300 1546 6.176e-03

"""

STS_rho = np.array([1.874e-07,
                    2.417e-06,
                    7.620e-06,
                    1.947e-05,
                    2.837e-05,
                    3.400e-05,
                    3.800e-05,
                    4.283e-05,
                    4.793e-05,
                    5.929e-05,
                    6.800e-05,
                    8.088e-05,
                    1.039e-04,
                    1.359e-04,
                    2.031e-04,
                    2.317e-04,
                    3.464e-04,
                    4.703e-04,
                    6.191e-04,
                    8.320e-04,
                    2.227e-03,
                    6.176e-03])

STS_vel = np.array([7456.0,
                    7432.0,
                    7465.0,
                    7394.0,
                    7290.0,
                    7173.0,
                    7047.0,
                    6913.0,
                    6771.0,
                    6611.0,
                    6429.0,
                    6210.0,
                    5948.0,
                    5674.0,
                    5221.0,
                    4992.0,
                    4569.0,
                    4084.0,
                    3598.0,
                    3097.0,
                    2223.0,
                    1546.0])

STS_h = altitude_from_exponential_atmosphere_density(STS_rho * rho_0, rho_0, H)

eg_vel = lifting_entry_velocity(state[2], rE, g_0, Beta, l_d_ratio, phi, H, rho_0)
plt.figure(0)
plt.xlabel(r'Velocity ($\frac{km}{s}$)')
plt.ylabel(r'Altitude ($km$)')
plt.plot(state[0] / 1000, state[2] / 1000, label='Planar EOM', c='b', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(STS_vel / 1000, STS_h / 1000, label='STS-5', c='r', linewidth=2, alpha=0.7)
plt.plot(eg_vel / 1000, state[2] / 1000, label='Analytic Equilibrium Glide', c='g', linestyle='-.', linewidth=2, alpha=0.7)
plt.legend()
plt.grid()
plt.show()
