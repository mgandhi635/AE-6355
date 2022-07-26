import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geopandas as gpd

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from Project.src.project_vars import APOLLO_4_params
from Project.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma,
                                       normalize_angle, inch_to_meter)
from Project.src.nonplanar_eom import non_planar_eom, altitude_zero_event


V_0 = inch_to_meter(36333*12)  # m/s
gamma_0 = np.radians(-7.350)
h_0 = APOLLO_4_params.atmosphere_altitude_planet
psi_0 = np.radians(90-66.481)

pos_final = np.degrees(normalize_angle(np.radians([-157.976, 32.465])))  # (long, lat)
pos_initial = [155.637, 23.398]

theta_0 = np.radians(pos_initial[0])
phi_0 = np.radians(pos_initial[1])
x_0 = np.array([V_0, gamma_0, h_0, psi_0, theta_0, phi_0])

sigma = np.radians(58)
u_0 = [0, 0, sigma, APOLLO_4_params.lift_drag_ratio]

t_span = (0, 710)
t_eval = np.linspace(t_span[0], t_span[-1], int(1e4))

exponential_atmosphere = True
std_atmosphere = False

# Solve numerically first
sol_exp = solve_ivp(non_planar_eom, t_span, x_0,
                    t_eval=t_eval, args=(u_0, APOLLO_4_params, exponential_atmosphere),
                    events=altitude_zero_event, max_step = 1)

sol_std = solve_ivp(non_planar_eom, t_span, x_0,
                    t_eval=t_eval, args=(u_0, APOLLO_4_params, std_atmosphere),
                    events=altitude_zero_event, max_step = 1)


state_exp = sol_exp.y
time_exp = sol_exp.t

state_std = sol_std.y
time_std = sol_std.t


plt.figure(0)
plt.xlabel(r'Velocity ($\frac{km}{s}$)')
plt.ylabel(r'Altitude ($km$)')
plt.plot(state_exp[0] / 1000, state_exp[2] / 1000, label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(state_std[0] / 1000, state_std[2] / 1000, label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()

plt.figure(1)
plt.xlabel(r'Time ($s$)')
plt.ylabel(r'Altitude ($kft$)')
plt.plot(time_exp, state_exp[2]*3.28084 / 1000, label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(time_std, state_std[2]*3.28084 / 1000, label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()

plt.figure(2)
plt.xlabel(r'Time ($s$)')
plt.ylabel(r'Velocity ($\frac{kft}{s}$)')
plt.plot(time_exp, state_exp[0]*3.28084 / 1000, label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(time_std, state_std[0]*3.28084 / 1000, label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()

plt.figure(3)
plt.xlabel(r'Time ($s$)')
plt.ylabel(r'Flight Path Angle ($^\circ$)')
plt.plot(time_exp, np.degrees(state_exp[1]), label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(time_std, np.degrees(state_std[1]), label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()


# plt.figure(4)
# plt.xlabel(r'Time ($s$)')
# plt.ylabel(r'Heading ($^\circ$)')
# plt.plot(time_exp, np.degrees(normalize_angle(state_exp[3])), label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
# plt.plot(time_std, np.degrees(normalize_angle(state_std[3])), label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)


# Create a new plot of the world
countries = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey")
ax = plt.gca()
ax.scatter(np.degrees(normalize_angle(state_exp[4])), np.degrees(normalize_angle(state_exp[5])), c='b', alpha=0.5)
ax.scatter(np.degrees(normalize_angle(state_std[4])), np.degrees(normalize_angle(state_std[5])), c='r', alpha=0.5)
ax.scatter(pos_final[0], pos_final[1], color='g', marker='*', s=50)

plt.show()
