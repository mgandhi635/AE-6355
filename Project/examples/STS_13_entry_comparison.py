import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geopandas as gpd

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from Project.src.project_vars import STS_13_params
from Project.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma, normalize_angle)
from Project.src.nonplanar_eom import non_planar_eom, altitude_zero_event


rE = 6378  # km
muE = 3.986e5  # km^3 / s^2
v_c = np.sqrt(muE / rE) * 1000  # m/s

pos_final = [242.163116-360, 34.930885]  # (long, lat)
pos_initial = [150-360, 0]

theta_0 = np.radians(pos_initial[0])
phi_0 = np.radians(pos_initial[1])

V_1 = 7397.1912
gamma_1 = np.radians(-2.036)
h_1 = 213241.64616
psi_1 = np.radians(90 - 59.723)
x_1 = np.array([V_1, gamma_1, h_1, psi_1, theta_0, phi_0])


sigma = np.radians(58)
u_0 = [0, 0, sigma, STS_13_params.lift_drag_ratio]

t_span = (0, 2500)
t_eval = np.linspace(t_span[0], t_span[-1], int(1e4))

exponential_atmosphere = True
std_atmosphere = False

# Solve numerically first
sol_exp = solve_ivp(non_planar_eom, t_span, x_1,
                    t_eval=t_eval, args=(u_0, STS_13_params, exponential_atmosphere),
                    events=altitude_zero_event, max_step = 1)

sol_std = solve_ivp(non_planar_eom, t_span, x_1,
                    t_eval=t_eval, args=(u_0, STS_13_params, std_atmosphere),
                    events=altitude_zero_event, max_step = 1)


state_exp = sol_exp.y
time_exp = sol_exp.t

state_std = sol_std.y
time_std = sol_std.t


# Load true data from STS_13
data = np.loadtxt('../STS13_Metric.txt')
true_time = data[:,0]
true_altitude = data[:,1]
true_velocity = data[:,2]
true_fpa = data[:,3]

plt.figure(0)
plt.xlabel(r'Velocity ($\frac{km}{s}$)')
plt.ylabel(r'Altitude ($km$)')
plt.plot(true_velocity/1000, true_altitude/1000, label='True', c='k', linestyle='-', linewidth=2, alpha=0.7)
plt.plot(state_exp[0] / 1000, state_exp[2] / 1000, label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(state_std[0] / 1000, state_std[2] / 1000, label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()

plt.figure(1)
plt.xlabel(r'Time ($s$)')
plt.ylabel(r'Altitude ($km$)')
plt.plot(true_time, true_altitude/1000, label='True', c='k', linestyle='-', linewidth=2, alpha=0.7)
plt.plot(time_exp, state_exp[2] / 1000, label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(time_std, state_std[2] / 1000, label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()

plt.figure(2)
plt.xlabel(r'Time ($s$)')
plt.ylabel(r'Velocity ($\frac{kft}{s}$)')
plt.plot(true_time, true_velocity/1000, label='True', c='k', linestyle='-', linewidth=2, alpha=0.7)
plt.plot(time_exp, state_exp[0] / 1000, label='Exp', c='b', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(time_std, state_std[0] / 1000, label='Std', c='r', linestyle='--', linewidth=2, alpha=0.7)
plt.grid()
plt.tight_layout()
plt.legend()

# plt.figure(3)
# plt.xlabel(r'Time ($s$)')
# plt.ylabel(r'Flight Path Angle ($^\circ$)')
# plt.plot(true_time, true_fpa/1000, label='True', c='k', linestyle='--', linewidth=2, alpha=0.5)
# plt.plot(time_exp, np.degrees(state_exp[1]), label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.5)
# plt.plot(time_std, np.degrees(state_std[1]), label='Std', c='r', linestyle='--', linewidth=2, alpha=0.5)
# plt.legend()


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