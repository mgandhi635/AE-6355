import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geopandas as gpd

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from Project.src.project_vars import STS_13_params
from Project.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma, normalize_angle)
from Project.edl_control.src.nonplanar_eom import non_planar_eom, altitude_zero_event

rE = 6378  # km
muE = 3.986e5  # km^3 / s^2
v_c = np.sqrt(muE / rE) * 1000  # m/s

V_0 = 7456  # m/s
gamma_0 = equilibrium_glide_gamma(V_0, v_c, STS_13_params.lift_drag_ratio, STS_13_params.scale_height, STS_13_params.radius_planet)
rho_0 = STS_13_params.density_planet  # kg/m^3
rho_init = 1.874e-07 * rho_0
h_0 = altitude_from_exponential_atmosphere_density(rho_init, rho_0, STS_13_params.scale_height)
psi_0 = np.radians(60)

pos_final = [242.15-360, 34.90]  # (long, lat)
pos_initial = [-150, 0]

theta_0 = np.radians(pos_initial[0])
phi_0 = np.radians(pos_initial[1])
# theta_0 = 0
# phi_0 = 0

x_0 = np.array([V_0, gamma_0, h_0, psi_0, theta_0, phi_0])

# Initial conditions from datasheet
V_1 = 7.416338e3
gamma_1 = np.radians(0)
h_1 = 8.69e5
psi_1 = np.radians(59.723)
x_1 = np.array([V_1, gamma_1, h_1, psi_1, theta_0, phi_0])


sigma = np.radians(68)
u_0 = [0, 0, sigma]

t_span = (0, 2.5e3)
t_eval = np.linspace(t_span[0], t_span[-1], int(1e6))

# Solve numerically first
sol = solve_ivp(non_planar_eom, t_span, x_1,
                t_eval=t_eval, args=(u_0, STS_13_params), events=altitude_zero_event, max_step = 1)

state = sol.y

plt.figure(0)
plt.xlabel(r'Velocity ($\frac{km}{s}$)')
plt.ylabel(r'Altitude ($km$)')
plt.plot(state[0] / 1000, state[2] / 1000, label='Planar EOM', c='b', linestyle='--', linewidth=2, alpha=0.7)

# Create a new plot of the world
countries = gpd.read_file(
               gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey")
ax = plt.gca()
ax.scatter(np.degrees(normalize_angle(state[4])), np.degrees(normalize_angle(state[5])))
ax.scatter(pos_final[0], pos_final[1], color='g', marker='*', s=20)

plt.show()
