import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geopandas as gpd

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from edl_control.src.project_vars import STS_13_params
from edl_control.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma)
from edl_control.src.nonplanar_eom import non_planar_eom, altitude_zero_event

rE = 6378  # km
muE = 3.986e5  # km^3 / s^2
v_c = np.sqrt(muE / rE) * 1000  # m/s

V_0 = 7456  # m/s
gamma_0 = equilibrium_glide_gamma(V_0, v_c, STS_13_params.lift_drag_ratio, STS_13_params.scale_height, STS_13_params.radius_planet)
rho_0 = STS_13_params.density_planet  # kg/m^3
rho_init = 1.874e-07 * rho_0
h_0 = altitude_from_exponential_atmosphere_density(rho_init, rho_0, STS_13_params.scale_height)
psi_0 = np.radians(56)

pos_final = [242.163116-360, 34.930885]  # (long, lat)
pos_initial = [150-360, 0]

theta_0 = np.radians(pos_initial[0])
phi_0 = np.radians(pos_initial[1])
# theta_0 = 0
# phi_0 = 0

x_0 = np.array([V_0, gamma_0, h_0, psi_0, theta_0, phi_0])

# Initial conditions from datasheet
V_1 = 7397.1912
gamma_1 = np.radians(-2.036)
h_1 = 213241.64616
psi_1 = np.radians(90-59.723)
x_1 = np.array([V_1, gamma_1, h_1, psi_1, theta_0, phi_0])


sigma = np.radians(56)
u_0 = [0, 0, sigma, STS_13_params.lift_drag_ratio]

t_span = (0, 2500)
t_eval = np.linspace(t_span[0], t_span[-1], int(5e2))

exponential_atmosphere = True
std_atmosphere = False


dynamic_args_std = (u_0, STS_13_params, exponential_atmosphere)

plt.grid()
# Create a new plot of the world
countries = gpd.read_file(
    gpd.datasets.get_path("naturalearth_lowres"))
fig = countries.plot(color="lightgrey")
ax = fig.axes
ax.set_title("Uncontrolled")
# ax.set_xlim(-125,-100)
# ax.set_ylim(30,45)



V_std = 500
gamma_std = 0.05
h_std = 5000
n_samples = 50
np.random.seed(5)

all_x_trajectories = np.array([None]*n_samples, dtype=np.ndarray)
all_u_trajectories = np.array([None]*n_samples, dtype=np.ndarray)
all_t_trajectories = np.array([None]*n_samples, dtype=np.ndarray)

for i in range(n_samples):
    # Initial conditions from datasheet
    V_1 = 7397.1912 - 0 * np.random.randn(1)[0]
    gamma_1 = np.radians(-2.036 + gamma_std * np.random.randn(1)[0])
    h_1 = 213241.64616 - h_std * np.random.randn(1)[0]
    psi_1 = np.radians(90 - 59.723)
    x_1 = np.array([V_1, gamma_1, h_1, psi_1, theta_0, phi_0])

    sol_std = solve_ivp(non_planar_eom, t_span, x_1,
                        t_eval=t_eval, args=dynamic_args_std,
                        events=altitude_zero_event, max_step=1)

    state_std = sol_std.y
    control_std = np.zeros((2,state_std.shape[1]))
    time_std = sol_std.t

    for t in range(state_std.shape[1]):
        control_std[:, t] = u_0[2:]

    all_x_trajectories[i] = state_std
    all_u_trajectories[i] = control_std
    all_t_trajectories[i] = time_std
    # np.savez('test.npz', state_std=state_std, time_std=time_std, control_std=u_0)

    # plt.figure(0)
    # plt.xlabel(r'Velocity ($\frac{km}{s}$)')
    # plt.ylabel(r'Altitude ($km$)')
    # # plt.plot(state_exp[0] / 1000, state_exp[2] / 1000, label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
    # plt.plot(state_std[0] / 1000, state_std[2] / 1000, label='Std', linestyle='--', linewidth=2, alpha=0.7)
    #
    # plt.figure(1)
    # plt.xlabel(r'Time ($s$)')
    # plt.ylabel(r'Altitude ($kft$)')
    # # plt.plot(time_exp, state_exp[2]*3.28084 / 1000, label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
    # plt.plot(time_std, state_std[2]*3.28084 / 1000, label='Std', linestyle='--', linewidth=2, alpha=0.7)
    #
    # # ax.scatter(np.degrees(normalize_angle(state_exp[4])), np.degrees(normalize_angle(state_exp[5])), c='b', alpha=0.5)
    # ax.scatter(np.degrees(normalize_angle(state_std[4])), np.degrees(normalize_angle(state_std[5])), alpha=0.2)
    # ax.scatter(pos_final[0], pos_final[1], color='g', marker='*', s=50)

# Save all the samples
np.save('../plotting/sts_13_exp_entry_x.npy', all_x_trajectories[:], allow_pickle=True)
np.save('../plotting/sts_13_exp_entry_u.npy', all_u_trajectories[:], allow_pickle=True)
np.save('../plotting/sts_13_exp_entry_t.npy', all_t_trajectories[:], allow_pickle=True)
# plt.show()
