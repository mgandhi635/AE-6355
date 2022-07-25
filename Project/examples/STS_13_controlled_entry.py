import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geopandas as gpd
from control import lqr

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from Project.src.project_vars import STS_13_params, EntryVehicleParams
from Project.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma,
                                       normalize_angle, inch_to_meter)
from Project.src.nonplanar_eom import non_planar_eom, nonplanar_eom_jacobians, altitude_zero_event


# Load the data
data = np.load('test.npz')

x_traj = data['state_std']
u_traj = data['control_std']

Q = 1*np.eye(6)
Q[0,0] = 2  # vel
Q[1,1] = 1  # gamma
Q[2,2] = 2  # height
Q[3,3] = 1e5  # heading
Q[4,4] = 20 # long
Q[5,5] = 20  # lat
R = 1e4*np.eye(2)

# Q = 1*np.eye(6)
# Q[0,0] = 100
# Q[1,1] = 50
# Q[2,2] = 1e-1
# Q[3,3] = 1e-1
# Q[4,4] = 200
# Q[5,5] = 200
# R = 1*np.eye(2)


pos_final = [242.163116-360, 34.930885]  # (long, lat)
pos_initial = [150-360, 0]
theta_0 = np.radians(pos_initial[0])
phi_0 = np.radians(pos_initial[1])

# Initial conditions from datasheet
x_final = np.array([25, np.radians(-70), 1e3, 0, np.radians(242.163116 - 360), np.radians(34.930885)])

sigma = np.radians(56)
u_0 = [0, 0, sigma, STS_13_params.lift_drag_ratio]

t_span = (0, 2500)
t_eval = np.linspace(t_span[0], t_span[-1], int(5e2))
exponential_atmosphere = True
std_atmosphere = False

# Solve numerically first
# sol_exp = solve_ivp(non_planar_eom, t_span, x_1,
#                     t_eval=t_eval, args=dynamic_args_exp,
#                     events=altitude_zero_event, max_step = 1)
# Precompute the control gains
K_vec = np.zeros((x_traj.shape[1], 2, 6))
for i in range(1,x_traj.shape[1]):
    curr_x = x_traj[:,-i]
    A, B = nonplanar_eom_jacobians(0, curr_x, u_0, STS_13_params, std_atmosphere)
    try:
        K, S, E = lqr(A, B, Q, R)
        K_vec[-i,:,:] = K
    except:
        K_vec[-i,:,:] = K_vec[-i+1,:,:]


dynamic_args_std = (u_0, STS_13_params, std_atmosphere,
                  K_vec, x_traj)

plt.grid()
# Create a new plot of the world
countries = gpd.read_file(
    gpd.datasets.get_path("naturalearth_lowres"))
fig = countries.plot(color="lightgrey")
ax = fig.axes
ax.set_title("Controlled")
# ax.set_xlim(-125,-100)
# ax.set_ylim(30,45)



V_std = 500
gamma_std = 0.05
h_std = 5000
n_samples = 50
np.random.seed(5)

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
    time_std = sol_std.t

    plt.figure(0)
    plt.xlabel(r'Velocity ($\frac{km}{s}$)')
    plt.ylabel(r'Altitude ($km$)')
    # plt.plot(state_exp[0] / 1000, state_exp[2] / 1000, label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
    plt.plot(state_std[0] / 1000, state_std[2] / 1000, label='Std', linestyle='--', linewidth=2, alpha=0.7)

    plt.figure(1)
    plt.xlabel(r'Time ($s$)')
    plt.ylabel(r'Altitude ($kft$)')
    # plt.plot(time_exp, state_exp[2]*3.28084 / 1000, label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
    plt.plot(time_std, state_std[2]*3.28084 / 1000, label='Std', linestyle='--', linewidth=2, alpha=0.7)

    # ax.scatter(np.degrees(normalize_angle(state_exp[4])), np.degrees(normalize_angle(state_exp[5])), c='b', alpha=0.5)
    ax.scatter(np.degrees(normalize_angle(state_std[4])), np.degrees(normalize_angle(state_std[5])), alpha=0.2)
    ax.scatter(pos_final[0], pos_final[1], color='g', marker='*', s=50)

plt.show()