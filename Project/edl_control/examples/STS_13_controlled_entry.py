import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import geopandas as gpd
from control import lqr

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from Project.src.project_vars import STS_13_params, EntryVehicleParams
from Project.edl_control.src.nonplanar_eom import non_planar_eom, nonplanar_eom_jacobians, altitude_zero_event, compute_control


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
R = np.eye(2)
R[0,0] = 1e4
R[1,1] = 1e4

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
    A, B = nonplanar_eom_jacobians(0, curr_x, u_0, STS_13_params, exponential_atmosphere)
    try:
        K, S, E = lqr(A, B, Q, R)
        K_vec[-i,:,:] = K
    except:
        K_vec[-i,:,:] = K_vec[-i+1,:,:]


dynamic_args_std = (u_0, STS_13_params, exponential_atmosphere,
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

all_x_trajectories = np.array([None]*n_samples, dtype=np.ndarray)
all_u_trajectories = np.array([None]*n_samples, dtype=np.ndarray)
all_t_trajectories = np.array([None]*n_samples, dtype=np.ndarray)

# Generate each sample
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
        control_std[:, t] = compute_control(state_std[:, t], u_0, x_traj, K_vec, STS_13_params)[2:]

    all_x_trajectories[i] = state_std
    all_u_trajectories[i] = control_std
    all_t_trajectories[i] = time_std


# Save all the samples
np.save('../plotting/sts_13_exp_controlled_entry_x.npy', all_x_trajectories[:], allow_pickle=True)
np.save('../plotting/sts_13_exp_controlled_entry_u.npy', all_u_trajectories[:], allow_pickle=True)
np.save('../plotting/sts_13_exp_controlled_entry_t.npy', all_t_trajectories[:], allow_pickle=True)

#     plt.figure(0)
#     plt.xlabel(r'Velocity ($\frac{km}{s}$)')
#     plt.ylabel(r'Altitude ($km$)')
#     # plt.plot(state_exp[0] / 1000, state_exp[2] / 1000, label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
#     plt.plot(state_std[0] / 1000, state_std[2] / 1000, label='Std', linestyle='-', linewidth=2, alpha=0.7)
#
#     fig1 = plt.figure(1)
#     ax1a = fig1.axes[0]
#     ax1b = ax1a.twinx()
#     plt.xlabel(r'Time ($s$)')
#     plt.ylabel(r'Altitude ($km$)')
#     ax1a.set_ylabel(r'Altitude ($km$)', color='r')
#     ax1b.set_ylabel(r'Velocity ($\frac{km}{s}$)', color='b')
#     # plt.plot(time_exp, state_exp[2]*3.28084 / 1000, label='Exp', c='b', linestyle='--', linewidth=2, alpha=0.7)
#     ax1a.plot(time_std, state_std[2] / 1000, label='Std', linestyle='-', linewidth=2, alpha=0.5, c='r')
#     ax1b.plot(time_std, state_std[0] / 1000, label='Std', linestyle='-', linewidth=2, alpha=0.5, c='b')
#
#     plt.figure(3)
#     plt.xlabel(r'Time ($s$)')
#     plt.ylabel(r'Roll Angle ($^\circ$)')
#     plt.plot(time_std, np.degrees(control_std[0]), label='Std', linestyle='-', linewidth=2, alpha=0.7)
#
#     plt.figure(4)
#     plt.xlabel(r'Time ($s$)')
#     plt.ylabel(r'L/D Ratio')
#     plt.plot(time_std, control_std[1], label='Std', linestyle='-', linewidth=2, alpha=0.7)
#
#     # ax.scatter(np.degrees(normalize_angle(state_exp[4])), np.degrees(normalize_angle(state_exp[5])), c='b', alpha=0.5)
#     ax.scatter(np.degrees(normalize_angle(state_std[4])), np.degrees(normalize_angle(state_std[5])), alpha=0.2)
#     ax.scatter(pos_final[0], pos_final[1], color='g', marker='*', s=50)
#
# plt.show()