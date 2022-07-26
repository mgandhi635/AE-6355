import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import geopandas as gpd

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from edl_control.src.project_vars import STS_13_params, EntryVehicleParams
from edl_control.src.project_utils import (normalize_angle)

# Load the data for standard atmosphere
x_traj_c = np.load('sts_13_controlled_entry_x.npy', allow_pickle=True)
u_traj_c = np.load('sts_13_controlled_entry_u.npy', allow_pickle=True)
t_traj_c = np.load('sts_13_controlled_entry_t.npy', allow_pickle=True)
#
x_traj_u = np.load('sts_13_entry_x.npy', allow_pickle=True)
u_traj_u = np.load('sts_13_entry_u.npy', allow_pickle=True)
t_traj_u = np.load('sts_13_entry_t.npy', allow_pickle=True)

# Load the data for exponential atmosphere
x_traj_ce = np.load('sts_13_exp_controlled_entry_x.npy', allow_pickle=True)
u_traj_ce = np.load('sts_13_exp_controlled_entry_u.npy', allow_pickle=True)
t_traj_ce = np.load('sts_13_exp_controlled_entry_t.npy', allow_pickle=True)

x_traj_ue = np.load('sts_13_exp_entry_x.npy', allow_pickle=True)
u_traj_ue = np.load('sts_13_exp_entry_u.npy', allow_pickle=True)
t_traj_ue = np.load('sts_13_exp_entry_t.npy', allow_pickle=True)


# Create a new plot of the world


def plot_earth(params: EntryVehicleParams):
    countries = gpd.read_file(
    gpd.datasets.get_path("naturalearth_lowres"))
    fig = countries.plot(color="lightgrey")
    gt_axis = fig.axes
    return fig, gt_axis


def plot_boundary_conditions(gt_axis, params: EntryVehicleParams):
    # Plot the target point
    gt_axis.scatter(np.degrees(params.desired_position[0]), np.degrees(params.desired_position[1]),
                 color='r', marker='*', s=50, label='Desired Landing Point', alpha=1)
    # Plot the starting point
    gt_axis.scatter(np.degrees(normalize_angle(params.initial_position[0])), np.degrees(params.initial_position[1]),
                 color='k', marker='*', s=50, label='Entry Point', alpha=1)


# First plot is the ground track
def plot_ground_track(axis, x_traj, label, color=None, alpha=0.002):
    x = x_traj[0]
    axis.scatter(np.degrees(normalize_angle(x[4])), np.degrees(normalize_angle(x[5])),
                 alpha=alpha, label=label, color=color)
    for i in range(1, x_traj.shape[0]):
        x = x_traj[i]
        axis.scatter(np.degrees(normalize_angle(x[4])), np.degrees(normalize_angle(x[5])),
                     alpha=alpha, color=color)


def compute_terminal_error(x_traj, position_desired):
    num_samples = x_traj.shape[0]
    x_position_final = np.zeros((2, num_samples))
    for i in range(num_samples):
        x_position_final[:,i] = x_traj[i][4:,-1]

    diff = np.sqrt(np.sum((np.degrees(x_position_final.T) - np.degrees(position_desired))**2, 1))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    return mean_diff, std_diff


def plot_alt_vel(axis, x_traj, color):
    for i in range(1, x_traj.shape[0]):
        x = x_traj[i]
        axis.plot(x[0] / 1000, x[2] / 1000, linestyle='--', linewidth=2, alpha=0.5, color=color)


def plot_vec(axis, t_traj, x_traj, idx, ylabel, color, label, scale=True):
    ax.set_xlabel(r'Time ($s$)')
    ax.set_ylabel(ylabel)
    for i in range(1, x_traj.shape[0]):
        t = t_traj[i]
        x = x_traj[i]
        if scale:
            x = x/1000
        else:
            x = np.degrees(x)
        axis.plot(t, x[idx], linestyle='-', linewidth=2, alpha=0.3, color=color, label=label)



# Compute Errors WRT final position
mean_c, std_c = compute_terminal_error(x_traj_c, STS_13_params.desired_position)
mean_u, std_u = compute_terminal_error(x_traj_u, STS_13_params.desired_position)

mean_ce, std_ce = compute_terminal_error(x_traj_ce, STS_13_params.desired_position)
mean_ue, std_ue = compute_terminal_error(x_traj_ue, STS_13_params.desired_position)

print("Controlled Mean Terminal Error:", mean_c)
print("UnControlled Mean Terminal Error:", mean_u)
print("Controlled Std Terminal Error:", std_c)
print("UnControlled Std Terminal Error:", std_u)

print("Exp Controlled Mean Terminal Error:", mean_ce)
print("Exp UnControlled Mean Terminal Error:", mean_ue)
print("Exp Controlled Std Terminal Error:", std_ce)
print("Exp UnControlled Std Terminal Error:", std_ue)

# Plot and compare ground tracks
gt1_fig, gt1_axes = plot_earth(STS_13_params)
gt2_fig, gt2_axes = plot_earth(STS_13_params)

plot_ground_track(gt1_axes, x_traj_u, 'STS_13 LQR Exp', 'g', .025)
plot_ground_track(gt2_axes, x_traj_c, 'STS_13 LQR Std', 'b', .025)

plot_boundary_conditions(gt1_axes, STS_13_params)
plot_boundary_conditions(gt2_axes, STS_13_params)

# gt1_axes.legend()
# gt2_axes.legend()

gt1_axes.set_xlim(-125,-100)
gt1_axes.set_ylim(25,50)

gt2_axes.set_xlim(-125,-100)
gt2_axes.set_ylim(25,50)

fig, ax = plt.subplots()
plot_alt_vel(ax, x_traj_u, 'g')
plot_alt_vel(ax, x_traj_c, 'b')
plt.grid()
plt.tight_layout()

fig, ax = plt.subplots()
plot_vec(ax, t_traj_u, x_traj_u, 0, r'Velocity ($\frac{km}{s})$', 'g', 'Uncontrolled')
plot_vec(ax, t_traj_c, x_traj_c, 0, r'Velocity ($\frac{km}{s})$', 'b', 'Controlled')
plt.grid()
plt.tight_layout()

fig, ax = plt.subplots()
plot_vec(ax, t_traj_u, x_traj_u, 1, r'Flight Path Angle ($^\circ$)', 'g', 'Uncontrolled', False)
plot_vec(ax, t_traj_c, x_traj_c, 1, r'Flight Path Angle ($^\circ$)', 'b', 'Controlled', False)
ax.plot(np.arange(0,2500), -90*np.ones(2500), 'r')
ax.set_ylim(-95,10)
plt.grid()
plt.tight_layout()

fig, ax = plt.subplots()
plot_vec(ax, t_traj_c, u_traj_c, 0, r'Roll Angle ($^\circ$)', 'b', 'Controlled', False)
ax.plot(np.arange(0,2000), 90*np.ones(2000), 'r')
ax.plot(np.arange(0,2000), -90*np.ones(2000), 'r')
plt.grid()
plt.tight_layout()

fig, ax = plt.subplots()
plot_vec(ax, t_traj_c, u_traj_c*1000, 1, r'$\frac{L}{D}$', 'b', 'Controlled', True)
ax.plot(np.arange(0,2000), STS_13_params.lift_drag_ratio*np.ones(2000), 'r')
ax.plot(np.arange(0,2000), 0*np.ones(2000), 'r')
# plot_vec(ax, t_traj_c, u_traj_c, 0, r'Roll Angle ($^\circ$)', 'b', 'Controlled', False)
# ax.set_ylim(-95,10)
plt.grid()
plt.tight_layout()


plt.show()