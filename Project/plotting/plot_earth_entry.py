import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import geopandas as gpd

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from Project.src.project_vars import STS_13_params, EntryVehicleParams
from Project.src.project_utils import (altitude_from_exponential_atmosphere_density, equilibrium_glide_gamma,
                                       normalize_angle, inch_to_meter)

# Load the data
x_traj_c = np.load('sts_13_controlled_entry_x.npy', allow_pickle=True)
u_traj_c = np.load('sts_13_controlled_entry_u.npy', allow_pickle=True)
t_traj_c = np.load('sts_13_controlled_entry_t.npy', allow_pickle=True)
#
# x_traj_u = np.load('sts_13_entry_x.npy', allow_pickle=True)
# u_traj_u = np.load('sts_13_entry_u.npy', allow_pickle=True)
# t_traj_u = np.load('sts_13_entry_t.npy', allow_pickle=True)

# Load the data
x_traj_u = np.load('sts_13_exp_controlled_entry_x.npy', allow_pickle=True)
u_traj_u = np.load('sts_13_exp_controlled_entry_u.npy', allow_pickle=True)
t_traj_u = np.load('sts_13_exp_controlled_entry_t.npy', allow_pickle=True)

# x_traj_u = np.load('sts_13_exp_entry_x.npy', allow_pickle=True)
# u_traj_u = np.load('sts_13_exp_entry_u.npy', allow_pickle=True)
# t_traj_u = np.load('sts_13_exp_entry_t.npy', allow_pickle=True)


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


def draw_error_band(ax, x, y, err, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


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
plt.show()