import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from util import (ballistic_eom, altitude_zero_event, allen_eggers_velocity,
                  compute_max_accel_numerical, compute_max_accel_allen_eggers, plot_problem_1)

rE = 6378  # km
muE = 3.986e5  # km^3 / s^2

Vc = np.sqrt(muE / rE)
V_atm = 1.4*Vc*1000  # m/s^2
Beta = 157  # kg / m^2
gamma = [-6, -10, -50]  # degrees
H = 7.2 # km
rho_0 = 1.225  # kg / m^3
g_0 = 9.806  # m/s^2
h_atm = 125*1000  # m

# Numerically integrate the ballistic equations of motion
t_span = (0, 1e3)
t_eval = np.linspace(t_span[0], t_span[-1], int(1e5))
title = r'Numerical Integration DOP853'
method = 'DOP853'

h_ae = np.linspace(0, h_atm, int(1e5))

# Solve for each condition in a loop
for g in gamma:
    x_initial = [V_atm, np.radians(g), h_atm]
    sol = solve_ivp(ballistic_eom, t_span, x_initial, method=method,
                    t_eval=t_eval, args=(rE, H, Beta, g_0, rho_0), events=altitude_zero_event)
    # print(sol)
    plot_problem_1(sol.y, g)
    # Compute the maximum deceleration and the velocity and altitude at which these occur
    n_max_ode, h_n_max_ode, v_n_max_ode = compute_max_accel_numerical(sol.y, rE, H, Beta, g_0, rho_0)
    print(f"\nMaximum deceleration info for gamma = {g} degrees using ODE:")
    print(f"Maximum deceleration {n_max_ode:.3f} (g)")
    print(f"Altitude of n_max: {h_n_max_ode/1000:.3f} (km)")
    print(f"Velocity of n_max: {v_n_max_ode/1000:.3f} (km/s)")


    # Solve allen eggers
    vel_ae = allen_eggers_velocity(h_ae, rho_0, H, Beta, np.radians(g), V_atm)
    gamma_ae = np.radians(g)*np.ones_like(h_ae)
    sol_ae = np.vstack([vel_ae, gamma_ae, h_ae])
    plot_problem_1(sol_ae, g, 0, 'ae')
    # Compute max deceleration using allen eggers:
    n_max_ae, h_n_max_ae, v_n_max_ae = compute_max_accel_allen_eggers(rho_0, H, Beta, np.radians(g), V_atm)
    print(f"\nMaximum deceleration info for gamma = {g} degrees using AE:")
    print(f"Maximum deceleration {n_max_ae:.3f} (g)")
    print(f"Altitude of n_max: {h_n_max_ae/1000:.3f} (km)")
    print(f"Velocity of n_max: {v_n_max_ae/1000:.3f} (km/s)")


for fig in plt.get_fignums():
    plt.figure(fig)
    plt.grid()

plt.show()
