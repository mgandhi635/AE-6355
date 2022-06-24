import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 13})

from util import (equilibrium_glide_gamma, equilibrium_glide_acceleration,
                  skipping_entry_gamma, skipping_entry_acceleration, skipping_entry_density,
                  skipping_entry_peak_acceleration, equilibrium_glide_peak_acceleration)


H = 7.2  # km
H_m = H * 1000
rE = 6378  # km
l_d_ratio = 2.0
# gamma_atm = -np.radians(25)
gamma_atm = -0.1
muE = 3.986e5  # km^3 / s^2
g_0 = 9.806  # m / s^2
v_c = np.sqrt(muE / rE) * 1000  # m/s
v_atm = 1.1*v_c  # m/s
Beta = 200


v_vec = np.linspace(1e4, 1.4*v_c, int(1e5))
eg_gamma = equilibrium_glide_gamma(v_vec, v_c, l_d_ratio, H, rE)
se_gamma = skipping_entry_gamma(v_vec, v_atm, l_d_ratio, gamma_atm)

plt.figure()
ax1 = plt.subplot()
ax2 = ax1.twinx()

ax1.plot(v_vec, np.degrees(eg_gamma), label=r"Equilibrium Glide $\gamma$", c='r')
ax1.legend()
ax2.plot(v_vec, np.degrees(se_gamma), label=r"Skipping Entry $\gamma$", c='b')
ax1.set_xlabel(r"Velocity $(\frac{m}{s}$)")
ax1.set_ylabel(r"Flight Path Angle $(^\circ$)")
ax1.set_ylim([-18,0])
ax2.set_ylim([-25,25])

ax2.legend()
plt.grid()

plt.figure()
ax1 = plt.subplot()
ax2 = ax1.twinx()

eg_n = equilibrium_glide_acceleration(v_vec, v_c, l_d_ratio)
se_rho = skipping_entry_density(se_gamma, gamma_atm, H_m, Beta, l_d_ratio)
se_n = skipping_entry_acceleration(v_vec, se_rho, Beta) / g_0

ax1.plot(v_vec, eg_n, label=r"Equilibrium Glide Acceleration", c='r')
ax1.legend()
ax2.plot(v_vec, se_n, label=r"Skipping Entry Acceleration", c='b')
plt.xlabel(r"Velocity $(\frac{m}{s})$")
plt.ylabel(r"Acceleration $(g)$")
# ax1.set_ylim([-18,0])
# ax2.set_ylim([-25,25])

ax2.legend()
plt.grid()

print(f"Peak deceleration skipping: {np.max(-se_n):.3f} (g's)")
print(f"Peak deceleration eq glide: {np.max(-eg_n):.3f} (g's)")


"""Comments
For equilibrium glide, the assumptions are a constant l_d_ratio, and a "small" flight path angle. As the velocity decreases,
the flight path angle increases. As the lift to drag ratio decreases towards 0, the flight path angle increases faster, which
is consistent with an entry vehicle having less lift. Gamma is proportional to 1 / velocity

For the skipping entry, as velocity decreases, the flight path angle becomes more shallow. Gamma is proportional to velocity squared.

For the equilibrium glide acceleration, the vehicle speeds up initially, and then decreases as velocity decreases. Again,
acceleration is proportional to velocity squared. The magnitude of acceleration is dependent on L/D, as the ratio increases,
the magnitude of acceleration decreases.

For the skipping entry, as velocity decreases, the acceleration follows a sinusoidal curve, due to the dependence of the 
density on cos(gamma).


"""

print(f"PART B")

peak_n_skipping = skipping_entry_peak_acceleration(v_atm, gamma_atm, l_d_ratio, g_0, H_m)
peak_n_eq_glide = equilibrium_glide_peak_acceleration(l_d_ratio)
print(f"Peak deceleration skipping: {peak_n_skipping:.3f} (g's)")
print(f"Peak deceleration eq glide: {peak_n_eq_glide:.3f} (g's)")

plt.show()

"""
Peak deceleration skipping: 5.418 (g's)
Peak deceleration eq glide: -1.118 (g's)

The equilibrium glide experiences a much lower magnitude of deceleration when compared to the skipping trajectory.
"""