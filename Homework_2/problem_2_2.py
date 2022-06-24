import numpy as np
import scipy as sp
from util import (sphere_volume, compute_ballistic_coeff, compute_constant_C,
                  allen_eggers_acceleration, allen_eggers_velocity)
# Meteorite parameters
rho_sphere = 3000  # kg/m^3
r_sphere = 5  # meters
gamma_atm = -np.radians(45)  #
v_atm = 15*1000  # m/s
rho_0 = 1.225  # kg / m^3
H = 7.2 # scale height km
H_m = H * 1000
g_0 = 9.806  # m / s^2


m_sphere = rho_sphere * sphere_volume(r_sphere)

Cd_sphere = 1
Beta_sphere = compute_ballistic_coeff(m_sphere, Cd_sphere, np.pi * r_sphere**2)
# print(Beta_sphere)
# Compute the constant C
# print(C_sphere)

C_sphere = compute_constant_C(rho_0, H, Beta_sphere, gamma_atm)
h_n_max = H_m * np.log(-2 * C_sphere)
# print(h_n_max)
# print(allen_eggers_acceleration(h_n_max, v_atm, gamma_atm, H, rho_0, Beta_sphere))

h_n_max = np.max([0.0, h_n_max])
print(f"Height of max acceleration: {h_n_max:.3f} km")
n_max = allen_eggers_acceleration(h_n_max, v_atm, gamma_atm, H, rho_0, Beta_sphere) / g_0
print(f"Magnitude of max acceleration: {n_max:.3f} g's")
v_n_max = allen_eggers_velocity(h_n_max, rho_0, H, Beta_sphere, gamma_atm, v_atm)
print(f"Velocity at max acceleration: {v_n_max/1000:.3f} km/s")
