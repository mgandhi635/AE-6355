import numpy as np
import scipy as sp
# import jax.numpy as jnp
import scipy.optimize

gamma_air = 1.4  # gas constant air
shock_angle = np.radians(90)
R_universal = 8.3145e3 # Newton * meter / Kelvin * kilomol
molar_mass_air = 28.97  # kg / kmol
molar_mass_N2 = 14.01  # kg / kmol


def cpg_pressure_ratio(gamma, m1, shock_angle):  # radians
    return 1 + 2*gamma / (gamma + 1) * (m1 ** 2 * np.sin(shock_angle) ** 2 - 1)  # p2 / p1


def cpg_density_ratio(gamma, m1, shock_angle):
    return ((gamma + 1) * m1**2 * np.sin(shock_angle)**2) / ((gamma - 1) * m1**2 * np.sin(shock_angle)**2 + 2)  # rho2 / rho1


def cpg_velocity_to_mach(vel, temp, gamma, gas_molar_mass):  # m/s, kelvin, -, kg / kmol
    return vel / np.sqrt(gamma * R_universal / gas_molar_mass * temp)


def cpg_temperature_ratio(gamma, m1, shock_angle):
    return cpg_pressure_ratio(gamma, m1, shock_angle) / cpg_density_ratio(gamma, m1, shock_angle)


# Problem #1
# Find pressure, density, and temperature ratio

h_peak_q = 75  # kilometers
u_1 = 10.72*1000  # m/s
T_1 = 208.4  # Kelvin
rho_1 = 3.99109e-5  # kg / m^3
p_1 = 2.3875  # N / m^2

# part a:
print("\nPART A:")
m_1_a = cpg_velocity_to_mach(u_1, T_1, gamma_air, molar_mass_air)
print(f"Upstream Mach number: {m_1_a}")

pressure_ratio_a = cpg_pressure_ratio(gamma_air, m_1_a, shock_angle)
print(f"Pressure ratio across normal shock: {pressure_ratio_a}")

density_ratio_a = cpg_density_ratio(gamma_air, m_1_a, shock_angle)
print(f"Density ratio across normal shock: {density_ratio_a}")

temperature_ratio_a = cpg_temperature_ratio(gamma_air, m_1_a, shock_angle)
print(f"Temperature ratio across normal shock: {temperature_ratio_a}")
print(f"Temperature downstream of normal shock {temperature_ratio_a * T_1}")
# part b
"""
Assume that air is approximated as pure diatomic nitrogen in local thermodynamic equilibrium
Use the following function for enthalpy
"""

cp_N2_200K = 1.039  # specific heat of nitrogen gas at 200 Kelvin in kJ / (kg K)
R_N2 = R_universal / molar_mass_N2


def enthalpy_N2(temp):  # Kelvin
    temp_ratio = 3390. / temp
    return (7. / 2. + (temp_ratio) / (np.exp(temp_ratio) - 1)) * R_N2 * temp

# Use the conservation of energy h + KE = constant
h_1 = cp_N2_200K * T_1
kinetic_1 = u_1 ** 2 / 2

density_ratio_b_est = 0.1
print("\nPART B:")

# Iterative procedure
# p_2_est = p_1 + rho_1 * u_1 ** 2 * (1 - density_ratio_b_est)
# h_2_est = h_1 + u_1 ** 2 / 2 * (1 - density_ratio_b_est ** 2)
#
# print(f"Pressure downstream of normal shock {p_2_est}")
# print(f"Enthalpy downstream of normal shock {h_2_est}")

# # Guess a temperature
# T_2_est = (T_1 * temperature_ratio_a)  #+ -34141.763
# h_2_check = enthalpy_N2(T_2_est)
# print(f"Temperature downstream of normal shock {T_2_est}")
# print(f"Enthalpy check downstream of normal shock {h_2_check}")
# print(f"Enthalpy difference: {h_2_check - h_2_est:e}")

def outer_loop(e0, p_1, rho_1, h_1, u_1, T_1):
    p_2_est = p_1 + rho_1 * u_1 ** 2 * (1 - e0)
    h_2_est = h_1 + u_1 ** 2 / 2 * (1 - e0 ** 2)
    T0 = T_1 + 5000  # Random guess from T_1

    def inner_loop(T):
        return h_2_est - enthalpy_N2(T)
    T_converged = sp.optimize.newton(inner_loop, T0)

    rho_2 = p_2_est / (R_N2 * T_converged)
    eps_check = rho_1 / rho_2
    return e0 - eps_check

density_ratio_b = sp.optimize.newton(outer_loop, density_ratio_b_est, args=(p_1, rho_1, h_1, u_1, T_1))

p_2_b = p_1 + rho_1 * u_1 ** 2 * (1 - density_ratio_b)
h_2_b = h_1 + u_1 ** 2 / 2 * (1 - density_ratio_b ** 2)
rho_2_b = rho_1 / density_ratio_b
T_2_b = p_2_b / (rho_2_b * R_N2)
print(f"Pressure downstream of normal shock {p_2_b}")
print(f"Enthalpy downstream of normal shock {h_2_b}")
print(f"Temperature downstream of normal shock {T_2_b}")


pressure_ratio_b = p_2_b / p_1
density_ratio_b = rho_2_b / rho_1
temperature_ratio_b = T_2_b / T_1
print(f"Pressure ratio across normal shock: {pressure_ratio_b}")

print(f"Density ratio across normal shock: {density_ratio_b}")

print(f"Temperature ratio across normal shock: {temperature_ratio_b}")

# Guess a temperature
# T_2_est = (T_1 * temperature_ratio_a)  #+ -34141.763
# h_2_check = enthalpy_N2(T_2_est)
# print(f"Temperature downstream of normal shock {T_2_est}")
# print(f"Enthalpy check downstream of normal shock {h_2_check}")
# print(f"Enthalpy difference: {h_2_check - h_2_est:e}")