import numpy as np
import scipy as sp

gamma_air = 1.4  # gas constant air
shock_angle = np.radians(90)
R_universal = 8.3145e3 # Newton * meter / Kelvin * kilomol
molar_mass_air = 28.97  # kg / kmol


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
m_1_a = cpg_velocity_to_mach(u_1, T_1, gamma_air, molar_mass_air)
print(f"Upstream Mach number: {m_1_a}")

pressure_ratio_a = cpg_pressure_ratio(gamma_air, m_1_a, shock_angle)
print(f"Pressure ratio across normal shock: {pressure_ratio_a}")

density_ratio_a = cpg_density_ratio(gamma_air, m_1_a, shock_angle)
print(f"Density ratio across normal shock: {density_ratio_a}")

temperature_ratio_a = cpg_temperature_ratio(gamma_air, m_1_a, shock_angle)
print(f"Temperature ratio across normal shock: {temperature_ratio_a}")

