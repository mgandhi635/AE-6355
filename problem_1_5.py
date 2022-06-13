import numpy as np
import scipy as sp
import scipy.optimize
from high_temp_shock_curve_fit import (GammaTildeCurveFit, HighTempCurveFit,
                                       compute_curve_fit_enthaply, compute_number_density)

R_universal = 8.3145e3 # Newton * meter / Kelvin * kilomol
molar_mass_air = 28.97  # kg / kmol
cp_air_200K = 1.0025 # specific heat of air at 200 Kelvin in kJ / (kg K)
R_air = R_universal / molar_mass_air

alt = 77.72  # km
u_1 = 9.75*1000  # m/s
p_1 = 1.556  # N/ m^2
rho_1 = 2.862e-5  # kg / m^3
T_1 = 189  # Kelvin
h_1 = cp_air_200K * T_1


def outer_loop_curve_fit(epsilon_c):
    # Given the estimated density ratio, compute the estimated density downstream of the shock
    rho_2_c_est = rho_1 / epsilon_c

    # Compute the estimated pressure downstream of the shock
    p_2_c_est = p_1 + rho_1 * u_1 ** 2 * (1 - epsilon_c)

    # Compute the estimated enthalpy downstream of the shock
    h_2_c_est = h_1 + u_1 ** 2 / 2 * (1 - epsilon_c ** 2)

    # Perform the curve fits on gamma tilde
    gamma_tilde_est = gamma_curve.fit(p_2_c_est, rho_2_c_est)

    # Compute the enthalpy from the curve fit
    h_2_c_check = compute_curve_fit_enthaply(p_2_c_est, rho_2_c_est, gamma_tilde_est[0])

    return h_2_c_check - h_2_c_est


# Part a Compute the conditions behind a normal shock
# initialize the curve fits
gamma_curve = GammaTildeCurveFit()
temp_curve = HighTempCurveFit()

epsilon_est = 0.1
epsilon = sp.optimize.newton(outer_loop_curve_fit, epsilon_est)

print(epsilon)

# Compute properties
u_2 = u_1 * epsilon
p_2 = p_1 + rho_1 * u_1 ** 2 * (1 - epsilon)
h_2 = h_1 + u_1 ** 2 / 2 * (1 - epsilon ** 2)
rho_2 = rho_1 / epsilon
T_2 = temp_curve.fit(p_2, rho_2)[0]

print("\nPART A:")
print(f"Temperature downstream of normal shock {T_2}")
print(f"Pressure downstream of normal shock {p_2}")
print(f"Density downstream of normal shock {rho_2:e}")
print(f"Velocity downstream of normal shock {u_2}")
print(f"Enthalpy downstream of normal shock {h_2}")
