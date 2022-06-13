import numpy as np
import scipy as sp
import scipy.optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
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
print(f"Enthalpy upstream of normal shock {h_1:e}")


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

# print(epsilon)

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
print(f"Enthalpy downstream of normal shock {h_2:e}")

# Compute Cp = (2 - ep)

print(f"\nPART: B")
# Given Parameters
Theta_XY = np.radians(60)
ep_b = 2
delta = np.radians(73)
R = 0.1
tau = np.radians(60)

# Dependent Parameters

tan_xy = np.tan(Theta_XY)
tan_delta = np.tan(delta)
sin_delta = np.sin(delta)
cos_delta = np.cos(delta)

# Compute geometric properties
Theta_XZ = np.arcsin( np.sin(Theta_XY) / sin_delta )
ep = tan_xy / np.tan(Theta_XZ)
xR = sin_delta / (2 * tan_xy) * (1 - (tan_xy**2 / tan_delta**2))
xC = (sin_delta - cos_delta * tan_xy) / (2 * tan_xy)
x0 = xC * (1 + tan_xy**2 / ep_b**2)
b = np.sqrt(x0 * (xC * tan_xy**2))
a = b / ep_b
c = b / ep
xN = x0 - a
xR2 = xR + R / sin_delta * (np.cos(tau - Theta_XY - delta) - np.cos(Theta_XY + delta))
xB = (xR * tan_delta) / (tan_delta - tan_xy)
phi_90 = np.pi/2
phi_m90 = -np.pi/2


x0_check = b**2 / (xC * tan_xy**2)
b_check = xC * tan_xy * np.sqrt(1 / (ep_b**2) * tan_xy**2 + 1)
tan_theta = tan_xy / np.sqrt((np.sin(phi_m90)**2 + ep**2 * np.cos(phi_m90)**2))
Theta = np.arctan(tan_theta)


def compute_xM(phi):
    xM = xR * sin_delta / (np.sin(phi) * cos_delta * tan_theta + sin_delta)
    return xM

# Double check values for sanity
assert np.isclose(x0,x0_check)
assert np.isclose(b,b_check)
assert np.isclose(0, cos_delta**2 - np.cos(Theta_XY)**2 + np.sin(Theta_XY)**2 / np.tan(Theta_XZ)**2)
assert np.isclose(xB, compute_xM(phi_m90))  # End of cone before skirt, bottom
assert np.isclose(xC, compute_xM(phi_90))  # End of cone before skirt, top


# Segment 1 & 2: Elliptical nose top and bottom
def y_nose(x, phi, x0=x0, a=a, b=b, c=c):
    t1 = ((x - x0) / a)**2
    t2 = np.sin(phi)**2 / b**2
    t3 = np.cos(phi)**2 / c**2

    r = np.sqrt((1 - t1) / (t2 + t3)) * np.sin(phi)
    return r


# Plot for sanity check
x_range_s1 = np.linspace(xN, xC)
y_s1 = y_nose(x_range_s1, phi_90)
y_s2 = y_nose(x_range_s1, phi_m90)
plt.figure()
plt.grid(which='both')
# plt.axis('equal')
plt.scatter(x_range_s1, y_s1)
plt.scatter(x_range_s1, y_s2)

# Segment 3: Elliptical Cone
x_range_s3 = np.linspace(xC, xB)


def y_cone(x, phi, ep=ep, Theta_XY=Theta_XY):
    tan_xy = np.tan(Theta_XY)
    tan_theta = tan_xy / np.sqrt((np.sin(phi) ** 2 + ep ** 2 * np.cos(phi) ** 2))
    return x * tan_theta * np.sin(phi)


y_s3 = y_cone(x_range_s3, phi_m90)
plt.scatter(x_range_s3, y_s3)


# Segments 4 and 5: Circular Arc Skirt - cyclindrical aft-body
def compute_xM2(x0S, r0S, xR2, phi, delta=delta):
    U = np.sin(delta) * (xR2 - x0S) - r0S*np.cos(delta)*np.sin(phi)
    V = np.sin(delta)**2 + np.cos(delta)**2 * np.sin(phi)**2
    W = np.sign(phi) * np.sqrt( (U * np.sin(delta))**2 - V * (U**2 - R**2 * np.sin(phi)**2 * np.cos(delta)**2))
    return x0S + (np.sin(delta) * U - W) / V


def y_skirt(x, phi, r0s, x0s, R):
    return (r0s + np.sqrt(R**2 - (x - x0s)**2))*np.sin(phi)


# Use the cylindrical aft body meaning the range ends at x0S

# Bottom Segment 4
xM_bot = compute_xM(phi_m90)
r0S_bot = xM_bot * tan_theta - R * np.cos(Theta)
x0S_bot = xM_bot + R * np.sin(Theta)
plt.scatter(x0S_bot, r0S_bot*np.sin(phi_m90),20,"k")

# Used for flat base
# xM2_bot = compute_xM2(x0S_bot, r0S_bot, xR2, phi_m90, delta)

x_range_s4 = np.linspace(xM_bot,x0S_bot)
y_s4 = y_skirt(x_range_s4, phi_m90, r0S_bot, x0S_bot, R)
plt.scatter(x_range_s4, y_s4)

# Top Segment 5
xM_top = compute_xM(phi_90)
r0S_top = xM_top * tan_theta - R * np.cos(Theta)
x0S_top = xM_top + R * np.sin(Theta)
plt.scatter(x0S_top, r0S_top*np.sin(phi_90),20,"k")

# Used for flat base
# xM2_top = compute_xM2(x0S_top, r0S_top, xR2, phi_90, delta)

x_range_s5 = np.linspace(xM_top,x0S_top)
y_s5 = y_skirt(x_range_s5, phi_90, r0S_top, x0S_top, R)
plt.scatter(x_range_s5, y_s5)

## Get an expression for the flat plate angle (beta) as a function of x for each segment
# Assume trim angle of attack is 4, from Ref 2, Figure 3, where Cm / Cm_ref ~ 0 for air
angle_of_attack = np.radians(4)

def get_alpha_beta(x,y, phi):
    beta = np.arctan2(y[1:] - y[0:-1], x[1:] - x[0:-1])
    if np.sign(phi) > 0:
        alpha = np.pi - beta
    else:
        alpha = -beta
    return beta, alpha-angle_of_attack, x[0:-1]


def compute_cp_cl_cd(cpmax, epsilon, alpha_l, alpha_u, x_chord, l_chord):
    cp = (cpmax - epsilon) * (np.sin(alpha_l)**2 + np.sin(alpha_u)**2)

    cl_top = np.sum(cp[:-1] * np.cos(alpha_u[:-1]) * (x_chord[1:] - x_chord[:-1]))
    cl_bot = np.sum(cp[:-1] * np.cos(alpha_l[:-1]) * (x_chord[1:] - x_chord[:-1]))
    cl = (cl_top + cl_bot) / l_chord

    cd_top = np.sum(cp[:-1] * np.sin(alpha_u[:-1]) * (x_chord[1:] - x_chord[:-1]))
    cd_bot = np.sum(cp[:-1] * np.sin(alpha_l[:-1]) * (x_chord[1:] - x_chord[:-1]))
    cd = (cd_top + cd_bot) / l_chord

    return cp, cl, cd


# Segment #1 top
beta_s1, alpha_s1, x1 = get_alpha_beta(x_range_s1, y_s1, phi_90)
# Segment #2 bottom
beta_s2, alpha_s2, x2 = get_alpha_beta(x_range_s1, y_s2, phi_m90)
# Segment #3 bottom
beta_s3, alpha_s3, x3 = get_alpha_beta(x_range_s3, y_s3, phi_m90)
# Segment #4 bottom
beta_s4, alpha_s4, x4 = get_alpha_beta(x_range_s4, y_s4, phi_m90)
# Segment #5 top
beta_s5, alpha_s5, x5 = get_alpha_beta(x_range_s5, y_s5, phi_90)

# Combine the top and bottom segments
x_l = np.hstack([x2, x3, x4])
alpha_l = np.hstack([alpha_s2, alpha_s3, alpha_s4])

x_u = np.hstack([x1, x5])
alpha_u = np.hstack([alpha_s1, alpha_s5])

# Currently we have a function for the top and the bottom. Let us use interpolate to get the top and bottom value at
# the same given x coordinate

f_alpha_l = interp1d(x_l, alpha_l, kind='linear')
f_alpha_u = interp1d(x_u, alpha_u, kind='linear', bounds_error=False, fill_value=alpha_u[-1])

x_chord = np.linspace(x_l[0],x_l[-1], 10000)
l_chord = x_chord[-1] - x_chord[0]
alpha_u_chord = f_alpha_u(x_chord)
alpha_l_chord = f_alpha_l(x_chord)

# plot
plt.figure()
plt.plot(x_chord, np.vstack([alpha_l_chord, alpha_u_chord]).T)
plt.plot(x_l, alpha_l)

# print(epsilon)
# Compute cp
cp_chord, cl_chord, cd_chord = compute_cp_cl_cd(2, epsilon, alpha_l_chord, alpha_u_chord, x_chord, l_chord)
plt.figure()
normalized_x = (x_chord-x_l[0]) / l_chord
plt.plot(x_chord, cp_chord)
plt.grid()
# plt.axis('equal')
plt.xlabel('X Chord Percentage')
plt.ylabel('Cp')

"""
Physically this distribution should a higher contribution of pressure from the bottom, thus being positive. Additionally
since the nose of the vehicle close to normal to the flow, it should experience the largest pressure, and we will see a
drop as the continue down the chord. The pressure should remain constant along the straight part of the cone, since the 
upper part of the body does not experience any flow (under the modified newtonian assumptions), and the pressure
decreases as the angle between the flow and normal vector decrease.
"""
print(f"PART: C")
"""
Cl = 1/c * cp(x)*cos(alpha(x))dx
Cd = 1/c * cp(x)*cos(alpha(x))dx

Here the chord length was used as c
"""
print(f"Cl: {cl_chord}")
print(f"Cd: {cd_chord}")

print(f"PART: D")


# Free molecular Cl and Cd = -0.025 and 1.8
# Use the bridging function!
def bridge_coefficient(c_continuum, c_free, kn):
    phi = (3/8 + 1/8 * np.log10(kn))*np.pi
    return c_continuum + (c_free - c_continuum)*np.sin(phi)**2


kn_range = np.logspace(-3, 1, 1000)

cl_cont = cl_chord
cl_free = -0.025

cd_cont = cd_chord
cd_free = 1.8

plt.figure()
plt.semilogx(kn_range, bridge_coefficient(cl_cont, cl_free, kn_range))
plt.grid()
plt.xlabel('Knudsen Number')
plt.ylabel('Cl')

plt.figure()
plt.semilogx(kn_range, bridge_coefficient(cd_cont, cd_free, kn_range))
plt.grid()
plt.xlabel('Knudsen Number')
plt.ylabel('Cd')
plt.show()
