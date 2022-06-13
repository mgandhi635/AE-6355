import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True

# # It's also possible to use the reduced notation by directly setting font.family:
# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "Helvetica"
# })

# Compute cpmax(1-ep)
# Find ep of a general blunted cone at an angle of attack, and then get Cl and Cd

# ep is from 2c
epsilon = 0.05886332692371807

# Plot the curves of Cp, Cl, and Cd from alpha = 0 - 20 degrees
alpha_range_deg = np.linspace(0.01,20,100)
alpha_range_rad = np.radians(alpha_range_deg)

# compute cp_max with modified Newtonian
cp_max = 2 - epsilon


# Pick a point on the body, use omega = 90 degrees
def cos_eta(alpha, dc, omega=np.pi/2):
    return np.cos(alpha)*np.sin(dc) - np.sin(alpha)*np.sin(omega)*np.cos(dc)


# Function for cp given a ratio and cone angle, and angle of attack
cpmax = 2


def cp_vec(xi, delta, alpha):
    CA = -1 / 4 * (2 * cpmax * (xi**2 * np.cos(delta)**2 - 1) * (np.sin(alpha)**2 * np.cos(delta)**2 +
         2 * np.cos(alpha)**2 * np.sin(delta)**2) + xi**2 * (epsilon - cpmax) *
         (2 * np.cos(alpha)**2 * (np.sin(delta)**4-1) - np.sin(alpha)**2 * np.cos(delta)**4))

    CN = (1/4) * np.sin(2*alpha) * np.cos(delta)**2 * (xi**2 * (epsilon - cpmax) * np.cos(delta)**2 -
        cpmax * (xi**2 * np.cos(2*delta) + xi**2 - cpmax))


    CP = np.sqrt(CA**2 + CN**2)
    CL = CN*np.cos(alpha) - CA*np.sin(alpha)
    CD = CN*np.sin(alpha) + CA*np.cos(alpha)
    return CP, CL, CD


def cp_vec_class(xi, delta, alpha):
    CA = (cpmax - epsilon) * ((1/2)*(1-np.sin(delta)**4)*xi**2 +
         ((np.sin(delta)**2 * np.cos(alpha)**2 + (1/2)*np.cos(delta)**2 * np.sin(alpha)**2) * (1 - xi**2*np.cos(delta)**2)))

    CN = (cpmax - epsilon) * (1 - xi**2 * np.cos(delta)**2) * (np.cos(delta)**2 * np.sin(alpha) * np.cos(alpha))

    CP = (cpmax - epsilon)*(np.cos(alpha)*np.sin(delta) - np.sin(alpha)*np.sin(np.pi/2) * (np.cos(delta)))**2
    CL = CN*np.cos(alpha) - CA*np.sin(alpha)
    CD = CN*np.sin(alpha) + CA*np.cos(alpha)
    return CP, CL, CD


def cp_point_class(rn, rc, delta, alpha, omega=np.pi/2):
    CP = (cpmax - epsilon)*(np.cos(alpha)*np.sin(delta) - np.sin(alpha)*np.sin(omega) * (np.cos(delta)))**2

    CN = CP*np.sin(omega)*(-1)*np.tan(delta) * rn / (np.pi * rc**2)
    CA = CP * np.tan(delta)**2 * rn / (np.pi * rc**2)

    CL = CN*np.cos(alpha) - CA*np.sin(alpha)
    CD = CN*np.sin(alpha) + CA*np.cos(alpha)
    return CP, CL, CD


def cp_estimate(xi, delta, alpha):

    CL = 2 * alpha * (1-xi**2)*(1 - delta**2)

    CD = xi**2 + (2*delta**2 + 3*alpha**2)*(1-xi**2)

    CA = CD - CL/alpha

    CP = np.sqrt(CL**2 + CA**2)

    return CP, CL, CD


# Case a
# dc = np.radians(70)
# xi = 0.6625 / (2.65 / 2)
# rn_a = 0.6625
# rc_a = 2.65 / 2
# cp_a_class, cl_a_class, cd_a_class = cp_vec_class(xi, dc, alpha_range_rad)
# cp_a_self, cl_a_self, cd_a_self = cp_vec(xi, dc, alpha_range_rad)
# cp_point_a, cl_point_a, cd_point_a = cp_point_class(rn_a, rc_a, dc, alpha_range_rad)


# plt.figure(1)
# plt.plot(alpha_range_deg, cp_a_class, label='cp_class')
# plt.plot(alpha_range_deg, cl_a_class, label='cl_class')
# plt.plot(alpha_range_deg, cd_a_class, label='cd_class')

# plt.plot(alpha_range_deg, cp_a_self, label='cp_self')
# plt.plot(alpha_range_deg, cl_a_self, label='cl_self')
# plt.plot(alpha_range_deg, cd_a_self, label='cd_self')

# plt.plot(alpha_range_deg, cp_point_a, label='cp_point')
# plt.plot(alpha_range_deg, cl_point_a, label='cl_point')
# plt.plot(alpha_range_deg, cd_point_a, label='cd_point')

# plt.legend()

# Data for cases
data = {"A": (70, 2.65, 0.6635),
         "B": (60, 0.827, 0.23),
         "C": (45, 0.35, 0.0875),
         "D": (20, 10, 1)}
         # "E": (20, 9, 2.5)}
        # "E1": (15, 10, 1),
        # "E2": (30, 0.35, 0.0875),
        # "E3": (15, 9, 1),
        # "E0": (15, 2.65, 0.6635)}

plt.figure(1)
plt.figure(2)
plt.ylim([-.50, 0.50])
plt.figure(3)

titles = [r"$\alpha$ vs $C_p$ at $\omega=90^{\circ}$", r"$\alpha$ vs $C_l$", r"$\alpha$ vs $C_d$"]

for case, value in data.items():
    cone_angle = np.radians(value[0])
    rc = value[1] / 2
    rn = value[2]
    xi = rn / rc
    cp, cl, cd = cp_vec_class(xi, cone_angle, alpha_range_rad)

    label = case + r": $\delta_c$ = " + str(value[0]) + r"$^{\circ}$,  $\xi$ = " + f"{xi:.2f}"

    plt.figure(1)
    plt.plot(alpha_range_deg, cp, label=label,linewidth=3, alpha=0.5)

    plt.figure(2)
    plt.plot(alpha_range_deg, cl, label=label,linewidth=3, alpha=0.5)

    plt.figure(3)
    plt.plot(alpha_range_deg, cd, label=label,linewidth=3, alpha=0.5)

for i in plt.get_fignums():
    plt.figure(i)
    plt.legend(bbox_to_anchor=(0.75, 1.0), loc=2, borderaxespad=-4)
    plt.grid()
    plt.xlabel(r"Angle of Attack ($^{\circ}$)")
    plt.ylabel("Coefficient")
    plt.title(titles[i-1])
plt.show()


"""
e) We assumed a modified Newtonian flow, which corrects for Cpmax using the density ratio epsilon. Newtonian flow is 
valid when the wave angle of the shock is close to the cone angle of the vehicle, such the flow is traversing along
the cone. As the angle of attack increases, the ratio of the cone angle and the wave angle should move away from 1,
making estimates less accurate. ince we neglected skin friction and parasite drag, the accuracy of the solution 
should decrease as it becomes closer to a flat plate. This means as the cone angle and bluntness parameter increase,
the solution becomes less accurate.

In terms of the order of magnitude, at large cone angles, the bluntness parameter does not matter as much for Cd. If the 
cone angle becomes smaller, the bluntess parameter impacts Cd much heavier. For example at a 20 degree cone angle, 
increasing the bluntness parameter from 0.2 - 0.56 increase Cd from .281 to .489 at 5 degrees alpha. 


"""
