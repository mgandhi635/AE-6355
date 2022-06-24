import numpy as np
import scipy as sp
import scipy.optimize
from util import general_cross_range


def cross_range_root_find(ld_ratio, phi, rp, cross_range_des):
    return cross_range_des - general_cross_range(ld_ratio, phi, rp)


rE = 6378  # km
phi = np.radians(45)
l_desired = 1500  # km

sol = sp.optimize.newton(cross_range_root_find, 1.0, args=(phi, rE, l_desired))
print(sol)

alpha = np.arctan(1 / sol)


Cn = 2*np.sin(alpha)**2
Cl = Cn*np.cos(alpha)
Cd = Cn*np.sin(alpha)

print(general_cross_range(sol, phi, rE))

print(np.degrees(alpha))
print(Cn)
print(Cl)
print(Cd)

