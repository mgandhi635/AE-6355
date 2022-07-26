import numpy as np
from Project.src.project_vars import EntryVehicleParams


def compute_gravity_from_altitude(altitude: float, params: EntryVehicleParams):
    """Gravity computation"""
    return params.gravity_planet * \
           (params.radius_planet ** 2) / (params.radius_planet + altitude) ** 2


def compute_density_from_altitude(altitude: float, params: EntryVehicleParams):
    """
    Altitude and scale height are in meters
    :param altitude:
    :param params:
    :return:
    """
    if altitude < params.atmosphere_altitude_planet:
        return params.density_planet * np.exp(-altitude / params.scale_height)
    else:
        return 0.0


# STANDARD ATMOSPHERE IMPLEMENTATION


# Constants
b = 3.31e-7
g0_prime = 9.806  # m^2/(s^2 m')
R_universal = 8.31432e3 # (N m) / (kmol K)
mw_0 = 28.9644  # kg / kmol
R_air = R_universal / mw_0
radius_earth = 6378*1000  # m
P_0 = 101325 # N/m^2


def compute_geopotential_altitude(r0: float, z: float):
    """
    Convert geometric altitude (meters) to geopotential altitude (meters')
    :param r0: planet radius (meters)
    :param z: geometric altitude (meters)
    :return: geopotential altitude (meters')
    """
    return (r0 * z) / (r0 + z)


def compute_molecular_scale_temp(Tm_i: float, Lh_i: float, h_i: float, h: float):
    """
    Compute the molecular scale temperature for geopotential altitude in layer i
    :param Tm_i: Base molecular scale temp (K)
    :param Lh_i: Thermal lapse rate (K/m')
    :param h_i: Layer geopotential altitude (m')
    :param h: Geopotential altitude (m')
    :return: Molecular scale temp
    """
    return Tm_i + Lh_i * (h - h_i)

# Added extra layer to blend 1962 and 1976 model
def get_std_atmosphere_1976_MST_layers():
    h_i = 1000 * np.array([0, 11, 20, 32, 47, 51, 71, 86, 90])
    Lh_i = np.array([-6.5, 0, 1, 2.8, 0, -2.8, -2, -1]) / 1000
    tmp = (h_i[1:] - h_i[:-1]) * Lh_i
    tmp = np.hstack((0, tmp))
    Tm_i = 288.15+np.cumsum(tmp)
    return {"h_i": h_i, "Lh_i": Lh_i, "Tm_i": Tm_i}


def get_std_atmosphere_1976_MST_layers_above_90():
    z_i = 1000 * np.array([90, 100, 110, 120, 150])
    Lz_i = np.array([3, 5, 10, 20]) / 1000
    tmp = (z_i[1:] - z_i[:-1]) * Lz_i
    tmp = np.hstack((0, tmp))
    Tm_i = 180.65+np.cumsum(tmp)
    return {"z_i": z_i, "Lz_i": Lz_i, "Tm_i": Tm_i}


std_atmosphere_1976_MST_layers = get_std_atmosphere_1976_MST_layers()
std_atmosphere_1976_MST_layers_above_90 = get_std_atmosphere_1976_MST_layers_above_90()

h_layers = std_atmosphere_1976_MST_layers["h_i"]
Lh_layers = std_atmosphere_1976_MST_layers["Lh_i"]
Tm_layers = std_atmosphere_1976_MST_layers["Tm_i"]

z_layers = std_atmosphere_1976_MST_layers_above_90["z_i"]
Lz_layers = std_atmosphere_1976_MST_layers_above_90["Lz_i"]
Tm_layers_above_86 = std_atmosphere_1976_MST_layers_above_90["Tm_i"]


def get_geopotential_altitude_index(h: float):
    """
    Return the index of base layer for a given geopotential altitude
    :param h: geopotential altitude between 0 m' and 90 m'
    :return:
    """
    assert np.logical_and(h >= h_layers[0], h <= h_layers[-1])
    return np.searchsorted(h_layers, h, 'left')-1


def get_geometric_altitude_index(z: float):
    """
    Return the index of base layer for a given geopotential altitude
    :param z: geometric altitude between 86 km and 150 km
    :return:
    """
    assert np.logical_and(z >= z_layers[0], z <= z_layers[-1])
    return np.searchsorted(z_layers, z, 'left')-1


def get_geopotential_MST_parameters(h: float, P_layers=None):
    """
    Compute the molecular scale temp and return the parameters
    :param h:
    :return:
    """
    idx = get_geopotential_altitude_index(h)
    h_i = h_layers[idx]
    Lh_i = Lh_layers[idx]
    Tm_i = Tm_layers[idx]
    Tm = compute_molecular_scale_temp(Tm_i, Lh_i, h_i, h)
    if P_layers is None:
        return Tm, h_i, Lh_i, Tm_i
    else:
        return Tm, h_i, Lh_i, Tm_i, P_layers[idx]



def get_geometric_MST_parameters(z: float, P_layers=None, rho_layers=None):
    """
    Compute the molecular scale temp and return the parameters
    :param z:
    :return:
    """
    idx = get_geometric_altitude_index(z)
    z_i = z_layers[idx]
    Lz_i = Lz_layers[idx]
    Tm_i = Tm_layers_above_86[idx]
    Tm = compute_molecular_scale_temp(Tm_i, Lz_i, z_i, z)
    if (P_layers is None) and (rho_layers is None):
        return Tm, z_i, Lz_i, Tm_i
    else:
        return Tm, z_i, Lz_i, Tm_i, P_layers[idx], rho_layers[idx]


def compute_pressure_density_below_90_km(h: float, h_i: float, Tm_i:float, Lh_i: float, P_i:float,
                                         R_universal: float=R_universal, mw_0: float=mw_0, g0_prime: float=g0_prime):
    """
    Compute the pressure from mo
    :param h:
    :param h_i:
    :param Tm_i:
    :param Lh_i:
    :param P_i:
    :param R_universal:
    :param mw_0:
    :param g0_prime:
    :return:
    """
    Tm = compute_molecular_scale_temp(Tm_i, Lh_i, h_i, h)
    if abs(Lh_i) > 0:
        power = (g0_prime * mw_0) / (R_universal * Lh_i)
        tmp = (Tm_i / Tm)**power
    else:
        tmp = np.exp((-g0_prime * mw_0 * (h - h_i)) / (R_universal * Tm_i))
    P = P_i*tmp
    rho = P * mw_0 / (R_universal * Tm)
    return P, rho


def compute_pressure_density_above_90_km(z: float, z_i: float, Tm_i:float, Lz_i: float, P_i: float, rho_i: float,
                                         R_specific: float=R_air, g0: float=g0_prime, b: float=b):
    """
    Compute the pressure from mo
    :param z:
    :param z_i:
    :param Tm_i:
    :param Lz_i:
    :param P_i:
    :param R_universal:
    :param mw_0:
    :param g0_prime:
    :return:
    """
    pow_pressure = -(g0 / (R_specific * Lz_i))*(1 + b * (Tm_i / Lz_i - z_i))
    pow_density = -(g0 / (R_specific * Lz_i))*(R_specific * Lz_i / g0 + 1 + b * (Tm_i / Lz_i - z_i))
    Tm = compute_molecular_scale_temp(Tm_i, Lz_i, z_i, z)
    Tm_ratio = Tm / Tm_i
    tmp = np.exp((g0 * b)/(R_specific * Lz_i)*(z - z_i))

    P = P_i * Tm_ratio**pow_pressure * tmp
    rho = rho_i * Tm_ratio**pow_density * tmp
    return P, rho


# Function to compute the base pressure at each layer
def get_base_pressures_below_90km():
    P_layers = [P_0]
    rho_layers = []
    for h in h_layers[1:]:
        P_i = P_layers[-1]
        Tm, h_i, Lh_i, Tm_i = get_geopotential_MST_parameters(h)
        P_new, rho_new = compute_pressure_density_below_90_km(h, h_i, Tm_i, Lh_i, P_i)
        P_layers.append(P_new)
        rho_layers.append(rho_new)
    return np.array(P_layers), rho_new


def get_base_pressures_and_densities_above_90km(P_0, rho_0):
    P_layers = [P_0]
    rho_layers = [rho_0]
    for z in z_layers[1:]:
        P_i = P_layers[-1]
        rho_i = rho_layers[-1]
        Tm, z_i, Lz_i, Tm_i = get_geometric_MST_parameters(z)
        P_new, rho_new = compute_pressure_density_above_90_km(z, z_i, Tm_i, Lz_i, P_i, rho_i)
        P_layers.append(P_new)
        rho_layers.append(rho_new)
    return np.array(P_layers), np.array(rho_layers)

P_layers, rho_0 = get_base_pressures_below_90km()
P_layers_above_90km, rho_layers_above_90km = get_base_pressures_and_densities_above_90km(P_layers[-1], rho_0)


def compute_density_std_atmosphere(z: float):
    # Check if above or below 90 km
    if 90000 <= z < 15000:
        # Get the parameters for the layer
        Tm, z_i, Lz_i, Tm_i, P_i, rho_i = get_geometric_MST_parameters(z, P_layers_above_90km, rho_layers_above_90km)
        P, rho = compute_pressure_density_above_90_km(z, z_i, Tm_i, Lz_i, P_i, rho_i)
    elif 0 <= z < 90000:
        # Get the geopotential altitude
        h = compute_geopotential_altitude(radius_earth, z)
        # Get the parameters for the layer
        Tm, h_i, Lh_i, Tm_i, P_i = get_geopotential_MST_parameters(h, P_layers)
        P, rho = compute_pressure_density_below_90_km(z, h_i, Tm_i, Lh_i, P_i)
    else:
        rho = 0
    return rho

"""
86-90 km Tm_i = 180.65, Lh_i = 0
90-150
"""