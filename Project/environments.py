import numpy as np
from project_vars import EntryVehicleParams


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
