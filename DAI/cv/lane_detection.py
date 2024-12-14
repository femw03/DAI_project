import numpy as np


def expected_deviation(distance, angle, correction_factor, boost_factor) -> float:
    """
    Given the angle that the car wants to drive calculated where the car will be from the center at a given distance
    """
    if abs(angle * correction_factor) < 1e-4:
        return 0  # Car is driving straight
    r = 1 / (angle * correction_factor)
    if distance > abs(r):
        return np.nan
    if r > 0:
        return boost_factor * (r - np.sqrt(r**2 - distance**2))
    else:
        return boost_factor * (r + np.sqrt(r**2 - distance**2))
