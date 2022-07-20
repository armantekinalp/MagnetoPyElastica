""" Handy utilities"""
__all__ = ["compute_ramp_factor"]

from elastica.utils import Tolerance


def compute_ramp_factor(time, ramp_interval, start_time, end_time):
    """
    This function returns a linear ramping up factor based on time, ramp_interval,
    start_time and end_time.

    Parameters
    ----------
    time : float
        The time of simulation.
    ramp_interval : float
        ramping time for magnetic field.
    start_time : float
        Turning on time of magnetic field.
    end_time : float
        Turning off time of magnetic field.

    Returns
    -------
    factor : float
        Ramp up factor.

    """
    factor = (time > start_time) * (time <= end_time) * min(
        1.0, (time - start_time) / (ramp_interval + Tolerance.atol())
    ) + (time > end_time) * max(
        0.0, -1 / (ramp_interval + Tolerance.atol()) * (time - end_time) + 1.0
    )
    return factor
