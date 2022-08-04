__doc__ = """ Module implementation for external magnetic fields for magnetic Cosserat rods."""
__all__ = [
    "BaseMagneticField",
    "ConstantMagneticField",
    "SingleModeOscillatingMagneticField",
]

from magneto_pyelastica.utils import compute_ramp_factor
import numpy as np


class BaseMagneticField:
    """
    This is the base class for external magnetic field objects.

    Notes
    -----
    Every new magnetic field class must be derived
    from BaseMagneticField class.

    """

    def __init__(self):
        """
        BaseMagneticField class does not need any input parameters.
        """

    def value(self, time: np.float64 = 0.0):
        """Returns the value of the magnetic field vector.

        In BaseMagneticField class, this routine simply passes.

        Parameters
        ----------
        time : float
            The time of simulation.

        Returns
        -------

        """


class ConstantMagneticField(BaseMagneticField):
    """
    This class represents a magnetic field constant in time.

        Attributes
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the constant magnetic field.
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

    """

    def __init__(self, magnetic_field_amplitude, ramp_interval, start_time, end_time):
        """

        Parameters
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the constant magnetic field.
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

        """
        super(ConstantMagneticField, self).__init__()
        self.magnetic_field_amplitude = magnetic_field_amplitude
        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

    def value(self, time: np.float64 = 0.0):
        """
        This function returns the value of the magnetic field vector based on the
        magnetic_field_amplitude.

        Parameters
        ----------
        time : float
            The time of simulation.

        Returns
        -------
        magnetic_field: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Value of the constant magnetic field.
        Notes
        -------
        Assumes only time dependence.

        """
        factor = compute_ramp_factor(
            time=time,
            ramp_interval=self.ramp_interval,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        return self.magnetic_field_amplitude * factor


class SingleModeOscillatingMagneticField(BaseMagneticField):
    """
    This class represents a magnetic field oscillating sinusoidally in time
    with one mode.

        Attributes
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the oscillating magnetic field.
        magnetic_field_angular_frequency: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Angular frequency of the oscillating magnetic field.
        magnetic_field_phase_difference: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Phase difference of the oscillating magnetic field.
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

    """

    def __init__(
        self,
        magnetic_field_amplitude,
        magnetic_field_angular_frequency,
        magnetic_field_phase_difference,
        ramp_interval,
        start_time,
        end_time,
    ):
        """

        Parameters
        ----------
        magnetic_field_amplitude: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Amplitude of the oscillating magnetic field.
        magnetic_field_angular_frequency: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Angular frequency of the oscillating magnetic field.
        magnetic_field_phase_difference: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Phase difference of the oscillating magnetic field.
        ramp_interval : float
            ramping time for magnetic field.
        start_time : float
            Turning on time of magnetic field.
        end_time : float
            Turning off time of magnetic field.

        """
        super(SingleModeOscillatingMagneticField, self).__init__()
        self.magnetic_field_amplitude = magnetic_field_amplitude
        self.magnetic_field_angular_frequency = magnetic_field_angular_frequency
        self.magnetic_field_phase_difference = magnetic_field_phase_difference
        self.ramp_interval = ramp_interval
        self.start_time = start_time
        self.end_time = end_time

    def value(self, time: np.float64 = 0.0):
        """
        This function returns the value of the sinusoidally oscillating magnetic field
        vector, based on amplitude, frequency and phase difference.

        Parameters
        ----------
        time : float
            The time of simulation.

        Returns
        -------
        magnetic_field: numpy.ndarray
            1D (dim,) array containing data with 'float' type.
            Value of the oscillatory magnetic field.
        Notes
        -------
        Assumes only time dependence.

        """
        factor = compute_ramp_factor(
            time=time,
            ramp_interval=self.ramp_interval,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        return (
            factor
            * self.magnetic_field_amplitude
            * np.sin(
                self.magnetic_field_angular_frequency * time
                + self.magnetic_field_phase_difference
            )
        )
