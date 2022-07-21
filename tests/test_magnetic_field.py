import numpy as np
import pytest
from magneto_pyelastica.magnetic_field import (
    BaseMagneticField,
    ConstantMagneticField,
    SingleModeOscillatingMagneticField,
)
from elastica.utils import Tolerance


@pytest.mark.parametrize("time", [0.0, 1.0, 2.0, 4.0, 8.0])
def test_base_magnetic_field(time):
    magnetic_field_object = BaseMagneticField()
    magnetic_field_value = magnetic_field_object.value(time=time)
    # base class does nothing!
    assert magnetic_field_value == None


@pytest.mark.parametrize("time", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("ramp_interval", [1.0, 2.0])
@pytest.mark.parametrize("start_time", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("end_time", [4.0, 8.0])
def test_constant_magnetic_field(time, ramp_interval, start_time, end_time):
    dim = 3
    magnetic_field_amplitude = np.random.rand(dim)
    magnetic_field_object = ConstantMagneticField(
        magnetic_field_amplitude=magnetic_field_amplitude,
        ramp_interval=ramp_interval,
        start_time=start_time,
        end_time=end_time,
    )
    magnetic_field_value = magnetic_field_object.value(time=time)
    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)
    correct_magnetic_field_value = correct_factor * magnetic_field_amplitude

    np.testing.assert_allclose(
        magnetic_field_value, correct_magnetic_field_value, atol=Tolerance.atol()
    )


@pytest.mark.parametrize("time", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("ramp_interval", [1.0, 2.0])
@pytest.mark.parametrize("start_time", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("end_time", [4.0, 8.0])
def test_single_mode_oscillating_magnetic_field(
    time, ramp_interval, start_time, end_time
):
    dim = 3
    magnetic_field_amplitude = np.random.rand(dim)
    magnetic_field_angular_frequency = np.random.rand(dim)
    magnetic_field_phase_difference = np.random.rand(dim)
    magnetic_field_object = SingleModeOscillatingMagneticField(
        magnetic_field_amplitude=magnetic_field_amplitude,
        magnetic_field_angular_frequency=magnetic_field_angular_frequency,
        magnetic_field_phase_difference=magnetic_field_phase_difference,
        ramp_interval=ramp_interval,
        start_time=start_time,
        end_time=end_time,
    )
    magnetic_field_value = magnetic_field_object.value(time=time)
    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)

    correct_magnetic_field_value = (
        correct_factor
        * magnetic_field_amplitude
        * np.sin(
            magnetic_field_angular_frequency * time + magnetic_field_phase_difference
        )
    )

    np.testing.assert_allclose(
        magnetic_field_value, correct_magnetic_field_value, atol=Tolerance.atol()
    )
