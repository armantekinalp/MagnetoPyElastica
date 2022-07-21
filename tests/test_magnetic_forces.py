import numpy as np
import pytest
from magneto_pyelastica.magnetic_field import BaseMagneticField, ConstantMagneticField
from magneto_pyelastica.magnetic_forces import MagneticForces
from elastica.utils import Tolerance


def mock_magnetic_rod_init(self):
    self.n_elems = 0.0
    self.external_forces = 0.0
    self.external_torques = 0.0
    self.director_collection = 0.0
    self.volume = 0.0


MockMagneticRod = type(
    "MockMagneticRod", (object,), {"__init__": mock_magnetic_rod_init}
)


@pytest.mark.parametrize("n_elems", [2, 4, 16])
def test_magnetic_forces_invalid_init(n_elems):
    dim = 3
    mock_rod = MockMagneticRod()
    mock_rod.external_torques = np.zeros((dim, n_elems))
    mock_rod.n_elems = n_elems
    mock_rod.director_collection = np.repeat(
        np.identity(dim)[:, :, np.newaxis], n_elems, axis=2
    )
    mock_rod.volume = np.ones((n_elems,))
    magnetic_field_object = BaseMagneticField()

    # invalid density
    magnetization_density = np.ones((n_elems + 1))
    correct_error_message = (
        "Invalid magnetization intensity! Should be either a float or "
        "an array of shape (num_rod_elements,)"
    )
    with pytest.raises(ValueError) as exc_info:
        _ = MagneticForces(
            external_magnetic_field=magnetic_field_object,
            magnetization_density=magnetization_density,
            magnetization_direction=np.ones((3,)),
            rod_volume=mock_rod.volume,
            rod_director_collection=mock_rod.director_collection,
        )
    assert exc_info.value.args[0] == correct_error_message
    # invalid direction
    magnetization_direction = np.ones((3, n_elems + 1))
    correct_error_message = (
        "Invalid magnetization direction! Should be either a (3,) array or "
        "an array of shape (3, num_rod_elements)"
    )
    with pytest.raises(ValueError) as exc_info:
        _ = MagneticForces(
            external_magnetic_field=magnetic_field_object,
            magnetization_density=1.0,
            magnetization_direction=magnetization_direction,
            rod_volume=mock_rod.volume,
            rod_director_collection=mock_rod.director_collection,
        )
    assert exc_info.value.args[0] == correct_error_message


@pytest.mark.parametrize("n_elems", [2, 4, 16])
@pytest.mark.parametrize("time", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("ramp_interval", [1.0, 2.0])
@pytest.mark.parametrize("start_time", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("end_time", [4.0, 8.0])
def test_magnetic_forces_apply_torques(
    n_elems, time, ramp_interval, start_time, end_time
):
    dim = 3
    mock_rod = MockMagneticRod()
    mock_rod.external_torques = np.zeros((dim, n_elems))
    mock_rod.n_elems = n_elems
    mock_rod.director_collection = np.repeat(
        np.identity(dim)[:, :, np.newaxis], n_elems, axis=2
    )
    mock_rod.volume = 2.0 * np.ones((n_elems,))
    magnetization_density = 3.0
    magnetization_vector = np.random.rand(dim) + Tolerance.atol()
    magnetization_vector /= np.linalg.norm(magnetization_vector)
    magnetic_field_amplitude = np.random.rand(dim)
    magnetic_field_object = ConstantMagneticField(
        magnetic_field_amplitude=magnetic_field_amplitude,
        ramp_interval=ramp_interval,
        start_time=start_time,
        end_time=end_time,
    )
    external_magnetic_field_forcing = MagneticForces(
        external_magnetic_field=magnetic_field_object,
        magnetization_density=magnetization_density,
        magnetization_direction=magnetization_vector,
        rod_volume=mock_rod.volume,
        rod_director_collection=mock_rod.director_collection,
    )

    external_magnetic_field_forcing.apply_torques(rod=mock_rod, time=time)

    correct_factor = 0.0
    if time > start_time:
        correct_factor = (time > start_time) * min(
            1.0, (time - start_time) / ramp_interval
        )
    if time > end_time:
        correct_factor = max(0.0, -1 / ramp_interval * (time - end_time) + 1.0)
    correct_magnetic_field_value = correct_factor * magnetic_field_amplitude
    # elemental_magnetisation_mag = mock_rod.volume * magnetization_density
    elemental_magnetisation_mag = 2.0 * 3.0
    correct_magnetic_field_torques = np.cross(
        elemental_magnetisation_mag * magnetization_vector, correct_magnetic_field_value
    ).reshape(dim, 1) * np.ones((n_elems,))

    # no effect on forces
    np.testing.assert_allclose(mock_rod.external_forces, 0.0, atol=Tolerance.atol())
    np.testing.assert_allclose(
        mock_rod.external_torques, correct_magnetic_field_torques, atol=Tolerance.atol()
    )
