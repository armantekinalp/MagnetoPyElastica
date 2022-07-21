__doc__ = """ Module implementation for external magnetic forces for magnetic Cosserat rods."""
__all__ = ["MagneticForces"]

from elastica.external_forces import NoForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica._linalg import _batch_cross, _batch_matvec, _batch_norm
from magneto_pyelastica.magnetic_field import BaseMagneticField
import numpy as np
from typing import Union


class MagneticForces(NoForces):
    """
    This class applies magnetic forces on a magnetic Cosserat rod, based on an
    external magnetic field.

        Attributes
        ----------
        external_magnetic_field: object
            External magnetic field object, that returns the value of the magnetic field vector
            via a .value() method.
        magnetization_collection: np.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Magnetization of the rod, defined on the elements, in the material frame.

    """

    def __init__(
        self,
        external_magnetic_field: BaseMagneticField,
        magnetization_density: Union[float, np.ndarray],
        magnetization_direction: np.ndarray,
        rod_volume: np.ndarray,
        rod_director_collection: np.ndarray,
    ):
        """
        Parameters
        ----------
        external_magnetic_field: object
            External magnetic field object, that returns the value of the
            magnetic field vector via a .value() method.
        magnetization_density: float or a np.ndarray
            Float number or 1D (n_elems) array containing data with 'float' type.
            Density of magnetization of the rod.
        magnetization_direction: np.ndarray
            1D (dim) or 2D (dim, n_elems) array containing data with 'float' type.
            Direction of magnetization of the rod in the lab frame.
        rod_volume: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element volumes.
        rod_director_collection: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Array containing rod elemental director matrices.

        """
        super(NoForces, self).__init__()
        self.external_magnetic_field = external_magnetic_field
        rod_n_elem = rod_volume.shape[0]

        # if fixed value, then expand to rod element size
        if magnetization_direction.shape == (3,) or magnetization_direction.shape == (
            3,
            rod_n_elem,
        ):
            magnetization_direction = magnetization_direction.reshape(3, -1) * np.ones(
                (rod_n_elem,)
            )
        else:
            raise ValueError(
                "Invalid magnetization direction! Should be either a (3,) array or "
                "an array of shape (3, num_rod_elements)"
            )
        # normalise for unit vectors
        magnetization_direction /= _batch_norm(magnetization_direction)
        # convert to local frame
        magnetization_direction_in_material_frame = _batch_matvec(
            rod_director_collection, magnetization_direction
        )

        if not (
            isinstance(magnetization_density, float)
            or magnetization_density.shape == (rod_n_elem,)
        ):
            raise ValueError(
                "Invalid magnetization intensity! Should be either a float or "
                "an array of shape (num_rod_elements,)"
            )

        self.magnetization_collection = (
            magnetization_density
            * rod_volume
            * magnetization_direction_in_material_frame
        )

    def apply_torques(self, rod: CosseratRod, time: np.float64 = 0.0):
        rod.external_torques += _batch_cross(
            self.magnetization_collection,
            # convert external_magnetic_field to local frame
            _batch_matvec(
                rod.director_collection,
                self.external_magnetic_field.value(time=time).reshape(
                    3, 1
                )  # broadcasting 3D vector
                * np.ones((rod.n_elems,)),
            ),
        )
