import numpy as np
from elastica.external_forces import NoForces


from numba import njit
from elastica._linalg import (
    _batch_norm,
    _batch_dot,
    _batch_product_i_k_to_ik,
    _batch_product_i_ik_to_k,
    _batch_product_k_ik_to_ik,
    _batch_vector_sum,
)

from elastica.interaction import (
    find_slipping_elements,
    node_to_element_position,
    node_to_element_mass_or_force,
    elements_to_nodes_inplace,
    node_to_element_velocity,
)


# base class for interaction
# only applies normal force no friction
class InteractionPlaneForRodTips:
    """
    If rod tips are in contact with the plane.
    """

    def __init__(self, k, nu, plane_origin, plane_normal):
        """ """
        self.k = k
        self.nu = nu
        self.plane_origin = plane_origin.reshape(3, 1)
        self.plane_normal = plane_normal.reshape(3)
        self.surface_tol = 1e-4

    def apply_normal_force(self, system):
        """ """
        return apply_normal_force_numba(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.k,
            self.nu,
            system.lengths,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
        )


@njit(cache=True)
def apply_normal_force_numba(
    plane_origin,
    plane_normal,
    surface_tol,
    k,
    nu,
    lengths,
    mass,
    position_collection,
    velocity_collection,
    internal_forces,
    external_forces,
):
    """ """

    # Compute plane response force
    nodal_total_forces = _batch_vector_sum(internal_forces, external_forces)
    element_total_forces = node_to_element_mass_or_force(nodal_total_forces)

    force_component_along_normal_direction = _batch_product_i_ik_to_k(
        plane_normal, element_total_forces
    )
    forces_along_normal_direction = _batch_product_i_k_to_ik(
        plane_normal, force_component_along_normal_direction
    )

    # If the total force component along the plane normal direction is greater than zero that means,
    # total force is pushing rod away from the plane not towards the plane. Thus, response force
    # applied by the surface has to be zero.
    forces_along_normal_direction[
        ..., np.where(force_component_along_normal_direction > 0)[0]
    ] = 0.0
    # Compute response force on the element. Plane response force
    # has to be away from the surface and towards the element. Thus
    # multiply forces along normal direction with negative sign.
    plane_response_force = -forces_along_normal_direction

    # Elastic force response due to penetration
    element_position = node_to_element_position(position_collection)
    distance_from_plane = _batch_product_i_ik_to_k(
        plane_normal, (element_position - plane_origin)
    )
    plane_penetration = np.minimum(distance_from_plane - lengths / 2, 0.0)
    elastic_force = -k * _batch_product_i_k_to_ik(plane_normal, plane_penetration)

    # Damping force response due to velocity towards the plane
    element_velocity = node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )
    normal_component_of_element_velocity = _batch_product_i_ik_to_k(
        plane_normal, element_velocity
    )
    damping_force = -nu * _batch_product_i_k_to_ik(
        plane_normal, normal_component_of_element_velocity
    )

    # Compute total plane response force
    plane_response_force_total = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane - lengths / 2) > surface_tol)[
        0
    ]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    # plane_response_force[..., no_contact_point_idx] = 0.0
    plane_response_force_total[..., no_contact_point_idx] = 0.0

    # Update the external forces
    elements_to_nodes_inplace(plane_response_force_total, external_forces)

    return (_batch_norm(plane_response_force_total), no_contact_point_idx)


class IsotropicFrictionalPlaneForRodTips(NoForces, InteractionPlaneForRodTips):
    """ """

    def __init__(
        self,
        k,
        nu,
        plane_origin,
        plane_normal,
        slip_velocity_tol,
        kinetic_mu,
        static_mu,
    ):
        """ """
        InteractionPlaneForRodTips.__init__(self, k, nu, plane_origin, plane_normal)
        self.slip_velocity_tol = slip_velocity_tol
        self.kinetic_mu = kinetic_mu
        self.static_mu = static_mu

    def apply_forces(self, system, time=0.0):
        """
        Call numba implementation to apply friction forces
        Parameters
        ----------
        system
        time

        Returns
        -------

        """
        isotropic_friction(
            self.plane_origin,
            self.plane_normal,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu,
            self.static_mu,
            system.lengths,
            system.mass,
            system.tangents,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
        )


@njit(cache=True)
def isotropic_friction(
    plane_origin,
    plane_normal,
    surface_tol,
    slip_velocity_tol,
    k,
    nu,
    kinetic_mu,
    static_mu,
    lengths,
    mass,
    tangents,
    position_collection,
    velocity_collection,
    internal_forces,
    external_forces,
):
    plane_response_force_mag, no_contact_point_idx = apply_normal_force_numba(
        plane_origin,
        plane_normal,
        surface_tol,
        k,
        nu,
        lengths,
        mass,
        position_collection,
        velocity_collection,
        internal_forces,
        external_forces,
    )

    # Kinetic friction
    axial_direction = tangents
    element_velocity = node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection
    )
    velocity_mag_along_axial_direction = _batch_dot(element_velocity, axial_direction)
    velocity_along_axial_direction = _batch_product_k_ik_to_ik(
        velocity_mag_along_axial_direction, axial_direction
    )
    # Compute the velocity perpendicular to the axial direction, since for friction forces for in plane velocities.
    velocity_perpendicular_to_axial_direction = (
        element_velocity - velocity_along_axial_direction
    )

    # Friction forces depends on the direction of velocity, in other words sign
    # of the velocity vector.
    # Call slip function to check if elements slipping or not
    slip_function_perpendicular_to_axial_direction = find_slipping_elements(
        velocity_perpendicular_to_axial_direction, slip_velocity_tol
    )

    # Compute unitized total velocity vector in plane since friction force is opposite to the motion direction.
    # unitized_total_velocity_in_plane = velocity_perpendicular_to_axial_direction
    # unitized_total_velocity_in_plane /= _batch_norm(
    #     unitized_total_velocity_in_plane + 1e-14
    # )
    l = _batch_norm(velocity_perpendicular_to_axial_direction + 1e-14)
    unitized_total_velocity_in_plane = velocity_perpendicular_to_axial_direction / l
    unitized_total_velocity_in_plane[:, l < 1e-8] = 0.0

    # Apply kinetic friction in axial direction.
    kinetic_friction_force_perpendicular_to_axial_direction = -(
        (1.0 - slip_function_perpendicular_to_axial_direction)
        * kinetic_mu
        * plane_response_force_mag
        * unitized_total_velocity_in_plane
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
    kinetic_friction_force_perpendicular_to_axial_direction[
        ..., no_contact_point_idx
    ] = 0.0
    elements_to_nodes_inplace(
        kinetic_friction_force_perpendicular_to_axial_direction, external_forces
    )

    # static friction
    nodal_total_forces = _batch_vector_sum(internal_forces, external_forces)
    element_total_forces = node_to_element_mass_or_force(nodal_total_forces)
    force_component_along_axial_direction = (
        _batch_dot(element_total_forces, axial_direction) * axial_direction
    )
    force_component_perpendicular_to_axial_direction = (
        element_total_forces - force_component_along_axial_direction
    )
    # force_component_sign_perpendicular_to_axial_direction = np.sign(
    #     force_component_perpendicular_to_axial_direction
    # )
    mag = _batch_norm(force_component_perpendicular_to_axial_direction + 1e-14)
    static_force_direction = force_component_perpendicular_to_axial_direction / mag
    static_force_direction[:, mag < 1e-8] = 0.0

    force_component_sign_perpendicular_to_axial_direction = np.sign(
        _batch_dot(
            force_component_perpendicular_to_axial_direction, static_force_direction
        )
    )

    max_friction_force = (
        slip_function_perpendicular_to_axial_direction
        * static_mu
        * plane_response_force_mag
    )
    # friction = min(mu N, pushing force)
    static_friction_force_perpendicular_to_axial_direction = -(
        np.minimum(
            # np.fabs(force_component_perpendicular_to_axial_direction),
            _batch_norm(force_component_perpendicular_to_axial_direction),
            max_friction_force,
        )
        * force_component_sign_perpendicular_to_axial_direction
        * static_force_direction
    )
    # If rod element does not have any contact with plane, plane cannot apply friction
    # force on the element. Thus let set static friction force to 0.0 for the no contact points.
    static_friction_force_perpendicular_to_axial_direction[
        ..., no_contact_point_idx
    ] = 0.0
    elements_to_nodes_inplace(
        static_friction_force_perpendicular_to_axial_direction, external_forces
    )
