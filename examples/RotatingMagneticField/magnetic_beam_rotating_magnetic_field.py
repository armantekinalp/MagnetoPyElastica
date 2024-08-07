import numpy as np
from elastica import *
from magneto_pyelastica import *
from examples.post_processing import (
    plot_video_with_surface,
    plot_tip_position_history,
)


class MagneticBeamSimulator(
    BaseSystemCollection, Constraints, Forcing, Damping, CallBacks
):
    pass


magnetic_beam_sim = MagneticBeamSimulator()
# setting up test params
n_elem = 25
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.5  # m
base_radius = 0.15  # m
base_area = np.pi * base_radius**2
volume = base_area * base_length
moment_of_inertia = np.pi / 4 * base_radius**4
density = 2.39e3  # kg/m3
E = 1.85e5  # Pa
shear_modulus = 6.16e4  # Pa

# Parameters are from
# Gu, Hongri, et al. "Magnetic cilia carpets with programmable metachronal waves." Nature communications 11.1 (2020).
angular_frequency = np.deg2rad(10.0)  # angular frequency of the rotating magnetic field
magnetic_field_strength = 80e-3  # 80mT
# MBAL2_EI is a non-dimensional number from
# Wang, Liu, et al. "Hard-magnetic elastica." Journal of the Mechanics and Physics of Solids 142 (2020).
MBAL2_EI = (
    3.82e-5 * magnetic_field_strength * 4e-3 / (1.85e5 * np.pi / 4 * (0.4e-3) ** 4)
)  # Magnetization magnitude * B * Length/(EI)
magnetization_density = (
    MBAL2_EI * E * moment_of_inertia / (volume * magnetic_field_strength * base_length)
)
magnetization_angle = np.deg2rad(90)
magnetization_direction = np.array(
    [np.sin(magnetization_angle), 0.0, np.cos(magnetization_angle)]
).reshape(3, 1) * np.ones((n_elem))

magnetic_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
magnetic_beam_sim.append(magnetic_rod)


# Add boundary conditions, one end of rod is clamped
magnetic_beam_sim.constrain(magnetic_rod).using(
    OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
# Set the constant magnetic field object
magnetic_field_object = SingleModeOscillatingMagneticField(
    magnetic_field_amplitude=magnetic_field_strength * np.array([1, 1e-2, 1]),
    magnetic_field_angular_frequency=np.array(
        [angular_frequency, 0, angular_frequency]
    ),
    magnetic_field_phase_difference=np.array([0, np.pi / 2, np.pi / 2]),
    ramp_interval=0.01,
    start_time=0.0,
    end_time=5e3,
)

# Apply magnetic forces
magnetic_beam_sim.add_forcing_to(magnetic_rod).using(
    MagneticForces,
    external_magnetic_field=magnetic_field_object,
    magnetization_density=magnetization_density,
    magnetization_direction=magnetization_direction,
    rod_volume=magnetic_rod.volume,
    rod_director_collection=magnetic_rod.director_collection,
)

# Add callbacks
class MagneticBeamCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["tangents"].append(system.tangents.copy())


# add damping
dl = base_length / n_elem
dt = 0.1 * dl
damping_constant = 0.5
magnetic_beam_sim.dampen(magnetic_rod).using(
    AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

# run simulation for two periods
final_time = 2 * 2 * np.pi / angular_frequency
total_steps = int(final_time / dt)
rendering_fps = 5
step_skip = int(1.0 / (rendering_fps * dt))

# Add call back for plotting time history of the rod
post_processing_dict = defaultdict(list)
magnetic_beam_sim.collect_diagnostics(magnetic_rod).using(
    MagneticBeamCallBack, step_skip=step_skip, callback_params=post_processing_dict
)


timestepper = PositionVerlet()
magnetic_beam_sim.finalize()
integrate(timestepper, magnetic_beam_sim, final_time, total_steps)

# Plot the magnetic rod time history
plot_video_with_surface(
    [post_processing_dict],
    fps=rendering_fps,
    step=4,
    x_limits=(-2, 2),
    y_limits=(-2, 2),
    z_limits=(-2, 2),
)
plot_tip_position_history(post_processing_dict)
