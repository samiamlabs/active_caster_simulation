import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd


def create_caster_constats():
    """Create caster constant tensor."""
    wheel_separation = 0.33
    wheel_offset = 0.04
    wheel_radius = 0.02

    fl_mount_point = np.array([wheel_separation / 2, wheel_separation / 2])
    fr_mount_point = np.array(
        [wheel_separation / 2, -wheel_separation / 2])
    rl_mount_point = np.array(
        [-wheel_separation / 2, wheel_separation / 2])
    rr_mount_point = np.array(
        [-wheel_separation / 2, -wheel_separation / 2])

    fl_h = np.linalg.norm(fl_mount_point)
    fl_beta = np.arctan2(fl_mount_point[0], fl_mount_point[1])
    fr_h = np.linalg.norm(fr_mount_point)
    fr_beta = np.arctan2(fl_mount_point[0], fr_mount_point[1])
    rl_h = np.linalg.norm(rl_mount_point)
    rl_beta = np.arctan2(rl_mount_point[0], rl_mount_point[1])
    rr_h = np.linalg.norm(rr_mount_point)
    rr_beta = np.arctan2(rl_mount_point[0], rr_mount_point[1])

    caster_names = ['fl', 'fr', 'rl', 'rr']
    caster_constants = pd.DataFrame([
        {'b': wheel_offset, 'beta': fl_beta, 'h': fl_h, 'r': wheel_radius},
        {'b': wheel_offset, 'beta': fr_beta, 'h': fr_h, 'r': wheel_radius},
        {'b': wheel_offset, 'beta': rl_beta, 'h': rl_h, 'r': wheel_radius},
        {'b': wheel_offset, 'beta': rr_beta, 'h': rr_h, 'r': wheel_radius},
    ], caster_names)

    return tf.constant(caster_constants.values, dtype=tf.float32)

# --- Joint Space ---


def joint_positions(observation, n_casters):
    """Positions of drive and steer joints."""
    with tf.name_scope('joint_positions'):
        return tf.reshape(tf.slice(observation,
                                   begin=[0, ],
                                   size=[n_casters*2, ]),
                          shape=[n_casters, 2],
                          name='joint_positions')


def joint_velocities(observation, n_casters):
    """Positions and velocities in x and y."""
    with tf.name_scope('joint_velocities'):
        return tf.reshape(tf.slice(observation,
                                   begin=[int(n_casters)*2, ],
                                   size=[n_casters*2, ]),
                          shape=[n_casters, 2],
                          name='joint_velocities')


def inertia(joint_positions=np.zeros(4)):
    """Create symmetric mass/inertia matrix in joint space."""
    # TODO: replace dummy
    with tf.name_scope('inertia'):
        mass = 5.0
        radius = 0.3
        height = 0.2
        return cylinder_inertia(mass, radius, height)


def cylinder_inertia(mass, radius, height):
    """Create a simple inertia matrix."""
    with tf.name_scope('cylinder_inertia'):
        return tf.constant(np.diag([(mass/12)*3*radius*radius*height*height,
                                    (mass/12)*3*radius*radius*height*height,
                                    mass*radius*radius/2]), dtype=tf.float32)


def cuboid_inertia(mass, x, y, z):  # or box inertia
    with tf.name_scope('cuboid_inertia'):
        return tf.constant(np.diag([(mass/12)*(y*y + z*z),
                                    (mass)/12*(x*x + z*z),
                                    (mass)/12*(x*x + y*y)]))


def inverse_jacobian(joint_positions, caster_constants):
    """Create mapping from operational space velocities to joint velocities."""
    with tf.name_scope('inverse_jacobian'):
        b = caster_constants[0]
        beta = caster_constants[1]
        h = caster_constants[2]
        r = caster_constants[3]

        phi = joint_positions[0]

        sin_phi = tf.sin(phi)
        cos_phi = tf.cos(phi)

        sin_beta = tf.sin(beta)
        cos_beta = tf.cos(beta)

        inverse_jacobian = tf.stack([
            [-sin_phi/b, cos_phi/b, h*(cos_beta*cos_phi+sin_beta*sin_phi)/b-1],
            [cos_phi/r,  sin_phi/r, h*(cos_beta*sin_phi-sin_beta*cos_phi)/r],
            [-sin_phi/b, cos_phi/b, h*(cos_beta*cos_phi+sin_beta*sin_phi)/b]
        ])

        return inverse_jacobian


def cc_coupling(joint_positions, joint_velocities):
    """Create joint space vector of centripital and Coriolis coupling terms."""
    # TODO: implement
    # create dummy vector
    with tf.name_scope('cc_coupling'):
        return tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

# --- Operational Space ---


def inertia_operational_space(inverse_jacobian, inertia):
    """Create mass/inertia matrix in operational space."""
    with tf.name_scope('inertia_operational_space'):
        return tf.transpose(inverse_jacobian) @ inertia @ inverse_jacobian


def cc_coupling_operational_space(cc_coupling, inverse_jacobian,
                                  inertia, base_velocity):
    """Operational space vector of centripital and Coriolis coupling terms."""
    with tf.name_scope('cc_coupling_operational_space'):
        cc_coupling = tf.expand_dims(cc_coupling, axis=1)
        base_velocity = tf.expand_dims(base_velocity, axis=1)
        cc_coupling_os = (tf.transpose(inverse_jacobian) @
                          (inertia @ inverse_jacobian @ base_velocity +
                           cc_coupling))

        return tf.transpose(cc_coupling_os)[0]


def force_vector_end_effector(inertia_operational_space, base_acceleration,
                              cc_coupling_operational_space):
    """Force vector at the origin of the end effector coordinate system."""
    with tf.name_scope('force_vector_end_effector'):
        cc_coupling_operational_space = (
            tf.expand_dims(cc_coupling_operational_space, axis=1))

        base_acceleration = tf.expand_dims(base_acceleration, axis=1)
        force_vector_end_effector = (inertia_operational_space @
                                     base_acceleration +
                                     cc_coupling_operational_space)

        return tf.transpose(force_vector_end_effector)[0]


def wheel_constraint(inverse_jacobian):
    """Create mapping from base velocity to joint speeds."""
    with tf.name_scope('wheel_constraint'):
        # TODO: Replace with improved version based on contact
        # point velocities/forces

        # Use the nonholonomic constraints from the inverse jacobian
        # return inverse_jacobian
        return tf.slice(inverse_jacobian, [0, 0], [2, 3])

# --- Controller ---


def base_position_integration_step(base_position, base_velocity, dt):
    """Integrate base velocity and update base_position variable."""
    # with tf.variable_scope("base", reuse=True):
    #     base_position = tf.get_variable('base_position', shape=3,
    #                                     initializer=tf.zeros_initializer())
    x, y, theta = tf.unstack(base_position)
    x_dot, y_dot, theta_dot = tf.unstack(base_velocity)

    # TODO: use Runge-Kutta or exact integration?
    x += x_dot*dt
    y += y_dot*dt
    theta += theta_dot*dt

    return base_position.assign(tf.stack([x, y, theta]))


def compensator(base_position_desired, base_velocity_desired,
                base_position, base_velocity, base_acceleration,
                kp, kv):
    """Compensator, control force for linearized unit mass system)."""
    # Kp = tf.constant(0.2, dtype=tf.float32)
    # Kv = tf.constant(0.6, dtype=tf.float32)

    return (- kp*(base_position - base_position_desired)
            - kv*(base_velocity - base_velocity_desired) + base_acceleration)


def joint_tourques(observation, base_velocity_desired):
    """PCV controller based on the operational space formulation."""
    caster_constants = create_caster_constats()
    n_casters = caster_constants.shape[0]
    joint_pos = joint_positions(observation, n_casters)
    joint_vel = joint_velocities(observation, n_casters)

    # TODO: replace with diff and integration of base_velocity_desired,
    # alt. implement trajectory interface
    with tf.name_scope('base_desired'):
        base_position_desired = tf.constant(np.arange(3.0),
                                            dtype=tf.float32,
                                            name='base_position_desired')

        base_acceleration_desired = tf.constant(np.arange(3.0),
                                                dtype=tf.float32,
                                                name='base_acceleration_desired')

    with tf.name_scope('base'):
        base_position = tf.constant(np.zeros([n_casters, 3]),
                                    tf.float32,
                                    name='base_position')

        base_velocity = tf.constant(np.zeros([n_casters, 3]),
                                    tf.float32,
                                    name='base_velocity')

        base_accelerations = tf.constant(np.zeros([n_casters, 3]),
                                         tf.float32,
                                         name='base_accelerations')

    inverse_jacobians = []
    inertias = []
    intertias_operational_space = []
    cc_couplings = []
    cc_couplings_operational_space = []
    force_vectors_end_effector = []
    wheel_constraints = []
    inverse_wheel_constraints = []
    for caster_index in range(n_casters):
        with tf.name_scope('caster'):
            with tf.name_scope('joint_space'):
                inverse_jacobians.append(
                    inverse_jacobian(joint_pos[caster_index],
                                     caster_constants[caster_index]))

                inertias.append(inertia(joint_pos[caster_index]))

                cc_couplings.append(cc_coupling(
                                    joint_pos[caster_index],
                                    joint_vel[caster_index]))

            with tf.name_scope('operational_space'):

                intertias_operational_space.append(
                    inertia_operational_space(inverse_jacobians[caster_index],
                                              inertias[caster_index]))

                cc_couplings_operational_space.append(
                    cc_coupling_operational_space(
                        cc_couplings[caster_index],
                        inverse_jacobians[caster_index],
                        inertias[caster_index],
                        base_velocity[caster_index]))

            force_vectors_end_effector.append(
                force_vector_end_effector(
                    intertias_operational_space[caster_index],
                    base_accelerations[caster_index],
                    cc_couplings_operational_space[caster_index]))

            wheel_constraints.append(inverse_jacobians[caster_index])

            inverse_wheel_constraints.append(
                tfp.math.pinv(wheel_constraints[caster_index]))

    cc_coupling_base = tf.reduce_sum(cc_couplings_operational_space,
                                     axis=0, name='cc_coupling_base')

    force_base = tf.reduce_sum(force_vectors_end_effector,
                               axis=0, name='force_base')

    inertia_base = tf.reduce_sum(intertias_operational_space,
                                 axis=0, name='inertia_base')

    kp = tf.constant(0.2, dtype=tf.float32)
    kv = tf.constant(0.6, dtype=tf.float32)

    comp = compensator(base_position_desired,
                       base_velocity_desired,
                       base_acceleration_desired,
                       base_position,
                       base_velocity,
                       kp, kv)

    return tf.constant(np.ones(n_casters*2), tf.float32)
