import tensorflow as tf
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


def inertia(joint_positions=np.zeros(4)):
    """Create symmetric mass/inertia matrix in joint space."""
    # TODO: replace dummy
    with tf.name_scope('inertia'):
        return tf.constant(np.diag([10.0, 10.0, 10.0]), dtype=tf.float32)


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
            [cos_phi/r,  sin_phi/r, h*(cos_beta*sin_phi+sin_beta*cos_phi)/r],
            [-sin_phi/b, cos_phi/b, h*(cos_beta*cos_phi+sin_beta*sin_phi)/b]
        ])

        return inverse_jacobian


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


def cc_coupling(joint_positions, joint_velocities):
    """Create joint space vector of centripital and Coriolis coupling terms."""
    # create dummy vector
    with tf.name_scope('cc_coupling'):
        return tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
