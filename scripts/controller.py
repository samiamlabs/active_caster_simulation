from caster_vrep_gym import CasterBaseVrepEnv

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
import pandas as pd

import shutil
import os

# tf.enable_eager_execution()
# eager_execution = True
eager_execution = False

logdir = 'Graph'
shutil.rmtree('./' + logdir, ignore_errors=True)
os.mkdir('./' + logdir)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)




def inertia(joint_positions):
    """Create symmetric mass/inertia matrix in joint space."""
    # TODO: replace dummy
    with tf.name_scope('inertia'):
        return tf.constant(np.diag([10.0, 10.0, 10.0]), dtype=tf.float32)


def inertia_operational_space(inverse_jacobian, inertia):
    """Create mass/inertia matrix in operational space."""
    with tf.name_scope('inertia_operational_space'):
        return tf.transpose(inverse_jacobian) @ inertia @ inverse_jacobian


def cc_coupling(joint_positions, joint_velocities):
    """Create joint space vector of centripital and Coriolis coupling terms."""
    # create dummy vector
    with tf.name_scope('cc_coupling'):
        return tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)


def cc_coupling_operational_space(cc_coupling, inverse_jacobian,
                                  inertia, base_velocities):
    """Operational space vector of centripital and Coriolis coupling terms."""
    with tf.name_scope('cc_coupling_operational_space'):
        return tf.transpose(inverse_jacobian) @ (inertia @ inverse_jacobian @
                                                 base_velocities + cc_coupling)


def force_vector_end_effector(inertia_operational_space, base_acceleration,
                              cc_coupling_operational_space):
    """Force vector at the origin of the end effector coordinate system."""
    with tf.name_scope('force_vector_end_effector'):
        return (inertia_operational_space @ base_acceleration
                + cc_coupling_operational_space)


def wheel_constraint(inverse_jacobian):
    """Create mapping from base velocity to joint speeds."""
    with tf.name_scope('wheel_constraint'):
        # TODO: Replace with improved version based on contact
        # point velocities/forces

        # Use the nonholonomic constraints from the inverse jacobian
        # return inverse_jacobian
        return tf.slice(inverse_jacobian, [0, 0], [2, 3])


def create_compensator(base_position_desired, base_velocity_desired,
                       base_position, base_velocity, base_accelerations):
    """Create compensator, control force for linearized unit mass system)."""
    Kp = tf.constant(0.1, dtype=tf.float32)
    Kv = tf.constant(0.1, dtype=tf.float32)

    return (- Kp*(base_position - base_position_desired)
            - Kv*(base_velocity - base_velocity_desired) + base_accelerations)


def pcv_controller():
    """PCV controller based on the operational space formulation."""
    caster_constants = tf.constant(create_caster_constats().values,
                                   dtype=tf.float32,
                                   name='caster_constants')

    n_casters = caster_constants.shape[0]

    observation = tf.placeholder(tf.float32,
                                 shape=[n_casters*2*2, ],
                                 name='observation')

    # q_dot
    with tf.name_scope('joint_velocities'):
        joint_velocities = tf.reshape(tf.slice(observation,
                                               begin=[int(n_casters)*2, ],
                                               size=[n_casters*2, ]),
                                      shape=[n_casters, 2],
                                      name='joint_velocities')

    # x_dot, y_dot, theta_dot
    # command = tf.placeholder(tf.float32, shape=[3, ], name='command')

    with tf.name_scope('base_position_desired'):
        base_position_desired = tf.constant(np.arange(3.0), dtype=tf.float32)

    with tf.name_scope('base_velocity_desired'):
        base_velocity_desired = tf.constant(np.arange(3.0), dtype=tf.float32)

    with tf.name_scope('base_acceleration_desired'):
        base_acceleration_desired = tf.constant(np.arange(3.0), dtype=tf.float32)

    base_position = tf.constant(np.zeros([n_casters, 3]),
                                tf.float32,
                                name='base_position')

    base_velocity = tf.constant(np.zeros([n_casters, 3]),
                                tf.float32,
                                name='base_velocity')

    with tf.name_scope('base_accelerations'):
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
    # inverse_wheel_constraints = []
    for caster_index in range(n_casters):
        with tf.name_scope('caster'):
            with tf.name_scope('joint_space'):
                inverse_jacobians.append(
                    inverse_jacobian(joint_positions[caster_index],
                                     caster_constants[caster_index]))

                inertias.append(inertia(joint_positions[caster_index]))

                cc_couplings.append(cc_coupling(
                                    joint_positions[caster_index],
                                    joint_velocities[caster_index]))

            with tf.name_scope('operational_space'):

                intertias_operational_space.append(
                    inertia_operational_space(inverse_jacobians[caster_index],
                                              inertias[caster_index]))

                cc_couplings_operational_space.append(
                    cc_coupling_operational_space(
                        tf.expand_dims(cc_couplings[caster_index], 1),
                        inverse_jacobians[caster_index],
                        inertias[caster_index],
                        tf.expand_dims(base_velocity[caster_index], 1)))

            force_vectors_end_effector.append(
                force_vector_end_effector(
                    intertias_operational_space[caster_index],
                    tf.expand_dims(base_accelerations[caster_index], 1),
                    tf.expand_dims(
                        cc_couplings_operational_space[caster_index], 1)))

            wheel_constraints.append(inverse_jacobians[caster_index])

            # inverse_wheel_constraints.append(
            #     tfp.math.pinv(wheel_constraints[caster_index]))

    cc_coupling_base = tf.reduce_sum(cc_couplings_operational_space,
                                     axis=0, name='cc_coupling_base')

    force_base = tf.reduce_sum(force_vectors_end_effector,
                               axis=0, name='force_base')

    inertia_base = tf.reduce_sum(intertias_operational_space,
                                 axis=0, name='inertia_base')

    compensator = create_compensator(base_position_desired,
                                     base_velocity_desired,
                                     base_acceleration_desired,
                                     base_position,
                                     base_velocity)

reset_graph()



# class ShapeTest(tf.test.TestCase):
#     def test_joint_positions_shape(self):
#         # feed_dict = {observation: np.arange(int(n_casters)*2.0*2.0)}
#         # self.assertAllEqual(compensator.eval(feed_dict=feed_dict),
#         #                         np.zeros(3, 3))
#
#         self.assertTrue(True)


# tf.test.main(argv=['first-arg-is-ignored'])

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()

# with tf.Session() as sess:
#     feed_dict = {observation: np.arange(int(n_casters)*2.0*2.0)}
#     # print(mass_matrices[0].eval(feed_dict={observation: np.zeros(8)}))
#     # print(intertias_operational_space[0].eval(feed_dict=feed_dict))
#     # print(joint_positions.eval(feed_dict=feed_dict))
#     # print(joint_velocities.eval(feed_dict=feed_dict))
#     # print(inertia_base.eval(feed_dict=feed_dict))
#     # print(force_vectors_end_effector[0].eval(feed_dict=feed_dict))
#     # print(force_base.eval(feed_dict=feed_dict))
#     # print(cc_couplings[0].eval(feed_dict=feed_dict))
#     # print(cc_couplings_operational_space[0].eval(feed_dict=feed_dict))
#     # print(inverse_jacobians[0].eval(feed_dict=feed_dict))
#     # print(wheel_constraints[0].eval(feed_dict=feed_dict))
#     # print(inverse_wheel_constraints[0].eval(feed_dict=feed_dict))
#     print(compensator[0].eval(feed_dict=feed_dict))

# def main(args):
#     """
#     Main function.
#
#     Agent does random actions with 'action_space.sample()'
#     """
#     # #modify: the env class name
#     env = CasterBaseVrepEnv()
#     for i_episode in range(1):
#         observation = env.reset()
#         total_reward = 0
#         for t in range(300):  # 10 ms per step
#             # action = env.action_space.sample()
#             action = np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1])
#             observation, reward, done, _ = env.step(action)
#             total_reward += reward
#             if done:
#                 break
#         print("Episode finished after {} timesteps.\tTotal reward: {}".format(
#             t + 1, total_reward))
#     env.close()
#     return 0
#
#
# if __name__ == '__main__':
#     import sys

    # sys.exit(main(sys.argv))
