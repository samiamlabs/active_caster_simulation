from caster_vrep_gym import CasterBaseVrepEnv

import numpy as np
import tensorflow as tf
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


def create_caster_constats():
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

    # print(caster_constants)
    return caster_constants


def inverse_jacobian(joint_positions, caster_constants):
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


def mass(joint_positions):
    # create dummmy matrix
    return tf.constant(np.diag([10.0, 10.0, 10.0]), dtype=tf.float32)


def mass_operational_space(inverse_jacobian, mass_matrix):
    return tf.transpose(inverse_jacobian) @ mass_matrix @ inverse_jacobian


# Centripital and coriolis coupling terms
def cc_coupling(joint_positions, joint_velocities):  # centripital and coriolis coupling terms
    # create dummy vector
    return tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

def cc_coupling_operational_space(joint_positions, joint_velocities, mass_matrix):
    # create dummy vector
    return tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)


reset_graph()

caster_constants = tf.constant(create_caster_constats().values,
                               dtype=tf.float32,
                               name='caster_constants')

n_casters = caster_constants.shape[0]

observation = tf.placeholder(tf.float32,
                             shape=[n_casters*2*2, ],
                             name='observation')

# q
with tf.name_scope('joint_positions'):
    joint_positions = tf.reshape(tf.slice(observation,
                                          begin=[0, ],
                                          size=[n_casters*2, ]),
                                 shape=[n_casters, 2],
                                 name='joint_positions')
# q_dot
with tf.name_scope('joint_velocities'):
    joint_velocities = tf.reshape(tf.slice(observation,
                                           begin=[int(n_casters)*2, ],
                                           size=[n_casters*2, ]),
                                  shape=[n_casters, 2],
                                  name='joint_velocities')

# x_dot, y_dot, theta_dot
command = tf.placeholder(tf.float32, shape=[3, ], name='command')

with tf.name_scope('base_velocities_desired'):
    base_velocities_desired = tf.slice(command, begin=[0, ], size=[3, ])

with tf.name_scope('inverse_jacobians'):
    inverse_jacobians = []
    for caster_index in range(n_casters):
        inverse_jacobians.append(
            inverse_jacobian(joint_positions[caster_index],
                             caster_constants[caster_index]))

with tf.name_scope('intertias'):
    masses = []
    for caster_index in range(n_casters):
        masses.append(mass(joint_positions[caster_index]))

with tf.name_scope('mass_operational_space'):
    operational_space_masses = []
    for caster_index in range(n_casters):
        operational_space_masses.append(
            mass_operational_space(inverse_jacobians[caster_index],
                                   masses[caster_index]))

with tf.name_scope('cc_coupling'):
    cc_couplings = []
    for caster_index in range(n_casters):
        cc_couplings.append(cc_coupling(
                            joint_positions[caster_index],
                            joint_velocities[caster_index]))

with tf.Session() as sess:
    # print(inverse_jacobians[0].eval(feed_dict={observation: np.zeros(8)}))
    # print(mass_matrices[0].eval(feed_dict={observation: np.zeros(8)}))
    print(operational_space_masses[0].eval(feed_dict={observation: np.arange(int(n_casters)*2.0*2.0)}))
    # print(cc_couplings[0].eval(feed_dict={observation: np.zeros(8)}))
    print(joint_positions.eval(feed_dict={observation: np.arange(int(n_casters)*2.0*2.0)}))
    print(joint_velocities.eval(feed_dict={observation: np.arange(int(n_casters)*2.0*2.0)}))

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()


def main(args):
    """
    Main function.

    Agent does random actions with 'action_space.sample()'
    """
    # #modify: the env class name
    env = CasterBaseVrepEnv()
    for i_episode in range(1):
        observation = env.reset()
        total_reward = 0
        for t in range(300):  # 10 ms per step
            # action = env.action_space.sample()
            action = np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1])
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print("Episode finished after {} timesteps.\tTotal reward: {}".format(
            t + 1, total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys
    # sys.exit(main(sys.argv))
