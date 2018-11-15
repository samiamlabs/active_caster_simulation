from caster_vrep_gym import CasterBaseVrepEnv

import numpy as np
import tensorflow as tf
# import pandas as pd

import shutil
import os

import controller_graph

logdir = 'Graph'
shutil.rmtree('./' + logdir, ignore_errors=True)
os.mkdir('./' + logdir)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def create_observation(n_casters):
    observation = np.empty(n_casters * 4)
    for caster_index in range(n_casters):
        index_offset_pos = caster_index * 2
        index_offset_vel = n_casters * 2 + caster_index * 2
        observation[index_offset_pos] = 0.1      # pos steer
        observation[index_offset_pos + 1] = 0.2  # pos drive
        observation[index_offset_vel] = 0.3      # vel steer
        observation[index_offset_vel + 1] = 0.4  # vel drive

    return tf.constant(observation, dtype=tf.float32)

reset_graph()

n_casters = 4
observation = create_observation(n_casters)

base_velocity_desired = tf.constant(np.zeros([n_casters, 3]),
                                    tf.float32,
                                    name='base_accelerations')

joint_tourques = controller_graph.joint_tourques(observation,
                                                 base_velocity_desired)

with tf.Session():
    print(joint_tourques.eval())

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()


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
#     sys.exit(main(sys.argv))
