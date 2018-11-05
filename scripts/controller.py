from caster_vrep_gym import CasterBaseVrepEnv

import numpy as np
import tensorflow as tf


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


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
            action = np.array([0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0])
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
    sys.exit(main(sys.argv))
