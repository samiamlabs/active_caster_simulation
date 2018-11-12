import tensorflow as tf
import numpy as np

import controller_graph
from tensorflow.python.framework import test_util


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class TestConstants(tf.test.TestCase):
    def test_defaults_yield_correct_shape(self):
        constants = controller_graph.create_caster_constats()
        self.assertShapeEqual(np.ones([4, 4]), constants)


class TestInertia(tf.test.TestCase):
    def test_defaults_yield_correct_shape_and_values(self):
        inertia = np.array([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])

        inertia_ = controller_graph.inertia()
        self.assertAllEqual(inertia, inertia_)


class TestJoint(tf.test.TestCase):
    def setUp(self):
        self.n_casters = 4
        self.observation = np.empty(self.n_casters * 4)
        for caster_index in range(self.n_casters):
            index_offset_pos = caster_index * 2
            index_offset_vel = self.n_casters * 2 + caster_index * 2
            self.observation[index_offset_pos] = 0.1      # pos drive
            self.observation[index_offset_pos + 1] = 0.2  # pos steer
            self.observation[index_offset_vel] = 0.3      # vel drive
            self.observation[index_offset_vel + 1] = 0.4  # vel steer

    def test_joint_positions_yield_correct_shape_and_values(self):
        joint_positions = np.array([[0.1, 0.2],
                                    [0.1, 0.2],
                                    [0.1, 0.2],
                                    [0.1, 0.2]])
        joint_positions_ = controller_graph.joint_positions(self.observation,
                                                            self.n_casters)
        self.assertAllEqual(joint_positions, joint_positions_)

    def test_joint_velocities_yield_correct_shape_and_values(self):
        joint_velocities = np.array([[0.3, 0.4],
                                     [0.3, 0.4],
                                     [0.3, 0.4],
                                     [0.3, 0.4]])
        joint_velocities_ = controller_graph.joint_velocities(self.observation,
                                                              self.n_casters)
        self.assertAllEqual(joint_velocities, joint_velocities_)


class TestJacobian(tf.test.TestCase):
    def setUp(self):
        reset_graph()
        self.n_casters = 4
        self.observation = np.empty(self.n_casters * 4)
        for caster_index in range(self.n_casters):
            index_offset_pos = caster_index * 2
            index_offset_vel = self.n_casters * 2 + caster_index * 2
            self.observation[index_offset_pos] = 0.1      # pos drive
            self.observation[index_offset_pos + 1] = 0.2  # pos steer
            self.observation[index_offset_vel] = 0.3      # vel drive
            self.observation[index_offset_vel + 1] = 0.4  # vel steer

    def test_defaults_yield_correct_shape_and_values(self):
        # TODO: check values
        inverse_jacobian = np.array([[-7.388006, 23.883413, 4.159784],
                                     [47.766827, 14.776011, 10.319568],
                                     [-7.388006, 23.883413, 5.159784]])

        caster_constants = controller_graph.create_caster_constats()
        observation = tf.constant(self.observation, dtype=tf.float32)
        joint_positions = controller_graph.joint_velocities(observation,
                                                            self.n_casters)
        inverse_jacobian_ = controller_graph.inverse_jacobian(
                                joint_positions[0], caster_constants[0])
        with self.test_session():
            self.assertAllClose(inverse_jacobian,
                                inverse_jacobian_.eval())


if __name__ == '__main__':
    tf.test.main()
