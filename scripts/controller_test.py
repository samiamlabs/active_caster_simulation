import tensorflow as tf
import numpy as np

import controller_graph
from tensorflow.python.framework import test_util


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


def create_inverse_jacobian():
    caster_constants = controller_graph.create_caster_constats()
    n_casters = 1
    observation = create_observation(n_casters)
    joint_positions = controller_graph.joint_positions(observation,
                                                       n_casters)
    return controller_graph.inverse_jacobian(joint_positions[0],
                                             caster_constants[0])

# --- Joint Space ---


class TestConstants(tf.test.TestCase):
    def test_defaults_yield_correct_shape(self):
        constants = controller_graph.create_caster_constats()
        self.assertShapeEqual(np.ones([4, 4]), constants)


class TestInertia(tf.test.TestCase):
    def test_defaults_yield_correct_shape_and_values(self):
        inertia = np.array([[0.0045, 0.0, 0.0],
                            [0.0, 0.0045, 0.0],
                            [0.0, 0.0, 0.225]])

        inertia_ = controller_graph.inertia()
        self.assertAllClose(inertia, inertia_)

    def test_cuboid_inertia_yield_correct_shape_and_values(self):
        inertia = np.array([[0.016667, 0.0, 0.0],
                            [0.0, 0.008333, 0.0],
                            [0.0, 0.0, 0.021667]])

        inertia_ = controller_graph.cuboid_inertia(mass=2.0,
                                                   x=0.2,
                                                   y=0.3,
                                                   z=0.1)

        self.assertAllClose(inertia, inertia_)


class TestJoint(tf.test.TestCase):
    def setUp(self):
        reset_graph()
        self.n_casters = 4
        self.observation = create_observation(self.n_casters)

    def test_joint_positions_yield_correct_shape_and_values(self):
        joint_positions = np.array([[0.1, 0.2],
                                    [0.1, 0.2],
                                    [0.1, 0.2],
                                    [0.1, 0.2]])
        joint_positions_ = controller_graph.joint_positions(self.observation,
                                                            self.n_casters)

        with self.test_session():
            self.assertAllClose(joint_positions, joint_positions_.eval())

    def test_joint_velocities_yield_correct_shape_and_values(self):
        joint_velocities = np.array([[0.3, 0.4],
                                     [0.3, 0.4],
                                     [0.3, 0.4],
                                     [0.3, 0.4]])
        joint_velocities_ = controller_graph.joint_velocities(self.observation,
                                                              self.n_casters)

        with self.test_session():
            self.assertAllClose(joint_velocities, joint_velocities_, atol=1e-1)


class TestJacobian(tf.test.TestCase):
    def setUp(self):
        reset_graph()
        self.n_casters = 4

    def test_defaults_yield_correct_shape_and_values(self):
        b = 0.04
        beta = 0.785398
        h = 0.233345
        r = 0.02

        phi = 0.1

        inverse_jacobian = np.array([
            [-np.sin(phi)/b, np.cos(phi)/b, h*(np.cos(beta)*np.cos(phi) + np.sin(beta)*np.sin(phi))/b -1],
            [np.cos(phi)/r, np.sin(phi)/r, h*(np.cos(beta)*np.sin(phi) - np.sin(beta)*np.cos(phi))/r],
            [-np.sin(phi)/b, np.cos(phi)/b, h*(np.cos(beta)*np.cos(phi) + np.sin(beta)*np.sin(phi))/b]])

        caster_constants = controller_graph.create_caster_constats()
        observation = create_observation(self.n_casters)
        joint_positions = controller_graph.joint_positions(observation,
                                                           self.n_casters)
        inverse_jacobian_ = controller_graph.inverse_jacobian(
                                joint_positions[0], caster_constants[0])
        with self.test_session():
            self.assertAllClose(inverse_jacobian,
                                inverse_jacobian_.eval(),
                                atol=1e-5)


class TestCcCoupling(tf.test.TestCase):
    def test_defaults_yield_correct_shape_and_values(self):
        # TODO: test with joints
        cc_coupling = np.array([0.0, 0.0, 0.0])

        joint_positions = controller_graph.joint_positions(create_observation(4), 4)
        joint_velocities = controller_graph.joint_velocities(create_observation(4), 4)

        with self.test_session():
            cc_coupling_ = controller_graph.cc_coupling(joint_positions, joint_velocities)

        self.assertAllClose(cc_coupling, cc_coupling_)

# --- Observational space ---


class TestInertiaOperationalSpace(tf.test.TestCase):
    def test_yield_correct_shape_and_values(self):
        # TODO: check values
        inertia_operational_space = np.array([[0.372715, -1.223058, -0.263053],
                                              [-1.223058, 12.439786, 2.25188],
                                              [-0.263053, 2.251882, 0.414613]])

        inertia = controller_graph.cylinder_inertia(mass=1.0,
                                                    radius=0.2,
                                                    height=0.1)

        inverse_jacobian = create_inverse_jacobian()

        inertia_operational_space_ = controller_graph.inertia_operational_space(
                                        inverse_jacobian, inertia)

        with self.test_session():
            self.assertAllClose(inertia_operational_space,
                                inertia_operational_space_.eval())


class TestCcCouplingOperationalSpace(tf.test.TestCase):
    def test_yield_correct_shape_and_range(self):
        n_casters = 1
        observation = create_observation(n_casters)
        joint_positions = controller_graph.joint_positions(observation, n_casters)
        joint_velocities = controller_graph.joint_velocities(observation, n_casters)

        inertia = controller_graph.cylinder_inertia(mass=1.0,
                                                    radius=0.2,
                                                    height=0.1)

        base_velocity = tf.constant(np.ones(3), tf.float32)
        cc_coupling = controller_graph.cc_coupling(joint_positions,
                                                   joint_velocities)
        inverse_jacobian = create_inverse_jacobian()

        cc_coupling_os = controller_graph.cc_coupling_operational_space(
                        cc_coupling, inverse_jacobian, inertia, base_velocity)

        with self.test_session():
            self.assertShapeEqual(np.ones(3), cc_coupling_os)
            self.assertAllInRange(cc_coupling,
                                  lower_bound=-100.0,
                                  upper_bound=100.0)


class TestForceVectorEndEffector(tf.test.TestCase):
    def test_yield_correct_shape_and_range(self):
        n_casters = 1
        observation = create_observation(n_casters)
        joint_positions = controller_graph.joint_positions(observation, n_casters)
        joint_velocities = controller_graph.joint_velocities(observation, n_casters)
        inverse_jacobian = create_inverse_jacobian()
        base_acceleration = tf.constant(np.ones(3), tf.float32)
        base_velocity = tf.constant(np.ones(3), tf.float32)
        inertia = controller_graph.cylinder_inertia(mass=1.0,
                                                    radius=0.2,
                                                    height=0.1)
        cc_coupling = controller_graph.cc_coupling(joint_positions,
                                                   joint_velocities)

        inertia_os = controller_graph.inertia_operational_space(
                                        inverse_jacobian, inertia)
        cc_coupling_os = controller_graph.cc_coupling_operational_space(
                        cc_coupling, inverse_jacobian, inertia, base_velocity)

        force_vector_end_effector = controller_graph.force_vector_end_effector(
            inertia_os, base_acceleration, cc_coupling_os)

        with self.test_session():
            self.assertShapeEqual(np.ones(3), force_vector_end_effector)
            self.assertAllInRange(cc_coupling,
                                  lower_bound=-100.0,
                                  upper_bound=100.0)


class TestWheelConstrain(tf.test.TestCase):
    def test_yield_correct_shape_and_range(self):
        wheel_constraint = np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
        inverse_jacobian = create_inverse_jacobian()

        wheel_constraint_ = controller_graph.wheel_constraint(inverse_jacobian)

        with self.test_session():
            self.assertShapeEqual(wheel_constraint, wheel_constraint_)
            self.assertAllInRange(wheel_constraint_,
                                  lower_bound=-100.0,
                                  upper_bound=100.0)

# --- Controller ---


class testIntegrator(tf.test.TestCase):
    def test_yield_correct_shape_and_range(self):
        base_position = np.array([0.0, 0.0, 0.0])

        # with tf.variable_scope('base'):
        base_position_ = tf.get_variable('base_position', shape=3,
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer)

        base_velocity = tf.constant(np.array([0.1, 0.2, 0.3]), tf.float32)
        dt = tf.constant(0.1, dtype=tf.float32)

        integration_step = controller_graph.base_position_integration_step(
                                    base_position_, base_velocity, dt)
        with self.test_session():
            tf.global_variables_initializer().run()
            integration_step.eval()
            self.assertShapeEqual(base_position, base_position_.read_value())
            self.assertAllInRange(base_position_.read_value(),
                                  lower_bound=-100.0, upper_bound=100.0)

    def test_correct_position_after_one_step(self):
        base_position = np.array([0.01, 0.02, 0.03])

        base_position_ = tf.get_variable('base_position', shape=3,
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer)

        base_velocity = tf.constant(np.array([0.1, 0.2, 0.3]), tf.float32)
        dt = tf.constant(0.1, dtype=tf.float32)

        integration_step = controller_graph.base_position_integration_step(
                                    base_position_, base_velocity, dt)

        with self.test_session():
            tf.global_variables_initializer().run()
            integration_step.eval()
            self.assertAllClose(base_position,
                                base_position_.read_value())

    def test_correct_position_after_multiple_steps(self):
        base_position = np.array([0.1, 0.2, 0.3])

        base_position_ = tf.get_variable('base_position', shape=3,
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer)

        base_velocity = tf.constant(np.array([0.1, 0.2, 0.3]), tf.float32)
        dt = tf.constant(0.1, dtype=tf.float32)

        integration_step = controller_graph.base_position_integration_step(
                                    base_position_, base_velocity, dt)

        with self.test_session():
            tf.global_variables_initializer().run()
            for count in range(10):
                integration_step.eval()
            self.assertAllClose(base_position,
                                base_position_.read_value())


class TestCompensator(tf.test.TestCase):
    def test_yield_correct_shape_and_range(self):
        base_position = tf.constant(np.zeros(3), tf.float32)
        base_velocity = tf.constant(np.zeros(3), tf.float32)
        base_acceleration = tf.constant(np.zeros(3), tf.float32)

        base_position_desired = tf.constant(np.ones(3), tf.float32)
        base_velocity_desired = tf.constant(np.ones(3), tf.float32)

        kp = tf.constant(0.2, dtype=tf.float32)
        kv = tf.constant(0.6, dtype=tf.float32)

        compensator = controller_graph.compensator(base_position_desired,
                                                   base_velocity_desired,
                                                   base_position,
                                                   base_velocity,
                                                   base_acceleration,
                                                   kp, kv)

        with self.test_session():
            self.assertShapeEqual(np.ones(3), compensator)
            self.assertAllInRange(compensator, lower_bound=-100,
                                  upper_bound=100)


class TestPcvController(tf.test.TestCase):
    def test_yield_correct_shape_and_range(self):
        n_casters = 4
        observation = create_observation(n_casters)

        base_velocity_desired = tf.constant(np.zeros([n_casters, 3]),
                                            tf.float32,
                                            name='base_accelerations')

        joint_tourques = controller_graph.joint_tourques(observation,
                                                         base_velocity_desired)
        with self.test_session():
            self.assertShapeEqual(np.ones(n_casters*2), joint_tourques)
            self.assertAllInRange(joint_tourques, lower_bound=-100,
                                  upper_bound=100)


if __name__ == '__main__':
    tf.test.main()
