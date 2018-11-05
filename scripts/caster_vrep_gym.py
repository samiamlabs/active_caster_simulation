import numpy as np

from gym import spaces
from vrep_env import vrep_env
from vrep_env import vrep

import os
vrep_scenes_path = os.environ['VREP_SCENES_PATH']


class CasterBaseVrepEnv(vrep_env.VrepEnv):
    metadata = {
        'render.modes': [],
    }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=vrep_scenes_path + '/caster_base.ttt'
    ):

        vrep_env.VrepEnv.__init__(self, server_addr, server_port, scene_path)

        joint_names = [
            'fl_caster_wheel_joint',
            'fr_caster_wheel_joint',
            'rl_caster_wheel_joint',
            'rr_caster_wheel_joint',
            'fl_caster_steer_joint',
            'fr_caster_steer_joint',
            'rl_caster_steer_joint',
            'rr_caster_steer_joint',
        ]

        shape_names = [
            'base_link_visual',
        ]

        self.joint_handles = list(map(self.get_object_handle, joint_names))
        self.shape_handles = list(map(self.get_object_handle, shape_names))

        num_actions = len(self.joint_handles)
        num_observations = (len(self.joint_handles))

        self.joints_max_velocity = 10000.0
        actions = np.array([self.joints_max_velocity] * num_actions)
        observations = np.array([np.inf] * num_observations)

        self.action_space = spaces.Box(-actions, actions)
        self.observation_space = spaces.Box(-observations, observations)

        print('CasterBaseVrepEnv: initialized')

    def _make_observation(self):
        """
        Query V-rep to make observation.

        The observation is stored in self.observation
        """
        observation_list = []

        # q (joint positions)
        for joint_handle in self.joint_handles:
            observation_list += [self.obj_get_joint_angle(joint_handle)]

        # delta q (joint_velocities)
        for joint_handle in self.joint_handles:
            observation_list += [self.obj_get_joint_velocity(joint_handle)]

        # ground truth odometry
        # for object_handle_index in self.shape_handles:
        #     observation_list += self.obj_get_position(
        #         object_handle_index, relative_to=vrep.sim_handle_parent)
        #     lin_vel, ang_vel = self.obj_get_velocity(object_handle_index)
        #     observation_list += ang_vel
        #     observation_list += lin_vel

        self.observation = np.array(observation_list).astype('float32')

    def obj_get_joint_velocity(self, handle):
        """Get velocity of joint from V-rep."""
        return self.RAPI_rc(vrep.simxGetObjectFloatParameter(
            self.cID, handle, 2012, self.opM_get))[0]

    def _make_action(self, action):
        """
        Query V-rep to make action.

        no return value
        """
        for joint_handle, joint_force in zip(self.joint_handles, action):
            self.obj_set_joint_force(joint_handle, -joint_force)

    def obj_set_joint_force(self, joint_handle, joint_force):
        """Apply force to joint."""
        joint_velocity = np.sign(joint_force) * 1000.0
        self.RAPI_rc(vrep.simxSetJointTargetVelocity(
                    self.cID,
                    joint_handle,
                    joint_velocity,
                    self.opM_set))

        return self.RAPI_rc(vrep.simxSetJointForce(
                            self.cID,
                            joint_handle,
                            abs(joint_force),
                            self.opM_set))

    def step(self, action):
        """Gym environment 'step'."""
        assert self.action_space.contains(
            action), "Action {} ({}) is invalid".format(action, type(action))

        # Actuate
        self._make_action(action)
        # Step
        self.step_simulation()
        # Observe
        self._make_observation()

        reward = 0.0

        done = False

        return self.observation, reward, done, {}

    def reset(self):
        """Gym environment 'reset'."""
        if self.sim_running:
            self.stop_simulation()

        self.dt = 0.01
        # need to set simulation time step to custom in v-rep (10 ms)
        self.set_float_parameter(
            vrep.sim_floatparam_simulation_time_step, self.dt)

        self.start_simulation()

        self._make_observation()
        return self.observation

    def render(self, mode='human', close=False):
        """Gym environment 'render'."""
        pass

    def seed(self, seed=None):
        """Gym environment 'seed'."""
        return []
