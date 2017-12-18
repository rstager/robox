from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

class RoboschoolReacher(RoboschoolMujocoXmlEnv):
    def __init__(self,xml='reacher8.xml'):
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),xml)
        RoboschoolMujocoXmlEnv.__init__(self, filename, 'body0', action_dim=1, obs_dim=1)
        tmp_obs=self._reset()
        print(tmp_obs.shape)
        high = np.ones([len(self.joints)])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([len(self.joints) * 4 + 6])
        high = np.inf * np.ones_like(tmp_obs)
        self.observation_space = gym.spaces.Box(-high, high)
        self.scene=None

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        self.jdict["target_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        #self.jdict["target_z"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.joints=[self.jdict["joint{}".format(i)] for i in range(100) if "joint{}".format(i) in self.jdict]
        self.lowerlimits,self.upperlimits,_,_= zip(*[joint.limits() for joint in self.joints])
        self.limitmask=np.greater(self.upperlimits,self.lowerlimits)

        for joint,ll,ul in zip(self.joints,self.lowerlimits,self.upperlimits):
            joint.reset_current_position(self.np_random.uniform(low=ll, high=ul),0)

        # joints[0] #base rotate
        # joints[1] # base elbow
        # joints[2] #shoulder rotate
        # joints[4] #shoulder bend
        # joints[5] # elbow bend
        # joints[6] #elbow twist
        # joints[7] #wrist
        # joints[8] #thumb


    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for joint,action in zip(self.joints,a):
            joint.set_motor_torque( 0.05*float(np.clip(action, -1, +1)) )

    def calc_state(self):
        #self.jointangledot = np.array([joint.current_relative_position() for joint in self.joints])
        self.jointangledot = np.array([joint.current_position() for joint in self.joints])
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        target_z=0
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
       #print("state {} {} {} {}".format(target_x,target_y,self.to_target_vec[0], self.to_target_vec[1]))
        # return np.array([target_x, target_y, self.to_target_vec[0], self.to_target_vec[1],
        #                  np.cos(self.jointangledot[0,0]),np.sin(self.jointangledot[0,0]),
        #                  self.jointangledot[0,1]*0.1,
        #                  self.jointangledot[1,0]/np.pi,
        #                  self.jointangledot[1,1]*0.1
        #                  ])
        return np.hstack([
            np.array([target_x,target_y,target_z,self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2]]),
            np.cos(self.jointangledot[:, 0]),
            np.sin(self.jointangledot[:, 0]),
            self.jointangledot[:, 0]/np.pi,
            self.jointangledot[:, 1]*.01,
            ])

    def calc_potential(self):
        return -100 * np.linalg.norm(self.to_target_vec)

    def _step(self, a):
        assert(not self.scene.multiplayer)
        self.apply_action(a)
        self.scene.global_step()

        state = self.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.calc_potential()
        #self.scene.cpp_world.test_window_print("a*jad {}".format(np.abs(a * self.jointangledot[:, 1])))
        #print("potent {} {} ecost {} {} {}".format((self.potential - potential_old),self.potential,np.abs(a * self.jointangledot[:, 1]),a,self.jointangledot[:,1]))
        electricity_cost = (
            -0.10 * (np.sum(np.abs(a * self.jointangledot[:, 1]*0.1)))  # work torque*angular_velocity
            - 0.1 * (np.sum(np.abs(a)))                                # stall torque require some energy
            )/len(self.joints)
        stuck=np.logical_and(self.limitmask,np.logical_or(np.greater_equal(self.jointangledot[:,0],np.array(self.upperlimits)-0.01),
                                                          np.less_equal(self.jointangledot[:,0],np.array(self.lowerlimits)+0.01)))
        stuck_joint_cost=-0.1 * np.sum(stuck)/len(self.joints)

        #stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.frame  += 1
        self.done   += 0 # (np.abs(self.potential)<5)
        self.reward += sum(self.rewards)
        self.HUD(state, a, False)
        #print("gymnreacher {} {} {} {}".format(self.rewards,a * self.jointangledot[:, 1],a,self.jointangledot[:,1]))
        return state, sum(self.rewards), self.done>0, {}

    def camera_adjust(self):
        x, y, z = self.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
