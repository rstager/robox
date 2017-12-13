from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

class RoboschoolReacher(RoboschoolMujocoXmlEnv):
    def __init__(self,xml='reacher8.xml'):
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)),xml)
        RoboschoolMujocoXmlEnv.__init__(self, filename, 'body0', action_dim=6, obs_dim=24)
        # for r in self.mjcf:
        #     for j in r.joints:
        #         print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    TARG_LIMIT = 0.27
    def robot_specific_reset(self):
        self.jdict["target_x"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-self.TARG_LIMIT, high=self.TARG_LIMIT ), 0)
        self.fingertip = self.parts["fingertip"]
        self.target    = self.parts["target"]
        self.joints=[self.jdict["joint{}".format(i)] for i in range(9) if "joint{}".format(i) in self.jdict]
        self.lowerlimits,self.upperlimits,_,_= zip(*[joint.limits() for joint in self.joints])
        self.np_random.uniform(low=-3.14, high=3.14)
        for joint,ll,ul in zip(self.joints,self.lowerlimits,self.upperlimits):
            joint.reset_current_position(self.np_random.uniform(low=ll, high=ul),0)

        # for joint,value in zip(self.joints,[1,1,1,-2.7,2.7,0,1,0]):
        #     joint.reset_current_position(value,0)
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
            joint.set_motor_torque( 0.001*float(np.clip(action, -1, +1)) )

    def calc_state(self):
        self.state = np.array([joint.current_relative_position() for joint in self.joints])
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        target_z=0
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
       #print("state {} {} {} {}".format(target_x,target_y,self.to_target_vec[0], self.to_target_vec[1]))
        return np.hstack([
            np.array([target_x,target_y,target_z,self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2]]),
            np.cos(self.state[:,0]),
            np.sin(self.state[:,0]),
            self.state[:, 1],
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

        electricity_cost = (
            -0.10*(np.sum(np.abs(a*self.state[:,1])))  # work torque*angular_velocity
            -0.01*(np.sum(np.abs(a)))                                # stall torque require some energy
            )
        stuck_joint_cost = -0.1*(np.sum(np.isclose(self.state[:,0],self.upperlimits,0.01))
                           +np.sum(np.isclose(self.state[:,0],self.lowerlimits,0.01)))
        # if stuck_joint_cost !=0 :
        #     print(np.isclose(self.state[:,0],self.upperlimits,0.01))
        #     print(np.isclose(self.state[:,0],self.lowerlimits,0.01))
        #     print("Stuck joint {}".format(stuck_joint_cost))
        #stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
        self.frame  += 1
        self.done   += 0
        self.reward += sum(self.rewards)
        self.HUD(state, a, False)
        return state, sum(self.rewards), False, {}

    def camera_adjust(self):
        x, y, z = self.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
