# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
import types

import gym
import tensorflow as tf
from gym import RewardWrapper, ObservationWrapper
from gym.spaces import Box
from pybullet_envs.bullet import KukaGymEnv

from rnr.util import mkchdir
import local_envs

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

#from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
import pybullet_envs
import time
import pickle


def callback(l,g):
    global trained_pi,global_render
    #print(l.keys())
    if 'iters_so_far' in l:
        iters=l['iters_so_far']
        if iters % 10 != 0:
            return
    if 'pi' in l:
        pi=l['pi']
    else:
        pi=l['policy']
    hid_layers=pi.hid_layers if hasattr(pi,'hid_layers') else None
    vhid_layers=pi.vhid_layers if hasattr(pi,'vhid_layers') else None
    env=l['env']
    dict = {}
    obfilter=l['obfilter']
    for tfv in tf.global_variables():
        dict[tfv.name] = tfv.eval()
    for tfv in tf.local_variables():
        dict[tfv.name] = tfv.eval()
    reward_scale=env.unwrapped._reward_scale if hasattr(env.unwrapped,'__reward_scale') else 1.0

    with open('config','wb') as f:
        pickle.dump((env.spec,type(pi),dict,hid_layers,vhid_layers,obfilter,reward_scale),f)

num_timesteps=1000
def i2g(self,sensors):
    #print("i2g {} {}".format(sensors.shape,self.unwrapped._cam_dist))
    return sensors

def main():
    spec=gym.spec('KukaI-v1')
    spec._kwargs.update({'renders':True,'isDiscrete':False})
    env = spec.make()
    env = ObservationWrapper(env)
    env._observation=types.MethodType(i2g, env)
    env = RewardWrapper(env)
    env._reward=lambda reward:reward/env.reward_scale
    env.reward_scale=100
    env.offset=0

    motorsIds = []

    dv = 1
    # motorsIds.append(env.unwrapped._p.addUserDebugParameter("posX", -dv, dv, 0))
    # motorsIds.append(env.unwrapped._p.addUserDebugParameter("posY", -dv, dv, 0))
    # motorsIds.append(env.unwrapped._p.addUserDebugParameter("posZ", -dv, dv, 0))
    # motorsIds.append(env.unwrapped._p.addUserDebugParameter("yaw", -dv, dv, 0))
    # motorsIds.append(env.unwrapped._p.addUserDebugParameter("fingerAngle", 0, 0.3, .3))

    from baselines.acktr.acktr_cont import learn
    from baselines.acktr.policies import GaussianMlpPolicy
    from baselines.acktr.value_functions import NeuralNetValueFunction
    with tf.Session(config=tf.ConfigProto()) as session:
        if isinstance(env.observation_space, Box):
            ob_dim = env.observation_space.shape[0]
        else:
            ob_dim = env.observation_space.spaces[0].shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim,hid_layers=[64]*env.action_space.shape[0])

        learn(env, policy=policy, vf=vf,
              gamma=0.99, lam=0.97, timesteps_per_batch=2500,
              desired_kl=0.002,
              num_timesteps=1e8, animate=False,callback=callback)



    done = False
    while (not done):

        action = []
        for motorId in motorsIds:
            action.append(env.unwrapped._p.readUserDebugParameter(motorId))

        state, reward, done, info = env.step(action)


if __name__ == "__main__":

    mkchdir('experiments/latest')
    with tf.device("/cpu:0"):
        main()
