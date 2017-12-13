#!/usr/bin/env python
# noinspection PyUnresolvedReferences
#import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
import os
from OpenGL import GLU
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
from baselines.pposgd.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
from baselines.trpo_mpi.trpo_mpi import global_render
import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import sys
num_cpu=1
import roboschool
from evalmodel import eval_model

trained_pi=None

def callback(l,g):
    global trained_pi,global_render
    iters=l['iters_so_far']
    global_render=False
    if iters%10 !=0:
        return
    global_render=True
    pi=l['pi']
    trained_pi=pi
    env=l['env']
    #U.save_state('./pi.tf')
    dict = {}
    for tfv in pi.get_variables():
        dict[tfv.name] = tfv.eval()
    with open('config','wb') as f:
        pickle.dump((env.spec.id,dict,pi.hid_layers,pi.vhid_layers),f)

def train(env_id, num_timesteps, seed):
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent":
        return
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    hid_layers=[128,64]
    vhid_layers=[128,64]
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_layers=hid_layers,vhid_layers=vhid_layers)

    print(logger.get_dir())
    env = bench.Monitor(env, osp.join(".", "{}.monitor.json".format(rank)), allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,callback=callback)

    eval_model(sess,env,trained_pi)

def main():
    gym.envs.registration.register(
        id='RoboschoolReacherTest-v1',
        entry_point='gym_nreacher:RoboschoolReacher',
        max_episode_steps=150,
        reward_threshold=18.0,
        kwargs={'xml':'reacher5.xml'}
    )
    with tf.device("/cpu:0"):
        train('RoboschoolReacherTest-v1', num_timesteps=1e7, seed=0)

if __name__ == '__main__':
    os.chdir('experiments/latest')

    main()
