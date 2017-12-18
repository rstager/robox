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
from baselines.pposgd import pposgd_simple
import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import sys

from rnr.util import mkchdir

num_cpu=1
import roboschool
from evalmodel import eval_model

trained_pi=None

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


    with open('config','wb') as f:
        pickle.dump((env.spec.id,type(pi),dict,hid_layers,vhid_layers,obfilter),f)

def train(env_id, num_timesteps, seed):
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent":
        return
    import baselines.common.tf_util as U
    sess = U.make_session(4)
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    hid_layers=[128,64]+[64]*env.action_space.shape[0]
    vhid_layers=[128,64]+[64]*env.action_space.shape[0]
    print(logger.get_dir())
    env = bench.Monitor(env, osp.join(".", "{}.monitor.json".format(rank)), allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)
    print("env:{}".format(env.spec.id))

    if False:
        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                             hid_layers=hid_layers, vhid_layers=vhid_layers)
        trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,callback=callback)
    elif False:
        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                             hid_layers=hid_layers, vhid_layers=vhid_layers)
        pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,callback=callback
        )
    elif True:
        from baselines.acktr.acktr_cont import learn
        from baselines.acktr.policies import GaussianMlpPolicy
        from baselines.acktr.value_functions import NeuralNetValueFunction
        with tf.Session(config=tf.ConfigProto()) as session:
            ob_dim = env.observation_space.shape[0]
            ac_dim = env.action_space.shape[0]
            with tf.variable_scope("vf"):
                vf = NeuralNetValueFunction(ob_dim, ac_dim)
            with tf.variable_scope("pi"):
                policy = GaussianMlpPolicy(ob_dim, ac_dim,hid_layers=[64]*env.action_space.shape[0])

            learn(env, policy=policy, vf=vf,
                  gamma=0.99, lam=0.97, timesteps_per_batch=2500,
                  desired_kl=0.002,
                  num_timesteps=num_timesteps, animate=False,callback=callback)
    else:
        pass

    eval_model(sess,env,trained_pi)

def main():
    import local_envs
    with tf.device("/cpu:0"):
        train('RoboschoolReacherTest4-v1', num_timesteps=1e8, seed=0)

if __name__ == '__main__':
    mkchdir('experiments/latest')

    main()
