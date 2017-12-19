from OpenGL import GLU
import os
import numpy as np
import gym
import pickle
import tensorflow as tf
import baselines.common.tf_util as U

import roboschool
from baselines.acktr.acktr_cont import rollout
from baselines.acktr.filters import ZFilter
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.pposgd.mlp_policy import MlpPolicy

tfvs={}


def eval_model(sess,env,pi, obfilter=None):
    #loader.restore(sess, tf.train.latest_checkpoint(model_dir_path))
    ob = env.reset()
    if obfilter: ob = obfilter(ob)
    prev_ob = np.float32(np.zeros(ob.shape))

    while True:
        still_open = env.render("human")
        if type(pi) == GaussianMlpPolicy:
            ac, ac_dist, logp = pi.act(np.concatenate([ob, prev_ob], -1))
            scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
            ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        else:
            ac, vpred = pi.act(True,ob)
        prev_ob = np.copy(ob)
        ob, rew, new, _ = env.step(ac)
        if obfilter: ob = obfilter(ob)
        if new:
            ob = env.reset()
            break

def run():
    filename='config'

    with open(filename,'rb') as f:
        env_id,pitype,tfv,hid_layers,vhid_layers,obfilter=pickle.load(f)
    print(pitype)
    env = gym.make(env_id)
    env.reset()
    print("envid={} obs {} act {}".format(env.spec.id,env.observation_space.shape,env.action_space.shape))
    #obfilter=ZFilter(env.observation_space.shape)
    with tf.device("/cpu:0"):
        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            if False:
                pi = MlpPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space,
                       hid_layers=hid_layers,vhid_layers=vhid_layers)
            else:
                with tf.variable_scope("pi"):
                    pi=GaussianMlpPolicy(env.observation_space.shape[0],env.action_space.shape[0],hid_layers=hid_layers)
            sess.run(tf.global_variables_initializer())
            while True:
                mtime = os.stat(filename).st_mtime
                with open(filename, 'rb') as f:
                    _,_, pi_values, _, _,obfilter = pickle.load(f)

                for tfv in tf.global_variables():
                    #print("{}={}".format(tfv.name,pi_values[tfv.name].shape,pi_values[tfv.name]))
                    tfv.assign(pi_values[tfv.name]).eval()
                    tfvs[tfv.name]=tfv

                while mtime==os.stat(filename).st_mtime:
                    #rollout(env, pi, 150, animate=True,obfilter=obfilter)
                    eval_model(sess,env,pi,obfilter=obfilter )


    env.close()

if __name__ == '__main__':
    import local_envs

    os.chdir('experiments/latest')
    run()