from OpenGL import GLU
import os

import gym
import pickle
import tensorflow as tf
import baselines.common.tf_util as U

import roboschool
from baselines.pposgd.mlp_policy import MlpPolicy


def eval_model(sess,env,pi):
    #loader.restore(sess, tf.train.latest_checkpoint(model_dir_path))
    ob = env.reset()
    while True:
        still_open = env.render("human")
        ac, vpred = pi.act(True, ob)
        ob, rew, new, _ = env.step(ac)
        if new:
            ob = env.reset()

def run():
    with open('config','rb') as f:
        env_id,pi_values,hid_layers,vhid_layers=pickle.load(f)
    env = gym.make(env_id)
    print("envid={}".format(env.spec.id))
    with tf.device("/cpu:0"):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            pi = MlpPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space,
                       hid_layers=hid_layers,vhid_layers=vhid_layers)

            for tfv in pi.get_variables():
                tfv.assign(pi_values[tfv.name]).eval()
            eval_model(sess,env,pi)
    env.close()

if __name__ == '__main__':
    gym.envs.registration.register(
        id='RoboschoolReacherTest-v1',
        entry_point='gym_nreacher:RoboschoolReacher',
        max_episode_steps=150,
        reward_threshold=18.0,
        kwargs={'xml':'reacher5.xml'}
    )
    os.chdir('experiments/latest')
    run()