from os import chdir

import gym
from math import pi
import numpy as np
import h5py
from rnr.gym import rnrenvs
import local_envs


def run(envname,maxidx=10000):
    env = gym.make(envname)
    env.reset()
    img=env.unwrapped.getCameraObservation()
    with h5py.File(env.env.spec.id + "_images", 'w') as f:
        obs = f.create_dataset('obs', (maxidx,) +img.shape,'f',chunks=(10,) +img.shape,compression="gzip")
        d2 = f.create_dataset('d2', (maxidx,), 'f')
        idx=0
        for i_episode in range(200):
            observation = env.reset()
            for t in range(1000):
                idx += 1
                if idx == maxidx:
                    return

                action = env.action_space.sample()
                observation, r0, done, info = env.step(action)
                img = env.unwrapped.getCameraObservation()
                env.render()

                d=1
                d2[idx] = d ** 2
                f.attrs["maxidx"] = idx

                print(idx,r0,observation,np.mean(img))
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

if __name__ == "__main__":
    rnrenvs()
    run('KukaI-v1',10000)