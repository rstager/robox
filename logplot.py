from rnr.util import mkchdir
import json
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from os import chdir
'''
{"env_id":"RoboschoolReacherTest3-v1","t_start":1513298334.5278396606,"gym_version":"0.9.4"}
{"l":150,"r":6.5733718575,"t":1.059802}
'''
def run():
    tbl={}
    with open('0.monitor.json') as f:
        for line in f:
            data =json.loads(line)
            for n,v in data.items():
                if n not in tbl:
                    tbl[n]=[]
                tbl[n].append(v)
    for n,v in tbl.items():
        print(n)

    plt.plot(tbl['r'], label='reward')
    N=100
    plt.plot(np.convolve(tbl['r'], np.ones((N,)) / N, mode='valid'), label='mean')

    plt.legend()
    plt.show(1)

    plt.show()

if __name__ == '__main__':
    chdir('experiments/latest')
    run()