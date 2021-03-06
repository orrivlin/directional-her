#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:56:22 2020

@author: orivlin
"""

import numpy as np
from envs.nav_2d import Nav2D
from envs.simple_env import Simple2D
from models.conv_nets import DiscreteActor
import matplotlib.pyplot as plt
from discrete_sac import DiscreteSAC
from time import sleep


#N = 10
#Nobs = 50
#Dobs = 1
#Rmin = 5
#env = Nav2D(N, Nobs, Dobs, Rmin)

N = 10
Nobs = 0
Dobs = 1
Rmin = 5
env = Simple2D(N, Nobs, Dobs, Rmin)

agent = DiscreteSAC(env)

state = env.reset()
init_state = state.copy()

for i in range(10):
    X = env.get_tensor(state, agent.device)
    action = agent.policy.act(X)
    state, reward, done, _ = env.step(action)
    plt.imshow(np.transpose(state, (1, 2, 0)))
    plt.show()
    print(0)
    sleep(0.2)

#for i in range(100):
#    agent.iteration()
