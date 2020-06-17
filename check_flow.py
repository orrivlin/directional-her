#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:56:22 2020

@author: orivlin
"""

import numpy as np
from envs.nav_2d import Nav2D
from models.conv_nets import DiscreteActor
import matplotlib.pyplot as plt
from discrete_sac import DiscreteSAC


N = 20
Nobs = 50
Dobs = 1
Rmin = 5
env = Nav2D(N, Nobs, Dobs, Rmin)

agent = DiscreteSAC(env)

state = env.reset()
init_state = state.copy()

for i in range(10):
    X = env.get_tensor(state, agent.device)
    action = agent.policy.act(X)
    state, reward, done, _ = env.step(action)


agent.train_episode()
