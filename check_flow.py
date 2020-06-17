#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:56:22 2020

@author: orivlin
"""

import numpy as np
from envs.nav_2d import Nav2D
from models.conv_nets import Conv2Vec
import matplotlib.pyplot as plt


N = 20
Nobs = 50
Dobs = 1
Rmin = 5
env = Nav2D(N, Nobs, Dobs, Rmin)
net = Conv2Vec(3, 4)

state = env.reset()
init_state = state.copy()

#fig = plt.figure()
#plt.imshow(init_state)

for i in range(10):
    X = env.get_tensor(state.copy())
    pi = net(X)
    action = np.random.randint(0, 4, (1,))[0]
    state, reward, done = env.step(state, action)
    #plt.imshow(state)
    #plt.show()
    
#fig2 = plt.figure()
#plt.imshow(state)