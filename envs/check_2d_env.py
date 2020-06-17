#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:25:04 2020

@author: orivlin
"""

import numpy as np
from nav_2d import Nav2D
import matplotlib.pyplot as plt


N = 20
Nobs = 50
Dobs = 1
Rmin = 5
env = Nav2D(N, Nobs, Dobs, Rmin)

state = env.reset()
init_state = state.copy()

fig = plt.figure()
plt.imshow(init_state)

for i in range(10):
    action = np.random.randint(0, 4, (1,))[0]
    state, reward, done = env.step(state, action)
    #plt.imshow(state)
    #plt.show()
    
fig2 = plt.figure()
plt.imshow(state)
