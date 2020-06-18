#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:33:44 2020

@author: orivlin
"""



import torch
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib as plt
from copy import deepcopy as dc
from gym import spaces



class Simple2D:
    def __init__(self,N,Nobs,Dobs,Rmin):
        self.N = N
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.Rmin = Rmin
        self.state_dim = [N,N,3]
        self.action_dim = 4
        self.scale = 10.0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, N, N))
        self.action_space = spaces.Discrete(4)
        
    def get_dims(self):
        return self.state_dim, self.action_dim
        
    def reset(self):
        grid = np.zeros((self.N,self.N,3))
        for i in range(self.Nobs):
            center = np.random.randint(0,self.N,(1,2))
            minX = np.maximum(center[0,0] - self.Dobs,1)
            minY = np.maximum(center[0,1] - self.Dobs,1)
            maxX = np.minimum(center[0,0] + self.Dobs,self.N-1)
            maxY = np.minimum(center[0,1] + self.Dobs,self.N-1)
            grid[minX:maxX,minY:maxY,0] = 1.0
            
        free_idx = np.argwhere(grid[:,:,0] == 0.0)
        start = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
        while (True):
            finish = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
            if ((start[0] != finish[0]) and (start[1] != finish[1]) and (np.linalg.norm(start - finish) >= self.Rmin)):
                break
        grid[start[0],start[1],1] = self.scale*1.0
        grid[finish[0],finish[1],2] = self.scale*1.0
        self.grid = grid
        self.n_step = 0
        return np.transpose(grid, (2, 0, 1))
    
    def step(self, action):
        grid = self.grid.copy()
        new_grid = dc(grid)
        done = False
        self.n_step += 1
        if self.n_step + 1 >= self.N:
            done = True
        reward = 0.0
        act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        pos = np.argwhere(grid[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(grid[:,:,2] == self.scale*1.0)[0]
        new_pos = pos + act[action]
                
        if (np.any(new_pos < 0.0) or np.any(new_pos > (self.N - 1)) or (grid[new_pos[0],new_pos[1],0] == 1.0)):
            self.grid = grid
            return np.transpose(grid, (2, 0, 1)), reward, done, dict()
        
        reward = 0
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = self.scale*1.0
        
        reward = np.linalg.norm([new_pos[0] - target[0], new_pos[1] - target[1]]) - np.linalg.norm([pos[0] - target[0], pos[1] - target[1]])
        if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
            reward = 1.0
            done = True
        self.grid = new_grid
        return np.transpose(grid, (2, 0, 1)), reward, done, dict()
    
    # def get_tensor(self):
    #     #S = torch.Tensor(self.grid).transpose(2,1).transpose(1,0).unsqueeze(0)
    #     S = torch.Tensor(self.grid).unsqueeze(0)
    #     return S

    def get_tensor(self, state, device):
        S = torch.Tensor(state).unsqueeze(0).to(device)
        return S
    
    def get_goal(self, state):
        return state[:, :, 2]
    
    def set_goal(self, state, goal):
        st = state.copy()
        st[:, :, 2] = goal
        return st
    
    def render(self,grid):
        plot = imshow(grid)
        return plot