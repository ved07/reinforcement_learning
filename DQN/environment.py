import torch
import torch.nn
import gym
from collections import deque
import Networks
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
env = gym.make('SpaceInvaders-v0')
env.reset()
screen = env.render(mode='rgb_array')

im = Image.fromarray(screen)
im.save("screen1.jpeg")
for x in range(4): env.step(env.action_space.sample()); env.render()
screen2 = env.render(mode='rgb_array')
im = Image.fromarray(screen2)
im.save("screen2.jpeg")
screen2 = torch.from_numpy(screen2)