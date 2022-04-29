# importing dependencies
import torch
import torch.nn
import gym
from collections import deque, namedtuple
import Networks
import random
import numpy as np

# initialise space invaders environment
env = gym.make('SpaceInvaders-v0')
env.reset()

# CONSTANTS
ACTION_SPACE = 6
STEPS = 4
EPSILON_DECAY = 0.9995
EPSILON_LIM = 1

targetDeepQ = Networks.DeepQ(input_dim=3, output_dim=ACTION_SPACE)
policyDeepQ = Networks.DeepQ(input_dim=3, output_dim=ACTION_SPACE)
epsilon = 1


def envStep(state, action):
    reward = 0
    for x in range(STEPS):
        observation, interim_rew, done, info = env.step(action)
        if done:
            reward = 0
        reward += interim_rew
    state_prime = env.render(mode='rgb_array').transpose((2, 0, 1))
    state_prime = torch.from_numpy(np.ascontiguousarray(state_prime, dtype=np.float32) / 255)
    sars = (state, action, reward, state_prime)
    return sars


def chooseAction(network, state, epsilonDecay, epsilonLimit, eps):
    Qvector = network(state)
    if random.randrange(0,1)>eps:
        action = env.action_space.sample()
    else:
        action = torch.argmax(Qvector)

    sars = envStep(state, action)
    if eps <= epsilonLimit:
        eps = epsilonLimit
    else:
        eps *= epsilonDecay
    return sars, eps, Qvector

transitionTuple = namedtuple('sars', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        self.memory.append(transitionTuple(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def experienceReplay(replayMemory, policyDQ):
    sars = replayMemory.sample(1)[0]
    policyDQ()