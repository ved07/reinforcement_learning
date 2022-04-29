import torch
import torch.nn
import gym
from collections import deque, namedtuple
import Networks
import random
import numpy as np

env = gym.make('SpaceInvaders-v0')
env.reset()
screen = env.render(mode='rgb_array').transpose((2, 0, 1))
screen = torch.from_numpy(np.ascontiguousarray(screen, dtype=np.float32)/255)
# the action space is one of 9 actions
actions = [i for i in range(9)]

targetDeepQ = Networks.DeepQ(input_dim=3, output_dim=9)
policyDeepQ = Networks.DeepQ(input_dim=3, output_dim=9)

STEPS = 4
def make_pred(state, action):
    reward = 0
    for x in range(STEPS):
        obs, rew, done, info = env.step(action)
        reward += rew
    state_prime = env.render(mode='rgb_array').transpose((2, 0, 1))
    state_prime = torch.from_numpy(np.ascontiguousarray(state_prime, dtype=np.float32) / 255)
    sars = (state, action, reward, state_prime)
    return sars


transition = namedtuple('sars', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        self.memory.append(transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
Memory = ReplayMemory(1000)


def experience_replay(memory, policyDQ, targetDQ, optimizer, GAMMA):
    sars = memory.sample(1)
    state, action, reward, state_prime = sars


def make_action(network, state, epsilon, epsilonDecay, epsilonLimit):
    Qvector = network(state)
    if random.randrange(0,1)>epsilon:
        action = env.action_space.sample()
    else:
        action = torch.argmax(Qvector)

    sars = make_pred(state, action)
    if epsilon <= epsilonLimit:
        epsilon = epsilonLimit
    else:
        epsilon *= epsilonDecay
    return sars, epsilon, Qvector




def train_model(n_epochs, targetDQ, policyDQ,optimizer, GAMMA, epsilon, epsilon_limit, epsilon_decay):
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        reward = 0
        done = False
        state = env.render(mode='rgb_array').transpose((2, 0, 1))
        state = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32) / 255)
        """Making optimal decision: implement epsilon greedy here"""
        sars, epsilon, Qvector = make_action(policyDQ, state,epsilon, epsilon_decay, epsilon_limit)
        action = sars[1]
        reward = sars[2]
        state_prime = sars[3]
        criterion = torch.nn.SmoothL1Loss()
        Qvalue = Qvector[action]
        TargetQvalue = reward + GAMMA*targetDQ(state_prime)
        loss = criterion(Qvalue, TargetQvalue)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            targetDQ.load_state_dict(policyDQ.state_dict())

        return policyDQ

