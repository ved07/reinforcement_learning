
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
