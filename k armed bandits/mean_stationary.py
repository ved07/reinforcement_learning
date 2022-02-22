"""
This file tracks the K-Armed-Bandits problem wherein there exist K 'levers' with random rewards normally
distributed about a mean. It employs a sample-average algorithm to construct estimations which aim to converge at the
mean. If the problem is not stationary then all the samples take a 'random walk' at each time step and a weighted sample
average is employed instead
"""
# Import Libraries
import random
import numpy as np
"""
Constructing the K-Armed-Bandits problem:
    There are K arms, defined by the constants NUM_ARMS
    The mean of the mean rewards is MEAN, and rewards are normally distributed with a standard deviation of STD_DEV
    Bound for arms mean defines the range of values in which the mean value for the K armed bandit can fall
"""
# set random seed
random.seed(10)

# mean value of all 'mean' rewards
MEAN = 0
# standard deviation for the normally distributed rewards about the randomly generated means
STD_DEV = 1
# number of arms for the k-armed bandit (i.e. k)
NUM_ARMS = 10
# plus minus bounds for the mean rewards that will be sampled from
ARMS_MEAN_BOUND = 4
# Boolean describing if the situation is stationary
STATIONARY = False
# change based on a walk
NON_STATIONARY_WALKS = 0.1
# Generate the mean randomly
arms_means = [random.uniform(-ARMS_MEAN_BOUND, ARMS_MEAN_BOUND) for x in range(NUM_ARMS-1)]
# Use pre-written means for testing
# arms_means = [1, -3, 0, 2, 3, 2, 3, -2, -4]

# final value is tuned so that the mean of all the arms is 0
arms_means.append(0-sum(arms_means))


# normal distribution sampling function (subroutine used so that other PDFs may be used instead)
def normal_dist(stdev, mean):
    return np.random.normal(mean, stdev)


# obtains reward by sampling from the distribution based on the selected arm
def get_reward(arm, distribution):
    reward_value = distribution(stdev=STD_DEV, mean=arms_means[arm])

    if not STATIONARY:
        # cause random walks through manipulating the mean values randomly, the problem is now non-stationary
        for i in range(len(arms_means)):
            arms_means[i] += random.uniform(-NON_STATIONARY_WALKS, NON_STATIONARY_WALKS)

    return reward_value

        



"""
Constructing estimator system.
Define NUM_EPOCHS and EPSILON_EXPLORATION for the model, this will loop n times, updating the mean estimations and thus
constructing an estimation of the whole k-armed system, the estimations are compared with the real means at the end.
"""
# Number of epochs or iteration the model will iterate for and the probability it explores rather than exploits
NUM_EPOCHS = 1000000
EPSILON_EXPLORATION = 0.01
ALPHA_DISCOUNT = 0.6
INITIAL_ESTIMATE = 5
estimates = [INITIAL_ESTIMATE*i for i in range(NUM_ARMS)]

total_reward = 0


def make_estimate(estimations):
    if random.uniform(0, 1) > EPSILON_EXPLORATION:
        greedy = max(estimations)
        return greedy, estimations.index(greedy)
    else:
        epsilon_index = random.randint(0, NUM_ARMS-1)
        exploration = estimations[epsilon_index]
        return exploration, epsilon_index


for epoch in range(1, 1+NUM_EPOCHS):
    estimate, index = make_estimate(estimates)

    reward = get_reward(index, normal_dist)
    total_reward += reward
    estimates[index] += 1/epoch * ALPHA_DISCOUNT * (reward-estimate)


print("the array of reward means: {} had an estimated reward array of {}\n".format(arms_means, estimates))
est_loss = [abs(arms_means[i]-estimates[i]) for i in range(10)]
print("estimates were off by: {} accruing a total loss of: {}\n".format(est_loss, sum(est_loss)))
max_index = arms_means.index(max(arms_means))
max_est_ind = estimates.index(max(estimates))
print("max index:{}\nestimated max index:{}\n".format(max_index, max_est_ind))
print("mean reward obtained: {}".format(total_reward/NUM_EPOCHS))
