"""
This file tracks the non-stationary K-Armed-Bandits problem wherein there exist K 'levers' with random rewards normally
distributed about a mean, at each step, each mean takes a 'random walk'.
It employs a weighted-sample average that focuses on the most

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

# Generate the mean randomly
arms_means = [random.uniform(-ARMS_MEAN_BOUND, ARMS_MEAN_BOUND) for x in range(NUM_ARMS - 1)]
# Use pre-written means for testing
# arms_means = [1, -3, 0, 2, 3, 2, 3, -2, -4]

# final value is tuned so that the mean of all the arms is 0
arms_means.append(0 - sum(arms_means))


# normal distribution sampling function (subroutine used so that other PDFs may be used instead)
def normal_dist(stdev, mean):
    return np.random.normal(mean, stdev)


# obtains reward by sampling from the distribution based on the selected arm
def get_reward(arm, means, distribution):
    return distribution(stdev=STD_DEV, mean=means[arm])


"""                                                                                                                     
Constructing estimator system.                                                                                          
Define NUM_EPOCHS and EPSILON_EXPLORATION for the model, this will loop n times, updating the mean estimations and thus 
constructing an estimation of the whole k-armed system, the estimations are compared with the real means at the end.    
"""
# Number of epochs or iteration the model will iterate for and the probability it explores rather than exploits
NUM_EPOCHS = 1000000
EPSILON_EXPLORATION = 0.3

estimates = [0 * i for i in range(NUM_ARMS)]

reward = 0


def make_estimate(estimations):
    if random.uniform(0, 1) > EPSILON_EXPLORATION:
        greedy = max(estimations)
        return greedy, estimations.index(greedy)
    else:
        epsilon_index = random.randint(0, NUM_ARMS - 1)
        exploration = estimations[epsilon_index]
        return exploration, epsilon_index


for epoch in range(1, 1 + NUM_EPOCHS):
    estimate, index = make_estimate(estimates)

    reward = get_reward(index, arms_means, normal_dist)
    estimates[index] += 1 / epoch * (reward - estimate)

print("the array of reward means: {} had an estimated reward array of {}".format(arms_means, estimates))
est_loss = [abs(arms_means[i] - estimates[i]) for i in range(10)]
print("estimates were off by: {} accruing a total loss of: {}".format(est_loss, sum(est_loss)))
max_index = arms_means.index(max(arms_means))
max_est_ind = estimates.index(max(estimates))
print("max index:{}\nestimated max index:{}".format(max_index, max_est_ind))
