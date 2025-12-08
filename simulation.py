import numpy as np
from bayesian_observer import BiasedBayesianObserver
import matplotlib.pyplot as plt

# helper function to clamp probabilities between 0 and 1 in drift model
def clamp(x):
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    return x

# inputs the ground truth probability of heads, prior, confirmation multiplier, and number of trials
# outputs a series of posteriors over trials as a np.array
def sim(ground_truth, prior, confirmation_multiplier, n_trials=100, drift=0):
    agent = BiasedBayesianObserver(prior, confirmation_multiplier)
    posterior = []
    for _ in range(n_trials):
        prediction = agent.prediction()
        evidence = np.random.binomial(n=1, p=ground_truth)
        agent.observe(evidence, prediction)
        posterior.append(agent.get_posterior())
    return np.array(posterior)


if __name__ == "__main__":
    # configuration
    ground_truth = 0.9
    prior = np.array([0, 0])
    confirmation_multiplier = 2
    n_trials = 100

    # run simulation
    series = sim(ground_truth, prior, confirmation_multiplier, n_trials)
    error = np.abs(series - ground_truth)
    # plot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(series, color="b")
    ax1.set_title("Posterior over Trials")
    ax1.hlines(ground_truth, 0, n_trials, colors="g", linestyles="dashed")

    ax2.plot(error, color="r")
    ax2.set_title("Error over Trials")
    ax2.hlines(0, 0, n_trials, colors="g", linestyles="dashed")
    plt.show()
