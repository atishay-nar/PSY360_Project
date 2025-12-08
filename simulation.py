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
def sim(ground_truth, prior, confirmation_multiplier, n_trials=100, drift=0.0):
    agent = BiasedBayesianObserver(prior, confirmation_multiplier)
    posterior = []
    truth_series = []
    for _ in range(n_trials):
        prediction = agent.prediction()
        evidence = np.random.binomial(n=1, p=ground_truth)
        agent.observe(evidence, prediction)
        posterior.append(agent.get_posterior())

        # optional drift in ground truth
        truth_series.append(ground_truth)
        if drift != 0.0:
            ground_truth = clamp(ground_truth + np.random.normal(0, drift))
    return np.array(posterior), np.array(truth_series)


if __name__ == "__main__":
    # configuration
    ground_truth = 0.5
    prior = np.array([0, 0])
    confirmation_multiplier = 1
    n_trials = 1000
    drift = 0.0

    # run simulation
    post_series, truth_series = sim(
        ground_truth, prior, confirmation_multiplier, n_trials, drift
    )
    error = np.abs(post_series - truth_series)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(post_series, color="b")
    ax1.plot(truth_series, color="g", linestyle="dashed")
    ax1.set_title("Posterior over Trials")

    ax2.plot(error, color="r")
    ax2.set_title("Error over Trials")
    ax2.hlines(0, 0, n_trials, colors="g", linestyles="dashed")
    plt.show()
