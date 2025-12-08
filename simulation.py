import numpy as np
from bayesian_observer import BiasedBayesianObserver
import matplotlib.pyplot as plt
# inputs the ground truth probability of heads, prior, confirmation multiplier, and number of trials
# outputs a series of posteriors over trials as a np.array
def sim(ground_truth, prior, confirmation_multiplier, n_trials=100):
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
    ground_truth = 0.5
    prior= np.array([0,0])
    confirmation_multiplier = 2
    n_trials = 1000

    # run simulation
    series = sim(ground_truth, prior, confirmation_multiplier, n_trials)
    error = np.abs(series - ground_truth)
    print(f"final result: posterior after {n_trials} trials: {series[-1]:.2f}")

    # plot
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(series, color='b')
    axes[1].plot(error, color='r')
    plt.show()