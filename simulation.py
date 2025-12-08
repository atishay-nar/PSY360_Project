import numpy as np
from bayesian_observer import BiasedBayesianObserver
import matplotlib.pyplot as plt
# inputs the ground truth probability of heads, prior, confirmation multiplier, and number of trials
# outputs a series of posteriors over trials as a np.array
def sim(ground_truth, prior, confirmation_multiplier, n_trials=100):
    agent = BiasedBayesianObserver(prior, confirmation_multiplier)
    posterior = []
    post_error = []
    for _ in range(n_trials):
        prediction = agent.prediction()
        evidence = np.random.binomial(n=1, p=ground_truth)
        agent.observe(evidence, prediction)
        posterior.append(agent.get_posterior())
        post_error.append(abs(ground_truth - agent.get_posterior()))
    return np.array(posterior), np.array(post_error)    

if __name__ == "__main__":
    # configuration
    ground_truth = 0.5
    prior= np.array([0,0])
    confirmation_multiplier = 1
    n_trials = 1000

    # run simulation
    series, error = sim(ground_truth, prior, confirmation_multiplier, n_trials)
    print(f"final result: posterior after {n_trials} trials: {series[-1]:.2f}")
    plt.plot(series)
    plt.plot(error)
    plt.show()