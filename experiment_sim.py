# run multiple simulations with different configurations, multiple times,
# average out MSE, and plot
import numpy as np
import matplotlib.pyplot as plt
from trial_sim import sim, mse
from scipy.stats import sem

# global configurations
ground_truth = 0.5
prior = np.array([8, 1])
n_flips = 100
n_walks = 300

# different configurations to test
# tuples of form (multiplier, drift)
configs = [(1, 0.0), (2, 0.0), (3, 0.0), 
           (1, 0.001), (2, 0.001), (3, 0.001), 
           (1, 0.005), (2, 0.005), (3, 0.005)] 

for config in configs:
    mult, drift = config
    mse_log = []
    for _ in range(n_walks):
        post_series, truth_series = sim(
            ground_truth, prior, mult, n_flips, drift
        )
        mse_log.append(mse(post_series, truth_series))
    avg_mse = np.mean(mse_log)
    stderr = sem(mse_log)
    print(f"Config (mult={mult}, drift={drift}): Avg MSE = {avg_mse:.4f} ± {stderr:.4f}")
