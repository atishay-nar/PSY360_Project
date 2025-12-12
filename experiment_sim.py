# run multiple simulations with different configurations, multiple times,
# average out MSE, and plot
import numpy as np
import matplotlib.pyplot as plt
from trial_sim import sim, mse
from scipy.stats import sem

# global configurations
ground_truth = 0.5
prior = np.array([1, 1])
n_flips = 1000
n_walks = 300

# different configurations to test
# tuples of form (multiplier, drift)
configs = [(1.00, 0.0), (1.1, 0.0), (1.25, 0.0), (1.50, 0.0), (1.75, 0.0), (2.00, 0.0)
           ,(1.00, 0.001), (1.1, 0.001), (1.25, 0.001), (1.50, 0.001), (1.75, 0.001), (2.00, 0.001)
           ,(1.00, 0.005), (1.1, 0.005), (1.25, 0.005), (1.50, 0.005), (1.75, 0.005), (2.00, 0.005)
           ,(1.00, 0.01), (1.1, 0.01), (1.25, 0.01), (1.50, 0.01), (1.75, 0.01), (2.00, 0.01)
          ] 

# store results by drift level for plots
results = {}

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

    # store mse and stderr for plotting later
    if drift not in results:
        results[drift] = {}
    results[drift][mult] = (avg_mse, stderr)
    print(f"Config (mult={mult}, drift={drift}): Avg MSE = {avg_mse:.4f} ± {stderr:.4f}")

    
#plot results

drift_levels = sorted(results.keys())
fig, ax = plt.subplots(2,2, figsize=(10, 10))
for i, drift in enumerate(drift_levels):
    mults = [str(i) for i in results[drift].keys()]
    avg_mses = [results[drift][m][0] for m in results[drift].keys()]
    stderrs = [results[drift][m][1] for m in results[drift].keys()]

    ax[i//2][i%2].bar(mults, avg_mses, yerr=stderrs, color='skyblue', capsize=5)
    ax[i//2][i%2].set_title(f'Drift = {drift}')
    ax[i//2][i%2].set_xlabel('Confirmation Multiplier')
    ax[i//2][i%2].set_ylabel('Average MSE')
plt.show()