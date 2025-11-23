import numpy as np
from bayesian_observer import BiasedBayesianObserver

if __name__ == "__main__":
    prior=np.array([0,0])
    agent = BiasedBayesianObserver(prior, confirmation_multiplier=2)