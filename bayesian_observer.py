import numpy as np
class BiasedBayesianObserver:
    def __init__(self, prior, confirmation_multiplier=1):
        # prior: np.array([k,l]) to represents prior observed tails and heads
        # confirmation_multiplier: factor to weight confirming evidence
        # we use MAP for our posterior probability estimation. first we initialize with our prior
        self.prior = prior
        self.confirmation_multiplier = confirmation_multiplier
        self.observations = np.array([0,0])
        self.posterior = prior / np.sum(prior)

    def observe(self, evidence, prediction):
        # if evidence confirms prediction, weight it more
        if evidence == prediction:
            self.observations[evidence] += self.confirmation_multiplier
        else:
            self.observations[evidence] += 1
        # update posterior distribution
        self.posterior = (self.observations[1] + self.prior[1]) / (np.sum(self.observations) + np.sum(self.prior))

    def prediction(self):
        # MAP estimation
        return np.random.binomial(1, self.posterior)
    
    def get_posterior(self):
        return self.posterior

if __name__ == "__main__":
    prior = np.array([1, 1])
    agent = BiasedBayesianObserver(prior)
    print(agent.prediction())