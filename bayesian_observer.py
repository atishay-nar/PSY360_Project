import numpy as np
class BiasedBayesianObserver:
    def __init__(self, prior, confirmation_multiplier=1):
        # prior: np.array([k,l]) to represents prior observed tails and heads
        # confirmation_multiplier: int factor to weight confirming evidence
        # we use MAP for our posterior probability estimation. first we initialize with our prior
        self.prior = prior
        self.confirmation_multiplier = confirmation_multiplier
        self.observations = np.array([0.0,0.0])
        # case where there is no prior
        if np.sum(prior) == 0:
            self.posterior = 0.5
        else:
            self.posterior = prior[1] / np.sum(prior)

    def observe(self, evidence, prediction):
        # if evidence confirms prediction, weight it more
        if evidence == prediction:
            self.observations[evidence] += self.confirmation_multiplier
        else:
            self.observations[evidence] += 1
        # update posterior distribution
        self.posterior = (self.observations[1] + self.prior[1]) / (np.sum(self.observations) + np.sum(self.prior))

    def prediction(self):
        # use posterior as bernoulli probability of a biased coin landing on heads
        return np.random.binomial(n=1, p=self.posterior)

    def get_posterior(self):
        return self.posterior

if __name__ == "__main__":
    prior = np.array([1.0, 1.0])
    agent = BiasedBayesianObserver(prior)
    print(f"prior: Heads: {prior[1]}, Tails: {prior[0]}")
    print(f"posterior with no observations: {agent.get_posterior()}")
    prediction = agent.prediction()
    print(f"prediction with no observations: {prediction}")
    # simulate some observations
    print("Lets say we flip the coin and observe a head")
    agent.observe(evidence=1,prediction=prediction)
    print(f"posterior after observing a head: {agent.get_posterior():.2f}")