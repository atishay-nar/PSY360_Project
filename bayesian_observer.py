import numpy as np
class BiasedBayesianObserver:
    def __init__(self, prior, confirmation_multiplier=1):
        # prior: numpy array representing prior belief regarding the distribution of an event
        # confirmation_multiplier: factor to weight confirming evidence
        self.prior = prior
        self.confirmation_multiplier = confirmation_multiplier

    def observe(self, evidence, prediction):
        if evidence == prediction:
            for i in range(self.confirmation_multiplier):
                self.observe(evidence, abs(prediction-1))
        else:
            # figure out bayesian mechanics

    def prediction(self):
        #figure out how to run prediction (MAP method?)

