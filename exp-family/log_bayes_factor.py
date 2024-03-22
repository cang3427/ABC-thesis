import numpy as np
from math import exp, pi, log, nan, inf
from scipy.special import gamma, loggamma
from abc_model_choice_exp_family import Model

def marginal_log_likelihood(data, model):
    n = len(data)
    if model == Model.EXP:
        return (loggamma(n + 1) - (n + 1) * log(1 + np.sum(data)))
    if model == Model.LOGNORM:
        logSum = np.sum(np.log(data))
        dataSum = logSum**2 / (2 * (n + 1)) - logSum - np.sum(np.log(data)**2) / 2
        return (dataSum - n / 2 * log(2*pi) - 0.5 * log(n + 1))
    if model == Model.GAMMA:
        return (np.sum(np.log(data)) + loggamma(2*n + 1) - (2*n + 1) * log(1 + np.sum(data)))
    
def posterior_prob(data, model):
    marginalLogLikelihoods = [marginal_log_likelihood(data, m) for m in Model]
    maxLogLikelihood = max(marginalLogLikelihoods)
    logPosteriorProb = marginalLogLikelihoods[model.value] - (maxLogLikelihood + log(np.sum([exp(marginalLogLikelihoods[i] - maxLogLikelihood) for i in range(3)])))

    return exp(logPosteriorProb)

def log_bayes_factor(data, m1, m2):
    return marginal_log_likelihood(data, m1) - marginal_log_likelihood(data, m2)
