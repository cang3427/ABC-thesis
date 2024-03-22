import numpy as np
from scipy.stats import norm
from math import sqrt, log, exp, pi

def exact_log_bayes_factor(sample, nullMean, variance, priorVarianceScale):
    n = len(sample)
    stdSampleMean = (np.mean(sample) - nullMean) / sqrt(variance / n)
    bayesFactor = 0.5 * log(n * priorVarianceScale + 1) - 0.5 * n / (n + 1/priorVarianceScale) * stdSampleMean ** 2
    return bayesFactor

def posterior_probs(sample, nullMean, variance, priorVarianceScale):
    bayesFactor = exp(exact_log_bayes_factor(sample, nullMean, variance, priorVarianceScale))
    posteriorProbM1 = bayesFactor / (bayesFactor + 1)
    
    return np.array([posteriorProbM1, 1 - posteriorProbM1])