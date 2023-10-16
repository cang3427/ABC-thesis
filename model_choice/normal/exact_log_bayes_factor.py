import numpy as np
from math import sqrt, log

def exact_log_bayes_factor(sample, nullMean, variance, priorVarianceScale):
    n = len(sample)
    stdSampleMean = (np.mean(sample) - nullMean) / sqrt(variance / n)
    bayesFactor = 0.5 * log(n * priorVarianceScale + 1) - 0.5 * n / (n + 1/priorVarianceScale) * stdSampleMean ** 2
    return bayesFactor