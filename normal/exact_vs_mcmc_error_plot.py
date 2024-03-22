from exact_log_bayes_factor import *
import os
import numpy as np
import matplotlib.pyplot as plt

OBSERVED_DIR = "model_choice/normal/observed_data"
SAVE_DIR = "model_choice/normal/plots/eq3_vs_neq3/exact_vs_mcmc"
MCMC_SAMPLES = 100_000
MCMC_PATH = "model_choice/normal/results/eq3_vs_neq3/mcmc/log_bayes_factors_" + str(MCMC_SAMPLES) + ".npy"
SIZES = np.linspace(10, 1000, 100).astype(int)
NULL_MEAN = 3
VARIANCE = 1
PRIOR_VARIANCE_SCALE = 100

if __name__ == '__main__':
    exactLogBayesFactors = np.zeros(len(SIZES))
    for i, size in enumerate(SIZES):
        observedPath = os.path.join(OBSERVED_DIR, "sample0size" + str(size) + ".npy")
        observed = np.load(observedPath)
        exactLogBayesFactors[i] = exact_log_bayes_factor(observed, NULL_MEAN, VARIANCE, PRIOR_VARIANCE_SCALE)
        
    mcmcLogBayesFactors = np.load(MCMC_PATH)
    errors = exactLogBayesFactors- mcmcLogBayesFactors
    plt.scatter(SIZES, errors, color = 'g')
    plt.axhline(linestyle = 'dashed', color = 'k')
    plt.xlabel('Sample Size')
    plt.ylabel('Log Bayes Factor Error (samples = ' + str(MCMC_SAMPLES) + ')')
    plt.ylim(-6, 6)
    plt.title(r'$Y \sim N(3, 1) \;, H_0: \mu = {m0}, H_1: \mu \neq {m0}, \; \sigma^2 = {sigmaSq}$ (known), c = {c}'.format(m0 = NULL_MEAN, sigmaSq = VARIANCE, c = PRIOR_VARIANCE_SCALE))
    plt.savefig(os.path.join(SAVE_DIR, "exact_vs_pymc_error_" + str(MCMC_SAMPLES) + ".png"))
