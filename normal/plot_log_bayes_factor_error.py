import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import *

NULL_MEAN = 3
VARIANCE = None
PRIOR_VARIANCE_SCALE = 100
SIZES = np.linspace(10, 1000, 100).astype(int)
EPSILONS = [0.01, 0.025, 0.05, 0.1]
if VARIANCE is None:
    DATA_DIR = "model_choice/normal/results/mean_test/eq3_vs_neq3/unknown_var"
    SAVE_DIR = "model_choice/normal/plots/mean_test/eq3_vs_neq3/unknown_var"
else:
    DATA_DIR = "model_choice/normal/results/mean_test/eq3_vs_neq3/known_var/with_mmd"
    SAVE_DIR = "model_choice/normal/plots/mean_test/eq3_vs_neq3/known_var"
    
if VARIANCE is None:
    benchmarkLogBayesFactors = np.load(os.path.join(DATA_DIR, "mcmc/log_bayes_factors_100000.npy"))
    benchmarkLabel = "PyMC"
    estimatedStr = "Estimated"
    abcStart = 0
else:
    benchmarkLogBayesFactors = np.load(os.path.join(DATA_DIR, "abc_log_bayes_factors_eps-" + str(0.01) + ".npy"))[0, :]
    benchmarkLabel = "Exact"
    estimatedStr = ""
    abcStart = 1

for epsilon in EPSILONS:  
    logBayesFactors = np.load(os.path.join(DATA_DIR, "abc_log_bayes_factors_eps-" + str(epsilon) + ".npy"))
    plt.axhline(y = 0, linestyle = 'dashed', color = 'k')
    for i in range(abcStart, logBayesFactors.shape[0]):
        logBayesFactorErrors = benchmarkLogBayesFactors - logBayesFactors[i, :]
        plt.plot(SIZES, logBayesFactorErrors, 'o', label = DistanceMetric(i-abcStart).name)
    plt.xlabel("Sample Size")
    plt.ylabel("Log Bayes Factor Error " + estimatedStr + "(" + r'$\varepsilon = {eps}$)'.format(eps = epsilon))
    plt.legend(loc = 'lower right')
    plt.ylim(-6, 6)
    plt.title(r'$Y \sim N(3, 1) \;, H_0: \mu = {m0}, H_1: \mu \neq {m0}, \; \sigma^2$ (unknown), c = {c}'.format(m0 = NULL_MEAN, sigmaSq = VARIANCE, c = PRIOR_VARIANCE_SCALE))
    plt.savefig(os.path.join(SAVE_DIR, "log_bayes_factor_error_plot_eps-" + str(epsilon) + ".png"))
    plt.clf()
        