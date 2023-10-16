import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import *

DATA_DIR = "model_choice/normal/results/eq3_vs_neq3"
SAVE_DIR = "model_choice/normal/plots/eq3_vs_neq3"
EPSILON = 0.01
NULL_MEAN = 3
VARIANCE = 1
PRIOR_VARIANCE_SCALE = 100
SIZES = np.linspace(10, 1000, 100)

logBayesFactors = np.load(os.path.join(DATA_DIR, "abc_log_bayes_factors_eps-" + str(EPSILON) + ".npy"))
exactLogBayesFactors = logBayesFactors[0, :]

colCount = 2
rowCount = np.round((logBayesFactors.shape()[0] - 1) / colCount)
fig, axs = plt.subplot((rowCount, colCount))
plt.plot(SIZES, exactLogBayesFactors, label = "Exact", linestyle = "dashed")
for i in range(1, exactLogBayesFactors.shape[0]):
    plt.plot(SIZES, exactLogBayesFactors[i, :], label = DistanceMetric(i-1).name)

plt.xlabel("Sample Size")
plt.ylabel("Log Bayes Factor (" + r'$eps = {}$)'.format(eps = EPSILON))
plt.legend()
plt.title(r'$Y \sim N(3, 1) \;, H_0: \mu = {m0}, H_1: \mu \neq {m0}, \; \sigma^2 = {sigmaSq}$ (known), c = {c}'.format(m0 = NULL_MEAN, sigmaSq = VARIANCE, c = PRIOR_VARIANCE_SCALE))
plt.show()
# plt.savefig(os.path.join(SAVE_DIR, "log_bayes_factor_plot_eps-" + str(EPSILON) + ".npy"))
