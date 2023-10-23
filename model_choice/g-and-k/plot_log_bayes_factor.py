import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import *

DATA_DIR = "model_choice/g-and-k/results/test_a/eq3_vs_neq3"
SAVE_DIR = "model_choice/g-and-k/plots/test_a/eq3_vs_neq3"
NULL_MEAN = 3
VARIANCE = 1
PRIOR_VARIANCE_SCALE = 100
SIZES = np.delete(np.linspace(10, 1000, 100).astype(int)[:50], 4)
EPSILONS = [0.01, 0.025, 0.05, 0.1]

for testDir in os.listdir(DATA_DIR):
    for epsilon in EPSILONS:
        logBayesFactors = np.load(os.path.join(DATA_DIR, testDir, "abc_log_bayes_factors_eps-" + str(epsilon) + ".npy"))
        for i in range(logBayesFactors.shape[0]):
            plt.plot(SIZES, logBayesFactors[i, :], label = DistanceMetric(i).name)
        plt.xlabel("Sample Size")
        plt.ylabel(r'Log Bayes Factor ($\varepsilon = {eps}$)'.format(eps = epsilon))
        plt.legend(loc = 'lower right')
        plt.ylim(-6, 6)
        paramTypes = []
        for i in range(3):
            paramType = "known"
            if testDir[i] == '0':
                paramType = "unknown"
            paramTypes.append(paramType)
                
        plt.title(r'$Y \sim gk(3, 1, 2, 0.5) \;, H_0: a = 3, H_1: a \neq 3, \; b$ {bType}, $g$ {gType}, $k$ {kType}'.format(bType = paramTypes[0], gType = paramTypes[1], kType = paramTypes[2]))
        plt.savefig(os.path.join(SAVE_DIR, testDir, "log_bayes_factor_plot_eps-" + str(epsilon) + ".png"))
        plt.clf()
