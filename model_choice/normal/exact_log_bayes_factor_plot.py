import numpy as np
import matplotlib.pyplot as plt
from exact_log_bayes_factor import *
import os

DATA_DIR = "g-and-k/model_choice/data/normal"
SAVE_DIR = "g-and-k/model_choice/results/normal_hypothesis_plots"

# Observed samples were taken from a N(3, 1)
sampleSizes = np.linspace(10, 1000, 100).astype(int)
m0 = 3
sigmaSq = 1
cs = [10**(x) for x in range(5)]
for c in cs:
    logBayesFactorMeans = []
    for size in sampleSizes:
        logBayesFactors = np.zeros(1)
        for i in range(1):
            # sample = np.load(os.path.join(DATA_DIR, "sample0size" + str(size) + ".npy"))
            sample = np.random.normal(3, 1, size)
            logBayesFactor = exact_log_bayes_factor(sample, m0, sigmaSq, c)
            logBayesFactors[i] = logBayesFactor
        logBayesFactorMeans.append(np.mean(logBayesFactors))
        
    plt.plot(sampleSizes, logBayesFactorMeans, label = str(c))

plt.axhline(y = 0, linestyle = 'dashed', color = 'y')
plt.ticklabel_format(style = "plain", axis = "y")
plt.legend(title = r'$c$')
plt.xlabel("n")
plt.ylabel("Mean Log Bayes Factor")
plt.title("Observed" + r'$\sim N(3, 1) \;, H_0: \mu = {m0}, H_1: \mu \neq {m0}, \; \sigma^2 = {sigmaSq}$ (known)'.format(m0 = m0, sigmaSq = sigmaSq))
# plt.savefig(os.path.join(SAVE_DIR, "null" + str(m0) + ".png"))
plt.show()

