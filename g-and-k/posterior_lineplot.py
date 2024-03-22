import numpy as np
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "model_choice/g-and-k/gk_test/results/m1_true"
PLOTS_DIR = "model_choice/g-and-k/gk_test/plots/m1_true"
EPSILON = 0.01
SIZES = [100, 1000]
NUM_OBSERVED = 100
metricDirs = ["cvm", "mmd", "wass", "qle"]

runIndices = list(range(1, NUM_OBSERVED + 1))
for size in SIZES:
    abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
    for metricDir in metricDirs:
        abcData = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
        posteriorProbs = abcData[:, 0] / np.sum(abcData, axis = 1)
        plt.plot(runIndices, posteriorProbs, 'o', label = metricDir.upper())

    for index in runIndices:
        plt.axvline(x = index, color = 'k')
        
    plt.xlabel("Run Number")
    plt.ylabel("Posterior Probability of M1")
    plt.legend(loc = "lower right")
    plt.savefig(os.path.join(PLOTS_DIR, "size_" + str(size), "posterior_lineplot_eps" + str(EPSILON) + ".png"))
    plt.clf()
