import numpy as np
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "normal/results/m2/unknown_var"
PLOTS_DIR = "normal/plots/m2/unknown_var"
EPSILONS = [0.01, 0.025, 0.05, 0.1]
SIZES = [100, 1000]
NUM_OBSERVED = 100

metricDirs = ["cvm", "mmd", "wass", "stat"]
metricColours = ["blue", "orange", "green", "red"]
for eps in EPSILONS:
    for size in SIZES:
        abcDataName = "size" + str(size) + "eps" + str(eps) + ".npy"
        posteriorProbs = []
        for metricDir in metricDirs:
            abcData = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
            posteriorProbs.append(abcData[:, 0] / np.sum(abcData, axis = 1))
            
        boxplot = plt.boxplot(posteriorProbs, labels = [metric.upper() for metric in metricDirs], patch_artist = True, medianprops = dict(color = 'black'))
        for box, color in zip(boxplot['boxes'], metricColours):
            box.set(facecolor = color)
        
        plt.ylim((0, 1))
        plt.xlabel('Methods')
        plt.ylabel('Posterior Probability of M1')
        plt.savefig(os.path.join(PLOTS_DIR, "size_" + str(size), "posteriors", "posterior_boxplot_eps" + str(eps) + ".png"))
        plt.clf()
