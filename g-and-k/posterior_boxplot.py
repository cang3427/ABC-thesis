import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *

RESULTS_DIR = "g-and-k/results/m2_true"
PLOTS_DIR = "g-and-k/plots/m2_true"
EPSILONS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
SIZES = [100, 1000]
NUM_OBSERVED = 100

metricDirs = ["cvm", "mmd", "wass", "qle"]
for size in SIZES:
    for eps in EPSILONS:
        posteriorsList = []
        for metricDir in metricDirs:            
            abcDataName = "size" + str(size) + "eps" + str(eps) + ".npy"
            posteriors = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
            posteriorsList.append(posteriors[:, 0])
            
        boxplot = plt.boxplot(posteriorsList, labels = [METRIC_LABELS[metric] for metric in metricDirs], 
                              patch_artist = True, 
                              boxprops = dict(facecolor = "lightgrey", edgecolor = 'k'), 
                              medianprops = dict(color = 'k', linewidth = 2), 
                              whiskerprops=dict(linestyle='--'),
                              widths = 0.75)
        
        plt.ylim((0, 1))
        plt.xlabel('ABC Method')
        plt.ylabel('Posterior Probability of ' + r'$M_1$')
        plt.savefig(os.path.join(PLOTS_DIR, "size_" + str(size), "boxplots", "posterior_boxplot_eps" + str(eps) + ".png"))
        plt.clf()
