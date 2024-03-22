import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *

RESULTS_DIR = "g-and-k/results/m2_true"
PLOTS_DIR = "g-and-k/plots/m2_true/combined"
EPSILONS = [0.01]
SIZES = [100, 1000]
NUM_OBSERVED = 100

metricDirs = ["cvm", "mmd", "wass", "qle"]
plotLabels = ["(a)", "(b)"]
for eps in EPSILONS:
    fig, axs = plt.subplots(1, len(SIZES), figsize = (10, 6))
    for ax, label, size in zip(axs, plotLabels, SIZES):
        posteriorsList = []
        for metricDir in metricDirs:            
            abcDataName = "size" + str(size) + "eps" + str(eps) + ".npy"
            posteriors = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
            posteriorsList.append(posteriors[:, 0])
            
        ax.boxplot(posteriorsList, labels = [METRIC_LABELS[metric] for metric in metricDirs], 
                   patch_artist = True, 
                   boxprops = dict(facecolor = "lightgrey", edgecolor = 'k'), 
                   medianprops = dict(color = 'k', linewidth = 2), 
                   whiskerprops=dict(linestyle='--'),
                   widths = 0.75)
        
        ax.set_ylim((0, 1))
        ax.set_xlabel(label)
    
    fig.supylabel('Posterior Probability of ' + r'$M_1$', fontsize = 'large')
    plt.savefig(os.path.join(PLOTS_DIR, "combined_posterior_boxplot_eps" + str(eps) + ".png"))
    plt.clf()
