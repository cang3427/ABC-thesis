import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *

RESULTS_DIR = "g-and-k/results/m2_true"
PLOTS_DIR = "g-and-k/plots/m2_true"
EPSILON = 0.01
SIZES = [100, 1000]
NUM_OBSERVED = 100

metricDirs = ["cvm", "mmd", "wass", "qle"]
fig, axs = plt.subplots(1, len(SIZES), figsize = (10, 6))
plotLabels = ["(a)", "(b)"]
for ax, label, size in zip(axs, plotLabels, SIZES):
    abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
    posteriorProbs = []
    dataDict = {}
    for metricDir in metricDirs:
        posteriors = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
        dataDict[METRIC_LABELS[metricDir]] = posteriors[:, 0]
        
    df = pd.DataFrame(dataDict)
    pairplot = sns.pairplot(df, diag_kws={'bins': 10})
    
    for i, axGrp in enumerate(pairplot.axes):
        for j, axSub in enumerate(axGrp):
            axSub.set_xlim((-0.1, 1.1))
            axSub.set_ylim((-0.1, 1.1))
            if i != j:
                axSub.plot([0, 1], [0, 1], color = 'k', zorder = -1)
    
    plt.show()
    # plt.savefig(os.path.join(PLOTS_DIR, "size_" + str(size), "pairplots", "posterior_pairplot_eps" + str(EPSILON) + ".png"))
    