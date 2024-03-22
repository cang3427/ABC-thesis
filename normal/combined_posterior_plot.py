import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "normal/plots/known_var/params_sampled"
RESULTS_DIR = "normal/results"
SIZE = 1000
EPSILON = 0.01
METRIC_DIRS = ["cvm", "mmd", "wass", "stat"]
COLORS = {"m1": 'k', "m2": 'gray'}

metricLabels = {"cvm": "CvM", "mmd": "MMD", "wass": "Wass", "stat": "Stat"}
modelLabels = {"m1": r'$M_0$', "m2": r'$M_1$'}
fig, axs = plt.subplots(2, 2, figsize=(8,8), layout = 'constrained')
for ax, metric in zip(axs.flat, METRIC_DIRS):
    ax.plot([0, 1], [0, 1], color = 'k')
    ax.set_aspect('equal')
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
    ax.set_title("ABC-" + metricLabels[metric], fontsize = "large")

abcDataName = "size" + str(SIZE) + "eps" + str(EPSILON) + ".npy"
for modelDir in os.listdir(RESULTS_DIR):
    if modelDir == "m1":
        resultsPath = os.path.join(RESULTS_DIR, modelDir, "known_var")
    else:
        resultsPath = os.path.join(RESULTS_DIR, modelDir, "known_var/params_sampled")

    truePosteriors = np.load(os.path.join(resultsPath, "true/posteriors_size" + str(SIZE)) + ".npy")
    for ax, metricDir in zip(axs.flat, METRIC_DIRS):  
        if modelDir == "m1":
            modelCounts = np.load(os.path.join(resultsPath, metricDir, abcDataName))
            totalCounts = np.sum(modelCounts, axis = 1)
            posteriorProbs = modelCounts[:, 0] / totalCounts
        else:
            posteriorProbs = np.load(os.path.join(resultsPath, metricDir, abcDataName))[:, 0]
        ax.plot(truePosteriors[:, 0], posteriorProbs, '*', color = COLORS[modelDir], label = modelLabels[modelDir])

for ax in axs.flat:
    ax.legend(loc = 'lower right')
fig.supxlabel("True Posterior Probability of " + r'$M_0$', fontsize = 'large')
fig.supylabel("ABC Posterior Probability of " + r'$M_0$', fontsize = 'large')
plt.savefig(os.path.join(PLOTS_DIR, "combined_posterior_plot_size" + str(SIZE) + ".png"))