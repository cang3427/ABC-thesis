import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "model_choice/exp-family/plots/1e6"
RESULTS_DIR = "model_choice/exp-family/results/1e6"
SIZES = [100, 1000]
EPSILON = 0.1

modelNames = ["Exponential", "Log-normal", "Gamma"]
coloursDict = {"cvm" : "blue", "mmd" : "orange", "wass" : "green", "stat" : "red"}
for model in os.listdir(RESULTS_DIR):
    modelDirPath = os.path.join(RESULTS_DIR, model)
    for size in SIZES:
        truePosterior = np.load(os.path.join(modelDirPath, "true/posteriors_size" + str(size)) + ".npy")
        abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
        fig, axs = plt.subplots(1, 3)
        for i in range(3):
            axs[i].plot([0, 1], [0, 1], color = 'k')
        for metricDir in os.listdir(modelDirPath):
            if metricDir == "true":
                continue            
            modelCounts = np.load(os.path.join(modelDirPath, metricDir, abcDataName))
            totalCounts = np.sum(modelCounts, axis = 1)
            for i in range(3):
                ax = axs[i]
                metricPosterior = modelCounts[:, i] / totalCounts
                ax.plot(truePosterior[:, i], metricPosterior, 'o', color = coloursDict[metricDir], label = metricDir.upper(), alpha = 0.5)
                ax.set_xlim((-0.1, 1.1))
                ax.set_ylim((-0.1, 1.1))
                ax.title.set_text(modelNames[i])
                
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'upper left', fontsize = 'small')
        fig.supxlabel("True Posterior Probability")
        fig.supylabel("ABC Posterior Probability")
        fig.tight_layout()
            
        figName = "combined_posterior_probs_eps" + str(EPSILON) + ".png" 
        plt.savefig(os.path.join(PLOTS_DIR, model, "size_" + str(size), "posteriors", figName))
        plt.close(fig)
