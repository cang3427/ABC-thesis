import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "model_choice/exp-family/plots"
RESULTS_DIR = "model_choice/exp-family/results"
SIZES = [100, 1000]
EPSILON = 0.1
PARAMS_DIR = "params_sampled"

modelColours = ["blue", "orange", "green"]
modelNames = ["Exponential", "Log-normal", "Gamma"]
for model in os.listdir(RESULTS_DIR):
    modelDirPath = os.path.join(RESULTS_DIR, model, PARAMS_DIR)
    for size in SIZES:
        truePosterior = np.load(os.path.join(modelDirPath, "true/posteriors_size" + str(size)) + ".npy")
        abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
        for metricDir in os.listdir(os.path.join(modelDirPath)):
            if metricDir == "true":
                continue            
            modelCounts = np.load(os.path.join(modelDirPath, metricDir, abcDataName))
            totalCounts = np.sum(modelCounts, axis = 1)
            fig, axs = plt.subplots(1, 3)
            for i in range(3):
                ax = axs[i]
                metricPosterior = modelCounts[:, i] / totalCounts
                ax.plot([0, 1], [0, 1], color = 'k')
                ax.plot(truePosterior[:, i], metricPosterior, 'o', color = modelColours[i])
                ax.set_xlim((-0.1, 1.1))
                ax.set_ylim((-0.1, 1.1))
                ax.title.set_text(modelNames[i])
                
            fig.supxlabel("True Posterior Probability")
            fig.supylabel("ABC Posterior Probability")
            fig.suptitle("ABC with " + metricDir.upper())
            fig.tight_layout()
            
            plotDir = os.path.join(PLOTS_DIR, model, PARAMS_DIR, "size_" + str(size), "posteriors", metricDir)
            if not os.path.exists(plotDir):
                os.mkdir(plotDir)
                
            figName = "posterior_probs_eps" + str(EPSILON) + ".png"            
            plt.savefig(os.path.join(plotDir, figName))
            plt.close(fig)
