import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "model_choice/exp-family/plots/1e6"
RESULTS_DIR = "model_choice/exp-family/results/1e6"
SIZES = [100, 1000]
EPSILON = 0.01

modelNames = ["Exponential", "Log-normal", "Gamma"]
modelComparisons = [(0, 1), (0, 2), (1, 2)]
for model in os.listdir(RESULTS_DIR):
    modelDirPath = os.path.join(RESULTS_DIR, model)
    for size in SIZES:
        trueLogBf = np.load(os.path.join(modelDirPath, "true/logbfs_size" + str(size)) + ".npy")
        abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
        for metricDir in os.listdir(os.path.join(modelDirPath)):
            if metricDir == "true":
                continue            
            modelCounts = np.load(os.path.join(modelDirPath, metricDir, abcDataName))
            fig, axs = plt.subplots(1, 3)
            for i in range(3):
                ax = axs[i]
                (m1, m2) = modelComparisons[i]
                metricLogBf = np.log(modelCounts[:, m1] / modelCounts[:, m2])
                ax.axline((0, 0), slope = 1, color = 'k')
                ax.plot(trueLogBf[:, i], metricLogBf, 'o')
                ax.title.set_text(modelNames[m1] + " vs " + modelNames[m2])
                ax.set_xlim((-1000, 1000))
                ax.set_ylim((-1000, 1000))
                
            handles, labels = ax.get_legend_handles_labels()
            fig.supxlabel("True Log BF")
            fig.supylabel("ABC Log BF")
            fig.suptitle("ABC with " + metricDir.upper())
            fig.tight_layout()
            
            figName = "bayes_factors_" + metricDir + ".png"            
            plt.savefig(os.path.join(PLOTS_DIR, model, "size_" + str(size), "bayes_factors", figName))
            plt.close(fig)
