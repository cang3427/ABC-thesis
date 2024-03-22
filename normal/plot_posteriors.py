import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "normal/plots/m2/known_var"
RESULTS_DIR = "normal/results/m2/known_var"
SIZES = [100, 1000]
EPSILONS = [0.01, 0.025, 0.05, 0.1]

coloursDict = {"cvm" : "blue", "mmd" : "orange", "wass" : "green", "stat" : "red"}
for size in SIZES:
    for eps in EPSILONS:
        truePosterior = np.load(os.path.join(RESULTS_DIR, "true/posteriors_size" + str(size)) + ".npy")
        abcDataName = "size" + str(size) + "eps" + str(eps) + ".npy"
        for metricDir in os.listdir(os.path.join(RESULTS_DIR)):           
            if metricDir == "true" or metricDir == "other":
                continue
            
            modelCounts = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
            totalCounts = np.sum(modelCounts, axis = 1)
            posteriorProb = modelCounts[:, 0] / totalCounts
            
            plt.plot([0, 1], [0, 1], color = 'k')
            plt.plot(truePosterior[:, 0], posteriorProb, 'o', color = coloursDict[metricDir])
            plt.xlim((-0.1, 1.1))
            plt.ylim((-0.1, 1.1))
            plt.xlabel("True Posterior Probability of M1")
            plt.ylabel("ABC Posterior Probability of M1")
            plt.title("ABC with " + metricDir.upper())
            
            plotDir = os.path.join(PLOTS_DIR, "size_" + str(size), "posteriors", metricDir)
            if not os.path.exists(plotDir):
                os.mkdir(plotDir)
                
            figName = "posterior_probs_eps" + str(eps) + ".png"            
            plt.savefig(os.path.join(plotDir, figName))
            plt.clf()
