import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "./model_choice/normal/plots/mean_test/eq3_vs_neq3/known_var"
RESULTS_DIR = "./model_choice/normal/results/mean_test/eq3_vs_neq3/known_var"
SIZES = [100]
EPSILON = 0.1

coloursDict = {"cvm" : "blue", "mmd" : "orange", "wass" : "green", "aux_" : "red"}
for size in SIZES:
    truePosterior = np.load(os.path.join(RESULTS_DIR, "true/posteriors_size" + str(size)) + ".npy")
    abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
    plt.plot([0, 1], [0, 1], color = 'k')
    for metricDir in os.listdir(os.path.join(RESULTS_DIR)):
        if metricDir == "true" or metricDir == "other":
            continue            
        modelCounts = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
        totalCounts = np.sum(modelCounts, axis = 1)
        posteriorProb = modelCounts[:, 0] / totalCounts     
        plt.plot(truePosterior[:, 0], posteriorProb, 'o', color = coloursDict[metricDir], label = metricDir.upper().rstrip("_"), alpha = 0.5)
        
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.legend(loc = "upper left")
    plt.xlabel("True Posterior Probability of M1")
    plt.ylabel("ABC Posterior Probability of M1")
    
    figName = "combined_posterior_probs_eps" + str(EPSILON) + ".png"
    plt.savefig(os.path.join(PLOTS_DIR, "size_" + str(size), "posteriors", figName))
    plt.clf()
