import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "./model_choice/normal/plots/mean_test/eq3_vs_neq3/known_var"
RESULTS_DIR = "./model_choice/normal/results/mean_test/eq3_vs_neq3/known_var"
SIZES = [100]
EPSILON = 0.1

coloursDict = {"cvm" : "blue", "mmd" : "orange", "wass" : "green"}
for size in SIZES:
    trueLogBayesFactors = np.load(os.path.join(RESULTS_DIR, "true/logbfs_size" + str(size)) + ".npy")
    abcDataName = "size" + str(size) + "eps" + str(EPSILON) + ".npy"
    plt.axline((0, 0), slope = 1, color = "k")
    for metricDir in os.listdir(os.path.join(RESULTS_DIR)):
        if metricDir == "true" or metricDir == "other":
            continue            
        modelCounts = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
        totalCounts = np.sum(modelCounts, axis = 1)
        logBayesFactors = np.log(modelCounts[:, 0] / modelCounts[:, 1])   
        plt.plot(trueLogBayesFactors, logBayesFactors, 'o', color = coloursDict[metricDir], label = metricDir.upper(), alpha = 0.5)
        
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.legend(loc = "upper left")
    plt.xlabel("True Log BF")
    plt.ylabel("ABC Log BF")
    
    figName = "combined_log_bayes_factors_eps" + str(EPSILON) + ".png"
    plt.savefig(os.path.join(PLOTS_DIR, "size_" + str(size), "bayes_factors", figName))
    plt.clf()
