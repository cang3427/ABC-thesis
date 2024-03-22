import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "./model_choice/normal/plots/m2/known_var"
RESULTS_DIR = "./model_choice/normal/results/m2/known_var"
SIZES = [100, 1000]
EPSILONS = [0.01, 0.025, 0.05, 0.1]

coloursDict = {"cvm" : "blue", "mmd" : "orange", "wass" : "green", "aux_" : "red"}
for size in SIZES:
    trueLogBayesFactors = np.load(os.path.join(RESULTS_DIR, "true/logbfs_size" + str(size)) + ".npy")
    for eps in EPSILONS:
        abcDataName = "size" + str(size) + "eps" + str(eps) + ".npy"
        for metricDir in os.listdir(os.path.join(RESULTS_DIR)):
            if metricDir == "true" or metricDir == "other":
                continue            
            modelCounts = np.load(os.path.join(RESULTS_DIR, metricDir, abcDataName))
            logBayesFactors = np.log(modelCounts[:, 0] / modelCounts[:, 1])        
            plt.axline((0, 0), slope = 1, color = "k")
            plt.plot(trueLogBayesFactors, logBayesFactors, 'o', color = coloursDict[metricDir])
            plt.xlim(0, 7)
            plt.ylim(0, 7)
            plt.xlabel("True Log BF")
            plt.ylabel("ABC Log BF")
            plt.title("ABC with " + metricDir.upper().rstrip("_"))
            
            plotDir = os.path.join(PLOTS_DIR, "size_" + str(size), "bayes_factors", metricDir)
            if not os.path.exists(plotDir):
                os.mkdir(plotDir)
                
            figName = "log_bayes_factors_eps" + str(eps) + ".png"     
            plt.savefig(os.path.join(plotDir, figName))
            plt.clf()
