import os
import numpy as np
from gk_bayes_factors import calculate_bayes_factor
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/results"
EPSILONS = [0.05, 0.01, 0.001]

rowCount = round(len(EPSILONS) / 2)
fig, axs = plt.subplots(rowCount, 2)
fig.tight_layout(rect = [0, 0.03, 1, 0.95])
sns.set_style('whitegrid')   
if len(EPSILONS) % 2:
    fig.delaxes(axs[rowCount - 1][1])
for testFolder in os.listdir(DATA_DIR):
    if not ("test" in testFolder):
        continue
    runDirectory = os.path.join(testFolder, "100/simple")
    runsPath = os.path.join(DATA_DIR, runDirectory)
    runs = os.listdir(runsPath)    
    bayesFactors = np.zeros((len(runs), len(EPSILONS)))
    for i in range(len(runs)):
        run = runs[i]
        runPath = os.path.join(runsPath, run)
        data = np.load(runPath)
        for j in range(len(EPSILONS)):
            epsilon = EPSILONS[j]
            bayesFactor = calculate_bayes_factor(data, epsilon)
            bayesFactors[i][j] = bayesFactor
            
    for i in range(len(EPSILONS)):
        epsilon = EPSILONS[i]
        epsilonBayesFactors = bayesFactors[:, i]
        ax = axs[i // 2][i % 2]
        sns.kdeplot(epsilonBayesFactors, cut = 0, ax = ax)
        ax.axvline(x = np.nanmean(epsilonBayesFactors), color = 'b', label = "Mean")   
        ax.axvline(x = np.nanmedian(epsilonBayesFactors), color = 'g', label = "Median") 
        ax.axvline(x = 1, color = 'y', linestyle = "dashed", label = "No Difference") 
        ax.legend(loc = 'upper right', prop={'size': 6})
        ax.title.set_text(r'$\varepsilon = {eps}$'.format(eps = epsilon))

    filenamePlot = runDirectory.replace("/", "-") + "-bayesdist.png"
    plt.savefig(os.path.join(SAVE_DIR, filenamePlot))
    plt.cla()