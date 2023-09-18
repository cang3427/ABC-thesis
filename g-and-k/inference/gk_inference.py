import numpy as np
import matplotlib.pyplot as plt
import math

abcResults = np.load("g-and-k/inference/data/sim100-abc1e7-prioruni.npy")
distances = abcResults[:, -1]
nonNanDistances = np.nan_to_num(distances, nan = math.inf)
threshold = np.quantile(nonNanDistances, 0.001)
posteriorTheta = abcResults[abcResults[:, -1] < threshold][:, :-1]

fig, axs = plt.subplots(2, 2)
trueParams = [3, 1, 2, 0.5]
meanEstimates = np.mean(posteriorTheta, axis = 0)
medianEstimates = np.median(posteriorTheta, axis = 0)
paramNames = ['a', 'b', 'g', 'k']
for i in range(4):
    ax = axs[i // 2][i % 2]
    ax.hist(posteriorTheta[:, i], bins = 40, density = True) 
    ax.axvline(x = meanEstimates[i], color = 'b', label = r'${param}_{{mean}}$'.format(param = paramNames[i]))   
    ax.axvline(x = medianEstimates[i], color = 'g', label = r'${param}_{{med}}$'.format(param = paramNames[i]))   
    ax.axvline(x = trueParams[i], color = 'r', linestyle = 'dashed', label = r'${param}_{{true}}$'.format(param = paramNames[i]))
    ax.title.set_text(paramNames[i])
    ax.legend(loc = 'upper right')
    
# plt.savefig("g-and-k/inference/results/sim100-abc1e7-prioruni-eps0.001.png")
plt.show()