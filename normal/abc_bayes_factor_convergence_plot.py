import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from abc_results import *

RUN_PATH = "model_choice/normal/runs"

runSizes = np.linspace(10_000, 1_000_000, 1000).astype(int)
for path in os.listdir(RUN_PATH):
    run = np.load(os.path.join(RUN_PATH, path))
    posteriors = np.zeros(len(runSizes))
    for i in range(len(runSizes)):
        print(runSizes[i])
        modelCounts = abc_results(run[:runSizes[i]], 1)
        posteriors[i] = modelCounts[0] / np.sum(modelCounts)
    plt.plot(runSizes, posteriors, label = path)

plt.legend(loc = 'upper left')    
plt.show()