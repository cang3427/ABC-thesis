import os
import matplotlib.pyplot as plt
import numpy as np
from abc_results import abc_results

RUN_DIR = "model_choice/exp-family/runs/wass"

sizes = np.linspace(10_000, 1_000_000, 100).astype(int)
for run in os.listdir(RUN_DIR):
    print(run)
    data = np.load(os.path.join(RUN_DIR, run))    
    posteriors = np.zeros(len(sizes))
    for i in range(len(sizes)):
        size = sizes[i]
        dataUsed = data[:size]
        acceptedCounts = abc_results(dataUsed, 0.01)
        posteriors[i] = acceptedCounts[0] / sum(acceptedCounts)
    
    plt.plot(sizes, posteriors, label = run)

plt.legend()
plt.show()