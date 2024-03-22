import numpy as np

def abc_results(run, epsilon = 0.01):
    results = np.zeros(3)
    distances = run[:, 1]
    threshold = np.quantile(distances, epsilon)
    for i in range(3):
        modelDistances = run[run[:, 0] == i][:, 1]
        results[i] = np.sum(modelDistances < threshold)
    
    return results