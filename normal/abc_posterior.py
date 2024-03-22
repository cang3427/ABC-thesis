import numpy as np
import math

def abc_posterior(run, epsilon = 0.01):
    distances = np.nan_to_num(run[:, 1], nan = math.inf)
    threshold = np.quantile(distances, epsilon)
    posterior = np.zeros(2)
    posterior[0] = np.sum(distances[run[:, 0] == 0] < threshold) / np.sum(distances < threshold)
    posterior[1] = 1 - posterior[0]
    
    return posterior