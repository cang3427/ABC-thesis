import numpy as np
import math
import os

def abc_log_bayes_factor(results, epsilon = 0.01):
    distances = results[:, -1]
    nonNanDistances = np.nan_to_num(distances, nan = math.inf)
    threshold = np.quantile(nonNanDistances, epsilon)
    nullDistances = results[results[:, 0] == 0][:, -1]
    alternativeDistances = results[results[:, 0] == 1][:, -1]
    bayesFactor = np.sum(nullDistances < threshold) / np.sum(alternativeDistances < threshold)
    return math.log(bayesFactor)
