import numpy as np
from math import inf

def abc_posterior(models: np.ndarray, distances: np.ndarray, num_models: int, distance_quantile: float = 0.01) -> np.ndarray:
    # Convert any nan values to infinity
    distances = np.nan_to_num(distances, nan=np.inf)
    
    # Calculate the distance threshold using the chosen distance quantile
    threshold = np.quantile(distances, distance_quantile)
    
    # Calculate the posterior probability for each model
    posterior = np.zeros(num_models)
    total_accepted = np.sum(distances <= threshold)
    for i in range(num_models):
        posterior[i] = np.sum(distances[models == i] <= threshold) / total_accepted
    
    return posterior

