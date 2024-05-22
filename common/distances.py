import numpy as np
from scipy.stats import rankdata
from scipy.stats import wasserstein_distance as wass_distance
from scipy.spatial.distance import pdist, cdist
from enum import Enum

class Distance(Enum):
    CVM = 0
    MMD = 1
    WASS = 2
    STAT = 3
    MMD_LOG = 4
    WASS_LOG = 5
    
DISTANCE_LABELS = {Distance.CVM: "CvM", Distance.MMD: "MMD", Distance.WASS: "Wass", Distance.STAT: "Stat", Distance.MMD_LOG: "MMD (log)", Distance.WASS_LOG: "Wass (log)"}

def cramer_von_mises_distance(observed_sample: np.ndarray, simulated_sample: np.ndarray) -> float:
    # Calculating ranks in combined sample
    observed_size = len(observed_sample)
    simulated_size = len(simulated_sample)    
    combined_sample = np.concatenate((observed_sample, simulated_sample))
    combined_ranks = rankdata(combined_sample)
    observed_ranks = np.sort(combined_ranks[:observed_size])
    simulated_ranks = np.sort(combined_ranks[observed_size:])
    
    # Calculating distance
    observed_indices = np.array(range(1, observed_size + 1))
    simulated_indices = np.array(range(1, simulated_size + 1))
    rank_sum = observed_size * sum((observed_ranks - observed_indices)**2) + simulated_size * sum((simulated_ranks - simulated_indices)**2)
    size_prod = observed_size * simulated_size
    size_sum = observed_size + simulated_size
    distance = rank_sum / (size_prod * size_sum) - (4 * size_prod - 1) / (6 * size_sum)
        
    return distance

def wasserstein_distance(observed_sample: np.ndarray, simulated_sample: np.ndarray, observed_is_sorted=True) -> float:
    observed_size = len(observed_sample)
    simulated_size = len(simulated_sample)
    if observed_size == simulated_size:
        if observed_is_sorted:
            sorted_observed = observed_sample
        else:
            sorted_observed = np.sort(observed_sample, axis=0)
            
        sorted_simulated = np.sort(simulated_sample, axis=0)
        
        # Calculating wass distance = mean difference between order statistics
        distance = np.mean(np.abs(sorted_observed - sorted_simulated))
        return distance

    return wass_distance(observed_sample.flatten(), simulated_sample.flatten())

def gaussian_kernel(sq_distances, sigma) -> np.ndarray:
    return np.exp(-sq_distances / (2 * sigma))

def maximum_mean_discrepancy(observed_sample: np.ndarray, simulated_sample: np.ndarray, observed_sq_distances=None, sigma=None) -> float: 
    # Calculate the distances between the samples in the observed data if they have not been passed
    if observed_sq_distances is None:
        observed_sq_distances = pdist(observed_sample, 'sqeuclidean')
    
    # Calculate sigma for the gaussian kernel if it has not been passed
    if sigma is None:
        sigma = np.median(observed_sq_distances) ** 0.5   
        
    # Calculate distances between samples in the simulated data and between samples with the observed data
    simulated_sq_distances = pdist(simulated_sample, 'sqeuclidean')
    mixed_sq_distances = cdist(observed_sample, simulated_sample, 'sqeuclidean')
    
    # Calculate the MMD
    distance = (np.mean(gaussian_kernel(observed_sq_distances, sigma)) +  
                np.mean(gaussian_kernel(simulated_sq_distances, sigma)) - 
                2 *  np.mean(gaussian_kernel(mixed_sq_distances, sigma)))
    
    return distance
