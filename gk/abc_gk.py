import random
import os
import numpy as np
import time
from common.distances import *
from gk.gk_funcs import gk_sample

def abc_gk(observed_data: np.ndarray, distance_type: Distance, num_iter: int = 100) -> np.ndarray:
    start_time = time.time()
    
    # Store relevant calcultions for observed data
    if distance_type == Distance.WASS:
        observed_data = np.sort(observed_data, axis=0)
    elif distance_type == Distance.MMD:
        observed_sq_distances = pdist(observed_data, 'sqeuclidean')
        sigma = np.median(observed_sq_distances) ** 0.5
    elif distance_type == Distance.STAT:
        observed_quantiles = np.quantile(observed_data, [0.1, 0.9])
        
    simulation_size = observed_data.shape
    model_choices = np.zeros(num_iter)
    distances = np.zeros(num_iter) 
    times = np.zeros(num_iter)
    for i in range(num_iter):          
        model = random.randint(0, 1)
        theta1 = 0
        theta2 = np.random.uniform(-0.5, 5)
        # Sample g parameter for second model
        if model == 1:
            theta1 = np.random.uniform(0.0, 4.0)

        # Calculating distance
        theta = (0.0, 1.0, theta1, theta2)
        sample = gk_sample(theta, simulation_size)  
        if distance_type == Distance.CVM:
            distance = cramer_von_mises_distance(observed_data, sample)
        elif distance_type == Distance.WASS:
            distance = wasserstein_distance(observed_data, sample)
        elif distance_type == Distance.MMD:
            distance = maximum_mean_discrepancy(observed_data, sample, observed_sq_distances, sigma)
        elif distance_type == Distance.STAT:
            quantiles = np.quantile(sample, [0.1, 0.9])
            distance = np.sum(np.abs(observed_quantiles - quantiles))

        model_choices[i] = model
        distances[i] = distance
        times[i] = time.time() - start_time

    results = np.column_stack((model_choices, distances, times))
    return results

def main(distance: Distance, observed_path: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return
    
    observed_data = np.load(observed_path)
    results = abc_gk(observed_data, distance)
    np.save(save_path, results)
