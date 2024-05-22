import random
import os
import time
import numpy as np
from common.distances import *
from enum import Enum
    
def abc_normal(observed_data: np.ndarray, distance_type: Distance, null_mean: float = 3.0, var: float = 1.0, prior_var_scale: float = 100.0, num_iter: int = 1_000_000) -> np.ndarray:
    start_time = time.time()
    simulation_size = (len(observed_data), 1)
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(simulation_size)
    
    # Calculate summary statistics for observed data if using summary statistic ABC
    if distance_type == Distance.STAT:
        observed_mean = np.mean(observed_data)
        if var is None:
            observed_var = np.var(observed_data, ddof=1)

    # Sort observed data for wasserstein distance
    if distance_type == Distance.WASS:
        observed_data = np.sort(observed_data, axis=0)

    # Calculate distance between points for observed data for MMD and sigma for the Gaussian kernel
    if distance_type == Distance.MMD:
        observed_sq_distances = pdist(observed_data, 'sqeuclidean')
        sigma = np.sqrt(np.median(observed_sq_distances))
    
    # Define values for priors depending on if var is known or unknown (None)
    if var is None:
        sd = None
        prior_mean_sd = prior_var_scale**0.5
    else:
        sd = var**0.5
        prior_mean_sd = (var * prior_var_scale)**0.5       
        distances = np.zeros(num_iter)  

    model_choices = np.zeros(num_iter) 
    distances = np.zeros(num_iter) 
    times = np.zeros(num_iter)
    for i in range(num_iter): 
        # Selecting the model to sample from and defining the mean
        model = random.randint(0, 1)
        if model == 0:
            model_mean = null_mean
        else:
            model_mean = np.random.normal(null_mean, prior_mean_sd)
            
        if sd is None:
            prior_sd = np.random.gamma(0.1, 10)
        else: 
            prior_sd = sd
        
        # Simulate sample from sampled model
        simulated_sample = np.random.normal(loc=model_mean, scale=prior_sd, size=simulation_size)
        
        # Calculate the chosen distance for the simulated sample
        if distance_type == Distance.STAT:
            simulated_mean = np.mean(simulated_sample)
            distance = (simulated_mean - observed_mean)**2
            if var is None:     
                simulated_var = np.var(simulated_sample, ddof=1)
                distance += (simulated_var - observed_var)**2   
        elif distance_type == Distance.CVM:
            distance = cramer_von_mises_distance(observed_data, simulated_sample=simulated_sample)
        elif distance_type == Distance.WASS:
            distance = wasserstein_distance(observed_data, simulated_sample=simulated_sample)
        elif distance_type == Distance.MMD:
            distance = maximum_mean_discrepancy(observed_data, simulated_sample=simulated_sample, observed_sq_distances=observed_sq_distances, sigma=sigma)
        
        # Store current values   
        model_choices[i] = model
        distances[i] = distance
        times[i] = time.time() - start_time
        
    results = np.column_stack((model_choices, distances, times))
    return results

def main(null_mean: float, var: float, prior_var_scale: float, distance: Distance, observed_path: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return
    observed_data = np.load(observed_path)
    results = abc_normal(observed_data, distance_type=distance, null_mean=null_mean, var=var, prior_var_scale=prior_var_scale)
    np.save(save_path, results)
