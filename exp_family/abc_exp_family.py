import os
import time
import numpy as np
from common.distances import *
from exp_family.model import Model

def abc_exp_family(observed_data: np.ndarray, distance_type: Distance, num_iter = 1_000_000) -> np.ndarray:
    start_time = time.time()
    simulation_size = (len(observed_data), 1)
    # Log data if specified
    if distance_type == Distance.MMD_LOG or distance_type == Distance.WASS_LOG:
        observed_data = np.log(observed_data)
    
    # Store relevant calculations for observed data
    if distance_type == Distance.MMD or distance_type == Distance.MMD_LOG:
        observed_sq_distances = pdist(observed_data, 'sqeuclidean')
        sigma = np.median(observed_sq_distances)**0.5
    elif distance_type == Distance.WASS or distance_type == Distance.WASS_LOG:
        observed_data = np.sort(observed_data, axis=0)
    elif distance_type == Distance.STAT:
        log_observed = np.log(observed_data)
        observed_stats = np.array([np.sum(observed_data), np.sum(log_observed), np.sum(log_observed**2)])
    
    model_choices = np.zeros(num_iter)
    distances = np.zeros(num_iter)
    times = np.zeros(num_iter)
    models = [m for m in Model]
    for i in range(num_iter):
        # Select model randomly and generate sample, taking caution with parametrisation
        model = np.random.choice(models)
        if model == Model.EXP:
            theta = np.random.exponential()
            sample = np.random.exponential(1 / theta, size=simulation_size)
        elif model == Model.LNORM:
            theta = np.random.normal()
            sample = np.random.lognormal(theta, 1, size=simulation_size)
        else:
            theta = np.random.exponential()
            sample = np.random.gamma(2, 1 / theta, size=simulation_size)
        
        # Take log of data if specified              
        if distance_type == Distance.MMD_LOG or distance_type == Distance.WASS_LOG:
            sample = np.log(sample)
        
        # Calculate distances
        if distance_type == Distance.CVM:
            distance = cramer_von_mises_distance(observed_data, sample)
        elif distance_type == Distance.MMD or distance_type == Distance.MMD_LOG:
            distance = maximum_mean_discrepancy(observed_data, sample, observed_sq_distances, sigma)
        elif distance_type == Distance.WASS or distance_type == Distance.WASS_LOG:
            distance = wasserstein_distance(observed_data, sample)
        elif distance_type == Distance.STAT:
            log_sample = np.log(sample)
            sample_stats = np.array([np.sum(sample), np.sum(log_sample), np.sum(log_sample**2)]) 
            distance = np.sum((observed_stats - sample_stats)**2)   
            
        model_choices[i] = model.value
        distances[i] = distance
        times[i] = time.time() - start_time

    results = np.column_stack((model_choices, distances, times))
    return results

def main(distance: Distance, observed_path: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return
    
    observed_data = np.load(observed_path)
    results = abc_exp_family(observed_data, distance)
    np.save(save_path, results)
