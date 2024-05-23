import os
import time
import math
from toad.toad_utils import *
from common.distances import *
from typing import List

def abc_toad(observed_data: np.ndarray, distance_type: Distance, num_toads: int = 66, num_days: int = 63, lags: List[int] = [1, 2, 4, 8], num_iter: int = 100_000):
    start_time = time.time()
    observed_summaries = summarise_sample(observed_data, lags)
    nlags = len(lags)
    if distance_type == Distance.WASS_LOG or distance_type == Distance.MMD_LOG:
        observed_summaries = [(data[0], np.log(data[1])) for data in observed_summaries]
    
    if distance_type == Distance.WASS or distance_type == Distance.WASS_LOG:
        observed_summaries = [(data[0], np.sort(data[1], axis=0)) for data in observed_summaries]
    elif distance_type == Distance.MMD or distance_type == Distance.MMD_LOG:
        observed_sq_distances = [pdist(observed_summaries[i][1], 'sqeuclidean') for i in range(nlags)]
        sigmas = [np.median(obsserved_sq_distances[i])**0.5 for i in range(nlags)]
    elif distance_type == Distance.STAT:
        observed_stats = get_statistics(observed_summaries)
        
    model_choices = np.zeros(num_iter)
    if distance_type == Distance.STAT:
        distances = np.zeros((num_iter, 12 * nlags)) 
        stats = np.zeros((num_iter, 12 * nlags))
    else:
        distances = np.zeros((num_iter, 8))
    times = np.zeros(num_iter)
    models = [m for m in Model]
    for i in range(num_iter):  
        model = np.random.choice(models)
        alpha = np.random.uniform(1, 2)
        gamma = np.random.uniform(10, 100)
        p0 = np.random.uniform(0, 1)
        d0 = None
        if model == Model.DISTANCE:
            d0 = np.random.uniform(20, 2000)
        
        sample = toad_movement_sample(model, alpha, gamma, p0, d0, num_toads, num_days)
        summaries = summarise_sample(sample, lags)
        if distance_type == Distance.WASS_LOG or distance_type == Distance.MMD_LOG:
            summaries = [(data[0], np.log(data[1])) for data in summaries]
    
        if distance_type == Distance.STAT:
            stats[i] = get_statistics(summaries)
            distance_list = (observed_stats - stats[i])**2
        else:     
            return_count_distances = [abs(observed_summaries[i][0] - summaries[i][0]) for i in range(nlags)]  
            if distance_type == Distance.CVM:
                non_return_distances = [cramer_von_mises_distance(observed_summaries[i][1], summaries[i][1]) if summaries[i][1].size > 0 else np.nan for i in range(nlags)]
            elif distance_type == Distance.WASS or distance_type == Distance.WASS_LOG:
                non_return_distances = [wasserstein_distance(observed_summaries[i][1], summaries[i][1]) if summaries[i][1].size > 0 else np.nan for i in range(nlags)]
            elif distance_type == Distance.MMD or distance_type == Distance.MMD_LOG:
                non_return_distances = [maximum_mean_discrepancy(observed_summaries[i][1], summaries[i][1], observed_sq_distances[i], sigmas[i]) if summaries[i][1].size > 0 else np.nan for i in range(nlags)]

            distance_list = return_count_distances + non_return_distances

        model_choices[i] = model.value
        distances[i] = distance_list
        times[i] = time.time() - start_time

    if distance_type == Distance.STAT:
        mads = np.nanmedian(np.abs(stats - np.nanmedian(stats, axis=0)), axis=0)
        distances /= mads**2
        distances = np.sum(distances, axis=1)
    
    results = np.column_stack((model_choices, distances, times))
    return results

def main(distance: Distance, observed_path: str, save_path: str):
    if os.path.exists(save_path):
        return
    observed_data = np.load(observed_path)
    results = abc_toad(observed_data, distance)
    np.save(save_path, results)
