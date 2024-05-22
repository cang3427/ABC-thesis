import numpy as np
import random
import math
import time
from enum import Enum
from scipy.stats import levy_stable
from typing import List, Tuple

class Model(Enum):
    RANDOM = 0
    NEAREST = 1
    DISTANCE = 2

def distance_based_probs(position: float, refuge_locations: np.ndarray, p0: float, d0: float) -> np.ndarray:
    # Calculating individual return probabilties based on the current position compared to the 
    # refuge locations for the distance-based return model
    refuge_distances = np.abs(position - refuge_locations)
    
    return p0 * np.exp(-refuge_distances / d0)

def toad_movement_sample(model: Model, alpha: float, gamma: float, p0: float, d0: float = None, num_toads: int = 66, num_days: int = 63) -> np.ndarray:
    
    # Initial setup, storing 0 as initial refuge for distance-based return
    toad_positions = np.zeros((num_days, num_toads))
    if model == Model.DISTANCE:
        refuge_counts = np.ones(num_toads, dtype=int)
        refuge_locations = np.zeros((num_days, num_toads))
    else:
        no_return_probs = 1 - p0
    
    # Pre-calculating step sizes for each toad over the tracking period
    steps = levy_stable.rvs(alpha, 0, scale=gamma, size=(num_days - 1, num_toads))
    
    # Main loop, all toads are handled in one loop to make use of vectorised calculations
    for i in range(1, num_days):
        # Calculating new position
        new_pos = toad_positions[i - 1] + steps[i - 1]
        
        # Calculating no return probability for distance-based return model (not constant)
        if model == Model.DISTANCE:
            refuge_probs = [distance_based_probs(new_pos[j], refuge_locations[:refuge_counts[j], j], p0, d0) for j in range(num_toads)]
            no_return_probs = np.array([np.prod(1 - refuge_probs[j]) for j in range(num_toads)])
        
        # Separating toads which are return and not returning for the current day
        no_return_flag = np.random.uniform(size=num_toads) < no_return_probs
        no_return_ids = np.nonzero(no_return_flag)[0]
        return_ids = np.nonzero(~no_return_flag)[0]
        
        # Updating toad position for non returning toads to th new positions
        toad_positions[i, no_return_ids] = new_pos[no_return_ids]
        
        if model == Model.RANDOM:
            # Randomly selecting a location among all previous locations for returning toads
            return_location_ids = np.random.randint(0, i, size=return_ids.shape)
            toad_positions[i, return_ids] = toad_positions[return_location_ids, return_ids]
        elif model == Model.NEAREST:
            # Determining nearest return location for each return toad
            return_location_ids = np.argmin(np.abs(new_pos[return_ids] - toad_positions[:i, return_ids]), axis=0)
            toad_positions[i, return_ids] = toad_positions[return_location_ids, return_ids]
        else:
            # Randomly selecting previous location using distance-based probabilities for returning toads
            # and updating refuge locations and counts for non-return toads
            return_location_ids = [np.random.choice(list(range(refuge_counts[j])), p=refuge_probs[j] / np.sum(refuge_probs[j])) for j in return_ids]
            toad_positions[i, return_ids] = refuge_locations[return_location_ids, return_ids]
            refuge_locations[refuge_counts[no_return_ids], no_return_ids] = new_pos[no_return_ids]
            refuge_counts[no_return_ids] += 1
            
    return toad_positions

def summarise_sample(toad_data: np.ndarray, lags: List[int]) -> List[Tuple[int, np.ndarray]]:
    summaries = []
    # Calculating data for each lag
    for lag in lags:
        num_days, num_toads = np.shape(toad_data)
        
        # Calculating the lag differenced toad matrix and then splitting into return counts
        # and non return data
        diffs = np.abs(toad_data[lag:, :] - toad_data[:(num_days - lag), :]).flatten()
        return_count = np.sum(diffs < 10.0)
        non_return_data = diffs[diffs >= 10.0]            
        summaries.append((return_count, np.reshape(non_return_data, (len(non_return_data), 1))))
        
    return summaries

def get_statistics(summarised_data: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    nlags = len(summarised_data)
    stats = np.zeros(12 * nlags)
    for i in range(nlags):
        # Storing count data for current lag
        stats[i * 12] = summarised_data[i][0]
        
        # Edge case for empty non-return data when p0 close to 1
        if summarised_data[i][1].size == 0:
            stats[(i * 12 + 1):(i * 12 + 12)] = np.nan
            continue
        
        # Storing log of quantile differences and median
        quantiles = np.quantile(summarised_data[i][1], np.arange(0, 1.1, 0.1))
        stats[(i * 12 + 1):(i * 12 + 11)] = np.log(np.diff(quantiles))
        stats[i * 12 + 11] = quantiles[5]
    
    # Edge case when quantile differences are zero for small non-return data
    if np.any(np.isinf(stats)):
        stats[np.where(np.isinf(stats))] = np.nan
    
    return stats
