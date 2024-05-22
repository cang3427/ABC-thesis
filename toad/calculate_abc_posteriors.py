import numpy as np
import os, sys
import math
import random
from common.distances import Distance
from common.abc_posterior import abc_posterior
from toad.toad_utils import Model

NUM_RUNS = 100
RUN_DIR = "./toad/runs"
RESULTS_DIR = "./toad/results"
DISTANCES = [Distance.CVM, Distance.WASS, Distance.WASS_LOG, Distance.STAT]
DISTANCE_QUANTILES = [0.001, 0.005, 0.01]
STAT_WEIGHT = 0.2
MODEL = Model.RANDOM

def weighted_distances(distances: np.ndarray, stat_weight: float) -> np.ndarray:
    nlags = int(distances.shape[1] / 2)
    return_distances = np.sum(distances[:, :nlags], axis=1)
    non_return_distances = np.sum(distances[:, nlags:], axis=1)
    return_distances /= np.nanmax(return_distances)
    non_return_distances /= np.nanmax(non_return_distances)
    distances = stat_weight * return_distances + (1 - stat_weight) * non_return_distances
    
    return distances

if __name__ == "__main__":
    model_dir = MODEL.name.lower()
    run_model_dir = os.path.join(RUN_DIR, model_dir)
    if not os.path.isdir(run_model_dir):
        sys.exit("Error: Run data does not exist. Complete ABC runs before running this script.")
    
    results_model_dir = os.path.join(RESULTS_DIR, model_dir)
    if not os.path.isdir(results_model_dir):
        os.makedirs(results_model_dir)
        
    for distance in DISTANCES:
        distance_dir = distance.name.lower()
        run_distance_dir = os.path.join(run_model_dir, distance_dir)
        results_distance_dir = os.path.join(results_model_dir, distance_dir)
        if not os.path.isdir(results_distance_dir):
            os.mkdir(results_distance_dir)
        
        for distance_quantile in DISTANCE_QUANTILES:
            posteriors = np.zeros((NUM_RUNS, 3))
            for i in range(NUM_RUNS):
                run = np.load(os.path.join(run_distance_dir, f"run{i}.npy"))
                run[np.isinf(run)] = np.nan
                if distance == Distance.STAT:
                    distances = run[:, 1]
                else:
                    distances = weighted_distances(run[:, 1:-1], STAT_WEIGHT)
                    
                posteriors[i] = abc_posterior(run[:, 0], distances, 3, distance_quantile)
            
            np.save(os.path.join(results_distance_dir, f"posteriors_{distance_quantile}q.npy"), posteriors)
