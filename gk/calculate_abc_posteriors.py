import numpy as np
import os, sys
from common.abc_posterior import abc_posterior

NUM_RUNS = 100
RUN_DIR = "./gk/runs"
SAVE_DIR = "./gk/results"
MODEL = 0
DISTANCE_QUANTILES = [0.01]
SIZES = [100, 1000]

if __name__ == "__main__":
    model_dir = f"m{MODEL}"
    run_model_dir = os.path.join(RUN_DIR, model_dir) 
    if not os.path.isdir(run_model_dir):        
        sys.exit("Error: Run data does not exist. Complete ABC runs before running this script.")
    
    save_model_dir = os.path.join(SAVE_DIR, model_dir)
    for size in SIZES:
        size_dir = f"size_{size}"
        run_size_dir = os.path.join(run_model_dir, size_dir)
        save_size_dir = os.path.join(save_model_dir, size_dir)
        if not os.path.isdir(save_size_dir):
            os.makedirs(save_size_dir)
        
        # Calculating posteriors from each ABC method with different distance thresholds
        # for each observed dataset
        for distance_dir in os.listdir(run_size_dir):
            for distance_quantile in DISTANCE_QUANTILES:
                posteriors = np.zeros((NUM_RUNS, 2))
                for i in range(NUM_RUNS):                
                    run_path = os.path.join(run_size_dir, distance_dir, f"run{i}.npy")
                    run = np.load(run_path)
                    posteriors[i] = abc_posterior(run[:, 0], run[:, 1], 2, distance_quantile=distance_quantile)
                
                distance_path = os.path.join(save_size_dir, distance_dir)
                if not os.path.isdir(distance_path):
                    os.mkdir(distance_path)
                    
                savePath = os.path.join(distance_path, f"posteriors_{distance_quantile}q.npy")
                np.save(savePath, posteriors)
