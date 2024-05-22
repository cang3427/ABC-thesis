import numpy as np
import os, sys
from common.distances import Distance
from toad.run_jobs import run_jobs
from multiprocessing import cpu_count

OBSERVED_PATH = "./toad/data/observed_data.npy"
SAVE_PATH = "./toad/runs/true"
DISTANCES = [Distance.CVM, Distance.WASS_LOG, Distance.STAT]
NUM_WORKERS = min(len(DISTANCES), cpu_count() - 1)

if __name__ == "__main__":
    if not os.path.exists(OBSERVED_PATH):
        sys.exit("Error: Observed data does not exist. Prepare observed data before running this script.")  
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    save_args = [os.path.join(SAVE_PATH, f"run_{d.name.lower()}.npy") for d in DISTANCES]
    observed_args = [OBSERVED_PATH] * len(DISTANCES)
    run_args = zip(DISTANCES, observed_args, save_args)
    run_jobs(run_args, NUM_WORKERS)
