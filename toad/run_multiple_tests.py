import os, sys
import numpy as np
from common.distances import Distance 
from toad.run_jobs import run_jobs
from multiprocessing import cpu_count
from toad.toad_utils import Model
from typing import Iterator, List, Tuple

OBSERVED_DIR = "./toad/simulated_data"
SAVE_DIR = "./toad/runs"
NUM_WORKERS = cpu_count() - 1
NUM_OBSERVED = 100
MODEL = Model.RANDOM

def generate_args(distances: List[Distance]) -> Iterator[Tuple[Distance, str, str]]:
    observed_model_dir = os.path.join(OBSERVED_DIR, MODEL.name.lower())
    save_model_dir = os.path.join(SAVE_DIR, MODEL.name.lower())
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)
    
    distance_args = []
    observed_args = []
    save_args = []
    for distance in distances:   
        distance_dir = os.path.join(save_model_dir, distance.name.lower())
        if not os.path.isdir(distance_dir):
            os.mkdir(distance_dir)  

        distance_args += [distance] * NUM_OBSERVED
        for i in range(NUM_OBSERVED):
            observed_path = os.path.join(observed_model_dir, f"sample{i}.npy")
            save_path = os.path.join(distance_dir, f"run{i}.npy")
            save_args.append(save_path)
            observed_args.append(observed_path)

    return zip(distance_args, observed_args, save_args)
    
if __name__ == "__main__":       
    if not os.path.isdir(OBSERVED_DIR):
        sys.exit("Error: Observed data does not exist. Generate observed data before running this script.")

    distances = [Distance.CVM, Distance.WASS, Distance.WASS_LOG, Distance.STAT]
    run_args = generate_args(distances)
    run_jobs(run_args, NUM_WORKERS)
