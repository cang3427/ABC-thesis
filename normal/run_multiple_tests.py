import os, sys
import numpy as np
from multiprocessing import Pool, Manager, Queue, Lock, cpu_count
from normal.abc_normal import main
from typing import List, Any, Iterator, Tuple
from common.distances import Distance

OBSERVED_DIR = "./normal/observed_data"
SAVE_DIR = "./normal/runs"
NUM_WORKERS = cpu_count() - 1
NUM_OBSERVED = 100
MODEL = 0
NULL_MEAN = 3
VAR = 1
PRIOR_VAR_SCALE = 100
DISTANCES = [Distance.CVM, Distance.MMD, Distance.WASS, Distance.STAT]
SIZES = [100, 1000]

def generate_args(distances: List[Distance], observed_sizes = List[int]) -> Iterator[Tuple[Distance, str, str]]:
    observed_path_args = []
    save_path_args = []
    model_dir = f"m{MODEL}"
    run_dir = os.path.join(SAVE_DIR, "unknown_var" if VAR is None else "known_var", model_dir)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
        
    for size in observed_sizes:
        size_dir = f"size_{size}"
        observed_size_dir = os.path.join(OBSERVED_DIR, model_dir, size_dir)
        run_size_dir = os.path.join(run_dir, size_dir)
        if not os.path.isdir(run_size_dir):
            os.mkdir(run_size_dir)
            
        for i in range(NUM_OBSERVED):
            for distance in distances:                    
                distance_dir = os.path.join(run_size_dir, distance.name.lower())
                if not os.path.isdir(distance_dir):
                    os.mkdir(distance_dir)  
                    
                observed_path_args.append(os.path.join(observed_size_dir, f"sample{i}.npy"))
                run_filename = f"run{i}.npy"        
                save_path = os.path.join(distance_dir, run_filename)
                save_path_args.append(save_path)
    
    distance_args = distances * (len(observed_sizes) * NUM_OBSERVED)
    return zip(distance_args, observed_path_args, save_path_args)

def worker_process(queue: Queue, lock: Lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(NULL_MEAN, VAR, PRIOR_VAR_SCALE, *args)
    
if __name__ == "__main__":       
    if not os.path.isdir(OBSERVED_DIR):
        sys.exit("Error: Observed data does not exist. Generate observed data before running this script.")
    
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    run_args = generate_args(DISTANCES, SIZES)
    with Manager() as manager:
        taskQueue = manager.Queue()
        taskLock = manager.Lock()
        for runArg in run_args:
            taskQueue.put(runArg)
            
        with Pool(NUM_WORKERS) as pool:
            for _ in range(NUM_WORKERS):
                pool.apply_async(worker_process, (taskQueue, taskLock))
            pool.close()
            pool.join()
