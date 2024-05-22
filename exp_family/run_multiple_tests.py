import os, sys
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from exp_family.abc_exp_family import main
from typing import List, Any, Iterator, Tuple 
from common.distances import Distance
from exp_family.model import Model

NUM_WORKERS = cpu_count() - 1
NUM_OBSERVED = 100
OBSERVED_DIR = "./exp_family/observed_data"
SAVE_DIR = "./exp_family/runs"
MODEL = Model.EXP

def generate_args(distances: List[Distance], observed_sizes: List[int]) -> Iterator[Tuple[Distance, str, str]]:
    observed_args = []
    save_args = []
    distance_args = []
    model_dir = MODEL.name.lower()
    run_dir = os.path.join(SAVE_DIR, model_dir)
    for size in observed_sizes:
        size_dir = f"size_{size}"
        observed_size_dir = os.path.join(OBSERVED_DIR, model_dir, size_dir)
        run_size_dir = os.path.join(run_dir, size_dir)
        if not os.path.isdir(run_size_dir):
            os.makedirs(run_size_dir)
            
        for distance in distances:                
            distance_dir = os.path.join(run_size_dir, distance.name.lower())
            if not os.path.isdir(distance_dir):
                os.mkdir(distance_dir) 
            distance_args += [distance] * NUM_OBSERVED
            for i in range(NUM_OBSERVED):   
                observed_args.append(os.path.join(observed_size_dir, f"sample{i}.npy"))
                run_filename = f"run{i}.npy"        
                save_path = os.path.join(distance_dir, run_filename)
                save_args.append(save_path)

    return zip(distance_args, observed_args, save_args)
    
def worker_process(queue: Queue, lock: Lock) -> None:
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(*args)
    
if __name__ == "__main__":   
    if not os.path.isdir(OBSERVED_DIR):
        sys.exit("Error: Observed data does not exist. Generate observed data before running this script.")
    
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
            
    distances = [distance for distance in Distance]
    observed_sizes = [100, 1000]
    run_args = generate_args(distances, observed_sizes)
    with Manager() as manager:
        task_queue = manager.Queue()
        task_lock = manager.Lock()
        for run_arg in run_args:
            task_queue.put(run_arg)           
        with Pool(NUM_WORKERS) as pool:
            for _ in range(NUM_WORKERS):
                result = pool.apply_async(worker_process, (task_queue, task_lock))              
            pool.close()
            pool.join()
