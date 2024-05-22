import os, sys
import numpy as np
from gk.abc_gk import main
from multiprocessing import Pool, Manager, cpu_count, Queue, Lock
from typing import List, Iterator, Tuple
from common.distances import Distance

OBSERVED_DIR = "./gk/observed_data"
SAVE_DIR = "./gk/runs"
NUM_WORKERS = cpu_count() - 1
NUM_OBSERVED = 100
MODEL = 0

def generate_args(distances: List[Distance], sample_sizes: List[int]) -> Iterator[Tuple[Distance, str, str]]:
    distance_args = []
    observed_args = []
    save_args = []
    model_dir = f"m{MODEL}"
    for sample_size in sample_sizes:
        size_dir = f"size_{sample_size}"
        observed_size_dir = os.path.join(OBSERVED_DIR, model_dir, size_dir)
        run_size_dir = os.path.join(SAVE_DIR, model_dir, size_dir)
        if not os.path.isdir(run_size_dir):
            os.makedirs(run_size_dir)
            
        for distance in distances:
            distance_dir = os.path.join(run_size_dir, distance.name.lower())
            distance_args += [distance] * NUM_OBSERVED
            if not os.path.isdir(distance_dir):
                os.mkdir(distance_dir)
                
            for i in range(NUM_OBSERVED):
                observed_path = os.path.join(observed_size_dir, f"sample{i}.npy")
                save_path = os.path.join(distance_dir, f"run{i}.npy")
                observed_args.append(observed_path)
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

    distances = [Distance.CVM, Distance.MMD, Distance.WASS, Distance.STAT]
    sample_sizes = [100, 1000]
    run_args = generate_args(distances, sample_sizes)
    with Manager() as manager:
        task_queue = manager.Queue()
        task_lock = manager.Lock()
        for run_arg in run_args:
            task_queue.put(run_arg)
            
        with Pool(NUM_WORKERS) as pool:
            for _ in range(NUM_WORKERS):
                pool.apply_async(worker_process, (task_queue, task_lock))                
            pool.close()
            pool.join()
