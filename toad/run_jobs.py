from multiprocessing import Pool, Manager, Queue, Lock
from toad.abc_toad import main
from typing import Any, Iterator, Tuple
from common.distances import Distance
    
def worker_process(queue: Queue, lock: Lock) -> None:
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(*args)
    
def run_jobs(args: Iterator[Tuple[Distance, str, str]], num_workers: int = 1) -> None:
    with Manager() as manager:
        taskQueue = manager.Queue()
        taskLock = manager.Lock()
        for arg in args:
            taskQueue.put(arg)           
        with Pool(num_workers) as pool:
            for _ in range(num_workers):
                result = pool.apply_async(worker_process, (taskQueue, taskLock))              
            pool.close()
            pool.join()
