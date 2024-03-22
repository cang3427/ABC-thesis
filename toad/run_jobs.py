from multiprocessing import Pool, Manager
from abc_toad import main
    
def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
def run_jobs(args, workers):
    with Manager() as manager:
        taskQueue = manager.Queue()
        taskLock = manager.Lock()
        for arg in args:
            taskQueue.put(arg)           
        with Pool(workers) as pool:
            for _ in range(workers):
                result = pool.apply_async(worker_process, (taskQueue, taskLock))              
            pool.close()
            pool.join()
