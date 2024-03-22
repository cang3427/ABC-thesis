from multiprocessing import Pool, Manager
from abc_mc_gk_test import *
import os, sys
import numpy as np

NUM_WORKERS = 32
NUM_OBSERVED = 100
OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data/gk_test/m1"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/gk_test/m1_true"

def generate_args(distanceMetrics, sampleSizes):
    distanceArgs = []
    observedArgs = []
    saveArgs = []
    for sampleSize in sampleSizes:
        for i in range(NUM_OBSERVED):       
            observedPath = os.path.join(OBSERVED_DIR, "sample" + str(i) + "size" + str(sampleSize) + ".npy")
            runName = "run" + str(i) + "size" + str(sampleSize) + ".npy"
            for metric in distanceMetrics:
                observedArgs.append(observedPath)
                distanceArgs.append(metric)
                if metric == DistanceMetric.AUXILIARY:
                    metricDir = "aux"
                elif metric == DistanceMetric.CVM:
                    metricDir = "cvm"
                elif metric == DistanceMetric.WASS:
                    metricDir = "wass"
                elif metric == DistanceMetric.MMD:
                    metricDir = "mmd"      
                elif metric == DistanceMetric.QUANTILE:
                    metricDir = "qle"   
                saveDir = os.path.join(SAVE_DIR, metricDir)
                if not os.path.isdir(saveDir):
                    os.mkdir(saveDir)                    
                savePath = os.path.join(saveDir, runName)
                saveArgs.append(savePath)
    
    return zip(distanceArgs, observedArgs, saveArgs)
    
def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
if __name__ == "__main__":       
    distanceMetrics = [DistanceMetric.WASS]
    sampleSizes = [100, 1000]
    runArgs = generate_args(distanceMetrics, sampleSizes)

    with Manager() as manager:
        taskQueue = manager.Queue()
        taskLock = manager.Lock()
        for runArg in runArgs:
            taskQueue.put(runArg)
            
        with Pool(NUM_WORKERS) as pool:
            for _ in range(NUM_WORKERS):
                pool.apply_async(worker_process, (taskQueue, taskLock))                
            pool.close()
            pool.join()
