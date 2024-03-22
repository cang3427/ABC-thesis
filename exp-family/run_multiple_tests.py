from multiprocessing import Pool, Manager
from abc_model_choice_exp_family import *
import os, sys
import numpy as np

NUM_WORKERS = 32
NUM_OBSERVED = 100
OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/observed_data/exp/params_sampled"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/runs/exp/params_sampled"

def generate_args(distanceMetrics, sampleSizes):
    distanceArgs = []
    observedArgs = []
    saveArgs = []
    for sampleSize in sampleSizes:
        for i in range(NUM_OBSERVED):       
            observedPath = os.path.join(OBSERVED_DIR, "sample" + str(i) + "size" + str(sampleSize) + ".npy")
            runName = "run" + str(i) + "size" + str(sampleSize) + ".npy"
            for metric in distanceMetrics:                
                if metric == DistanceMetric.CVM:
                    metricDir = "cvm"
                elif metric == DistanceMetric.WASS:
                    metricDir = "wass"
                elif metric == DistanceMetric.MMD:
                    metricDir = "mmd"   
                elif metric == DistanceMetric.STATS:
                    metricDir = "stat"
                else: 
                    continue
                
                saveDir = os.path.join(SAVE_DIR, metricDir)
                if not os.path.isdir(saveDir):
                    os.mkdir(saveDir)                    
                savePath = os.path.join(saveDir, runName)
                saveArgs.append(savePath)
                observedArgs.append(observedPath)
                distanceArgs.append(metric)
    
    return zip(distanceArgs, observedArgs, saveArgs)
    
def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
if __name__ == "__main__":       
    distanceMetrics = [metric for metric in DistanceMetric]
    sampleSizes = [100, 1000]
    runArgs = generate_args(distanceMetrics, sampleSizes)
    with Manager() as manager:
        taskQueue = manager.Queue()
        taskLock = manager.Lock()
        for runArg in runArgs:
            taskQueue.put(runArg)           
        with Pool(NUM_WORKERS) as pool:
            for _ in range(NUM_WORKERS):
                result = pool.apply_async(worker_process, (taskQueue, taskLock))              
            pool.close()
            pool.join()
