import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import DistanceMetric 
import numpy as np
from run_jobs import run_jobs

NUM_WORKERS = 32
NUM_OBSERVED = 100
OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/test_data/m3"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/runs/m3"

def generate_args(distanceMetrics):
    distanceArgs = []
    observedArgs = []
    saveArgs = []

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

        for i in range(NUM_OBSERVED):
            observedPath = os.path.join(OBSERVED_DIR, "sample" + str(i) + ".npy")
            runName = "run" + str(i) + ".npy"                  
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
    runArgs = generate_args(distanceMetrics)
    run_jobs(runArgs, NUM_WORKERS)
