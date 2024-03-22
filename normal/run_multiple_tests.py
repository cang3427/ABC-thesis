from multiprocessing import Pool, Manager
from abc_model_choice_normal import DistanceMetric, main
import os
import numpy as np

NUM_WORKERS = 32
NUM_OBSERVED = 100
OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data/m2/params_sampled/unknown_var"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/runs/m2"

def generate_args(nullMeans, variances, priorVarianceScales, distanceMetrics, observedSizes):
    nullMeanArgs = []
    varianceArgs = []
    priorVarianceScaleArgs = []
    distanceMetricArgs = []
    observedPathArgs = []
    savePathArgs = []
    for size in observedSizes:
        for i in range(NUM_OBSERVED):
            for nullMean, variance, priorVarianceScale in zip(nullMeans, variances, priorVarianceScales):
                if variance is None:
                    test_dir = os.path.join(SAVE_DIR, "unknown_var/params_sampled")
                else:
                    test_dir = os.path.join(SAVE_DIR, "known_var/params_sampled")
                for distanceMetric in distanceMetrics:                    
                    if distanceMetric == DistanceMetric.STATS:
                        distanceDir = os.path.join(test_dir, "stat")
                    elif distanceMetric == DistanceMetric.CVM:
                        distanceDir = os.path.join(test_dir, "cvm")
                    elif distanceMetric == DistanceMetric.WASS:
                        distanceDir = os.path.join(test_dir, "wass")
                    elif distanceMetric == DistanceMetric.MMD:
                        distanceDir = os.path.join(test_dir, "mmd")  
                    else:
                        continue
                    if not os.path.isdir(distanceDir):
                        os.mkdir(distanceDir)  
                    observedPathArgs.append(os.path.join(OBSERVED_DIR, "sample" + str(i) + "size" + str(size) + ".npy"))
                    nullMeanArgs.append(nullMean)
                    varianceArgs.append(variance)
                    priorVarianceScaleArgs.append(priorVarianceScale)   
                    runFilename = "run" + str(i) + "size" + str(size) + ".npy"         
                    savePath = os.path.join(distanceDir, runFilename)
                    savePathArgs.append(savePath)
                    distanceMetricArgs.append(distanceMetric)
    
    return zip(nullMeanArgs, varianceArgs, priorVarianceScaleArgs, distanceMetricArgs, observedPathArgs, savePathArgs)

def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
if __name__ == "__main__":       
    nullMeans = [3]
    variances = [None]
    priorVarianceScales = [100]
    distanceMetrics = [metric for metric in DistanceMetric]
    observedSizes = [100, 1000]
    runArgs = generate_args(nullMeans, variances, priorVarianceScales, distanceMetrics, observedSizes)
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
