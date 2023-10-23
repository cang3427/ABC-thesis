from multiprocessing import Pool, Manager
from abc_model_choice_normal import DistanceMetric, main
import os
import numpy as np

NUM_WORKERS = 5
NUM_OBSERVED = 1
OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/runs/eq3_vs_neq3"

def generate_args(nullMean, variance, priorVarianceScale, distanceMetrics, observedSizes):
    nullMeanArgs = []
    varianceArgs = []
    priorVarianceScaleArgs = []
    distanceMetricArgs = []
    observedPathArgs = []
    savePathArgs = []
    for i in range(NUM_OBSERVED):
        for distanceMetric in distanceMetrics:
            distanceMetricArgs += [distanceMetric] * len(observedSizes)
            for size in observedSizes:
                observedPathArgs.append(os.path.join(OBSERVED_DIR, "sample" + str(i) + "size" + str(size) + ".npy"))
                nullMeanArgs.append(nullMean)
                varianceArgs.append(variance)
                priorVarianceScaleArgs.append(priorVarianceScale)   
                runFilename = "run" + str(i) + "size" + str(size) + ".npy"
                if distanceMetric == DistanceMetric.AUXILIARY:
                    savePath = os.path.join(SAVE_DIR, "aux", runFilename)
                elif distanceMetric == DistanceMetric.CVM:
                    savePath = os.path.join(SAVE_DIR, "cvm", runFilename)
                elif distanceMetric == DistanceMetric.WASS:
                    savePath = os.path.join(SAVE_DIR, "wass", runFilename)
                elif distanceMetric == DistanceMetric.MMD:
                    savePath = os.path.join(SAVE_DIR, "mmd", runFilename)           
                savePathArgs.append(savePath)
    
    return zip(nullMeanArgs, varianceArgs, priorVarianceScaleArgs, distanceMetricArgs, observedPathArgs, savePathArgs)

def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
if __name__ == "__main__":       
    nullMean = 3
    variance = 1
    priorVarianceScale = 100
    distanceMetrics = [DistanceMetric.MMD]
    observedSizes = np.linspace(10, 1000, 100).astype(int)
    runArgs = generate_args(nullMean, variance, priorVarianceScale, distanceMetrics, observedSizes)
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
