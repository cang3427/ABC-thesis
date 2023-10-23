from multiprocessing import Pool, Manager
from abc_model_choice_gk import main, ParameterName, DistanceMetric
import os, sys
import numpy as np

NUM_WORKERS = 32
NUM_OBSERVED = 1
OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/test_a/eq3_vs_neq3"

def generate_args(paramAs, paramBs, paramsGs, paramKs, testParams, distanceMetrics, sampleSizes):
    paramAArgs = []
    paramBArgs = []
    paramGArgs = []
    paramKArgs = []
    testParamArgs = []
    distanceMetricArgs = []
    observedPathArgs = []
    savePathArgs = []
    for i in range(NUM_OBSERVED):
        for j in range(len(sampleSizes)):
            observedPath = os.path.join(OBSERVED_DIR, "sample" + str(i) + "size" + str(sampleSizes[j]) + ".npy")
            observedPathArgs += [observedPath] * (len(testParams) * len(distanceMetrics))
            runName = "run" + str(i) + "size" + str(sampleSizes[j]) + ".npy"
            for k in range(len(testParams)):
                paramAArgs += [paramAs[k]] * len(distanceMetrics)
                paramBArgs += [paramBs[k]] * len(distanceMetrics)
                paramGArgs += [paramGs[k]] * len(distanceMetrics)
                paramKArgs += [paramKs[k]] * len(distanceMetrics)
                testParamArgs += [testParams[k]] * len(distanceMetrics)
                params = [paramAs[k], paramBs[k], paramGs[k], paramKs[k]]
                testDir = ""
                for paramIdx in range(4):
                    if paramIdx == testParams[k].value:
                        continue
                    testDir += str(int(params[paramIdx] != None))                 
                
                for l in range(len(distanceMetrics)):
                    metric = distanceMetrics[l]
                    distanceMetricArgs.append(metric)
                    if metric == DistanceMetric.AUXILIARY:
                        metricDir = "aux"
                    elif metric == DistanceMetric.CVM:
                        metricDir = "cvm"
                    elif metric == DistanceMetric.WASS:
                        metricDir = "wass"
                    elif metric == DistanceMetric.MMD:
                        metricDir = "mmd"                  
                    savePathArgs.append(os.path.join(SAVE_DIR, testDir, metricDir, runName))
    
    return zip(paramAArgs, paramBArgs, paramGArgs, paramKArgs, testParamArgs, distanceMetricArgs, observedPathArgs, savePathArgs)               

def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
if __name__ == "__main__":       
    paramAs = [3, 3, 3, 3]
    paramBs = [1, None, None, None]
    paramGs = [2, 2, None, None]
    paramKs = [0.5, 0.5, 0.5, None]
    testParams = [ParameterName.AA] * 4
    distanceMetrics = [DistanceMetric.AUXILIARY, DistanceMetric.CVM, DistanceMetric.WASS]
    sampleSizes = np.linspace(10, 1000, 100).astype(int)
    runArgs = generate_args(paramAs, paramBs, paramGs, paramKs, testParams, distanceMetrics, sampleSizes)
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
