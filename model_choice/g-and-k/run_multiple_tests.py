from multiprocessing import Pool, Manager
from abc_model_choice_auxiliary import ParameterType, DistributionType, main
import os
import numpy as np

NUM_WORKERS = 16
NUM_OBSERVED = 100
NUM_TESTS = 4
PARAM_NAMES = ["a", "b", "g", "k"]
DIST_TYPE = DistributionType.GANDK

def generate_args(nullParams, nullTypes, alternativeParams, alternativeTypes, testIndices, observedSizes):
    nullParamArgs = []
    nullTypeArgs = []
    alternativeParamArgs = []
    alternativeTypeArgs = []
    testIndexArgs = []
    observedSizeArgs = []
    observedPathArgs = []
    savePathArgs = []
    for i in range(NUM_OBSERVED):
        for j in range(len(observedSizes)):
            observedSize = observedSizes[j]
            observedSizeArgs += [observedSize]*NUM_TESTS
            if (DIST_TYPE == DistributionType.GANDK):
                observedPathArgs += [os.path.join("../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data", str(observedSize), "sample" + str(i) + ".npy")]*NUM_TESTS
            elif (DIST_TYPE == DistributionType.NORMAL):
                observedPathArgs += [os.path.join("../../project/RDS-FSC-ABCMC-RW/normal/observed_data", "sample" + str(i) +  "size" + str(observedSize) + ".npy")]*NUM_TESTS
            for k in range(NUM_TESTS):
                nullParamArgs.append(nullParams[k])
                nullTypeArgs.append(nullTypes[k])
                alternativeParamArgs.append(alternativeParams[k])
                alternativeTypeArgs.append(alternativeTypes[k])
                testIndexArgs.append(testIndices[k])
                if nullTypes[k] == ParameterType.EQUAL:
                    if (DIST_TYPE == DistributionType.GANDK):
                        savePath = os.path.join("../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice", "test_" + PARAM_NAMES[testIndices[k]], str(observedSize), "simple", "run" + str(i))
                    elif (DIST_TYPE == DistributionType.NORMAL):
                        savePath = os.path.join("../../project/RDS-FSC-ABCMC-RW/normal/model_choice", "test_" + PARAM_NAMES[testIndices[k]], "simple", "run" + str(i) + "size" + str(observedSize))
                else: 
                    if (DIST_TYPE == DistributionType.GANDK):
                        savePath = os.path.join("../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice", "test_" + PARAM_NAMES[testIndices[k]], str(observedSize), "composite", "run" + str(i))
                    elif (DIST_TYPE == DistributionType.NORMAL):                        
                        savePath = os.path.join("../../project/RDS-FSC-ABCMC-RW/normal/model_choice", "test_" + PARAM_NAMES[testIndices[k]], "composite", "run" + str(i) + "size" + str(observedSize))
                savePath += ".npy"
                savePathArgs.append(savePath)
    
    distTypeArgs = [DIST_TYPE]*NUM_OBSERVED*len(observedSizes)*NUM_TESTS
    return zip(nullParamArgs, alternativeParamArgs, nullTypeArgs, alternativeTypeArgs, testIndexArgs, observedSizeArgs, distTypeArgs, observedPathArgs, savePathArgs)

def worker_process(queue, lock):
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(args)
    
if __name__ == "__main__":       
    nullParams = [3, 1, 2, 0.5]
    nullTypes = [ParameterType.EQUAL]*NUM_TESTS
    alternativeParams = [5, 3, 4, 2]
    alternativeTypes = [ParameterType.EQUAL]*NUM_TESTS
    testIndices = [0, 1, 2, 3]
    observedSizes = [100]
    runArgs = generate_args(nullParams, nullTypes, alternativeParams, alternativeTypes, testIndices, observedSizes)
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
