from multiprocessing import Pool
from abc_model_choice_auxiliary import ParameterType, main
import os
import numpy as np

NUM_THREADS = 16
NUM_OBSERVED = 100
NUM_OBSERVED_SIZE = 1
NUM_TESTS = 4
PARAM_NAMES = ["a", "b", "g", "k"]
OBSERVED_DATA_PATH = "../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data"
SAVE_DATA_PATH = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice"

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
        sampleFileName = "sample" + str(i) + ".npy"    
        for j in range(NUM_OBSERVED_SIZE):
            observedSize = observedSizes[j]
            observedSizeArgs += [observedSize]*NUM_TESTS
            observedPathArgs += [os.path.join(OBSERVED_DATA_PATH, str(observedSize), sampleFileName)]*NUM_TESTS
            for k in range(NUM_TESTS):
                nullParamArgs.append(nullParams[k])
                nullTypeArgs.append(nullTypes[k])
                alternativeParamArgs.append(alternativeParams[k])
                alternativeTypeArgs.append(alternativeTypes[k])
                testIndexArgs.append(testIndices[k])
                if nullTypes[k] == ParameterType.EQUAL:
                    savePath = os.path.join(SAVE_DATA_PATH, "test_" + PARAM_NAMES[testIndices[k]], str(observedSize), "simple", "run" + str(i))
                else: 
                    savePath = os.path.join(SAVE_DATA_PATH, "test_" + PARAM_NAMES[testIndices[k]], str(observedSize), "composite", "run" + str(i))
                savePathArgs.append(savePath)
    
    return zip(nullParamArgs, alternativeParamArgs, nullTypeArgs, alternativeTypeArgs, testIndexArgs, observedSizeArgs, observedPathArgs, savePathArgs)
    
if __name__ == "__main__":       
    nullParams = [3, 1, 2, 0.5]
    nullTypes = [ParameterType.EQUAL]*4
    alternativeParams = [5, 3, 4, 2]
    alternativeTypes = [ParameterType.EQUAL]*4
    testIndices = [0, 1, 2, 3]
    observedSizes = [100]
    runArgs = generate_args(nullParams, nullTypes, alternativeParams, alternativeTypes, testIndices, observedSizes)
    with Pool(NUM_THREADS) as pool:
        pool.map(main, runArgs)
        pool.close()
        pool.join()
