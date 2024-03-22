import numpy as np
import os
from abc_results import abc_results

NUM_RUNS = 100
RUN_DIR = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/runs"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/results/params_sampled"

epsilons = [0.01, 0.025, 0.05, 0.1]
sizes = [100, 1000]
for model in os.listdir(RUN_DIR):
    if model == "gamma":
        continue
    for size in sizes:  
        modelRunPath = os.path.join(RUN_DIR, model, "params_sampled")
        for metricDir in os.listdir(modelRunPath):
            for eps in epsilons:
                results = np.zeros((NUM_RUNS, 3))
                for i in range(NUM_RUNS):                
                    runPath = os.path.join(modelRunPath, metricDir, "run" + str(i) + "size" + str(size) + ".npy")
                    run = np.load(runPath)
                    results[i] = abc_results(run, eps)
                
                metricPath = os.path.join(SAVE_DIR, model, metricDir)
                if not os.path.isdir(metricPath):
                    os.mkdir(metricPath)
                    
                savePath = os.path.join(metricPath, "size" + str(size) + "eps" + str(eps) + ".npy")
                np.save(savePath, results)
