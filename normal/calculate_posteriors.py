import numpy as np
import os
from abc_posterior import abc_posterior

NUM_RUNS = 100
RUN_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/runs/m2/unknown_var/params_sampled"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/results/m2/unknown_var/params_sampled"

epsilons = [0.01, 0.025, 0.05, 0.1]
sizes = [100, 1000]
for size in sizes:  
    for metricDir in os.listdir(RUN_DIR):
        for eps in epsilons:
            results = np.zeros((NUM_RUNS, 2))
            for i in range(NUM_RUNS):                
                runPath = os.path.join(RUN_DIR, metricDir, "run" + str(i) + "size" + str(size) + ".npy")
                run = np.load(runPath)
                if metricDir == "stat" and "unknown_var" in SAVE_DIR:
                    run[:, 1] = run[:, 1]**2 + run[:, 2]**2
                results[i] = abc_posterior(run, eps)
            
            metricPath = os.path.join(SAVE_DIR, metricDir)
            if not os.path.isdir(metricPath):
                os.mkdir(metricPath)
                
            savePath = os.path.join(metricPath, "size" + str(size) + "eps" + str(eps) + ".npy")
            np.save(savePath, results)
