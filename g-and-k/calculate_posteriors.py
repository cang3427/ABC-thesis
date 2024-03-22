import numpy as np
import os
from abc_posterior import abc_posterior

NUM_RUNS = 100
RUN_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/gk_test/m2_true"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/gk_test/results/m2_true"

if __name__ == "__main__":
    epsilons = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    sizes = [100, 1000]
    for size in sizes:  
        for metricDir in os.listdir(RUN_DIR):
            if metricDir == "aux":
                continue
            for eps in epsilons:
                results = np.zeros((NUM_RUNS, 2))
                for i in range(NUM_RUNS):                
                    runPath = os.path.join(RUN_DIR, metricDir, "run" + str(i) + "size" + str(size) + ".npy")
                    run = np.load(runPath)
                    results[i] = abc_posterior(run, eps)
                
                metricPath = os.path.join(SAVE_DIR, metricDir)
                if not os.path.isdir(metricPath):
                    os.mkdir(metricPath)
                    
                savePath = os.path.join(metricPath, "size" + str(size) + "eps" + str(eps) + ".npy")
                np.save(savePath, results)
