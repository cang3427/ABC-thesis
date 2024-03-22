import numpy as np
import os
import math
from toad_utils import *
import matplotlib.pyplot as plt

NUM_RUNS = 100
RUN_DIR = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/runs/m1"
RESULTS_DIR = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/results/m1"
EPSILONS = [0.01, 0.025, 0.05, 0.1]
ALPHA = 1.7
GAMMA = 35
PROB_0 = 0.6
DIST_0 = 750

def get_statistics_weights(model):
    d0 = None
    if model == Model.DISTANCE:
        d0 = DIST_0
        
    summaryStats = np.zeros((10, 48))
    for i in range(10):
        print(i)
        toadData = toad_movement_sample(model, ALPHA, GAMMA, PROB_0, d0)
        summaryStats[i] = get_statistics(summarise_sample(toadData, [1, 2, 4, 8]))
        
    return np.std(summaryStats, axis = 0)
        
def abc_posterior_probs(run, epsilon, distanceWeights):
    distances = run[:, 1:-1]
    combinedDistances = np.sum(distances / distanceWeights, axis = 1)
    if np.isnan(combinedDistances).any():
        combinedDistances = np.nan_to_num(combinedDistances, nan = math.inf)
    
    threshold = np.quantile(combinedDistances, epsilon)
    posteriorProbs = np.zeros(3)
    acceptedCount = np.sum(combinedDistances < threshold)
    for i in range(3):
        modelDistances = combinedDistances[run[:, 0] == i]
        posteriorProbs[i] = np.sum(modelDistances < threshold) / acceptedCount
    
    return posteriorProbs

if __name__ == "__main__":
    run = np.load("toad/runs/true/run_wass.npy")
    run[np.isinf(run)] = np.nan
    weights = np.nanstd(run[:, 1:-1], axis = 0)
    for eps in [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]:
        print(abc_posterior_probs(run[:30000], eps, weights))
        
    # for metricDir in os.listdir(os.path.join(RUN_DIR)):
    #     if metricDir == "stat":
    #         if os.path.dirname(RUN_DIR) == "m1":
    #             distanceWeights = get_statistics_weights(Model.RANDOM)
    #         elif os.path.dirname(RUN_DIR) == "m2":
    #             distanceWeights = get_statistics_weights(Model.NEAREST)
    #         else:
    #             distanceWeights = get_statistics_weights(Model.DISTANCE)
    #     else:
    #         distanceWeights = np.nanstd(run[:, 1:-1], axis = 0)
            
    #     for eps in EPSILONS:
    #         results = np.zeros((NUM_RUNS, 3))
    #         for i in range(NUM_RUNS):                
    #             runPath = os.path.join(RUN_DIR, metricDir, "run" + str(i) + ".npy")
    #             run = np.load(runPath)
    #             results[i] = abc_posterior_probs(run, eps, distanceWeights)
            
    #         metricPath = os.path.join(RESULTS_DIR, metricDir)
    #         if not os.path.isdir(metricPath):
    #             os.mkdir(metricPath)
                
    #         savePath = os.path.join(metricPath, "posterior_probs_eps" + str(eps) + ".npy")
    #         np.save(savePath, results)