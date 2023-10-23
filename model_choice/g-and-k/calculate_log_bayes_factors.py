import numpy as np
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
from abc_log_bayes_factor import *

RUNS_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/test_a/eq3_vs_neq3"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/model_choice/results/test_a/eq3_vs_neq3"
SIZES = np.delete(np.linspace(10, 1000, 100).astype(int)[:50], 4)
EPSILONS = [0.01, 0.025, 0.05, 0.1]
DISTANCE_METRICS = [DistanceMetric.AUXILIARY, DistanceMetric.CVM, DistanceMetric.WASS]

for testDir in os.listdir(RUNS_DIR):
    abcLogBayesFactors = np.zeros((len(DISTANCE_METRICS), len(SIZES), len(EPSILONS)))    
    for i in range(len(DISTANCE_METRICS)):
        metric = DISTANCE_METRICS[i]
        if metric == DistanceMetric.AUXILIARY:
            runsUsingMetricDir = os.path.join(RUNS_DIR, testDir, "aux")
        elif metric == DistanceMetric.CVM:
            runsUsingMetricDir = os.path.join(RUNS_DIR, testDir, "cvm")
        elif metric == DistanceMetric.WASS:
            runsUsingMetricDir = os.path.join(RUNS_DIR, testDir, "wass")
        elif metric == DistanceMetric.MMD:
            runsUsingMetricDir = os.path.join(RUNS_DIR, testDir, "mmd")
        
        for j in range(len(SIZES)):
            runPath = os.path.join(runsUsingMetricDir, "run0size" + str(SIZES[j]) + ".npy")
            run = np.load(runPath)
            
            for k in range(len(EPSILONS)):
                logBayesFactor = abc_log_bayes_factor(run, EPSILONS[k])
                abcLogBayesFactors[i, j, k] = logBayesFactor
    
    for k in range(len(EPSILONS)):
        data = abcLogBayesFactors[:, :, k]
        np.save(os.path.join(SAVE_DIR, testDir, "abc_log_bayes_factors_eps-" + str(EPSILONS[k]) + ".npy"), data)