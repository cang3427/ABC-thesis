import numpy as np
import sys, os
from exact_log_bayes_factor import *
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
from abc_log_bayes_factor import *

OBS_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data"
RUNS_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/runs/eq3_vs_neq3"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/results/eq3_vs_neq3"

sizes = np.linspace(10, 1000, 100).astype(int)
epsilons = [0.01, 0.025, 0.05, 0.1]
distanceMetrics = [DistanceMetric.AUXILIARY, DistanceMetric.CVM, DistanceMetric.WASS]
abcLogBayesFactors = np.zeros((len(distanceMetrics), len(sizes), len(epsilons)))
for i in range(len(distanceMetrics)):
    metric = distanceMetrics[i]
    if metric == DistanceMetric.AUXILIARY:
        runsUsingMetricDir = os.path.join(RUNS_DIR, "aux")
    elif metric == DistanceMetric.CVM:
        runsUsingMetricDir = os.path.join(RUNS_DIR, "cvm")
    elif metric == DistanceMetric.WASS:
        runsUsingMetricDir = os.path.join(RUNS_DIR, "wass")
    elif metric == DistanceMetric.MMD:
        runsUsingMetricDir = os.path.join(RUNS_DIR, "mmd")
    
    for j in range(len(sizes)):
        runPath = os.path.join(runsUsingMetricDir, "run0size" + str(sizes[j]) + ".npy")
        run = np.load(runPath)
        
        for k in range(len(epsilons)):
            logBayesFactor = abc_log_bayes_factor(run, epsilons[k])
            abcLogBayesFactors[i, j, k] = logBayesFactor

exactLogBayesFactors = np.zeros(len(sizes))
nullMean = 3
variance = 1
priorVarianceScale = 100
for i in range(len(sizes)):
    observedPath = os.path.join(OBS_DIR, "sample0size" + str(sizes[i]) + ".npy")
    observedData = np.load(observedPath)
    exactLogBayesFactors[i] = exact_log_bayes_factor(observedData, nullMean, variance, priorVarianceScale)
    
exactLogBayesFactors = np.reshape(exactLogBayesFactors, (1, len(sizes)))
for k in range(len(epsilons)):
    data = np.concatenate((exactLogBayesFactors, abcLogBayesFactors[:, :, k]), axis = 0)
    np.save(os.path.join(SAVE_DIR, "abc_log_bayes_factors_eps-" + str(epsilons[k]) + ".npy"), data)