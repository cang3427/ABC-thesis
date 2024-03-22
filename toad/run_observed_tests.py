import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import DistanceMetric
from run_jobs import run_jobs
from abc_toad import *

OBSERVED_PATH = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/true_data/observed_data.npy"
SAVE_PATH = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/runs/true"
# DISTANCE_METRICS = [DistanceMetric.WASS, DistanceMetric.STATS]
# NUM_WORKERS = 2

if __name__ == "__main__":
    results = abc_toad(np.load(OBSERVED_PATH), DistanceMetric.STATS)
    np.save(os.path.join(SAVE_PATH, "run_stat.npy"), results)
    # saveArgs = []
    # for metric in DISTANCE_METRICS:
    #     if metric == DistanceMetric.CVM:
    #         suff = "cvm"
    #     elif metric == DistanceMetric.WASS:
    #         suff = "wass"
    #     elif metric == DistanceMetric.MMD:
    #         suff = "mmd"   
    #     elif metric == DistanceMetric.STATS: 
    #         suff = "stat"
    #     else:
    #         continue        
    #     saveArgs.append(os.path.join(SAVE_PATH, "run_" + suff + ".npy"))
        
    # runArgs = zip(DISTANCE_METRICS, [OBSERVED_PATH] * len(DISTANCE_METRICS), saveArgs) 
    # run_jobs(runArgs, NUM_WORKERS)
