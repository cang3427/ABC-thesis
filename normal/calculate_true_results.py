import numpy as np
import os
from exact_log_bayes_factor import *

NULL_MEAN = 3
VAR = 1
VAR_SCALE = 100
NUM_RUNS = 100
OBS_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data/m2/params_sampled/known_var"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/results/m2/known_var/params_sampled/true"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

sizes = [100, 1000]
for size in sizes:
    posteriorProbs = np.zeros((NUM_RUNS, 2))
    logBayesFactors = np.zeros((NUM_RUNS))
    for i in range(NUM_RUNS):
        dataPath = os.path.join(OBS_DIR, "sample" + str(i) + "size" + str(size) + ".npy")
        data = np.load(dataPath)
        posteriorProbs[i] = posterior_probs(data, NULL_MEAN, VAR, VAR_SCALE)
        logBayesFactors[i] = exact_log_bayes_factor(data, NULL_MEAN, VAR, VAR_SCALE)
        
    np.save(os.path.join(SAVE_DIR, "posteriors_size" + str(size) + ".npy"), posteriorProbs)
    np.save(os.path.join(SAVE_DIR, "logbfs_size" + str(size) + ".npy"), logBayesFactors)
