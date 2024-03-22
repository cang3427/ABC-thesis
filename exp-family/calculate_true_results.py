import numpy as np
import os
from log_bayes_factor import *
from abc_model_choice_exp_family import Model

NUM_RUNS = 100
OBS_DIR = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/observed_data/exp/params_sampled"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/results/params_sampled/exp/true"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

sizes = [100, 1000]
for size in sizes:
    posteriorProbs = np.zeros((NUM_RUNS, 3))
    logBayesFactors = np.zeros((NUM_RUNS, 3))
    for i in range(NUM_RUNS):
        dataPath = os.path.join(OBS_DIR, "sample" + str(i) + "size" + str(size) + ".npy")
        data = np.load(dataPath)
        for m in Model:
            posteriorProbs[i, m.value] = posterior_prob(data, m)
            
        logBayesFactors[i, 0] = log_bayes_factor(data, Model.EXP, Model.LOGNORM)
        logBayesFactors[i, 1] = log_bayes_factor(data, Model.EXP, Model.GAMMA)
        logBayesFactors[i, 2] = log_bayes_factor(data, Model.LOGNORM, Model.GAMMA)

    np.save(os.path.join(SAVE_DIR, "posteriors_size" + str(size) + ".npy"), posteriorProbs)
    np.save(os.path.join(SAVE_DIR, "logbfs_size" + str(size) + ".npy"), logBayesFactors)
