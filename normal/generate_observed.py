import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 

SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data/m2/unknown_var/params_sampled"
OBSERVED_SIZES = [100, 1000]
NUM_TO_GENERATE = 100
NULL_MEAN = 3
VAR_SCALE = 100

for observedSize in OBSERVED_SIZES:
    for i in range(NUM_TO_GENERATE):
        sd = np.random.gamma(0.1, 10)
        mean = np.random.normal(NULL_MEAN, VAR_SCALE**0.5)
        observed = np.random.normal(mean, sd, observedSize).reshape((observedSize, 1))
        savePath = os.path.join(SAVE_DIR, "sample" + str(i) + "size" + str(observedSize) + ".npy")
        np.save(savePath, observed)
