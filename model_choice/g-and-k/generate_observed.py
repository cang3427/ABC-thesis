import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import gk_sample

SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data"
OBSERVED_SIZES = np.linspace(10, 1000, 100).astype(int)
PARAMS = [3, 1, 2, 0.5]
NUM_TO_GENERATE = 1

for observedSize in OBSERVED_SIZES:
    for i in range(NUM_TO_GENERATE):
        observed = gk_sample(PARAMS, observedSize)
        savePath = os.path.join(SAVE_DIR, "sample" + str(i) + "size" + str(observedSize) + ".npy")
        np.save(savePath, observed)