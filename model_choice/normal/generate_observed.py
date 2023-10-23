import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import normal_sample

SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data"
OBSERVED_SIZES = np.linspace(10, 1000, 100).astype(int)
MEAN = 3
SD = 1
NUM_TO_GENERATE = 100

for observedSize in OBSERVED_SIZES:
    for i in range(NUM_TO_GENERATE):
        observed = normal_sample(MEAN, SD, observedSize)
        savePath = os.path.join(SAVE_DIR, "sample" + str(i) + "size" + str(observedSize) + ".npy")
        if os.path.exists(savePath):
            continue
        np.save(savePath, observed)
