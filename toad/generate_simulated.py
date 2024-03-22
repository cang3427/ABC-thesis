import os
import numpy as np
from toad_utils import toad_movement_sample, Model

DATA_DIR = "../../project/RDS-FSC-ABCMC-RW/toad/model_choice/test_data"
NUM_TO_SAMPLE = 100
ALPHA = 1.7
GAMMA = 35
PROB_0 = 0.6
DIST_0 = 750

for model in Model:
    d0 = None
    if model == Model.RANDOM:
        saveDir = os.path.join(DATA_DIR, "m1")
    if model == Model.NEAREST:
        saveDir = os.path.join(DATA_DIR, "m2")
    if model == Model.DISTANCE:
        saveDir = os.path.join(DATA_DIR, "m3")
        d0 = DIST_0
    
    for i in range(NUM_TO_SAMPLE):
        sample = toad_movement_sample(model, ALPHA, GAMMA, PROB_0, d0)
        np.save(os.path.join(saveDir, "sample" + str(i) + ".npy"), sample)
