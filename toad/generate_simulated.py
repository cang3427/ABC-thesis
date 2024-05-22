import os
import numpy as np
from toad.toad_utils import toad_movement_sample, Model

DATA_DIR = "./toad/simulated_data"
NUM_TO_SAMPLE = 100
MODEL_PARAMS = {Model.RANDOM: (1.7, 34, 0.6), Model.NEAREST: (1.83, 46, 0.65), Model.DISTANCE: (1.65, 32, 0.43, 758)}
MODEL = Model.RANDOM

if __name__ == "__main__":
    model_dir = os.path.join(DATA_DIR, MODEL.name.lower())
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    for i in range(NUM_TO_SAMPLE):
        sample = toad_movement_sample(MODEL, *MODEL_PARAMS[MODEL])
        np.save(os.path.join(model_dir, f"sample{i}.npy"), sample)
