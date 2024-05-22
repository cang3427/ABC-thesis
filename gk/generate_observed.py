import numpy as np
import os
from gk.gk_funcs import gk_sample

SAVE_DIR = "./gk/observed_data"
OBSERVED_SIZES = [100, 1000]
NUM_TO_GENERATE = 100
MODEL_PARAMS = {0: (2,), 1: (1, 2)}
MODEL = 0

if __name__ == "__main__":
    model_dir = os.path.join(SAVE_DIR, f"m{MODEL}")        
    for size in OBSERVED_SIZES:
        model_size_dir = os.path.join(model_dir, f"size_{size}")
        if not os.path.isdir(model_size_dir):
            os.makedirs(model_size_dir)
            
        for i in range(NUM_TO_GENERATE):
            if MODEL == 0:
                observed = gk_sample((0, 1, 0) + MODEL_PARAMS[MODEL], (size, 1))
            else:
                observed = gk_sample((0, 1) + MODEL_PARAMS[MODEL], (size, 1))
                
            save_path = os.path.join(model_size_dir, f"sample{i}.npy")
            np.save(save_path, observed)
