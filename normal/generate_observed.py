import numpy as np
import os

OBSERVED_SIZES = [100, 1000]
NUM_TO_GENERATE = 100
NULL_MEAN = 3
VAR = 1
PRIOR_VAR_SCALE = 100
MODEL = 1
SAVE_DIR = f"./normal/observed_data/m{MODEL}"

if __name__ == "__main__":
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    sd = VAR**0.5   
    for observed_size in OBSERVED_SIZES:
        size_dir = os.path.join(SAVE_DIR, f"size_{observed_size}")
        if not os.path.isdir(size_dir):
            os.mkdir(size_dir)
        
        for i in range(NUM_TO_GENERATE):
            if MODEL == 0:
                mean = NULL_MEAN
            else:
                mean = np.random.normal(NULL_MEAN, PRIOR_VAR_SCALE**0.5 * sd)
                
            observed = np.random.normal(mean, sd, (observed_size, 1))
            save_path = os.path.join(size_dir, f"sample{i}.npy")
            
            np.save(save_path, observed)
