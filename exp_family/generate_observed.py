import numpy as np
import os
from exp_family.model import Model

SAVE_DIR = f"./exp_family/observed_data"
NUM_OBSERVED = 100
OBSERVED_SIZES = [100, 1000]
MODEL = Model.EXP
MODEL_MEAN = 2.0

if __name__ == "__main__":
    model_dir = os.path.join(SAVE_DIR, MODEL.name.lower())
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        
    if MODEL_MEAN <= 0:
        sys.exit("Error: Mean must be greater than 0.")  
        
    for size in OBSERVED_SIZES:
        size_dir = os.path.join(model_dir, f"size_{size}")
        if not os.path.isdir(size_dir):
            os.mkdir(size_dir)
        
        # Generating observed datasets under the model based on the specified mean
        for i in range(NUM_OBSERVED):
            if MODEL == Model.EXP:
                sample = np.random.exponential(MODEL_MEAN, (size, 1))
            elif MODEL == Model.LNORM:
                sample = np.random.lognormal(np.log(MODEL_MEAN) - 0.5, 1.0, (size, 1))
            else:
                sample = np.random.gamma(2.0, MODEL_MEAN / 2.0, (size, 1))
                
            np.save(os.path.join(size_dir, f"sample{i}.npy"), sample)
