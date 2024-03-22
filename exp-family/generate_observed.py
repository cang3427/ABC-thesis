import numpy as np
import os

SAVE_PATH = "../../project/RDS-FSC-ABCMC-RW/exp-family/model_choice/observed_data"
NUM_OBSERVED = 100
OBSERVED_SIZES = [100, 1000]
USE_PRIOR = True

if __name__ == "__main__":
    modelDirs = ["exp", "lnorm"]
    for model in modelDirs:
        modelPath = os.path.join(SAVE_PATH, model)
        if not os.path.isdir(modelPath):
            os.mkdir(modelPath)
        for size in OBSERVED_SIZES:
            for i in range(NUM_OBSERVED):
                if model == "exp":
                    if USE_PRIOR:
                        theta = np.random.exponential()
                        sample = np.random.exponential(theta, size)
                    else:
                        sample = np.random.exponential(0.5, size)
                elif model == "lnorm":
                    if USE_PRIOR:
                        theta = np.random.exponential()
                        sample = np.random.lognormal(theta, 1.0, size)
                    else:
                        sample = np.random.lognormal(np.log(2.0) - 0.5, 1.0, size)
                else:
                    if USE_PRIOR:
                        theta = 1 / np.random.exponential()
                        sample = np.random.gamma(2, theta, size)
                    else:
                        sample = np.random.gamma(2.0, 1.0, size)
                        
                if USE_PRIOR:
                    saveDir = os.path.join(modelPath, "params_sampled")
                else:
                    saveDir = os.path.join(modelPath, "params_fixed")
                    
                np.save(os.path.join(saveDir, "sample" + str(i) + "size" + str(size) + ".npy"), sample)