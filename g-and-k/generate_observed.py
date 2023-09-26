import numpy as np
from utils import gk_sample

observedSizes = [100]
params = [3, 1, 2, 0.5]
numToGenerate = 100
for observedSize in observedSizes:
    for i in range(numToGenerate):
        observed = gk_sample(observedSize, params)
        np.save("../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data/" + str(observedSize) + "/sample" + str(i) + ".npy", observed)