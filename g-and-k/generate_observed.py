import numpy as np
from utils import gk_sample

observedSize = 1000
params = [3, 1, 2, 0.5]
numToGenerate = 10
for i in range(numToGenerate):
    observed = gk_sample(observedSize, params)
    np.save("g-and-k/observed_data/" + str(observedSize) + "/sample" + str(i) + ".npy", observed)
    