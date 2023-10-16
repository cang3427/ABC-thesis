import numpy as np
from utils import gk_sample, normal_sample, DistributionType

observedSizes = np.linspace(10, 1000, 100).astype(int)
params = [3, 1]
numToGenerate = 1
distributionType = DistributionType.NORMAL
for observedSize in observedSizes:
    for i in range(numToGenerate):
        if (distributionType == DistributionType.GANDK):
            observed = gk_sample(observedSize, params)
            savePath = "../../project/RDS-FSC-ABCMC-RW/g-and-k/observed_data/" + str(observedSize) + "/sample" + str(i) + ".npy"
        elif (distributionType == DistributionType.NORMAL):
            observed = normal_sample(observedSize, params)
            savePath = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data/" + "sample" + str(i) + "size" + str(observedSize) + ".npy"
        np.save(savePath, observed)