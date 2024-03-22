import numpy as np
import random
from enum import Enum
from scipy.stats import levy_stable
import math

class Model(Enum):
    RANDOM = 0
    NEAREST = 1
    DISTANCE = 2

def toad_movement_sample(model, alpha, gamma, p0, d0 = None, numToads = 66, numDays = 63):
    toadPositions = np.zeros((numDays, numToads))
    for j in range(numToads):
        if model == Model.DISTANCE:
            refugeCount = 1
            refugeLocations = np.zeros(numDays)
        for i in range(1, numDays):
            newPos = toadPositions[i - 1, j] + levy_stable.rvs(alpha, 0, scale = gamma)
            if model == Model.DISTANCE:
                refugeDistances = np.abs(newPos - refugeLocations[:refugeCount])
                refugeProbs = p0 * np.exp(-refugeDistances / d0)
                noReturnProb = np.prod(1 - refugeProbs)
            else:
                noReturnProb = 1 - p0
            
            # No return vs return
            noReturn = np.random.uniform() < noReturnProb
            if noReturn:
                toadPositions[i, j] = newPos
                if model == Model.DISTANCE:
                    refugeLocations[refugeCount] = newPos
                    refugeCount += 1
            else:
                if model == Model.RANDOM:
                    returnIdx = 0
                    if i > 1:
                        returnIdx = random.randint(0, i - 1)
                    toadPositions[i, j] = toadPositions[returnIdx, j]
                elif model == Model.NEAREST:
                    returnIdx = np.argmin(np.abs(newPos - toadPositions[:i, j]))
                    toadPositions[i, j] = toadPositions[returnIdx, j]
                else:
                    returnIdx = np.random.choice(list(range(refugeCount)), p = refugeProbs / np.sum(refugeProbs))
                    toadPositions[i, j] = refugeLocations[returnIdx]

    return toadPositions

# Robust to NaN values in observed sample as filtering numpy arrays automatically ignores
# these values
def summarise_sample(toadData, lags):
    summaries = []
    for lag in lags:
        numDays, numToads = np.shape(toadData)
        diffs = np.abs(toadData[lag:, :] - toadData[:(numDays-lag), :]).flatten()
        returnCount = np.sum(diffs < 10.0)
        nonReturnData = diffs[diffs >= 10.0]
        summaries.append([returnCount, np.reshape(nonReturnData, (len(nonReturnData), 1))])
    return summaries

def get_statistics(summarisedData):
    stats = np.zeros(12 * len(summarisedData))
    for i in range(len(summarisedData)):
        stats[i*12] = summarisedData[i][0]
        if summarisedData[i][1].size == 0:
            stats[(i*12 + 1):(i*12 + 12)] = math.inf
        else:
            quantiles = np.quantile(summarisedData[i][1], np.arange(0, 1.1, 0.1))
            stats[(i*12 + 1):(i*12 + 11)] = np.log(np.diff(quantiles))
            stats[i*12 + 11] = quantiles[5]
    
    return stats
