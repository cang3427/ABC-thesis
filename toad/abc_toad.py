import random
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
import time
import math
from toad_utils import *

def abc_toad(observedData, distanceMetric, numToads = 66, numDays = 63, lags = [1, 2, 4, 8], abcIterations = 100_000):
    startTime = time.time()
    observedSummaries = summarise_sample(observedData, lags)
    nlags = len(lags)
    if distanceMetric == DistanceMetric.MMD:
        observedSqDistances = [pdist(observedSummaries[i][1], 'sqeuclidean') for i in range(nlags)]
        sigmas = [np.median(observedSqDistances[i]) ** 0.5 for i in range(nlags)]
    elif distanceMetric == DistanceMetric.STATS:
        observedStats = get_statistics(observedSummaries)
        
    simulationSize = len(observedData)
    modelChoices = np.zeros(abcIterations)
    if distanceMetric == DistanceMetric.STATS:
        distances = np.zeros((abcIterations, 48)) 
    else:
        distances = np.zeros((abcIterations, 8))
    times = np.zeros(abcIterations)
    for i in range(abcIterations):  
        model = Model(random.randint(0, 2))
        alpha = np.random.uniform(1, 2)
        gamma = np.random.uniform(10, 100)
        p0 = np.random.uniform(0, 1)
        d0 = None
        
        if model == Model.DISTANCE:
            d0 = np.random.uniform(20, 2000)
                
        sample = toad_movement_sample(model, alpha, gamma, p0, d0)
        summaries = summarise_sample(sample, lags)
    
        if distanceMetric == DistanceMetric.STATS:
            stats = get_statistics(summaries)
            distanceList = (observedStats - stats)**2
        else:     
            returnCountDistances = [abs(observedSummaries[i][0] - summaries[i][0]) for i in range(nlags)]  
            if distanceMetric == DistanceMetric.CVM:
                nonReturnDistances = [cramer_von_mises_distance(observedSummaries[i][1], summaries[i][1]) if summaries[i][1].size > 0 else math.inf for i in range(nlags)]
            elif distanceMetric == DistanceMetric.WASS:
                nonReturnDistances = [wasserstein_distance(observedSummaries[i][1], summaries[i][1]) if summaries[i][1].size > 0 else math.inf for i in range(nlags)]
            elif distanceMetric == DistanceMetric.MMD:
                nonReturnDistances = [maximum_mean_discrepancy(observedSummaries[i][1], summaries[i][1], sigmas[i], observedSqDistances[i]) if summaries[i][1].size > 0 else math.inf for i in range(nlags)]

            distanceList = returnCountDistances + nonReturnDistances

        modelChoices[i] = model.value
        distances[i] = distanceList
        times[i] = time.time() - startTime

    results = np.column_stack((modelChoices, distances, times))
    return results

def main(args):
    (distanceMetric, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_toad(observedData, distanceMetric)
    np.save(savePath, results)
