import random
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
from enum import Enum
import time

class Model(Enum):
    EXP = 0
    LOGNORM = 1
    GAMMA = 2
    
def abc_model_choice_exp_family(observedData, distanceMetric, abcIterations = 1_000_000):
    startTime = time.time()
    simulationSize = len(observedData)
    observedData = np.reshape(observedData, (simulationSize, 1))
    if distanceMetric == DistanceMetric.AUXILIARY:
        auxiliaryModel = fit_gaussian_mixture_EM(observedData, 3)  
        weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
        
    if distanceMetric == DistanceMetric.WASS:
        observedData = np.sort(observedData, axis = 0)

    if distanceMetric == DistanceMetric.MMD:
        observedSqDistances = pdist(observedData, 'sqeuclidean')
        sigma = np.median(observedSqDistances) ** 0.5
        
    if distanceMetric == DistanceMetric.STATS:
        logObserved = np.log(observedData)
        observedStats = np.array([np.sum(observedData), np.sum(logObserved), np.sum(logObserved**2)])
        
    modelChoices = np.zeros(abcIterations)
    distances = np.zeros(abcIterations) 
    times = np.zeros(abcIterations)
    for i in range(abcIterations):          
        model = Model(random.randint(0, 2))
        if model == Model.EXP:
            theta = np.random.exponential()
            sample = np.random.exponential(theta, simulationSize)
        elif model == Model.LOGNORM:
            theta = np.random.normal()
            sample = np.random.lognormal(theta, 1, simulationSize)
        else:
            theta = np.random.exponential()
            sample = np.random.gamma(2, 1 / theta, simulationSize)
        sample = np.reshape(sample, (simulationSize, 1))
            
        if distanceMetric == DistanceMetric.CVM:
            distance = cramer_von_mises_distance(observedData, sample)
        elif distanceMetric == DistanceMetric.WASS:
            distance = wasserstein_distance(observedData, sample)
        elif distanceMetric == DistanceMetric.MMD:
            distance = maximum_mean_discrepancy(observedData, sample, sigma, observedSqDistances)
        elif distanceMetric == DistanceMetric.STATS:
            logSample = np.log(sample)
            sampleStats = np.array([np.sum(sample), np.sum(logSample), np.sum(logSample**2)]) 
            distance = np.sum((observedStats - sampleStats)**2)        

        modelChoices[i] = model
        distances[i] = distance
        times[i] = time.time() - startTime

    results = np.column_stack((modelChoices, distances, times))
    return results

def main(args):
    (distanceMetric, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_model_choice_exp_family(observedData, distanceMetric)
    np.save(savePath, results)
