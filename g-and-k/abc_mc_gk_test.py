import random
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
import time
from gk_utils import *
    
def abc_mc_gk_test(observedData, distanceMetric, abcIterations = 1_000_000):
    startTime = time.time()
    if distanceMetric == DistanceMetric.AUXILIARY:
        auxiliaryModel = fit_gaussian_mixture_EM(observedData, 3)
        weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
        
    if distanceMetric == DistanceMetric.WASS:
        observedData = np.sort(observedData, axis = 0)

    if distanceMetric == DistanceMetric.MMD:
        observedSqDistances = pdist(observedData, 'sqeuclidean')
        sigma = np.median(observedSqDistances) ** 0.5
        
    if distanceMetric == DistanceMetric.QUANTILE:
        observedQuantiles = np.quantile(observedData, [0.1, 0.9])
        
    simulationSize = len(observedData)
    modelChoices = np.zeros(abcIterations)
    distances = np.zeros(abcIterations) 
    times = np.zeros(abcIterations)
    for i in range(abcIterations):          
        model = random.randint(0, 1)
        theta1 = 0
        theta2 = np.random.uniform(-0.5, 5)
        if model == 1:
            theta1 = np.random.uniform(0, 4)

        theta = (0, 1, theta1, theta2)
        sample = gk_sample(theta, simulationSize)
            
        if distanceMetric == DistanceMetric.AUXILIARY:            
            distance = auxiliary_distance(sample, auxiliaryModel, weightMatrix)  
        elif distanceMetric == DistanceMetric.CVM:
            distance = cramer_von_mises_distance(observedData, sample)
        elif distanceMetric == DistanceMetric.WASS:
            distance = wasserstein_distance(observedData, sample)
        elif distanceMetric == DistanceMetric.MMD:
            distance = maximum_mean_discrepancy(observedData, sample, sigma, observedSqDistances)
        elif distanceMetric == DistanceMetric.QUANTILE:
            quantiles = np.quantile(sample, [0.1, 0.9])
            distance = np.sum(np.abs(observedQuantiles - quantiles))

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
    results = abc_mc_gk_test(observedData, distanceMetric)
    np.save(savePath, results)
