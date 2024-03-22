import random
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
from enum import Enum
from math import sqrt
import time
    
def abc_model_choice_normal(observedData, nullMean, variance, priorVarianceScale, distanceMetric, abcIterations = 1_000_000):
    startTime = time.time()
    if distanceMetric == DistanceMetric.STATS:
        observedMean = np.mean(observedData)
        if variance is None:
            observedVar = np.var(observedData, ddof = 1)
        
    if distanceMetric == DistanceMetric.WASS:
        observedData = np.sort(observedData, axis = 0)

    if distanceMetric == DistanceMetric.MMD:
        observedSqDistances = pdist(observedData, 'sqeuclidean')
        sigma = np.sqrt(np.median(observedSqDistances))
        
    modelChoices = np.zeros(abcIterations)  
    times = np.zeros(abcIterations)
    simulationSize = len(observedData)
    
    if variance is None:
        sd = None
        priorMeanSd = priorVarianceScale**0.5
        if distanceMetric == DistanceMetric.STATS:
            distances = np.zeros((abcIterations, 2))
        else:
            distances = np.zeros(abcIterations)
    else:
        sd = variance**0.5
        priorMeanSd = (variance * priorVarianceScale)**0.5       
        distances = np.zeros(abcIterations)  

    for i in range(abcIterations): 
        # Selecting the model to sample from and defining the mean
        model = random.randint(0, 1)
        if model == 0:
            modelMean = nullMean
        else:
            modelMean = np.random.normal(nullMean, priorMeanSd)
            
        if sd is None:
            priorSd = np.random.gamma(0.1, 10)
        else: 
            priorSd = sd
        
        # Simulate sample from sampled model
        simulatedSample = np.random.normal(modelMean, priorSd, simulationSize).reshape((simulationSize, 1))
        
        # Calculate the chosen distance for the simulated sample
        if distanceMetric == DistanceMetric.STATS:
            simulatedMean = np.mean(simulatedSample)
            if variance is None:     
                simulatedVar = np.var(simulatedSample, ddof = 1)
                distance = np.array([abs(simulatedMean - observedMean), abs(simulatedVar - observedVar)])   
            else:
                distance = abs(simulatedMean - observedMean)
        elif distanceMetric == DistanceMetric.CVM:
            distance = cramer_von_mises_distance(observedData, simulatedSample)
        elif distanceMetric == DistanceMetric.WASS:
            distance = wasserstein_distance(observedData, simulatedSample)
        elif distanceMetric == DistanceMetric.MMD:
            distance = maximum_mean_discrepancy(observedData, simulatedSample, sigma, observedSqDistances)
        
        # Store current values   
        modelChoices[i] = model
        distances[i] = distance
        times[i] = time.time() - startTime
        
    results = np.column_stack((modelChoices, distances, times))
    return results

def main(args):
    (nullMean, variance, priorVarianceScale, distanceMetric, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_model_choice_normal(observedData, nullMean, variance, priorVarianceScale, distanceMetric)
    np.save(savePath, results)
