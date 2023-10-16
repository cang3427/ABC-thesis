import random
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
from enum import Enum
from math import sqrt

class ParameterType(Enum):
    EQUAL = 0
    LESS = 1
    GREATER = 2
    NOTEQUAL = 3
    
def abc_model_choice_normal(observedData, nullMean, variance, priorVarianceScale, distanceMetric, numComp = 3, abcIterations = 10_000_000):
    if (distanceMetric == DistanceMetric.AUXILIARY):
        auxiliaryModel = fit_gaussian_mixture_EM(observedData, numComp)  
        weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
        
    if distanceMetric == DistanceMetric.WASS:
        observedData = np.sort(observedData)

    if distanceMetric == DistanceMetric.MMD:
        observedSqDistances = pdist(observedData, 'sqeuclidean')
        sigma = np.median(observedSqDistances) ** 0.5
        
    modelChoices = np.zeros(abcIterations)   
    thetas = np.zeros((abcIterations, 2))
    distances = np.zeros(abcIterations) 
    simulationSize = len(observedData)
    sd = sqrt(variance)
    priorSd = sqrt(variance * priorVarianceScale)        
    for i in range(abcIterations): 
        print(i)    
        # Selecting the model to sample from and defining the mean
        model = random.randint(0, 1)
        if model == 0:
            modelMean = nullMean
        else:
            modelMean = np.random.normal(nullMean, priorSd)
        
        # Simulate sample from sampled model
        simulatedSample = normal_sample(modelMean, sd, simulationSize)
        
        # Calculate the chosen distance for the simulated sample
        if distanceMetric == DistanceMetric.AUXILIARY:            
            statistic = gaussian_mixture_score(simulatedSample, auxiliaryModel)
            distance = np.linalg.multi_dot([statistic, weightMatrix, statistic.T])  
        elif distanceMetric == DistanceMetric.CVM:
            distance = cramer_von_mises_distance(observedData, simulatedSample)
        elif distanceMetric == DistanceMetric.WASS:
            distance = wasserstein_distance(observedData, simulatedSample)
        elif distanceMetric == DistanceMetric.MMD:
            distance = maximum_mean_discrepancy(observedData, simulatedSample, sigma, observedSqDistances)
        
        # Store current values   
        modelChoices[i] = model
        thetas[i] = np.array([modelMean, variance])
        distances[i] = distance

    results = np.column_stack((np.reshape(modelChoices, (len(modelChoices), 1)), thetas, np.reshape(distances, (len(distances), 1))))
    return results

def main(args):
    (nullMean, variance, priorVarianceScale, distanceMetric, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_model_choice_normal(observedData, nullMean, variance, priorVarianceScale, distanceMetric)
    np.save(savePath, results)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)