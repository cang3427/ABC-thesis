import random
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
from enum import Enum

class ParameterName(Enum):
    AA = 0
    BB = 1
    GG = 2
    KK = 3

def abc_model_choice_gk(observedData, paramA, paramB, paramG, paramK, paramToTest, distanceMetric, numComp = 3, abcIterations = 10_000_000):
    if distanceMetric == DistanceMetric.AUXILIARY:
        auxiliaryModel = fit_gaussian_mixture_EM(observedData, numComp)  
        weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
        
    if distanceMetric == DistanceMetric.WASS:
        observedData = np.sort(observedData)

    if distanceMetric == DistanceMetric.MMD:
        observedSqDistances = pdist(observedData, 'sqeuclidean')
        sigma = np.median(observedSqDistances) ** 0.5
        
    modelChoices = np.zeros(abcIterations)   
    thetas = np.zeros((abcIterations, 4))
    distances = np.zeros(abcIterations) 
    simulationSize = len(observedData)
    for i in range(abcIterations):             
        # Selecting the model to sample from
        model = random.randint(0, 1)
        theta = np.zeros(4)           
        if ParameterName.AA == paramToTest:
            if model == 0:
                theta[0] = paramA
            else:
                theta[0] = np.random.normal(paramA, 10**0.5)
        elif paramA is None:
            theta[0] = np.random.normal(0, 10**0.5)
        else:
            theta[0] = paramA
            
        if ParameterName.BB == paramToTest:
            if model == 0:
                theta[1] = paramB
            else:
                theta[1] = paramB + np.random.gamma(0.1, 0.1)
        elif paramB is None:
            theta[1] = np.random.gamma(0.1, 0.1)
        else:
            theta[1] = paramB
            
        if ParameterName.GG == paramToTest:
            if model == 0:
                theta[2] = paramG
            else:
                theta[2] = np.random.normal(paramG, 10**0.5)
        elif paramG is None:
            theta[2] = np.random.normal(0, 10**0.5)
        else:
            theta[2] = paramG
            
        if ParameterName.KK == paramToTest:
            if model == 0:
                theta[3] = paramK
            else:
                theta[3] = np.random.uniform(-0.5, 10)
        elif paramK is None:
            theta[3] = np.random.uniform(-0.5, 10)
        else:
            theta[3] = paramK
            
        simulatedSample = gk_sample(theta, simulationSize)
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
        thetas[i] = theta
        distances[i] = distance

    results = np.column_stack((np.reshape(modelChoices, (len(modelChoices), 1)), thetas, np.reshape(distances, (len(distances), 1))))
    return results

def main(args):
    (paramA, paramB, paramG, paramK, testParam, distanceMetric, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_model_choice_gk(observedData, paramA, paramB, paramG, paramK, testParam, distanceMetric)
    np.save(savePath, results)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    