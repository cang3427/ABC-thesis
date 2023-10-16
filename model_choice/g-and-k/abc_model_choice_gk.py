import random
import sys, os, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
from enum import Enum

class ParameterType(Enum):
    EQUAL = 0
    LESS = 1
    GREATER = 2
    NOTEQUAL = 3
    
def abc_model_choice_gk(observedData, nullTestParam, alternativeTestParam, nullType, alternativeType, paramToTestIdx, distanceMetric, lower = 0, upper = 10, numComp = 3, abcIterations = 10_000_000):
    if (distanceMetric == DistanceMetric.AUXILIARY):
        auxiliaryModel = fit_gaussian_mixture_EM(observedData, numComp)  
        weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
        
    modelChoices = np.zeros(abcIterations)   
    thetas = np.zeros((abcIterations, 4))
    distances = np.zeros(abcIterations) 
    simulationSize = len(observedData)
    for i in range(abcIterations):             
        # Selecting the model to sample from
        model = random.randint(0, 1)
        if (model == 0):
            fixedParam = nullTestParam
            testType = nullType
        else:
            fixedParam = alternativeTestParam
            testType = alternativeType
            
        # Selecting test parameter based on the type of test associated with the sampled model
        if (testType == ParameterType.EQUAL):
            testParam = fixedParam
        elif (testType == ParameterType.LESS):
            testParam = np.random.uniform(lower, fixedParam)
        elif (testType == ParameterType.GREATER):
            testParam = np.random.uniform(fixedParam, upper)
        else:
            testParam = np.random.uniform(lower, upper, 1)
            
        # Generating other proposal parameters from a U(lower, upper) prior
        otherParams = np.random.uniform(lower, upper, 3)
        thetaProp = np.insert(otherParams, paramToTestIdx, testParam)
        
        # Generating the summary statistic for a simulated sample with
        # the proposal parameters (score of auxiliarly model at the
        # MLE for the observed sample)
        simulatedSample = gk_sample(simulationSize, thetaProp)
        if distanceMetric == DistanceMetric.AUXILIARY:            
            statistic = gaussian_mixture_score(simulatedSample, auxiliaryModel)
            distance = np.linalg.multi_dot([statistic, weightMatrix, statistic.T])  
        elif distanceMetric == DistanceMetric.CVM:
            distance = cramer_von_mises_distance(observedData, simulatedSample)
        elif distanceMetric == DistanceMetric.WASS:
            distance = wasserstein_distance(observedData, simulatedSample)
        elif distanceMetric == DistanceMetric.MMD:
            distance = maximum_mean_discrepancy(observedData, simulatedSample)
        
        # Store current values   
        modelChoices[i] = model
        thetas[i] = thetaProp
        distances[i] = distance

    results = np.column_stack((np.reshape(modelChoices, (len(modelChoices), 1)), thetas, np.reshape(distances, (len(distances), 1))))
    return results

def main(args):
    (nullParam, alternativeParam, nullType, alternativeType, testIndex, observedSize, distType, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_model_choice_gk(observedData, nullParam, alternativeParam, nullType, alternativeType, testIndex, observedSize, distType)
    np.save(savePath, results)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)

observedData = np.load("g-and-k/model_choice/data/sample0size100.npy")
nullTestParam = 0
alternativeTestParam = 0
nullType = ParameterType.EQUAL
alternativeType = ParameterType.NOTEQUAL
paramIdx = 0
simSize = 100
distType = DistributionType.NORMAL
abc_model_choice(observedData, nullTestParam, alternativeTestParam, nullType, alternativeType, paramIdx, simSize, distType)
