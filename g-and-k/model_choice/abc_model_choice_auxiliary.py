import random
import sys, os, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
from enum import Enum
import time
    
class ParameterType(Enum):
    EQUAL = 0
    LESS = 1
    GREATER = 2
    NOTEQUAL = 3
    
def abc_model_choice_auxiliary(observedData, nullTestParam, alternativeTestParam, nullType, alternativeType, paramToTestIdx, simulationSize, distType, numComp = 3, abcIterations = 10_000_000, lower = 0, upper = 10):
    # Fitting auxiliary model to the observed data
    auxiliaryModel = fit_gaussian_mixture_EM(observedData, numComp)  

    # Calculating weight matrix which is the inverse of the observed information
    # using the MLEs
    weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
    
    if (distType == DistributionType.GANDK):
        numParams = 4
    elif (distType == DistributionType.NORMAL):
        numParams = 2
        
    modelChoices = np.zeros(abcIterations)   
    thetas = np.zeros((abcIterations, numParams))
    distances = np.zeros(abcIterations) 
    start = time.time()
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
        otherParams = np.random.uniform(lower, upper, numParams - 1)
        thetaProp = np.insert(otherParams, paramToTestIdx, testParam)
        
        # Generating the summary statistic for a simulated sample with
        # the proposal parameters (score of auxiliarly model at the
        # MLE for the observed sample)
        if (distType == DistributionType.GANDK):
            simulatedSample = gk_sample(simulationSize, thetaProp)
        elif (distType == DistributionType.NORMAL):
            simulatedSample = normal_sample(simulationSize, thetaProp)
        statistic = gaussian_mixture_score(simulatedSample, auxiliaryModel)
        
        # Distance function (Mahalanobis distance) for the summary statistic 
        # (note that we  do not need to consider the observed summary statistic
        # as in this case it is 0 since the score is 0 at the MLE with the observed
        # data )
        propDist = np.linalg.multi_dot([statistic, weightMatrix, statistic.T])     
        
        # Store current values   
        modelChoices[i] = model
        thetas[i] = thetaProp
        distances[i] = propDist 

    results = np.column_stack((np.reshape(modelChoices, (len(modelChoices), 1)), thetas, np.reshape(distances, (len(distances), 1))))
    return results

def main(args):
    (nullParam, alternativeParam, nullType, alternativeType, testIndex, observedSize, distType, observedPath, savePath) = args
    if os.path.exists(savePath):
        return
    observedData = np.load(observedPath)
    results = abc_model_choice_auxiliary(observedData, nullParam, alternativeParam, nullType, alternativeType, testIndex, observedSize, distType)
    np.save(savePath, results)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)