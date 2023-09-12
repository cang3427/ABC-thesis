import sys, os, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from utils import *
import warnings
warnings.filterwarnings("ignore")
    
def gk_abc(observedData, simulationSize, numComp, abcIterations, lower = 0, upper = 10):
    numParams = 4    
    # Fitted auxiliary model to the observed data
    auxiliaryModel = fit_gaussian_mixture_EM(observedData, numComp)  

    # Calculating weight matrix which is the inverse of the observed information
    # using the MLEs
    weightMatrix = np.linalg.inv(gaussian_mixture_information(observedData, auxiliaryModel))
    thetas = np.zeros((abcIterations, numParams))
    distances = np.zeros(abcIterations)    
    for i in range(abcIterations):  
        # Generating proposal parameters from a U(lower, upper) prior
        thetaProp = np.random.uniform(lower, upper, numParams)
        print(thetaProp)
        
        # Generating the summary statistic for a simulated sample with
        # the proposal parameters (score of auxiliarly model at the
        # MLE for the observed sample)
        simulatedSample = gk_sample(simulationSize, thetaProp)
        statistic = gaussian_mixture_score(simulatedSample, auxiliaryModel)
        
        # Distance function (Mahalanobis distance) for the summary statistic 
        # (note that we  do not need to consider the observed summary statistic
        # as in this case it is 0 since the score is 0 at the MLE with the observed
        # data )
        propDist = np.linalg.multi_dot([statistic, weightMatrix, statistic.T])     

        # Store current values      
        thetas[i] = thetaProp
        distances[i] = propDist  
    
    results = np.column_stack((thetas, np.reshape(distances, (len(distances), 1))))
    return results

observedSample = np.load("g-and-k/observed_data/100/sample0.npy")
simulationSize = 100
uniLower = 0
uniUpper = 10
numComp = 3
abcIterations = 10000000
abcData = gk_abc(observedSample, simulationSize, numComp, abcIterations, uniLower, uniUpper)
np.save("g-and-k/inference/data/sim100-abc1e7-prioruni(1).npy", abcData)