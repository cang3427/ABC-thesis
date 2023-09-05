from scipy.stats import norm
from math import exp
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def gkSample(numSamples, params, c = 0.8):
    normalSamples = np.random.normal(size = numSamples)
    return [[gkQuantile(normalSample, params, c)] for normalSample in normalSamples]

def gkQuantile(normalSample, params, c = 0.8):    
    (a, b, g, k) = list(params)
    val = a + b * (1 + c * (1 - exp(-g * normalSample)) / (1 + exp(-g * normalSample))) * (1 + normalSample**2)**k * normalSample
    return val

def fitGaussianMixtureEM(observedData, numComponents, tol = 1e-7, maxIterations = 10000, nInit = 10):
    gm = GaussianMixture(numComponents, tol = tol, max_iter = maxIterations, n_init = nInit)
    return gm.fit(observedData)
    
def gaussianMixtureScore(data, gmModel):    
    y = np.reshape(data, len(data))
    w = gmModel.weights_
    mu = gmModel.means_
    sig2 = gmModel.covariances_  
    numComponents = len(w)  
    numParams = numComponents * 3 - 1
    scoreVec = np.zeros(numParams)    
    probs = np.exp(gmModel.score_samples(data))
    
    # Calculation of weight derivatives
    for i in range(numComponents - 1):
        weightDerivatives = (norm.pdf(y, mu[i], sig2[i]**0.5) - norm.pdf(y, mu[-1], sig2[-1]**0.5)).flatten() / probs
        scoreVec[i] = sum(weightDerivatives)
        
    # Calculation of mean derivatives
    for i in range(numComponents):
        meanDerivatives = (w[i] * (y - mu[i]) / sig2[i] * norm.pdf(y, mu[i], sig2[i]**0.5)).flatten() / probs
        scoreVec[numComponents + i - 1] = sum(meanDerivatives)
        
    # Calculation of variance derivatives
    for i in range(numComponents):
        varianceDerivatives = (w[i] * (-1 / (2*sig2[i]) + (y - mu[i])**2 / (2*sig2[i]**2)) * norm.pdf(y, mu[i], sig2[i]**0.5)[0]).flatten() / probs
        scoreVec[2*numComponents + i - 1] = sum(varianceDerivatives)
    
    return scoreVec

def gaussianMixtureInformation(data, gmModel):
    y = data
    w = gmModel.weights_
    mu = gmModel.means_
    sig2 = gmModel.covariances_  
    numComponents = len(w)  
    numParams = numComponents * 3 - 1
    scoreVec = np.zeros(numParams)    
    infoMat = np.zeros((numParams, numParams))
    
    for i in range(len(y)):        
        grad = np.zeros(numParams)
        denom = exp(gmModel.score_samples([y[i]]))
        # Calculation of individual weight derivatives
        for j in range(numComponents - 1):
            grad[j] = (norm.pdf(y[i], mu[j], sig2[j]**0.5) - norm.pdf(y[i], mu[-1], sig2[-1]**0.5)) / denom
        
        # Calculation of individual mean derivatives
        for j in range(numComponents):
            grad[numComponents + j - 1] = (w[j] * (y[i] - mu[j]) / sig2[j] * norm.pdf(y[i], mu[j], sig2[j]**0.5)) / denom
            
        # Calculation of individual variance derivatives
        for j in range(numComponents):
            grad[2*numComponents + j - 1] = (w[j] * (-1 / (2*sig2[j]) + (y[i] - mu[j])**2 / (2*sig2[j]**2)) * norm.pdf(y[i], mu[j], sig2[j]**0.5)) / denom
            
        infoMat += np.outer(grad, grad) 
        
    return infoMat

def logUniformTransform(theta, lower, upper):
    logTrans = np.log((theta - lower) / (upper - theta))
    return logTrans
    
def logUniformTransformInverse(theta, lower, upper):
    expTheta = np.exp(theta)
    logTransInverse = (upper * expTheta + lower) / (1 + expTheta)
    return logTransInverse   

def logUniformTransformPdf(theta, lower, upper):
    expTheta = np.exp(theta)
    mass = np.prod(expTheta / (1 + expTheta)**2)
    return mass
    
def gkABC(observedData, simulationSize, numComp, abcIterations, lower = 0, upper = 10, covRw = [], printof = 10000):

    numParams = 4
    if len(covRw) == 0:
        covRw = np.identity(numParams) * 5
    
    # Fitted auxiliary model to the observed data
    auxiliaryModel = fitGaussianMixtureEM(observedData, numComp)  

    # Calculating weight matrix which is the inverse of the observed information
    # using the MLEs
    weightMatrix = np.linalg.inv(gaussianMixtureInformation(observedData, auxiliaryModel))
    thetas = np.zeros((abcIterations, numParams))
    distances = np.zeros(abcIterations)
    
    for i in range(abcIterations):  
        print(i)
        # Generating (transformed) proposal parameters 
        thetaTransProp = np.random.multivariate_normal(np.zeros(numParams), covRw, 1)[0]
        
        # Inverse transforming proposal to get the actual proposal parameters
        thetaProp = logUniformTransformInverse(thetaTransProp, lower, upper)
        
        # Generating the summary statistic for a simulated sample with
        # the proposal parameters (score of auxiliarly model at the
        # MLE for the observed sample)
        simulatedSample = gkSample(simulationSize, thetaProp)
        statistic = gaussianMixtureScore(simulatedSample, auxiliaryModel)
        
        # Distance function (Mahalanobis distance) for the summary statistic 
        # (note that we  do not need to consider the observed summary statistic
        # as in this case it is 0 since the score is 0 at the MLE with the observed
        # data )
        propDist = statistic @ weightMatrix @ statistic.T     

        # Store current values      
        thetas[i] = thetaProp
        distances[i] = propDist  
    
    return [thetas, distances]

params = [3, 1, 2, 0.5]
simulationSize = 100
observedSample = gkSample(simulationSize, params)
uniLower = 0
uniUpper = 10
numComp = 3
abcIterations = 10000000

# covRw = np.array([[0.00549166661341732, 0.0112801170236118, -0.0255611780526848, -0.0163180936020067],
#                    [0.0112801170236118, 0.0926757047779224, 0.0263799869794522, -0.165807183816090],
#                    [-0.0255611780526848, 0.0263799869794522, 0.292641868551302, -0.0435074048753841],
#                    [-0.0163180936020067, -0.165807183816090, -0.0435074048753841, 0.540229899133093]])

(thetas, dists) = gkABC(observedSample, simulationSize, numComp, abcIterations, uniLower, uniUpper)
abcData = np.column_stack((thetas, np.reshape(dists, (len(dists), 1))))
np.save("./data/sim100-abc1e7-cov5I.npy", abcData)