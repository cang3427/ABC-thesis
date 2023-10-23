import numpy as np
from scipy.stats import norm, rankdata
from scipy.spatial.distance import pdist, cdist
from math import exp
from sklearn.mixture import GaussianMixture
from enum import Enum

class DistanceMetric(Enum):
    AUXILIARY = 0
    CVM = 1
    WASS = 2
    MMD = 3

def normal_sample(mean, sd, numSamples):
    normalSamples = np.random.normal(mean, sd, numSamples)
    return np.reshape(normalSamples, (numSamples, 1))
    
def gk_sample(params, numSamples, c = 0.8):
    normalSamples = np.random.normal(size = numSamples)
    return [[gk_quantile(normalSample, params, c)] for normalSample in normalSamples]

def gk_quantile(normalSample, params, c = 0.8):    
    (a, b, g, k) = list(params)
    val = a + b * (1 + c * (1 - exp(-g * normalSample)) / (1 + exp(-g * normalSample))) * (1 + normalSample**2)**k * normalSample
    return val

def fit_gaussian_mixture_EM(observedData, numComponents, tol = 1e-7, maxIterations = 10000, nInit = 10):
    gm = GaussianMixture(numComponents, tol = tol, max_iter = maxIterations, n_init = nInit)
    return gm.fit(observedData)
    
def gaussian_mixture_score(data, gmModel):    
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
        scoreVec[i] = np.sum(weightDerivatives)
        
    # Calculation of mean derivatives
    for i in range(numComponents):
        meanDerivatives = (w[i] * (y - mu[i]) / sig2[i] * norm.pdf(y, mu[i], sig2[i]**0.5)).flatten() / probs
        scoreVec[numComponents + i - 1] = np.sum(meanDerivatives)
        
    # Calculation of variance derivatives
    for i in range(numComponents):
        varianceDerivatives = (w[i] * (-1 / (2*sig2[i]) + (y - mu[i])**2 / (2*sig2[i]**2)) * norm.pdf(y, mu[i], sig2[i]**0.5)[0]).flatten() / probs
        scoreVec[2*numComponents + i - 1] = np.sum(varianceDerivatives)
    
    return scoreVec

def gaussian_mixture_information(data, gmModel):
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

def cramer_von_mises_distance(observedSample, simulatedSample):
    sampleSize = len(observedSample)
    combinedSample = np.concatenate((observedSample, simulatedSample))
    combinedRanks = rankdata(combinedSample)
    observedRanks = np.sort(combinedRanks[:sampleSize])
    simulatedRanks = np.sort(combinedRanks[sampleSize:])
    indices = np.array(range(1, sampleSize + 1))
    rankSum = sum((observedRanks - indices)**2) + sum((simulatedRanks - indices)**2)
    distance = rankSum / (2 * sampleSize**2) - (4 * sampleSize**2 - 1) / (12 * sampleSize)
    
    return distance

def wasserstein_distance(observedSample, simulatedSample, observedIsSorted = True):
    if observedIsSorted:
        sortedObserved = observedSample
    else:
        sortedObserved = np.sort(observedSample)
        
    sortedSimulated = np.sort(simulatedSample)
    distance = np.mean(np.absolute(sortedObserved - sortedSimulated))
    return distance

def gaussian_kernel(sqDistances, sigma):
    return np.exp(-sqDistances / (2 * sigma))

def maximum_mean_discrepancy(observedSample, simulatedSample, sigma = None, observedSqDistances = None):
    if sigma is None:
        sigma = np.median(observedSqDistances) ** 0.5    
    if observedSqDistances is None:
        observedSqDistances = pdist(observedSample, 'sqeuclidean')
    simulatedSqDistances = pdist(simulatedSample, 'sqeuclidean')
    mixedSqDistances = cdist(observedSample, simulatedSample, 'sqeuclidean')
    distance = np.mean(gaussian_kernel(observedSqDistances, sigma)) +  np.mean(gaussian_kernel(simulatedSqDistances, sigma)) - 2 *  np.mean(gaussian_kernel(mixedSqDistances, sigma))
    return distance
