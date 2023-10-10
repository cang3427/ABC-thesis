import numpy as np
import os
from scipy.stats import norm
from abc_model_choice_auxiliary import ParameterType, main
from bayes_factor import calculate_bayes_factor
import matplotlib.pyplot as plt
import math

OBSERVED_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/observed_data"
DATA_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/model_choice/test_a/composite"
SAVE_DIR = "../../project/RDS-FSC-ABCMC-RW/normal/results"

def exact_log_bayes_factor_normal(observed, nullParam, alternativeParam, nullType, alternativeType, paramToTestIndex, iterations = 10_000, lower = 0, upper = 10):
    sampleSize = len(observed)
    commonParamSamples = np.random.uniform(lower, upper, iterations)
    if nullType == ParameterType.EQUAL:
        nullParamSamples = np.full(iterations, nullParam)
    elif nullType == ParameterType.LESS:
        nullParamSamples = np.random.uniform(lower, nullParam, iterations)
    elif nullType == ParameterType.GREATER:
        nullParamSamples = np.random.uniform(nullParam, upper, iterations)
    else:
        nullParamSamples = np.random.uniform(lower, upper, iterations)
        
    if alternativeType == ParameterType.EQUAL:
        alternativeParamSamples = np.full(iterations, alternativeParam)
    elif alternativeType == ParameterType.LESS:
        alternativeParamSamples = np.random.uniform(lower, alternativeParam, iterations)
    elif alternativeType == ParameterType.GREATER:
        alternativeParamSamples = np.random.uniform(alternativeParam, upper, iterations)
    else:
        alternativeParamSamples = np.random.uniform(lower, upper, iterations)

    if paramToTestIndex == 0:
        nullProposals = np.column_stack((nullParamSamples, np.sqrt(commonParamSamples)))
        alternativeProposals = np.column_stack((alternativeParamSamples, np.sqrt(commonParamSamples)))
    else:
        nullProposals = np.column_stack((commonParamSamples, np.sqrt(nullParamSamples)))
        alternativeProposals = np.column_stack((commonParamSamples, np.sqrt(alternativeParamSamples)))
        
    nullLogLikelihoods = np.zeros(iterations)
    alternativeLogLikelihoods = np.zeros(iterations)    
    for i in range(iterations):
        nullLogLikelihoods[i] = np.sum(norm.logpdf(observed, loc = nullProposals[i, 0], scale = nullProposals[i, 1]))
        alternativeLogLikelihoods[i] = np.sum(norm.logpdf(observed, loc = alternativeProposals[i, 0], scale = alternativeProposals[i, 1]))

    logBayesFactor = np.mean(nullLogLikelihoods) - np.mean(alternativeLogLikelihoods)
    return logBayesFactor

sampleSizes = np.linspace(10, 1000, 100).astype(int)
exactLogBayesFactors = []
for size in sampleSizes:
    sample = np.load(os.path.join(OBSERVED_DIR, "sample0size" + str(size) + ".npy"))
    exactLogBayesFactor = exact_log_bayes_factor_normal(sample, 5, 5, ParameterType.LESS, ParameterType.GREATER, 0)
    exactLogBayesFactors.append(exactLogBayesFactor)
    
epsilons = [0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001]   
for eps in epsilons:
    abcLogBayesFactors = []
    for size in sampleSizes:
        abcData = np.load(os.path.join(DATA_DIR, "run0size" + str(size) + ".npy"))
        abcBayesFactor = calculate_bayes_factor(abcData, eps)
        abcLogBayesFactor = math.log(abcBayesFactor)
        abcLogBayesFactors.append(abcLogBayesFactor)

    plt.plot(exactLogBayesFactors, exactLogBayesFactors, linestyle = "solid", color = "green")
    plt.scatter(exactLogBayesFactors, abcLogBayesFactors, color = 'b')
    plt.xlabel("true logBF")
    plt.ylabel("estimated logBF")
    plt.title(r'$\varepsilon = {eps}$'.format(eps = eps))
    plt.savefig(os.path.join(SAVE_DIR, "logBF_plot_eps" + str(eps) + ".png"))
    plt.clf()
