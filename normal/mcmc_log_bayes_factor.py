import pymc as pm
import numpy as np
import os
import time

OBSERVED_DIR = "model_choice/normal/observed_data"
SAVE_DIR = "model_choice/normal/results/mean_test/eq3_vs_neq3/unknown_var/mcmc"
SIZES = np.linspace(610, 1000, 40).astype(int)
NULL_MEAN = 3
VARIANCE = None
PRIOR_VARIANCE_SCALE = 100
MCMC_SAMPLES = 100_000

def mcmc_log_bayes_factor(observedData, nullMean, variance, priorVarianceScale, runs):
    with pm.Model() as model:
        modelIndex = pm.Bernoulli('modelIndex', p = 0.5)
        if variance is None:
            std = pm.Gamma('sigma', alpha = 0.1, beta = 0.1)
            alternativeMean = pm.Normal('alternativePriorMean', mu = nullMean, sigma = priorVarianceScale**0.5)
        else:
            alternativeMean = pm.Normal('alternativePriorMean', mu = nullMean, sigma = (variance * priorVarianceScale)**0.5)
            std = variance**0.5
        
        model0 = pm.Normal('nullModel', mu = nullMean, sigma = std)        
        model1 = pm.Normal('alternativeModel', mu = alternativeMean, sigma = std)
        
        model = pm.Deterministic('selectedModel', pm.math.switch(modelIndex, model1, model0))  
        y = pm.Normal('y', model, observed = observedData)   
        step0 = pm.CategoricalGibbsMetropolis(vars = [modelIndex])
        step1 = pm.NUTS()
        trace = pm.sample(runs, step = [step0, step1])
        pM1 = trace['posterior']['modelIndex'].mean()
        pM0 = 1 - pM1
        logBayesFactor = np.log(pM0 / pM1).item()
        
        return logBayesFactor

if __name__ == '__main__':
    mcmcLogBayesFactors = np.load("model_choice/normal/results/mean_test/eq3_vs_neq3/unknown_var/mcmc/log_bayes_factors_100000_cp60.npy")  
    mcmcLogBayesFactors.resize((100))  
    startTime = time.time()
    for i, size in enumerate(SIZES):
        j = i + 60
        print(str(j) + ": " + str(time.time() - startTime))
        if (j % 10) == 0 and j > 0:
            checkpointName = "log_bayes_factors_" + str(MCMC_SAMPLES) + "_cp" + str(j) + ".npy"
            checkpointPath = os.path.join(SAVE_DIR, checkpointName)
            np.save(checkpointPath, mcmcLogBayesFactors[:j])
        observedPath = os.path.join(OBSERVED_DIR, "sample0size" + str(size) + ".npy")
        observed = np.load(observedPath)
        mcmcLogBayesFactors[j] = mcmc_log_bayes_factor(observed, NULL_MEAN, VARIANCE, PRIOR_VARIANCE_SCALE, MCMC_SAMPLES)
        
    savePath = os.path.join(SAVE_DIR, "log_bayes_factors_" + str(MCMC_SAMPLES) + ".npy")
    np.save(savePath, mcmcLogBayesFactors)
    