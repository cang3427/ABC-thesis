import numpy as np
import os, sys
import numpy as np
from scipy.special import loggamma
from exp_family.model import Model

NUM_RUNS = 100
OBSERVED_DIR = "./exp_family/observed_data"
SAVE_DIR = "./exp_family/results"
MODEL = Model.EXP
SIZES = [100, 1000]

def marginal_log_likelihood(data: np.ndarray, model: Model) -> float:
    n = len(data)
    if model == Model.EXP:
        return (loggamma(n + 1) - (n + 1) * np.log(1 + np.sum(data)))
    if model == Model.LNORM:
        log_sum = np.sum(np.log(data))
        data_sum = log_sum**2 / (2 * (n + 1)) - log_sum - np.sum(np.log(data)**2) / 2
        return (data_sum - n / 2 * np.log(2*np.pi) - 0.5 * np.log(n + 1))
    if model == Model.GAMMA:
        return (np.sum(np.log(data)) + loggamma(2*n + 1) - (2*n + 1) * np.log(1 + np.sum(data)))
    
def posterior_prob(data: np.ndarray, model: Model) -> float:
    # Stable calculation of posterior
    marginals_log = np.array([marginal_log_likelihood(data, m) for m in Model])
    max_marginal_log = np.max(marginals_log)
    log_posterior_prob = marginals_log[model.value] - (max_marginal_log + np.log(np.sum(np.exp(marginals_log - max_marginal_log))))

    return np.exp(log_posterior_prob)

def log_bayes_factor(data: np.ndarray, m1: Model, m2: Model) -> float:
    return marginal_log_likelihood(data, m1) - marginal_log_likelihood(data, m2)

if __name__ == "__main__":
    if not os.path.isdir(OBSERVED_DIR):
        sys.exit("Error: Observed data does not exist. Generate observed data before running this script.")
    
    model_dir = MODEL.name.lower()
    for size in SIZES:
        posterior_probs = np.zeros((NUM_RUNS, 3))
        log_bayes_factors = np.zeros((NUM_RUNS, 3))
        size_dir = f"size_{size}"
        observed_size_dir = os.path.join(OBSERVED_DIR, model_dir, size_dir)
        true_dir = os.path.join(SAVE_DIR, model_dir, size_dir, "true")
        if not os.path.isdir(true_dir):
            os.makedirs(true_dir)
        
        # Calculating true posterior for each observed dataset and saving
        for i in range(NUM_RUNS):
            data_path = os.path.join(observed_size_dir, f"sample{i}.npy")
            data = np.load(data_path)
            posterior_probs[i] = [posterior_prob(data, m) for m in Model]                
            log_bayes_factors[i, 0] = log_bayes_factor(data, Model.EXP, Model.LNORM)
            log_bayes_factors[i, 1] = log_bayes_factor(data, Model.EXP, Model.GAMMA)
            log_bayes_factors[i, 2] = log_bayes_factor(data, Model.LNORM, Model.GAMMA)

        np.save(os.path.join(true_dir, "posteriors.npy"), posterior_probs)
        np.save(os.path.join(true_dir, "logbfs.npy"), log_bayes_factors)
