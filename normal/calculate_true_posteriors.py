import numpy as np
import os, sys

OBSERVED_DIR = "./normal/observed_data"
RESULTS_DIR = "./normal/results/known_var"
NULL_MEAN = 3
VAR = 1
VAR_SCALE = 100
NUM_RUNS = 100
MODEL = 0
SIZES = [100, 1000]

def log_bayes_factor(sample: np.ndarray, null_mean: float, var: float, prior_var_scale: float) -> float:
    n = len(sample)
    std_sample_mean = (np.mean(sample) - null_mean) / np.sqrt(var / n)
    log_bayes_factor = (0.5 * np.log(n * prior_var_scale + 1) - 
                        0.5 * n / (n + 1 / prior_var_scale) * std_sample_mean**2)
    return log_bayes_factor

def posterior_probs(sample: np.ndarray, null_mean: float, var: float, prior_var_scale: float) -> np.ndarray:
    bf = np.exp(log_bayes_factor(sample, null_mean, var, prior_var_scale))
    posterior_prob_m0 = 1 - 1 / (bf + 1)
    
    return np.array([posterior_prob_m0, 1 - posterior_prob_m0])

if __name__ == "__main__":
    if not os.path.isdir(OBSERVED_DIR):
        sys.exit("Error: Observed data does not exist. Generate observed data before running this script.")
    
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    model_dir = f"m{MODEL}"
    observed_model_dir = os.path.join(OBSERVED_DIR, model_dir)
    results_model_dir = os.path.join(RESULTS_DIR, model_dir)
    if not os.path.isdir(results_model_dir):
        os.mkdir(results_model_dir)
        
    for size in SIZES:
        size_dir = f"size_{size}"
        observed_size_dir = os.path.join(observed_model_dir, size_dir)
        true_size_dir = os.path.join(results_model_dir, size_dir, "true")
        if not os.path.isdir(true_size_dir):
            os.makedirs(true_size_dir)
            
        posterior_probs_arr = np.zeros((NUM_RUNS, 2))
        log_bayes_factors_arr = np.zeros((NUM_RUNS))
        for i in range(NUM_RUNS):
            data_path = os.path.join(observed_size_dir, f"sample{i}.npy")
            data = np.load(data_path)
            posterior_probs_arr[i] = posterior_probs(data, NULL_MEAN, VAR, VAR_SCALE)
            log_bayes_factors_arr[i] = log_bayes_factor(data, NULL_MEAN, VAR, VAR_SCALE)
            
        np.save(os.path.join(true_size_dir, f"posteriors.npy"), posterior_probs_arr)
        np.save(os.path.join(true_size_dir, f"logbfs.npy"), log_bayes_factors_arr)
