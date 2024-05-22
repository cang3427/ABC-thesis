import numpy as np
import matplotlib.pyplot as plt
import os, sys
from common.distances import Distance, DISTANCE_LABELS

RESULTS_DIR = "./normal/results"
PLOTS_DIR = f"./normal/plots"
VAR_DIR = "known_var"
SIZE = 100
DISTANCES = [Distance.CVM, Distance.MMD, Distance.WASS] + ([Distance.STAT] if VAR_DIR == "known_var" else [])
NUM_RUNS = 100
if VAR_DIR == "known_var" or (SIZE != 100 and SIZE != 1000):
    DISTANCE_QUANTILES = {Distance.CVM: 0.001, Distance.MMD: 0.001, Distance.WASS: 0.001, Distance.STAT: 0.001}
else:
    if SIZE == 100:
        DISTANCE_QUANTILES = {Distance.CVM: 0.001, Distance.MMD: 0.001, Distance.WASS: 0.005}
    elif SIZE == 1000:
        DISTANCE_QUANTILES = {Distance.CVM: 0.005, Distance.MMD: 0.001, Distance.WASS: 0.005}
        
if __name__ == "__main__":
    results_var_dir = os.path.join(RESULTS_DIR, VAR_DIR)
    if not os.path.isdir(results_var_dir):
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    plots_var_dir = os.path.join(PLOTS_DIR, VAR_DIR)
    if not os.path.isdir(plots_var_dir):
        os.makedirs(plots_var_dir)

    if VAR_DIR == "known_var":
        true_posteriors_subpath = "true/posteriors.npy"
    else:
        true_posteriors_subpath = "stat/posteriors_0.001q.npy"
    
    size_dir = f"size_{SIZE}"
    results = np.zeros((len(DISTANCES) + 1, 3))
    results[0, :2] = 0.0
    for model_dir in os.listdir(results_var_dir):
        true_posteriors = np.load(os.path.join(results_var_dir, model_dir, size_dir, true_posteriors_subpath))
        results[0, 2] += np.sum(np.argmax(true_posteriors, axis=1) != (0 if model_dir == "m0" else 1))
        
    for model_dir in os.listdir(results_var_dir):
        results_size_dir = os.path.join(results_var_dir, model_dir, size_dir)
        true_posteriors = np.load(os.path.join(results_size_dir, true_posteriors_subpath))
        for i, distance in enumerate(DISTANCES):
            posteriors_filename = f"posteriors_{DISTANCE_QUANTILES[distance]}q.npy"
            posteriors = np.load(os.path.join(results_size_dir, distance.name.lower(), posteriors_filename))
            error = np.abs(posteriors[:, 0] - true_posteriors[:, 0])
            results[i + 1, 0] += np.sum(error)
            results[i + 1, 1] += np.sum(error**2)
            results[i + 1, 2] += np.sum(np.argmax(posteriors, axis=1) != (0 if model_dir == "m0" else 1))

    results = np.round(results / (2 * NUM_RUNS), 5)
    
    exact_suff = "" if VAR_DIR == "known_var" else "(approx.)"
    row_labels = [f"Exact {exact_suff}"] + [f"ABC-{DISTANCE_LABELS[d]}" for d in DISTANCES]
    col_labels = ["MAE", "MSE", "PER"]
    num_columns = results.shape[1]
    default_col_width = 0.5 / (num_columns + 0.5)
    col_widths = [default_col_width] * num_columns
    fig = plt.figure()
    table = plt.table(cellText=results, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2) 
    plt.axis('off')
    plt.savefig(os.path.join(plots_var_dir, f"results_table_size{SIZE}"))
