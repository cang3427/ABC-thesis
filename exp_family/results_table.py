import numpy as np
import os, sys
import matplotlib.pyplot as plt
from common.distances import Distance, DISTANCE_LABELS
from exp_family.model import Model

RESULTS_DIR = "./exp_family/results"
PLOTS_DIR = "./exp_family/plots"
SIZE = 100
DISTANCE_QUANTILE = 0.0001
DISTANCES = [Distance.CVM, Distance.MMD, Distance.MMD_LOG, Distance.WASS, Distance.WASS_LOG, Distance.STAT]
NUM_RUNS = 100

if __name__ == "__main__":
    if not os.path.isdir(RESULTS_DIR):     
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    results = np.zeros((len(DISTANCES) + 1, 3))
    abc_posteriors_filename = f"posteriors_{DISTANCE_QUANTILE}q.npy"
    for model in Model:
        model_size_dir = os.path.join(RESULTS_DIR, model.name.lower(), f"size_{SIZE}")
        true_posteriors = np.load(os.path.join(model_size_dir, "true/posteriors.npy"))
        results[0, 2] += np.sum(np.argmax(true_posteriors, axis=1) != model.value)
        
        # Calculating error of each ABC method under the current model
        for i, distance in enumerate(DISTANCES):
            posteriors = np.load(os.path.join(model_size_dir, distance.name.lower(), abc_posteriors_filename))
            error = np.abs(posteriors - true_posteriors)
            results[i + 1, 0] += np.sum(error)
            results[i + 1, 1] += np.sum(error**2)
            results[i + 1, 2] += np.sum(np.argmax(posteriors, axis = 1) != model.value)
    
    # Gives the average total error across the observed datasets
    results = np.round(results / (3 * NUM_RUNS), 5)
    
    row_labels = ["Exact"] + [f"ABC-{DISTANCE_LABELS[d]}" for d in DISTANCES]
    col_labels = ["MAE", "MSE", "PER"]
    num_columns = results.shape[1]
    default_col_width = 0.5 / (num_columns + 0.5)  # Adding extra space for row labels
    col_widths = [default_col_width] * num_columns
    fig = plt.figure()
    table = plt.table(cellText=results, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2) 
    plt.axis('off')
    plt.savefig(os.path.join(PLOTS_DIR, f"results_table_size{SIZE}"))