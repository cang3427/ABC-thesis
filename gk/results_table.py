import numpy as np
import os, sys
import matplotlib.pyplot as plt
from common.distances import Distance, DISTANCE_LABELS

RESULTS_DIR = "./gk/results"
PLOTS_DIR = "./gk/plots"
SIZE = 100
DISTANCE_QUANTILE = 0.01
DISTANCES = [Distance.CVM, Distance.MMD, Distance.WASS, Distance.STAT]
NUM_RUNS = 100

if __name__ == "__main__":
    if not os.path.isdir(RESULTS_DIR):     
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    results = np.zeros((len(DISTANCES), 1))
    abc_posteriors_filename = f"posteriors_{DISTANCE_QUANTILE}q.npy"
    
    # Calculating errors under each model
    for model_idx, model in enumerate(os.listdir(RESULTS_DIR)):
        model_size_dir = os.path.join(RESULTS_DIR, model, f"size_{SIZE}")
                        
        # Calculating PER of each ABC method under the current model
        for i, distance in enumerate(DISTANCES):
            posteriors = np.load(os.path.join(model_size_dir, distance.name.lower(), abc_posteriors_filename))
            results[i, 0] += np.sum(np.argmax(posteriors, axis = 1) != model_idx)
    
    results = np.round(results / (2 * NUM_RUNS), 5)
    row_labels = [f"ABC-{DISTANCE_LABELS[d]}" for d in DISTANCES]
    col_labels = ["PER"]
    num_columns = results.shape[1]
    default_col_width = 0.5 / (num_columns + 0.5)
    col_widths = [default_col_width] * num_columns
    fig = plt.figure()
    table = plt.table(cellText=results, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2) 
    plt.axis('off')
    plt.savefig(os.path.join(PLOTS_DIR, f"results_table_size{SIZE}"))
