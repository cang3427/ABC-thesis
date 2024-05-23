import numpy as np
import os, sys
import matplotlib.pyplot as plt
from toad.calculate_abc_posteriors import weighted_distances
from common.abc_posterior import abc_posterior
from common.distances import Distance, DISTANCE_LABELS

PLOTS_DIR = "./toad/plots/true"
RUN_DIR = "./toad/runs/true"
DISTANCES = [Distance.CVM, Distance.WASS_LOG, Distance.STAT]
DISTANCE_QUANTILE = 0.001
PAPER_POSTERIORS = [0.15, 0.0, 0.85]
STAT_WEIGHT = 0.2

if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        sys.exit("Error: Run data does not exist. Complete ABC runs before running this script.")
    
    if not os.path.isdir(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    results = np.zeros((len(DISTANCES) + 1, 3))
    results[0] = PAPER_POSTERIORS
    for i, distance in enumerate(DISTANCES):
        run = np.load(os.path.join(RUN_DIR, f"run_{distance.name.lower()}.npy"))
        run[np.isinf(run)] = np.nan
        if distance == Distance.STAT:
            distances = run[:, 1]
        else:
            distances = weighted_distances(run[:, 1:-1], STAT_WEIGHT)
        
        results[i + 1] = abc_posterior(run[:, 0], distances, 3, DISTANCE_QUANTILE)
    
    results = np.round(results, 2)
    row_labels = ["Paper"] + [f"ABC-{DISTANCE_LABELS[d]}" for d in DISTANCES]
    col_labels = [f"$M_{{{i + 1}}}$" for i in range(3)]
    num_columns = results.shape[1]
    default_col_width = 0.5 / (num_columns + 0.5)
    col_widths = [default_col_width] * num_columns
    fig = plt.figure()
    table = plt.table(cellText=results, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2) 
    plt.axis('off')
    plt.savefig(os.path.join(PLOTS_DIR, f"results_table.png"))
