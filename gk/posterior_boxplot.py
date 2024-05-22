import numpy as np
import os, sys
import matplotlib.pyplot as plt
from string import ascii_lowercase
from common.distances import Distance, DISTANCE_LABELS

RESULTS_DIR = "./gk/results"
PLOTS_DIR = "./gk/plots"
DISTANCE_QUANTILE = 0.01
SIZES = [100, 1000]
NUM_OBSERVED = 100
DISTANCES = [Distance.CVM, Distance.MMD, Distance.WASS, Distance.STAT]
MODEL = 0

if __name__ == "__main__":
    model_dir = f"m{MODEL}"
    results_model_dir = os.path.join(RESULTS_DIR, model_dir)
    if not os.path.isdir(results_model_dir):     
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    plot_model_dir = os.path.join(PLOTS_DIR, model_dir)
    if not os.path.isdir(plot_model_dir):
        os.makedirs(plot_model_dir)
    
    # Generating boxplot for each sample size
    plot_labels = [f"({ascii_lowercase[i]})" for i in range(len(SIZES))]
    fig, axs = plt.subplots(1, len(SIZES), figsize=(5 * len(SIZES) + 1, 5))
    fig.subplots_adjust(left=0.1, right=0.9)  # Adjust the left margin
    for ax, label, size in zip(axs, plot_labels, SIZES):
        # Collecting posteriors from each ABC method
        results_size_dir = os.path.join(results_model_dir, f"size_{size}")
        abc_data_name = f"posteriors_{DISTANCE_QUANTILE}q.npy"
        posteriors_list = []
        for distance in DISTANCES:            
            posteriors = np.load(os.path.join(results_size_dir, distance.name.lower(), abc_data_name))
            posteriors_list.append(posteriors[:, 0])
            
        ax.boxplot(posteriors_list, 
                   labels=[f"ABC-{DISTANCE_LABELS[d]}" for d in DISTANCES], 
                   patch_artist=True, 
                   boxprops=dict(facecolor="lightgrey", edgecolor='k'), 
                   medianprops=dict(color='k', linewidth=2), 
                   whiskerprops=dict(linestyle='--'),
                   widths=0.75)
        
        ax.set_ylim((0, 1))
        ax.set_xlabel(label, fontsize='large', labelpad=10)
        ax.tick_params(axis='x', labelsize='medium')
    
    fig.supylabel('Posterior Probability of ' + r'$M_1$', fontsize='large')
    plt.savefig(os.path.join(plot_model_dir, "posterior_boxplot.png"))
