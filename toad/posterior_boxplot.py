import numpy as np
import os, sys
import matplotlib.pyplot as plt
from common.distances import Distance, DISTANCE_LABELS
from toad.toad_utils import Model

RESULTS_DIR = "toad/results"
PLOTS_DIR = "toad/plots"
DISTANCES = [Distance.CVM, Distance.WASS, Distance.WASS_LOG, Distance.STAT]
DISTANCE_QUANTILES = [0.01, 0.005, 0.001]
MODEL = Model.RANDOM
PLOT_WIDTH = 4 * len(DISTANCE_QUANTILES) + 2
PLOT_HEIGHT = 4

if __name__ == "__main__":
    model_dir = MODEL.name.lower()
    results_model_dir = os.path.join(RESULTS_DIR, model_dir)
    if not os.path.isdir(results_model_dir):     
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    plot_model_dir = os.path.join(PLOTS_DIR, model_dir)
    if not os.path.isdir(plot_model_dir):
        os.makedirs(plot_model_dir)
        
    fig, axs = plt.subplots(1, len(DISTANCE_QUANTILES), figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    for ax, distance_quantile in zip(axs, DISTANCE_QUANTILES):
        abc_data_name = f"posteriors_{distance_quantile}q.npy"
        posteriors_list = []
        for distance in DISTANCES:            
            posteriors = np.load(os.path.join(results_model_dir, distance.name.lower(), abc_data_name))
            posteriors_list.append(posteriors[:, MODEL.value])
            
        ax.boxplot(posteriors_list, labels=[f"ABC-{DISTANCE_LABELS[d]}" for d in DISTANCES], 
                patch_artist=True, 
                boxprops=dict(facecolor="lightgrey", edgecolor='k'), 
                medianprops=dict(color='k', linewidth=2), 
                whiskerprops=dict(linestyle='--'),
                widths=0.75)
        
        ax.tick_params(axis='x', labelsize=8.5)
        
        ax.set_ylim((0, 1))
        ax.set_title(r'$\varepsilon=$' + f"{distance_quantile * 100}% quantile", pad=10)
    
    model_label = f"$M_{{{MODEL.value + 1}}}$"      
    axs[0].set_ylabel(f"Posterior Probability of {model_label}", fontsize='large', labelpad=10)
    plt.savefig(os.path.join(plot_model_dir, f"posterior_boxplot.png"))
