import numpy as np
import matplotlib.pyplot as plt
import os, sys
from common.distances import Distance, DISTANCE_LABELS

PLOTS_DIR = f"./normal/plots"
RESULTS_DIR = "./normal/results"
MODEL = 0
VAR_DIR = "known_var"
SIZE = 100
DISTANCE_QUANTILES = [0.01, 0.005, 0.001]
DISTANCES = [Distance.CVM, Distance.MMD, Distance.WASS] + ([Distance.STAT] if VAR_DIR == "known_var" else [])
PLOT_WIDTH = 3 * len(DISTANCES) - 1
PLOT_HEIGHT = 3 * len(DISTANCE_QUANTILES) - 1

if __name__ == "__main__":
    model_dir = f"m{MODEL}"
    results_size_dir = os.path.join(RESULTS_DIR, VAR_DIR, model_dir, f"size_{SIZE}")
    if not os.path.isdir(results_size_dir):     
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    plot_model_dir = os.path.join(PLOTS_DIR, VAR_DIR, model_dir)
    if not os.path.isdir(plot_model_dir):
        os.makedirs(plot_model_dir)
    
    fig, axs = plt.subplots(len(DISTANCE_QUANTILES), len(DISTANCES), figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    for ax, distance in zip(axs[0], DISTANCES):
        ax.set_title(f"ABC-{DISTANCE_LABELS[distance]}", fontsize="large")
        
    for ax, distance_quantile in zip(axs[:, 0], DISTANCE_QUANTILES):
        ax.set_ylabel(r'$\varepsilon=$' + f"{distance_quantile * 100}%" + "quantile", fontsize="large")

    if VAR_DIR == "known_var":
        true_posteriors = np.load(os.path.join(results_size_dir, "true/posteriors.npy"))[:, 0]
        true_label = "True"
    else:
        true_posteriors = np.load(os.path.join(results_size_dir, "stat/posteriors_0.001q.npy"))[:, 0]
        true_label = "Approximate"
        
    for ax_set, distance_quantile in zip(axs, DISTANCE_QUANTILES):
        abc_data_name = f"posteriors_{distance_quantile}q.npy"
        for ax, distance in zip(ax_set, DISTANCES):
            posterior_probs = np.load(os.path.join(results_size_dir, distance.name.lower(), abc_data_name))[:, 0]
            ax.plot(true_posteriors, posterior_probs, 'o', color='blue', alpha=0.5, zorder=1)

    for ax in axs.flat:
        ax.plot([0, 1], [0, 1], color='k', zorder=-1)
        ax.set_aspect('equal')
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
    fig.supxlabel(f"{true_label} Posterior Probability of " + r'$M_0$', fontsize='x-large')
    fig.supylabel("ABC Posterior Probability of " + r'$M_0$', fontsize='x-large')
    plt.savefig(os.path.join(plot_model_dir, f"posterior_plot_size{SIZE}.png"))
