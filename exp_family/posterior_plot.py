import numpy as np
import matplotlib.pyplot as plt
import os, sys
from exp_family.model import Model
from common.distances import Distance, DISTANCE_LABELS

PLOTS_DIR = "./exp_family/plots"
RESULTS_DIR = "./exp_family/results"
MODEL = Model.EXP
SIZE = 100
DISTANCE_QUANTILES = [0.001, 0.0005, 0.0001]
DISTANCES = [Distance.CVM, Distance.MMD_LOG, Distance.WASS_LOG, Distance.STAT]
PLOT_WIDTH = 3 * len(DISTANCES) - 1
PLOT_HEIGHT = 3 * len(DISTANCE_QUANTILES) - 1

if __name__ == "__main__":
    model_dir = MODEL.name.lower()
    results_size_dir = os.path.join(RESULTS_DIR, model_dir, f"size_{SIZE}")
    if not os.path.isdir(results_size_dir):     
        sys.exit("Error: Results do not exist. Calculate ABC posteriors before running this script.")    
    
    plot_model_dir = os.path.join(PLOTS_DIR, model_dir)
    if not os.path.exists(plot_model_dir):
        os.makedirs(plot_model_dir)
    
    # Num. thresholds X Num. ABC methods subplot
    fig, axs = plt.subplots(len(DISTANCE_QUANTILES), len(DISTANCES), figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)

    for ax in axs.flat:
        ax.plot([0, 1], [0, 1], color = 'k', zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # Adding labels for each ABC method for the columns
    for ax, distance in zip(axs[0], DISTANCES):
        ax.set_title("ABC-" + DISTANCE_LABELS[distance], fontsize="large")
    
    # Adding labels for each distance thresholds for the rows 
    for ax, distance_quantile in zip(axs[:, 0], DISTANCE_QUANTILES):
        ax.set_ylabel(r'$\varepsilon=$' + f"{distance_quantile * 100}% quantile", fontsize="large")

    # Plotting true posteriors vs approximations of true model for each distance threshold and ABC method
    model_idx = MODEL.value
    true_posteriors = np.load(os.path.join(results_size_dir, "true",  "posteriors.npy"))[:, model_idx]
    for ax_set, distance_quantile in zip(axs, DISTANCE_QUANTILES):
        abc_data_name = f"posteriors_{distance_quantile}q.npy"
        for ax, distance in zip(ax_set, DISTANCES):
            posterior_probs = np.load(os.path.join(results_size_dir, distance.name.lower(), abc_data_name))[:, model_idx]            
            ax.plot(true_posteriors, 
                    posterior_probs, 'o', 
                    color='blue',
                    alpha=0.5,
                    zorder=1)

    model_label = f"$M_{{{model_idx + 1}}}$"
    fig.supxlabel(f"True Posterior Probability of {model_label}", fontsize='x-large')
    fig.supylabel(f"ABC Posterior Probability of {model_label}", fontsize='x-large')
    plt.savefig(os.path.join(plot_model_dir, f"posterior_plot_size{SIZE}.png"))
