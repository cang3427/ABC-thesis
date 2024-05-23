import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import numpy as np
from toad.toad_utils import summarise_sample

OBSERVED_PATH = "./toad/data/observed_data.npy"
PLOT_DIR = "./toad/plots/density"
LAGS = [1, 2, 4, 8]

if __name__ == "__main__":
    if not os.path.exists(OBSERVED_PATH):
        sys.exit("Error: Observed data does not exist. Prepare observed data before running this script.")        
    sample = np.load(OBSERVED_PATH)
    sample_data = summarise_sample(sample, lags=LAGS)
    
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    fig, axs = plt.subplots(1, len(LAGS), figsize=(16, 6))  
    for i in range(len(LAGS)):
        ax = axs[i]
        sns.kdeplot(sample_data[i][1], ax=ax, legend=False)
        ax.lines[0].set_color('black')
        
    for ax in axs:
        ax.set_xlim(0, 1000)
        
    for i, ax in enumerate(axs):
        ax.set_xlabel(f"Lag {LAGS[i]}", fontsize='xx-large')
        ax.set_yticks([0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012])
        ax.yaxis.set_tick_params(labelsize=12) 
        ax.xaxis.set_tick_params(labelsize=12) 
        if i > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Density", fontsize='xx-large')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "toad_density.png"))