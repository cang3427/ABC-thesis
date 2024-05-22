import numpy as np
import matplotlib.pyplot as plt
import os
from gk.gk_funcs import gk_log_likelihood

PLOT_DIR = "gk/plots/density"
G_VALUES = [1, 2.5, 3.5]

if __name__ == "__main__":
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    support = np.arange(-2.5, 2.5, 0.01)
    plt.plot(support, np.exp([gk_log_likelihood(x,(0, 1, 0, 2)) for x in support]), color='k', label=r'$g = 0$', zorder=1)
    for g in G_VALUES:
        label = None
        if g == G_VALUES[0]:
            label = r'$g > 0$'

        plt.plot(support, np.exp([gk_log_likelihood(x, (0, 1, g, 2)) for x in support]), color='grey', label=label, zorder=0)

    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(os.path.join(PLOT_DIR, "gk_density.png"))
