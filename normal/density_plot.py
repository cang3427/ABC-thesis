import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

PLOT_DIR = "./normal/plots/density"
NULL_MEAN = 3
ALTERNATIVE_MEANS = [0, 2, 6]
VAR = 1

if __name__ == "__main__":
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    sd = np.sqrt(VAR)
    support = np.arange(NULL_MEAN - 6.5, NULL_MEAN + 6.5, 0.001)
    plt.plot(support, norm.pdf(support, NULL_MEAN, sd), color = 'k', label = r'$\mu = $' + str(NULL_MEAN), zorder = 1)
    for alt_mean in ALTERNATIVE_MEANS:
        label = None
        if alt_mean == ALTERNATIVE_MEANS[0]:
            label = r'$\mu \neq $' + str(NULL_MEAN)
            
        plt.plot(support, norm.pdf(support, alt_mean, sd), color = 'grey', label = label, zorder = -1)

    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(os.path.join(PLOT_DIR, "normal_density.png"))