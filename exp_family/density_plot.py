import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import expon, lognorm, gamma

PLOT_DIR = "./exp_family/plots/density"
MODEL_MEAN = 2.0

if __name__ == "__main__":
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    # Plotting densities of each model under the chosen mean
    support = np.arange(0.01, 10, 0.001)
    plt.plot(support, expon.pdf(support, scale=MODEL_MEAN), color='k', label=f"Exp({1 / MODEL_MEAN})", zorder=2)
    plt.plot(support, lognorm.pdf(support, s=1.0, scale=np.exp(np.log(MODEL_MEAN) - 0.5)), color='dimgrey', label=f"Lognormal(log({MODEL_MEAN}) - 0.5, 1)", zorder=1)
    plt.plot(support, gamma.pdf(support, a=2.0, scale=MODEL_MEAN / 2), color='silver', label=f"Gamma(2, {MODEL_MEAN / 2})", zorder=0)

    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig(os.path.join(PLOT_DIR, "exp_family_density.png"))
