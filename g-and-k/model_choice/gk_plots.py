import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

abcResults = np.load("g-and-k/model_choice/data/test_a/sim100-abc1e7-3vs5.npy")
distances = abcResults[:, -1]
nonNanDistances = np.nan_to_num(distances, nan = math.inf)
threshold = np.quantile(nonNanDistances, 0.01)

nullDistances = abcResults[abcResults[:, 0] == 0][:, -1]
alternativeDistances = abcResults[abcResults[:, 0] == 1][:, -1]
sns.set_style('whitegrid')
sns.kdeplot(nullDistances, label = "Null")
sns.kdeplot(alternativeDistances, label = "Alternative")
plt.ticklabel_format(style = "plain", axis = "x")
plt.ticklabel_format(style = "plain", axis = "y")
plt.legend()
plt.show()