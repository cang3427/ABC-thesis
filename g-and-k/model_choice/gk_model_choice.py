import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

abcResults = np.load("g-and-k/model_choice/data/sim100-abc1e6.npy")
distances = abcResults[:, -1]
nonNanDistances = np.nan_to_num(distances, nan = math.inf)
threshold = np.quantile(nonNanDistances, 0.01)

nullResults = abcResults[abcResults[:, 0] == 0]
alternativeResults = abcResults[abcResults[:, 0] == 1]

nullProb = len(nullResults[nullResults[:, -1] < threshold]) / len(abcResults)
alternativeProb = len(alternativeResults[alternativeResults[:, -1] < threshold]) / len(abcResults)
print(nullProb)
print(alternativeProb)

sns.set_style('whitegrid')
sns.kdeplot(nullResults[:, -1], label = "Null")
sns.kdeplot(alternativeResults[:, -1], label = "Alternative")
plt.ticklabel_format(style = "plain", axis = "x")
plt.legend()
plt.show()