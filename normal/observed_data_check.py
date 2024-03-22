import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = "./model_choice/normal/observed_data/sample18size100.npy"

data = np.load(DATA_DIR)
sns.set_style('whitegrid')
sns.kdeplot(np.array(data), bw_method = 0.5)
plt.show()