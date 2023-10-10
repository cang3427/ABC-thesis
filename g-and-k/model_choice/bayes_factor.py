import numpy as np
import math
import os

def calculate_bayes_factor(results, epsilon = 0.01):
    distances = results[:, -1]
    nonNanDistances = np.nan_to_num(distances, nan = math.inf)
    threshold = np.quantile(nonNanDistances, epsilon)
    nullDistances = results[results[:, 0] == 0][:, -1]
    alternativeDistances = results[results[:, 0] == 1][:, -1]
    bayesFactor = np.sum(nullDistances < threshold) / np.sum(alternativeDistances < threshold)
    return bayesFactor

# dataPath = "g-and-k/model_choice/data"
# labels = ["Parameter", "Null", "Alternative", "Bayes Factor"]
# results = [labels]
# for folder in os.listdir(dataPath):
#     param = folder[-1]
#     folderPath = os.path.join(dataPath, folder)
#     for runName in os.listdir(folderPath):
#         runPath = os.path.join(folderPath, runName)
#         tests = os.path.splitext(runName)[0].split('-')[-1].split('vs')
#         testNames = []
#         for test in tests:
#             if "leq" in test:
#                 testNames.append("< " + test[-1])
#             elif "geq" in test:
#                 testNames.append("> " + test[-1])
#             else:
#                 testNames.append(test)
#         (null, alternative) = testNames
        
#         runResults = np.load(runPath)
#         bayesFactor = calculate_bayes_factor(runResults)
        
#         results.append([param, null, alternative, bayesFactor])

# resultsTable = tabulate(results, tablefmt = "tsv")
# textFile = open("g-and-k/model_choice/results/bayes_factors.csv", "w")
# textFile.write(resultsTable)
# textFile.close()

