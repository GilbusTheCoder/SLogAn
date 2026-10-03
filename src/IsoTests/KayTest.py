''' 
Use this file for testing and importing whatever you need tested. We all have our 
own files to (hopefully) avoid a ton of merging that would arise when using a shared
main.py '''

import os
import sys
import numpy as np

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path: sys.path.append(PARENT_DIR)
SIBNET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../NeuralNet"))
if SIBNET_DIR not in sys.path: sys.path.append(SIBNET_DIR)

# Only importing our own Model 2 clustering class
from ClusterNN import ClusterNN


# Test Script for Model 2 (DBSCAN Clustering)

# 1. load the extracted adfa dataset
print("loading data files...")
X = np.load("X_adfa.npy")
y = np.load("y_adfa.npy")

# 2. we only care about clustering the attacks (label == 1)
X_attacks = X[y == 1]
print(f"got {len(X_attacks)} attack traces to cluster")

# 3. train dbscan
# eps is neighbor radius, min_samples is points needed to form a group
cluster_bot = ClusterNN(eps=2.5, min_samples=5)
cluster_bot.Train(X_attacks)

# 4. print out quick results for our report
stats = cluster_bot.Evaluate(X_attacks)
print("\n Model 2 Clustering Results")
for key, val in stats.items():
    print(f"{key}: {val}")

# 5. save the 2d cluster scatter plot
cluster_bot.Plot(X_attacks, save_name="kay_attack_clusters.png")
print("\nall done! plot saved to kay_attack_clusters.png")
