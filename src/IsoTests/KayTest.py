"""
KayTest.py
Model 2: Unsupervised Attack Clustering (DBSCAN)
- Isolates attack traces (y = 1) from the ADFA-LD dataset
- Performs density-based spatial clustering without target labels
- Exports a 2D PCA projection plot (kay_attack_clusters.png)
- Inspects and prints cluster distributions, elevated syscalls, and suppression signals
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# 1. Load data and filter attacks
X = np.load("X_adfa.npy")
y = np.load("y_adfa.npy")
X_att = X[y == 1]

# 2. Fit DBSCAN
db = DBSCAN(eps=0.15, min_samples=5).fit(X_att)
labels = db.labels_

# 3. PCA & Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_att)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=25)
plt.title("ADFA-LD Attack Clusters (DBSCAN)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster ID")
plt.savefig("kay_attack_clusters.png")
plt.close()

# 4. Cluster inspection and feature deviations
global_mean = np.mean(X_att, axis=0)
for c in sorted(set(labels)):
    mask = (labels == c)
    count = int(np.sum(mask))
    pct = (count / len(X_att)) * 100
    name = f"Cluster {c}" if c != -1 else "Noise (-1)"
    
    print(f"\n{name}: {count} traces ({pct:.1f}%)")
    if c == -1:
        continue
        
    diff = np.mean(X_att[mask], axis=0) - global_mean
    top_pos = np.argsort(diff)[-3:][::-1]
    top_neg = np.argsort(diff)[:2]
    
    print(f"  Top elevated syscall IDs: {top_pos.tolist()}")
    print(f"  Top suppressed syscall IDs: {top_neg.tolist()}")
