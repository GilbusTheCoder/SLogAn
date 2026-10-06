"""
KayTest.py
Model 2: Unsupervised Attack Clustering (DBSCAN)
- Isolates attack traces (y = 1) from the ADFA-LD dataset
- Performs density-based spatial clustering without target labels
- Exports a 2D PCA projection plot (kay_attack_clusters.png)
- Inspects and prints cluster distributions, elevated syscalls, and suppression signals
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Resolve file paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add src to path to import syscall mappings if available
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    import ADFALDTranslator as AT
    SYSCALL_MAP = getattr(AT, "syscalls", {})
except ImportError:
    SYSCALL_MAP = {}

# 1. Load Preprocessed Datasets
x_path = os.path.join(PROJECT_ROOT, "X_adfa.npy")
y_path = os.path.join(PROJECT_ROOT, "y_adfa.npy")

if not os.path.exists(x_path) or not os.path.exists(y_path):
    raise FileNotFoundError(
        f"Missing processed dataset files. Please run build_dataset.py in {PROJECT_ROOT} first."
    )

X = np.load(x_path)
y = np.load(y_path)

# 2. Isolate Malicious Traces (Unsupervised within Attack Class)
attack_indices = (y == 1)
X_att = X[attack_indices]
total_attacks = len(X_att)

print(f"Loaded {total_attacks} attack traces across {X_att.shape[1]} system call dimensions.")

# 3. Model Configuration: DBSCAN
# eps: neighborhood radius, min_samples: core density threshold
eps_val = 0.80
min_samples_val = 5

db = DBSCAN(eps=eps_val, min_samples=min_samples_val, metric="euclidean")
labels = db.fit_predict(X_att)

# 4. Clustering Metrics
unique_labels = sorted(set(labels))
n_clusters = len([c for c in unique_labels if c != -1])
n_noise = np.sum(labels == -1)

print(f"\nDBSCAN Results (eps={eps_val}, min_samples={min_samples_val}):")
print(f"  - Number of Clusters Formed: {n_clusters}")
print(f"  - Isolated Noise Points: {n_noise} ({(n_noise / total_attacks) * 100:.2f}%)")

# Calculate silhouette score for non-noise samples if at least 2 clusters exist
valid_mask = (labels != -1)
if len(set(labels[valid_mask])) > 1:
    sil_score = silhouette_score(X_att[valid_mask], labels[valid_mask])
    print(f"  - Silhouette Score (excluding noise): {sil_score:.4f}")

# 5. Dimensionality Reduction (PCA) & Visualization Export
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_att)

plt.figure(figsize=(10, 7))

# Plot non-noise clusters
for cid in unique_labels:
    if cid == -1:
        continue
    mask = (labels == cid)
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cid} (n={np.sum(mask)})",
        s=30,
        alpha=0.85
    )

# Plot noise instances
noise_mask = (labels == -1)
if np.any(noise_mask):
    plt.scatter(
        X_pca[noise_mask, 0],
        X_pca[noise_mask, 1],
        color="grey",
        marker="x",
        s=25,
        alpha=0.45,
        label=f"Noise (n={n_noise})"
    )

plt.title("Model 2: DBSCAN Clustering on Unlabelled ADFA-LD Attack Traces (PCA Projection)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

output_plot_path = os.path.join(SCRIPT_DIR, "kay_attack_clusters.png")
plt.savefig(output_plot_path, dpi=300)
plt.close()
print(f"Cluster visualization exported to: {output_plot_path}")

# 6. Cluster Inspection & Behavioral Deviation Breakdown
print("\n" + "=" * 65)
print("  DBSCAN CLUSTER BEHAVIORAL INSPECTION (RUBRIC METRICS)")
print("=" * 65)

global_mean = np.mean(X_att, axis=0)

for cid in unique_labels:
    mask = (labels == cid)
    size = np.sum(mask)
    pct = (size / total_attacks) * 100
    cluster_title = f"Cluster {cid}" if cid != -1 else "Noise (Label -1)"

    print(f"\n{cluster_title}: {size} traces ({pct:.1f}% of attack dataset)")

    if cid == -1:
        print("  - Dispersed distribution; no dense consensus vector (heterogeneous/multi-stage attacks).")
        continue

    cluster_mean = np.mean(X_att[mask], axis=0)
    diff = cluster_mean - global_mean

    # Extract 3 most elevated and 2 most suppressed/absent system calls
    top_elevated = np.argsort(diff)[-3:][::-1]
    top_suppressed = np.argsort(diff)[:2]

    print("  Top Elevated System Calls (vs. Global Attack Mean):")
    for sc in top_elevated:
        sc_name = SYSCALL_MAP.get(sc, f"syscall_{sc}")
        print(f"    + ID {sc:3d} ({sc_name:<16}): Avg = {cluster_mean[sc]:.4f} (Deviation: +{diff[sc]:.4f})")

    print("  Top Suppressed/Absent System Calls (vs. Global Attack Mean):")
    for sc in top_suppressed:
        sc_name = SYSCALL_MAP.get(sc, f"syscall_{sc}")
        print(f"    - ID {sc:3d} ({sc_name:<16}): Avg = {cluster_mean[sc]:.4f} (Deviation: {diff[sc]:.4f})")

print("=" * 65 + "\n")
