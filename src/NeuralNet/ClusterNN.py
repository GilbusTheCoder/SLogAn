'''
ClusterNN.py - Model 2: Clustering attacks
Using DBSCAN so we can find attack groups without labels.
Also added simple K-Means just to compare if needed.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class ClusterNN:
    def __init__(self, eps=2.5, min_samples=5, use_kmeans=False, k=4):
        # basic settings for dbscan / kmeans
        self.eps = eps
        self.min_samples = min_samples
        self.use_kmeans = use_kmeans
        self.k = k

        self.scaler = StandardScaler()
        self.model = None
        self.labels = None

    def Train(self, X):
        # if input is 3D from sliding window, flatten to 2D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        # scale features so big numbers dont mess up distance
        X_scaled = self.scaler.fit_transform(X)

        if self.use_kmeans:
            # simple kmeans fallback
            self.model = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            self.labels = self.model.fit_predict(X_scaled)
        else:
            # our main dbscan clustering
            self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.labels = self.model.fit_predict(X_scaled)

        return self.labels

    def Evaluate(self, X):
        # calculate quick stats to see how it did
        if self.labels is None:
            print("Run Train() first!")
            return {}

        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X)

        # -1 means noise/outlier in dbscan
        noise_count = int(np.sum(self.labels == -1))
        total_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        noise_percent = (noise_count / len(self.labels)) * 100

        # silhouette score only makes sense if we have >= 2 real clusters
        sil = 0.0
        if total_clusters > 1 and (len(self.labels) - noise_count) > total_clusters:
            clean_mask = self.labels != -1
            sil = float(silhouette_score(X_scaled[clean_mask], self.labels[clean_mask]))

        results = {
            "Total Attacks": len(self.labels),
            "Clusters Found": total_clusters,
            "Noise Points (-1)": noise_count,
            "Noise %": round(noise_percent, 2),
            "Silhouette": round(sil, 4)
        }
        return results

    def Plot(self, X, save_name="attack_clusters.png"):
        # make a 2d scatter plot using PCA so we can put it in the report
        if self.labels is None:
            print("Run Train() first!")
            return

        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X)

        # drop to 2 components for plotting
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)

        plt.figure(figsize=(7, 5))
        for cluster in sorted(set(self.labels)):
            idx = (self.labels == cluster)
            if cluster == -1:
                # noise is black dots
                plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c="black", s=15, alpha=0.4, label="Noise (-1)")
            else:
                plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=30, alpha=0.8, label=f"Cluster {cluster}")

        plt.title("Attack Clusters (Model 2: DBSCAN)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
        print(f"saved plot to {save_name}")
