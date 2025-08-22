import numpy as np
from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        """Run KMeans clustering on the dataset."""
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.model.fit(X)
        labels = self.model.labels_
        centers = self.model.cluster_centers_
        return labels, centers
