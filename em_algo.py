import numpy as np
from sklearn.mixture import GaussianMixture

class EMClustering:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        """Run Expectation Maximization (Gaussian Mixture Model) clustering."""
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            random_state=self.random_state
        )
        self.model.fit(X)
        labels = self.model.predict(X)
        centers = self.model.means_
        return labels, centers
