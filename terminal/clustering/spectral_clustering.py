import numpy as np
from clustering_utils import *
from sklearn.cluster import KMeans


class SpectralClustering:

    def __init__(self, clusters=2, k=3, affinity="nearest_neighbors") -> None:
        self._k = k
        self._clusters = clusters
        self._affinity = self._affinity_guard(affinity)

    def _create_graph_laplacian(self, matrix):
        degree_matrix = np.zeros_like(matrix)
        diag_vals = np.sum(matrix, axis=1)
        np.fill_diagonal(degree_matrix, diag_vals)
        L = degree_matrix - matrix
        return L
    
    def fit(self, X):
        adj = self._affinity(X, k=self._k)
        L = self._create_graph_laplacian(adj)
        w, v = np.linalg.eigh(L)
        idcs = np.argsort(w)
        eigenvalues = w[idcs][:self._clusters]
        eigenvectors = v[:, idcs][:, :self._clusters]
        labels = KMeans(n_clusters=self._clusters).fit_predict(eigenvectors)
        return labels
    
    @staticmethod
    def _affinity_guard(affinity):
        allowed = {"nearest_neighbors", "rbf"}
        assert affinity in allowed, f"the affinity has to be one of {allowed}"
        return create_nearest_neighbour_adjacency_matrix if affinity == "nearest_neighbors" else create_exponential_adjacency_matrix