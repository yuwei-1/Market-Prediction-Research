from sklearn.neighbors import KDTree
import numpy as np


def create_nearest_neighbour_adjacency_matrix(X, k=3, include_self=False, **kwargs):
    n = X.shape[0]
    tree = KDTree(X)
    if not include_self:
        _, ind = tree.query(X, k=k+1)
    else:
        _, ind = tree.query(X, k=k)
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        adj_matrix[i][ind[i]] = 1
    return adj_matrix

def create_exponential_adjacency_matrix(X, sigma=0.01, **kwargs):
    n = X.shape[0]
    tree = KDTree(X)
    dist, _ = tree.query(X, k=n)
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        adj_matrix[i] = np.exp(-(dist[i]**2)/(2*sigma**2))
    return adj_matrix