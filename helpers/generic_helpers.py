import networkx as nx
import numpy as np

def get_G_from_L(L):
    """Gets networkx graph from graph laplacian"""
    adj = np.copy(-L)
    adj[np.diag_indices(L.shape[0])] = 0
    return nx.from_numpy_array(adj)


def get_hard_proj(P):
    """Projects matrix P with soft constraints onto hard constraints"""
    idx = P.argmax(0)
    P = np.zeros_like(P)
    P[idx,range(P.shape[1])] = 1.
    return P.astype(int)
