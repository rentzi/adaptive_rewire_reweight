"""
Provide functions for cleaning and summarizing empirical connectome data.
"""
import numpy as np
import networkx as nx

def get_adj(edge_counts):
    """
    Construct an adjacency matrix from an edge list.
    args:
        edge_counts: Edge list containing presynaptic nodes, postsynaptic nodes,
            and edge weights.
    returns:
        A: Adjacency matrix.
    """
    edge_counts.columns = ['pre', 'post', 'weight']
    G = nx.from_pandas_edgelist(edge_counts, source='pre', target='post', edge_attr='weight', create_using=nx.DiGraph())
    adj_matrix_T = nx.to_numpy_array(G, weight='weight')
    A = adj_matrix_T.T
    return A

def adjacency_stats(A):
    """
    Compute basic properties of a network.
    args:
        A: Adjacency matrix.
    returns:
        n: Number of nodes.
        edges: Number of edges.
        density: Network density.
        W: Total edge weight.
        avg_weight: Average edge weight.
    """
    n = A.shape[0]
    possible_edges = n * (n - 1)
    edges = np.sum(A>0)
    density = edges/possible_edges
    weights = A[A>0]
    W = np.sum(A)
    avg_weight = weights.mean()
    return n, edges, density, W, avg_weight