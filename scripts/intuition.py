"""
Provide functions for intuitive interpretation of the model mechanisms.
"""
import numpy as np
import networkx as nx
import sys
sys.path.append('.')
from scripts import indices

def step_mat(A):
    """
    Compute the number of steps along the shortest directed path between node pairs.
    args:
        A: adjacency matrix of the network
    returns:
        steps: Matrix in which element (i, j) gives the number of steps
            from node j to node i. A value of 0 indicates that no path exists.
    """
    steps = np.zeros(A.shape)
    G = indices.convert_from_adj2networkX(A, weighted=True)
    shortest_path_length = dict(nx.shortest_path_length(G))
    for s in shortest_path_length.keys():
        for t in shortest_path_length[s].keys():
            if(shortest_path_length[s][t]>0):
                steps[t, s] = shortest_path_length[s][t]
    return steps

def fisher_z_average_safe(r_values, n_values, epsilon=1e-6):
    """
    Compute a sample-size-weighted average correlation coefficient using
    the Fisher z transformation.
    args:
        r_values: Correlation coefficients.
        n_values: Sample sizes associated with the correlation coefficients.
        epsilon: Small value used to avoid numerical overflow near -1 and 1.
    returns:
        r_avg: Weighted average correlation coefficient.
    """
    r_values = np.clip(r_values, -1 + epsilon, 1 - epsilon)
    z = np.arctanh(r_values)
    weights = np.array(n_values) - 3
    z_avg = np.sum(weights * z) / np.sum(weights)
    r_avg = np.tanh(z_avg)
    return r_avg

def kernel_weight_relation(A, kernel_mat, flag):
    """
    Compute the average correlation between node states and connection weights.
    For advection, correlations are computed across the out-links of each node.
    For consensus, correlations are computed across the in-links of each node.
    args:
        A: Adjacency matrix of the network.
        kernel_mat: Consensus or advection kernel.
        flag: Type of dynamics; must be 'Consensus' or 'Advection'.
    returns:
        weight_ls: Connection weights pooled across the relevant in- or out-links.
        kernel_ls: Corresponding node-state values.
        rho: Average correlation coefficient across nodes.
    """
    n = A.shape[0] # number of nodes
    rval = np.zeros(n); nval = np.zeros(n)

    # for each node, connection weights and neighbors' states
    weight_ls = []; kernel_ls = []
    for l in range(n): # consider each node l
        if flag == 'Advection': # if advection is used as the dynamics of node states, consider the node l's out-links
            ind_connection = (A[:, l]>0) # true if exists l's out-link to a node
            positive_weight = A[ind_connection, l] # weights of l's out-links
            positive_kernel_val = kernel_mat[ind_connection, l] # l's out-neighbors' states if initially l's state is 1 and others' states are 0

        elif flag == 'Consensus': # if consensus is used as the dynamics of node states, consider the node l's in-links
            ind_connection = (A[l, :]>0) # true if exists l's in-link to a node
            positive_weight = A[l, ind_connection] # weights of l's in-links
            positive_kernel_val = kernel_mat[l, ind_connection]  # l's state if initially one of l's in-neighbors' state is 1 and others' states are 0

        # Compute the correlation coefficient between weights and node states if node l has at least 2 in/out-links.
        if len(positive_weight)>1:
            rval[l] = np.corrcoef(positive_kernel_val, positive_weight)[0, 1]
            nval[l] = len(positive_weight)
        weight_ls.extend(positive_weight.tolist())
        kernel_ls.extend(positive_kernel_val.tolist())

    # average correlation coefficient
    rval = rval[nval>1]; nval = nval[nval>1]
    rho = fisher_z_average_safe(rval, nval)
    return weight_ls, kernel_ls, rho

def increment_weight_relation(A, kernel_mat, eta, flag):
    """
    Compute the average correlation between weight increments and connection weights.
    For advection, correlations are computed across the out-links of each node.
    For consensus, correlations are computed across the in-links of each node.
    args:
        A: Adjacency matrix of the network.
        kernel_mat: Consensus or advection kernel.
        eta: Learning rate.
        flag: Type of dynamics; must be 'Consensus' or 'Advection'.
    returns:
        weight_ls: Connection weights pooled across the relevant in- or out-links.
        weight_increment_ls: Corresponding accumulated weight increments.
        rho: Average correlation coefficient across nodes.
    """
    n = A.shape[0] # number of nodes
    rval = np.zeros(n); nval = np.zeros(n)

    # taking each node l as the source of activity, compute weight increments
    delta_w = np.zeros((n, n, n))
    for l in range(n):
        x = kernel_mat[:, l]
        Delta = eta * x * x[:, np.newaxis]
        Delta[A == 0] = 0
        delta_w[l, :, :] = Delta

    weight_ls = []; weight_increment_ls = []
    for s in range(n): # consider each node s
        if flag == 'Advection': # if advection is used as the dynamics of node states, consider the node s's out-links
            ind_connection = (A[:, s]>0)
            positive_weight = A[ind_connection, s] # weights of s's out-links
            weight_increments  = delta_w[:, ind_connection, s] # increments of weights of s's out-links

        elif flag == 'Consensus': # if consensus is used as the dynamics of node states, consider the node l's in-links
            ind_connection = (A[s, :]>0)
            positive_weight = A[s, ind_connection] # weights of s's in-links
            weight_increments  = delta_w[:, s, ind_connection] # increments of weights of s's in-links

        accumulated_weight_increments = np.sum(weight_increments, 0) # sum of weight increments across all node as source

        # Compute the correlation coefficient between weights and increments if node s has at least 2 in/out-links.
        if len(positive_weight)>1:
            rval[s] = np.corrcoef(accumulated_weight_increments, positive_weight)[0, 1]
            nval[s] = len(positive_weight)
        weight_ls.extend(positive_weight.tolist())
        weight_increment_ls.extend(accumulated_weight_increments.tolist())

    # average correlation coefficient
    rval = rval[nval>1]; nval = nval[nval>1]
    rho = fisher_z_average_safe(rval, nval)
    return weight_ls, weight_increment_ls, rho