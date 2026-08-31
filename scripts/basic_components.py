"""
Provide basic functions used across all simulations.
"""

import numpy as np
from scipy.linalg import expm
import warnings

def generate_rand_adj(n_nodes, edges, weight_distribution, **kwargs):
    """
    Generate an adjacency matrix of a random directed weighted network.
    Adapted from Rentzeperis et al., 2022.
    args:
        n_nodes: Number of nodes.
        edges: Number of edges.
        weight_distribution: Distribution used to generate edge weights;
            must be 'binary', 'normal' or 'lognormal'.
        **kwargs:
            mu: The mu parameter, is not valid for binary.
            sig: The sig parameter, is not valid for binary.
    returns:
        adj_matrix: Adjacency matrix of the generated network.
    """
    # settings of weight distribution
    for (key, value,) in kwargs.items():
        if key == "mu":
            mu = value
        elif key == "sig":
            sig = value

    # set constant
    EPSILON = 0.05

    # set the max number of network edges
    max_connections = int(n_nodes * (n_nodes - 1))
    print("max connections:", max_connections,)
    print("weight distribution:", weight_distribution,)
    
    if edges > max_connections or edges < 0:
        print("Edge number out of range")
        return -1
        
    print("Generating random adjacency matrix ...")

    # sample weights from a lognormal distribution
    if weight_distribution == "lognormal":
        rand_weights = np.random.lognormal(mean=mu, sigma=sig, size=edges,)

    # ... from a normal distribution
    elif weight_distribution == "normal":
        rand_weights = np.random.normal(loc=mu, scale=sig, size=edges,)
        ind = np.where(rand_weights < 0)
        rand_weights[ind] = EPSILON

    # ... from a binary distribution
    elif weight_distribution == "binary":
        rand_weights = np.ones(edges)

    print("Weights generation: random weights: type:{}, ndim: {}, shape: {}, mean: {}, std: {}".format(type(rand_weights),rand_weights.ndim,rand_weights.shape,np.mean(rand_weights),np.std(rand_weights),))

    # Normalize weights such that their sum equals the number of edges
    if (weight_distribution == "normal") | (weight_distribution == "lognormal"):
        norm_factor = edges / np.sum(rand_weights)
        norm_rand_weights = rand_weights * norm_factor
        # [print info on normalized weights for validation]
        print("Weights normalization: normalized random weights: type:{}, ndim: {}, shape: {}, min: {}, max: {},mean: {},std: {},factor: {}".format(type(norm_rand_weights),norm_rand_weights.ndim,norm_rand_weights.shape,norm_rand_weights.min(),norm_rand_weights.max(),np.mean(norm_rand_weights),np.std(norm_rand_weights),norm_factor))
    else:
        norm_rand_weights = rand_weights

    # Get the indices of 1s of a matrix the same size as A with 1s everywhere except in the diagonal
    Aones = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    ind = np.where(Aones)

    # Pick a random sample of those indices (# edges)
    rand_max_con = np.random.permutation(max_connections)
    rand_edges_ind = (ind[0][rand_max_con[:edges]],ind[1][rand_max_con[:edges]],)

    # build the adjacency matrix w/ those indices
    adj_matx = np.zeros((n_nodes, n_nodes))
    adj_matx[rand_edges_ind] = norm_rand_weights
    print("Adjacent matrix: type:{}, ndim: {}, shape: {}, min: {}, max: {}".format(type(adj_matx), adj_matx.ndim, adj_matx.shape, adj_matx.min(), adj_matx.max(), ))

    print("Completed")
    return adj_matx

def consensus_kernel(weight_matx, tau):
    """
    Compute the consensus kernel.
    args:
        weight_matx: Adjacency matrix.
        tau: Time scale of the dynamics.
    returns:
        kernel: Consensus kernel.
    """
    # calculate the in degree Laplacian
    Din = np.diag(np.sum(weight_matx, axis=1))
    Lin = Din - weight_matx

    # calculate the consensus kernel
    kernel = expm(-tau * Lin)
    if np.any(kernel<0):
        #print(kernel[kernel<0], flush=True)
        warnings.warn('Negative consensus kernel!')
    kernel[kernel<1e-308] = 0 #avoid overflow in division
    return kernel

def advection_kernel(weight_matx, tau):
    """
    Compute the advection kernel.
    args:
        weight_matx: Adjacency matrix.
        tau: Time scale of the dynamics.
    returns:
        kernel: Advection kernel.
    """
    # calculate the out degree Laplacian
    Dout = np.diag(np.sum(weight_matx, axis=0))
    Lout = Dout - weight_matx
    
    # calculate the advection kernel
    kernel = expm(-tau * Lout)
    if np.any(kernel<0):
        #print(kernel[kernel<0], flush=True)
        warnings.warn('Negative advection kernel!')
    kernel[kernel<1e-308] = 0 #avoid overflow in division
    return kernel

def choose_rewire_vertex(A, nodes_rew, flag):
    """
    Select a node for rewiring and identify its connected and unconnected nodes.
    args:
        A: Adjacency matrix.
        nodes_rew: Rewirable nodes, i.e., with its in-/out-degree > 0 and < n-1.
        flag: Direction of the rewired link; must be 'in' or 'out'.
    returns:
        v: Node selected for rewiring.
        U_cp_v: Nodes connected to v in the specified direction.
        U_nc_v: Nodes not connected to v in the specified direction.
    """
    # randomly select node v from the rewirable nodes
    v = np.random.choice(nodes_rew)

    n = A.shape[0]
    all_vertices_ind = np.arange(n)
    no_v_vertices_ind = np.delete(all_vertices_ind, v)
    
    if flag=='in':
        U_cp_v = no_v_vertices_ind[np.where(A[v, no_v_vertices_ind]>0)[0]]
        U_nc_v = no_v_vertices_ind[np.where(A[v, no_v_vertices_ind]==0)[0]]
        return v, U_cp_v, U_nc_v
    elif flag=='out':
        U_cp_v = no_v_vertices_ind[np.where(A[no_v_vertices_ind, v]>0)[0]]
        U_nc_v = no_v_vertices_ind[np.where(A[no_v_vertices_ind, v]==0)[0]]
        return v, U_cp_v, U_nc_v
