"""
Provide functions for adaptive weight adjustment.
"""
import numpy as np
import sys
sys.path.append(".")
from scripts import basic_components as bcomp

############## Raw weight increment #############
def weight_increment(A, K, eta, u):
    """
    Compute raw weight increments for one step of adaptive weight adjustment.
    args:
        A: Adjacency matrix of the network.
        K: Consensus or advection kernel.
        eta: Learning rate.
        u: Source node(s) of activity.
    returns:
        Delta: Matrix of raw weight increments.
    """
    # node states
    cols = K[:, np.atleast_1d(u)]

    # compute weight increments, and only existing connections will receive increments
    Delta = eta * (cols @ cols.T)
    Delta[A==0] = 0
    return Delta

############## Normalization ###################
def in_normalization(A, Delta):
    """
    Normalize weights after applying the increments to keep in-strengths constant.
    args:
        A: Adjacency matrix before the weight-adjustment step.
        Delta: Matrix of weight increments.
    returns:
        A: Adjacency matrix after normalization.
        trunc_temp: Number of links truncated due to negative weights.
    """
    A_temp = A + Delta
    W_in = np.sum(A, axis=1, keepdims=False,)
    W_in_temp = np.sum(A_temp, axis=1, keepdims=False,)
    W_in_temp[W_in_temp==0] = 1 #make denomintor non-zero
    W_norm = W_in/W_in_temp
    A = A_temp*W_norm[:, np.newaxis]

    # truncate edges with negative weight
    trunc_temp = np.sum(A<0)
    A[A<0] = 0
    return A, trunc_temp

def out_normalization(A, Delta):
    """
    Normalize weights after applying the increments to keep out-strengths constant.
    args:
        A: Adjacency matrix before the weight-adjustment step.
        Delta: Matrix of weight increments.
    returns:
        A: Adjacency matrix after normalization.
        trunc_temp: Number of links truncated due to negative weights.
    """
    A_temp = A + Delta
    W_out = np.sum(A, axis=0, keepdims=False,)
    W_out_temp = np.sum(A_temp, axis=0, keepdims=False,)
    W_out_temp[W_out_temp==0] = 1 #make denomintor non-zero
    W_norm = W_out/W_out_temp
    A = A_temp*W_norm

    # truncate edges with negative weight
    trunc_temp = np.sum(A<0)
    A[A<0] = 0
    return A, trunc_temp

################ adaptive weight adjustment #############
def weight_adjust_Cout(A, eta, tau, u):
    """
    Perform one step of adaptive weight adjustment based on consensus dynamics.
    args:
        A: Adjacency matrix of the network.
        eta: Learning rate.
        tau: Time constant of the consensus dynamics.
        u: Source node(s) of activity.
    returns:
        A: Adjacency matrix after one step of adaptive weight adjustment.
        trunc_temp: Number of links truncated due to negative weights.
    """
    K = bcomp.consensus_kernel(A, tau)
    Delta = weight_increment(A, K, eta, u)
    A, trunc_temp = out_normalization(A, Delta)
    return A, trunc_temp

def weight_adjust_Ain(A, eta, tau, u):
    """
    Perform one step of adaptive weight adjustment based on advection dynamics.
    args:
        A: Adjacency matrix of the network.
        eta: Learning rate.
        tau: Time constant of the consensus dynamics.
        u: Source node(s) of activity.
    returns:
        A: Adjacency matrix after one step of adaptive weight adjustment.
        trunc_temp: Number of links truncated due to negative weights.
    """
    K = bcomp.advection_kernel(A, tau)
    Delta = weight_increment(A, K, eta, u)
    A, trunc_temp = in_normalization(A, Delta)
    return A, trunc_temp
