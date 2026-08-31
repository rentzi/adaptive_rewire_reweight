"""
Provide functions for adaptive and random rewiring of directed networks.
"""
import numpy as np
import sys
sys.path.append('.')
from scripts import basic_components as bcomp

######################## adaptive rewiring ######################
def rewire(A, conK, advK, n, flag):
    """
    Perform one step of adaptive rewiring.
    args:
        A: Adjacency matrix of the network.
        conK: Consensus kernel.
        advK: Advection kernel.
        n: Number of nodes in the network.
        flag: Direction of rewiring; must be 'in' or 'out'.
    returns:
        A: Adjacency matrix after one adaptive rewiring step.
    """

    # select the node v whose link will be rewired, get v's neighbors and the rest nodes
    # and assign node states according to the flag
    if flag == 'in': #rewire in-link
        deg_in = np.sum(A > 0, axis=1, keepdims=False,)
        nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
        if len(nodes_receiving)==0:
            print("All nodes have either 0 or (n-1) in-degree.", flush=True)
            return A

        v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)
        x = conK[v,:]
    elif flag == 'out': #rewire out-link
        deg_out = np.sum(A > 0, axis=0, keepdims=False,)
        nodes_sending = np.where((deg_out > 0) & (deg_out < n-1))[0]
        if len(nodes_sending) == 0:
            print("All nodes have either 0 or n-1 out-degree.", flush=True)
            return A

        v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)
        x = advK[:,v]

    states_cp = x[U_cp_v] # states of v's neighbors
    states_nc = x[U_nc_v] # states of other nodes

    # check there are more than one nodes in U_cp_v with minimum kernel value
    # if true, randomly choose among them
    state_min = np.min(states_cp)
    tie_cp = np.where(states_cp==state_min)[0]
    if len(tie_cp)>1:
        ind_minus = np.random.choice(tie_cp)
    else:
        ind_minus = tie_cp[0]

    # check there are more than one nodes in U_nc_v with maximum kernel value
    # if true, randomly choose among them
    state_max = np.max(states_nc)
    tie_nc = np.where(states_nc==state_max)[0]
    if len(tie_nc)>1:
        ind_add = np.random.choice(tie_nc)
    else:
        ind_add = tie_nc[0]

    j_minus = U_cp_v[ind_minus] # the node losing a link
    j_add = U_nc_v[ind_add] # the node getting a link

    # update the adjacency matrix
    if flag == 'in':
        A[v, j_add] = A[v, j_minus]
        A[v, j_minus] = 0
    elif flag == 'out':
        A[j_add, v] = A[j_minus, v]
        A[j_minus, v] = 0
    return A

######################## random rewire #################################
def random_rewire(A, n, flag):
    """
    Perform one step of random rewiring.
    args:
        A: Adjacency matrix of the network.
        n: Number of nodes in the network.
        flag: Direction of rewiring; must be 'in' or 'out'.
    returns:
        A: Adjacency matrix after one random rewiring step.
    """

    # select the node v whose link will be rewired, get v's neighbors and the rest nodes
    if flag == 'in':
        deg_in = np.sum(A > 0, axis=1, keepdims=False,)
        nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
        if len(nodes_receiving)==0:
            print("All nodes have either 0 or (n-1) in-degree.", flush=True)
            return A

        v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)
    elif flag == 'out': #rewire out-link
        deg_out = np.sum(A > 0, axis=0, keepdims=False,)
        nodes_sending = np.where((deg_out > 0) & (deg_out < n-1))[0]
        if len(nodes_sending) == 0:
            print("All nodes have either 0 or n-1 out-degree.", flush=True)
            return A

        v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

    # randomly choose which node loses a link and which one gets a link
    j_minus = np.random.choice(U_cp_v)
    j_add = np.random.choice(U_nc_v)

    # update the adjacency matrix
    if flag == 'in':
        A[v, j_add] = A[v, j_minus]
        A[v, j_minus] = 0
    elif flag == 'out':
        A[j_add, v] = A[j_minus, v]
        A[j_minus, v] = 0
    return A
