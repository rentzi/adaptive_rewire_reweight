"""
Provide functions for construction of null networks, network connectivity metrics,
and analysis of convergent-divergent units.
"""
import numpy as np
import pandas as pd
import networkx as nx
import sys
sys.path.append('.')
from scripts import rewire, basic_components as bcomp

################### Null networks ########################
def randomizeWeightsKeepTopology(A):
    """
    Generate a null network by randomly permuting edge weights while
    preserving the network topology.
    args:
        A: Adjacency matrix of the network.
    returns:
        A_perm: Adjacency matrix of the null network.
    """
    A_perm = A.copy()

    indNonZero = np.where(A > 0)  # finds the indices of the nonzero elements -weights

    # shuffled version of the weights
    weights = A[indNonZero]
    newWeights = np.random.permutation(weights)

    A_perm[indNonZero] = newWeights

    return A_perm

def rewire_only(A, init_mat, tau_rewire, tau_reweight, prand, final_step):
    """
    Generate a rewiring-only null network with fixed edge weights.
    The null network starts from the initial network topology, with weights
    randomly assigned from the final simulated network, and subsequently
    undergoes rewiring without adaptive weight adjustment.
    args:
        A: Adjacency matrix of the final simulated network.
        init_mat: Adjacency matrix of the initial network.
        tau_rewire: time constant for adaptive rewiring
        ttau_reweight: Time constant for adaptive weight adjustment in the
            original simulation.
        prand: Probability of random rewiring.
        final_step: Number of simulation steps.
    returns:
        null_init: Initial null network.
        null_A: Final rewiring-only null network.
    """
    # get the initial null network
    null_init = init_mat.copy() # initial random network
    w = A[A>0] # weights of the simulated network
    np.random.shuffle(w)
    null_init[null_init>0] = w # assign to the null network
    
    # rewire the initial null network
    rate = int(tau_rewire / tau_reweight)
    n = A.shape[0]
    null_A = null_init.copy()            
    for i in range(final_step):
        if (i+1)%rate==0:
            r1 = np.random.random_sample() # choose incoming or outgoing connections
            if r1<0.5:
                flag = 'in'
            else:
                flag = 'out'
            
            r2 = np.random.random_sample() # choose random or adaptive rewiring
            if r2<prand:
                null_A = rewire.random_rewire(A, n, flag)
            else:
                conK = bcomp.consensus_kernel(A, tau_rewire)
                advK = bcomp.advection_kernel(A, tau_rewire)
                null_A = rewire.rewire(A, conK, advK, n, flag)

    return null_init, null_A

################ Connectivity metrics ##########################
def convert_from_adj2networkX(A, weighted=True):
    """
    Convert an adjacency matrix to a NetworkX directed graph for computing path length.
    args:
        A: Adjacency matrix of the network.
        weighted: Whether to construct a weighted graph. If True, the edge
            attribute 'weight' is set to the inverse of the connection weight
            and is therefore interpreted as distance.
    returns:
        G: NetworkX DiGraph representation of the network.
    """
    edges_ind = np.where(A>0)
    num_edges = len(edges_ind[0])

    G = nx.DiGraph()   # Create a DiGraph

    # add nodes
    G.add_nodes_from(np.arange(A.shape[0]))

    # add edges
    edges_list = list()
    if weighted:
        for ind in np.arange(num_edges):
            edge_pair_w = (edges_ind[1][ind],edges_ind[0][ind],1/A[edges_ind[0][ind],edges_ind[1][ind]]) #distance = 1/weight
            edges_list.append(edge_pair_w)    
        
        G.add_weighted_edges_from(edges_list)
    else:
        for ind in np.arange(num_edges):
            edge_pair = (edges_ind[1][ind],edges_ind[0][ind])
            edges_list.append(edge_pair)
        
        G.add_edges_from(edges_list)    
    return G

def average_efficiency(A):
    """
    Compute the average efficiency of a weighted directed network.
    Connection distances are defined as the inverse of connection weights.
    args:
        A: Adjacency matrix of the network.
    returns:
        ave_eff: Average efficiency.
    """
    n = A.shape[0]
    G = convert_from_adj2networkX(A, weighted=True)
    shortest_path_length = dict(nx.shortest_path_length(G,weight='weight'))
    efficiency_sum = 0
    for key in shortest_path_length.keys():
        for d_ij in shortest_path_length[key].values():
            if d_ij>0:
                efficiency_sum += 1/d_ij
    ave_eff = efficiency_sum/(n*(n-1))

    return ave_eff

def cluster_coefficient(A):
    """
    Compute the average clustering coefficient of a weighted directed network.
    args:
        A: Adjacency matrix of the network
    returns:
        clust_coef: Average clustering coefficient.
    """
    G = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    clust_coef = nx.average_clustering(G, weight='weight')
    return clust_coef

################### Convergent hubs, divergent hubs & units #####################
def connectedness(A):
    """
    Determine reachability between all pairs of nodes.
    args:
        A: Adjacency matrix of the network.
    returns:
        connected_pairs: Binary matrix in which element (i, j) is 1 if a
            directed path from node i to node j exists, and 0 otherwise.
    """
    G =  convert_from_adj2networkX(A,True)
    len_paths = dict(nx.all_pairs_dijkstra_path_length(G))
    connected_pairs = np.zeros(A.shape)
    for s in len_paths.keys():
        for t in len_paths[s].keys():
            if(len_paths[s][t]>0):
                connected_pairs[s][t] = 1
    return  connected_pairs

def hub_number(A, thresh, binary_flag, axisUsed=0):
    """
    Compute the proportion of hubs in a directed network.
    args:
        A: Adjacency matrix of the network.
        thresh: Degree or strength threshold used to define hubs.
        binary_flag: If True, identify hubs based on degree; if False,
            identify hubs based on node strength.
        axisUsed: Axis used to calculate the degree or strength defining
            the hubs.
    returns:
        num_hubs: Proportion of nodes classified as hubs.
    """
    n = A.shape[0]
    if binary_flag==True:
        deg_u = np.sum(A>0,axis=axisUsed)
        deg_n = np.sum(A>0,axis=1-axisUsed)
    else:
        deg_u = np.sum(A,axis=axisUsed)
        deg_n = np.sum(A,axis=1-axisUsed)
        
    num_hubs = len(np.where((deg_u>=thresh)&(deg_n>0))[0])/n
    return num_hubs

# convergent-divergent unit
def cd_pairs(A, connected_pairs, thresh):
    """
    Identify convergent-divergent units in a network.
    args:
        A: Adjacency matrix of the network.
        connected_pairs: Binary reachability matrix indicating whether a
            directed path exists between each pair of nodes.
        thresh: Degree threshold used to identify convergent and divergent hubs.
    returns:
        connected_cd_pairs: Array containing pairs of convergent and divergent
            hubs that form convergent-divergent units. The first row contains
            convergent hubs and the second row contains the corresponding
            divergent hubs.
    """
    Topo_A = 1.0*(A>0)
    deg_in = np.sum(Topo_A,1)
    deg_out = np.sum(Topo_A,0)    
    candCon = np.where((deg_in>=thresh)&(deg_out>0))[0]
    candDiv = np.where((deg_out>=thresh)&(deg_in>0))[0]
    
    if (len(candCon)>0)&(len(candDiv)>0):
        cd_pairs_temp = np.where(connected_pairs[np.ix_(candCon,candDiv)]>0)
        cHubs = candCon[cd_pairs_temp[0]]
        dHubs = candDiv[cd_pairs_temp[1]]
        connected_cd_pairs = np.stack((cHubs,dHubs), axis=0)
    else:
        connected_cd_pairs = np.array([[],[]])
    return connected_cd_pairs
    
def find_interm_subG(A, connected_cd_pairs):
    """
    Identify intermediate subgraphs of convergent-divergent units.
    args:
        A: Adjacency matrix of the network.
        connected_cd_pairs: Pairs of convergent and divergent hubs forming
            convergent-divergent units.
    returns:
        interms: Dictionary mapping each convergent-divergent unit to the
            nodes in its intermediate subgraph.
    """
    n = A.shape[0]
    num_cd = connected_cd_pairs.shape[1]

    interms = {}
    for i in range(num_cd):
        c = connected_cd_pairs[0,i]
        d = connected_cd_pairs[1,i]
        interm = []
        B = A.copy()
        B[c,:] = 0 #cut in-links of c
        B[:,d] = 0 #cut out-links of d
        G = convert_from_adj2networkX(B,True)

        nodes = set(np.arange(n)).difference(set([c,d]))        #all nodes except c and d
        for k in nodes:
            if nx.has_path(G,c,k) & nx.has_path(G,k,d): # c->k->d exist
                interm.append(k)
        interms[i] = interm
    return interms

def interm_stats(A, interm_temp):
    """
    Compute summary statistics across intermediate subgraphs.
    args:
        A: Adjacency matrix of the network.
        interm_temp: Dictionary containing the nodes of the intermediate
            subgraphs.
    returns:
        subG_size: Mean relative size of the intermediate subgraphs.
        subG_density: Mean density of the intermediate subgraphs.
        subG_aveW: Mean edge weight within the intermediate subgraphs.
    """
    n = A.shape[0]
    subG_size = np.nan; subG_density = np.nan; subG_aveW = np.nan
    size_temp = []; density_temp = []; aveW_temp = []
    for l in interm_temp.keys():
        ind = interm_temp[l]
        size = len(ind)
        size_temp.append(size/n)
                    
        if size>1:
            subG_temp = A[np.ix_(ind, ind)]
            edges_num = np.sum(subG_temp>0)
            density_temp.append(edges_num/(size*(size-1)))
            if edges_num>0:
                aveW_temp.append(np.sum(subG_temp)/edges_num)

    if len(size_temp)>0:
        subG_size = np.mean(size_temp)
    if len(density_temp)>0:
        subG_density = np.mean(density_temp)
    if len(aveW_temp)>0:
        subG_aveW = np.mean(aveW_temp)

    return subG_size, subG_density, subG_aveW

def clean_group(x):
    """
    Remove NaN values from a list.
    args:
        x: Input values.
    returns:
        Array containing the non-NaN values of x.
    """
    s = pd.Series(x, dtype='float64')
    s = s.dropna()
    return s.to_numpy()