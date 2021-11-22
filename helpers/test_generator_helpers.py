import warnings
import random
from collections import deque
import networkx as nx
import numpy as np
import numpy.linalg as lg

def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x   + alpha * np.eye(len(x)) + np.ones([len(x),len(x)])/len(x)) 
        #x_reg = lg.inv(x   + np.ones([len(x),len(x)])/len(x))
    else:
        x_reg = lg.inv(x + alpha * np.eye(len(x)))
    return x_reg


def get_G_from_L(L):
    adj = np.copy(-L)
    adj[np.diag_indices(L.shape[0])] = 0
    return nx.from_numpy_array(adj)

def sbm_generator(n, pin=.95, pout=.1, ncomm=2, seed=None):
    """
    Generates connected stochastic block model graph with two clusters of the same size, 
    with a total of n nodes.
    """
    # Get evenly sized clusters
    a = n//ncomm
    b = n-a*(ncomm-1)
    sizes = np.repeat(a, ncomm)
    sizes[0] = b # Ensure that sum is n 
    prob = np.eye(ncomm) * pin + (1-np.eye(ncomm)) * pout
    
    # Generate connected sbm graph
    i=0
    while True:
        if seed is not None:
            sbm = nx.generators.community.stochastic_block_model(sizes, prob, seed=seed+i)
        else:
            sbm = nx.generators.community.stochastic_block_model(sizes, prob)
        if nx.is_connected(sbm):
            return sbm
        i+=1

def er_generator(n):
    """
    Generates connected ER graph with p=2*log(n)/n
    """
    # Choose p such that graph is connected whp
    p = 2*np.log(n) / n
    
    while True:
        er = nx.generators.erdos_renyi_graph(n, p)
        if nx.is_connected(er):
            return er
        
def permutation_generator(n):
    """
    Generates permutation matrix of size n*n
    """
    a = np.eye(n)
    reorder = np.random.permutation(n)
    perm = a[:,reorder]
    return perm

def otm_generator(m, n):
    """
    Generates a one-to-many assignment matrix of size m*n
    """
    assert n>=m
    
    out = np.zeros((m,n))
    
    # Ensure that every row has at least one entry
    ix = np.random.choice(n, m, replace=False)
    out[range(m),ix] = 1
    
    # Ensure that every column has exactly one entry
    no_entry = out.sum(axis=0) == 0
    ix = np.random.choice(m, no_entry.sum(), replace=True)
    out[ix, no_entry] = 1
    return out  

def edge_collapse_permutation(G, n, seed=None):
    """
    Returns permutation matrix when collapsing n edges of the graph.
    """
    if seed is not None:
        random.seed(seed)
    edges = random.sample(G.edges, n)
    
    # Construction of permutation matrix
    P = np.eye(len(G))
    lookup = np.arange(len(G))
    todelete = []
    for edge in edges:
        (fr, to) = edge
        fr = lookup[fr]
        to = lookup[to]
        P[fr,:] += P[to,:]
        lookup[to] = fr
        todelete.append(to)
    
    return np.delete(P, todelete, 0)    

def get_laplacian(G):
    """
    Returns laplacian of generated graph as numpy array
    """
    return nx.laplacian_matrix(G).todense()

def add_edges(G, n):
    """
    Randomly adds n edges to graph G IN PLACE
    """
    copied = G.copy()
    def add_single_edge(a,b):
        copied.add_edge(a,b)
    
    # Choose among random non-edges which to add
    non_edges = list(nx.non_edges(copied))
    sample = np.random.choice(len(non_edges), n, replace=False)
    [add_single_edge(*non_edges[ix]) for ix in sample]
    return copied

def remove_edges(G, n):
    """
    Randomly removes n edges from graph G, while ensuring that it remains connected
    """
    copied = G.copy()
    all_edges = list(nx.edges(copied))
    random.shuffle(all_edges)
    remove_candidates = deque(all_edges)
    n_removed = 0
    
    # Remove edges while there are candidate edges 
    while len(remove_candidates)>0 and n_removed<n:
        curr_edge = remove_candidates.pop()
        copied.remove_edge(*curr_edge)
        
        # Put edge back if it results in non-connected graph
        if not nx.is_connected(copied):
            copied.add_edge(*curr_edge)
        else:
            n_removed+=1
    
    if n_removed!=n:
        warnings.warn("Not enough edges could be removed.")
        return copied
    else:
        return copied