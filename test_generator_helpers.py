import warnings
import random
from collections import deque
import networkx as nx
import numpy as np



def sbm_generator(n, pin, pout, ncomm):
    """
    Generates connected stochastic block model graph with ncomm clusters of the same size, 
    with a total of n nodes.
    """
    # Get evenly sized clusters
    a = n//ncomm
    b = n-a*(ncomm-1)
    sizes = np.repeat(a, ncomm)
    sizes[0] = b # Ensure that sum is n 
    prob = np.eye(ncomm) * pin + (1-np.eye(ncomm)) * pout
    for i in range(1,ncomm):
        prob[i-1, i] = 3*pout
        prob[i, i-1] = 3*pout
    
    # Generate connected sbm graph
    while True:
        sbm = nx.generators.community.stochastic_block_model(sizes, prob)
        if nx.is_connected(sbm):
            return sbm

def ba_generator(n, m):
    """
    Generates connected BA graph
    """
    # Choose p such that graph is connected whp
    #p = 2*np.log(n) / n
    
    while True:
        ba = nx.generators.barabasi_albert_graph(n, m)
        if nx.is_connected(ba):
            return ba
        
        
def er_generator(n):
    """
    Generates connected ER graph with p=2*log(n)/n
    """
    # Choose p such that graph is connected whp
    p = 2*np.log(n) / n #2
    
    while True:
        er = nx.generators.erdos_renyi_graph(n, p)
        if nx.is_connected(er):
            return er
        
def permutation_generator(n, seed = None):
    """
    Generates permutation matrix of size n*n
    """
    if seed is not None:
        np.random.seed(seed)
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
    print(n)
    print(len(edges))
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
    print(len(todelete))
    print(np.delete(P, todelete, 0).shape)
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