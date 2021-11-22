import helpers.test_generator_helpers as h
import networkx as nx
import random
from collections import deque

def testgen_permutation(graph_generator=h.sbm_generator, n=10):
    """
    Using specified graph generator, generate a graph and permute it with
    a permutation matrix.
    
    Parameters
    ----------
    graph_generator:
        Graph generator function, taking as argument the desired size
    n:
        Size of generated graph
        
    Returns
    -------
    L1:
        Laplacian of first graph
    L2:
        Laplacian of second graph
    P:
        Identity matrix
    """
    P = h.permutation_generator(n)
    G = graph_generator(n)
    L1 = h.get_laplacian(G)
    L2 = P@L1@P.T

    return L1, L2, P


def testgen_otm_weighted(graph_generator=h.sbm_generator, n1=10, n2=13, seed=None):
    """
    Using specified graph generator, generate a graph of size n2, then collapse it 
    into a graph of size n1, and return the collapsing matrix P.
    
    Parameters
    ----------
    graph_generator:
        Graph generator function, taking as argument the desired size
    n1:
        Size of smaller graph
    n2:
        Size of larger graph
        
    Returns
    -------
    L1:
        Laplacian of first graph
    L2:
        Laplacian of second graph
    P:
        Identity matrix
    """
    assert n1<=n2, "n2 has to be larger than n1"
    G2 = graph_generator(n2, seed=seed)
    P = h.edge_collapse_permutation(G2, n2-n1, seed=seed)
    L2 = h.get_laplacian(G2)
    L1 = P@L2@P.T
    return L1, L2, P


def testgen_otm_unweighted(graph_generator=h.sbm_generator, n1=10, n2=13):
    """
    Using specified graph generator, generate a graph of size n2, then collapse it 
    into a graph of size n1 and set all edge weights to 1.
    
    Parameters
    ----------
    graph_generator:
        Graph generator function, taking as argument the desired size
    n1:
        Size of smaller graph
    n2:
        Size of larger graph
        
    Returns
    -------
    L1:
        Laplacian of first graph
    L2:
        Laplacian of second graph
    P:
        Identity matrix
    """
    assert n1<=n2, "n2 has to be larger than n1"
    G2 = graph_generator(n2)
    P = h.edge_collapse_permutation(G2, n2-n1)
    L2 = h.get_laplacian(G2)
    L1 = P@L2@P.T
    L1[L1<0] = -1
    return L1, L2, P


def create_perturbed_sbm(n, pin, pout, ncomm, nremove, ncollapse, seed=None):
    '''
    Creates perturbed stochastic block model: G2 is just SBM with specified
    number of communities ncomm and edge probabilities pin within, and pout between 
    communities.
    G1 is obtained by removing nremove edges from G2, before collapsing ncollapse
    edges. 
    '''
    g2 = h.sbm_generator(n, pin, pout, ncomm, seed=seed)
    g1 = g2.copy()
    
    # Create dictionary from node to its community
    comm_dict = dict()
    for comm, nodes in enumerate(g1.graph['partition']):
        for n in nodes:
            comm_dict[n] = comm
    
    # Remove nremove edges between communities
    all_edges = list(nx.edges(g1))
    random.seed(seed)
    random.shuffle(all_edges)
    remove_candidates = deque(all_edges)
    n_removed = 0
        
    # Remove edges while there are candidate edges 
    while len(remove_candidates)>0 and n_removed<nremove:
        (fr, to) = remove_candidates.pop()
        if comm_dict[fr] != comm_dict[to]:
            g1.remove_edge(fr, to)
            n_removed+=1

    
    # Collapse edges of G1
    P = h.edge_collapse_permutation(g1, ncollapse)
    
    # Get Laplacian
    L1 = P@h.get_laplacian(g1)@P.T
    L2 = h.get_laplacian(g2)
    
    return L1, L2, P
