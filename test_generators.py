from test_generator_helpers import *
import numpy as np

def testgen_permutation(graph_generator=ba_generator, n=10, ncomm = 2, seed = None):
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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    P = permutation_generator(n, seed)
    if graph_generator == sbm_generator:
        print('lal')
        G = graph_generator(n, 0.8, 0.05, ncomm)
    else:
        G = graph_generator(n, ncomm)
    L2 = get_laplacian(G)
    L1 = P@L2@P.T

    return L1, L2, P


def testgen_permutation_perturbed(graph_generator=sbm_generator, n=10, change = 5, ncomm = 2, seed = None):
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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    P = permutation_generator(n, seed)
    G = graph_generator(n, 0.8, 0.08, ncomm)
    G2 = remove_edges(G, change)
    L1 = get_laplacian(G)
    L2 = get_laplacian(G2)
    L1 = P@L1@P.T

    return L1, L2, P

def testgen_otm_weighted(graph_generator=sbm_generator, n1=10, n2=13, ncomm = 2):
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
    G2 = graph_generator(n2, 0.6, 0.4, ncomm)
    P = edge_collapse_permutation(G2, n2-n1)
    L2 = get_laplacian(G2)
    L1 = P@L2@P.T
    return L1, L2, P


def testgen_otm_unweighted(graph_generator=ba_generator, n=10, m=13, ncomm = 2, seed = None):
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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
        
    assert n<=m, "n2 has to be larger than n1"
    G2 = graph_generator(m, ncomm)
    P_true = edge_collapse_permutation(G2, m-n)
    idx = np.random.permutation(n)
    P = P_true[idx, :]
    L2 = get_laplacian(G2)
    L1 = P @ L2 @ P.T
    #L1[L1<0] = -1
    return L1, L2, P