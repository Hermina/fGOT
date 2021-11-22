import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


def show_network(G, y=None, labels=None, pos=None, ax=None, figsize=(5,5)):

    if ax is None:
        plt.figure(figsize=figsize)  # image is 8 x 8 inches
        plt.axis('off')
        ax = plt.gca()

    if pos is None:
        pos = nx.kamada_kawai_layout(G)
        
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8, cmap=plt.cm.RdYlGn, node_color=y, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
            
    if labels is None:
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold', font_size=15, ax=ax)
    else:
        labeldict = {}
        for i, v in enumerate(G.nodes):
            labeldict[v] = labels[i]
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold', font_size=15, labels=labeldict, ax=ax)


def G_from_L(L):
    A = L<0
    return nx.from_numpy_array(A)

def plot_permuted_graph(L1, L2, P_true, P_est=None, pos = None):    
    
    if P_est is None:
        P_est = np.eye(*P_true.shape)

    G1 = G_from_L(L1)
    G2 = G_from_L(L2)

    N_nodes = L1.shape[0]
    
    n1 = np.arange(1, N_nodes+1)
    n2 = (P_est.T@P_true@n1).astype(int)
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].axis('off')
    ax[1].axis('off')
    
    pos = nx.kamada_kawai_layout(G1)
    
    nx.draw(G1, pos=pos, ax=ax[0])
    show_network(G2, pos=pos, labels=n2, y=np.ones(N_nodes), ax=ax[1])
    
    plt.show()
    
    
def plot_collapsed_graph(L1, L2, P_true, P_est=None, pos = None):        
    if P_est is None:
        P_est = np.eye(*P_true.shape)
    
    G1 = G_from_L(L1)
    G2 = G_from_L(L2)

    n1_nodes = L1.shape[0]
    
    n1 = np.arange(1, n1_nodes+1)
    n2_imp = (P_est.T@n1).astype(int)
    n2_tru = (P_true.T@n1).astype(int)
    
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    
    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2, pos=pos1)
    
    nx.draw(G1, pos1, ax=ax[0], labels=dict(enumerate(n1)),     node_color='wheat', linewidths=1)
    nx.draw(G2, pos2, ax=ax[1], labels=dict(enumerate(n2_tru)), node_color='wheat', linewidths=1)
    nx.draw(G2, pos2, ax=ax[2], labels=dict(enumerate(n2_imp)), node_color='wheat', linewidths=1)
    
    ax[0].set_title('G1')
    ax[1].set_title('G2')
    ax[2].set_title('G2, imputed permutation')
    
    plt.show()

    
    
def show_graph(L, title, pos=None, labels=None, P=None, seed=42):
    """
    New version, shows edge weights if they are 1 or 2.
    """
    
    G = G_from_L(L)    
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)  # positions for all nodes
    
    # If no labels, numerate
    if labels is None and P is None:
        labels = np.arange(len(G)) + 1
    if labels is None and P is not None:
        labels = np.arange(P.shape[0]) + 1
    if P is not None:
        labels = (P.T@labels).astype(int)
    
    # Divide edges into those with weight one and two
    eTwo= [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1]
    eOne = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 1]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=eOne,
                           width=3)
    nx.draw_networkx_edges(G, pos, edgelist=eTwo,width=6)    
    nx.draw_networkx_labels(G, pos, labels=dict(enumerate(labels)),font_color='w',
                           font_weight='bold', font_size=16)
    
    plt.title(title, {'fontsize':20});
    plt.axis('off')

    
def show_p(P, title):
    """
    Shows Permutation matrix P
    """
    plt.imshow(P, cmap='BuGn')
    plt.axis('off')
    plt.title(title, {'fontsize':20})

    
def show_all(L1, L2, P, names=['Full Graph', 'Node 1 Collapsed', 'P'], pos=[None, None],
            labels=[None, None], Pplot=[None, None]):
    """
    Shows two matrices defined by Laplacian and Permutation.
    Note that plot order is L2, L1, P
    
    names: 
        Plot titles
    pos:
        List of two, positions for graph 2 and graph 1 (in that order)
    labels:
        Labels passed to show_graph for graph 2 and graph 1 (in that order)
    Pplot:
        P passed to show_graph for graph 2 and graph 1 (in that order)
    """    
    plt.subplot(131)
    show_graph(L1, names[1], seed=41, pos=pos[1], labels=labels[1], P=Pplot[1])
    plt.subplot(132)
    show_graph(L2, names[0], seed=41, pos=pos[0], labels=labels[0], P=Pplot[0])
    plt.subplot(133)
    show_p(P, names[2])
