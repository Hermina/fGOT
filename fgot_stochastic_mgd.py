
from sink_otm import *

from bregman import *

import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import random
import math
import numpy.linalg as lg
import scipy.linalg as slg
from matplotlib import pyplot as plt
from   numpy import linalg as LA

import networkx as nx
import time


def loss(DS, g1, g2, loss_type, epsilon = 5e-4):
    """
    Calculate loss, with the help of initially calculated params
    """
     
    if loss_type == 'w_simple':
        cost = - 2 * torch.trace( g1 @ DS @ g2 @ DS.t() )
        
    elif loss_type == 'l2':       
        cost = torch.sum((g1 @ DS - DS @ g2)**2, dim=1).sum()
        
    return cost



# Algorithm -- Stochastic Mirror gradient 
#===================================================================

def fgot_stochastic(g1, g2, tau=1, n_samples=10, epochs=1000, lr=.5, 
            std_init = 10, loss_type = 'w_simple', seed=42, verbose=True, tol = 1e-12, adapt_lr = False):   

    # Initialization
    torch.manual_seed(seed)
    n = g1.shape[0]
    m = g2.shape[0]
    if adapt_lr:
        lr = lr/(np.max(g1)*np.max(g2))
    g1 = to_torch(g1)
    g2 = to_torch(g2)
    g1 = g1 - 0.5*torch.diag(torch.diag(g1))
    g2 = g2 - 0.5*torch.diag(torch.diag(g2))
    
    
    mean = to_torch(np.outer(np.repeat(1/n, n), np.repeat(1/m, m)))
    mean = mean.requires_grad_()
    
    std  = std_init * torch.ones(n, m) 
    std  = std.requires_grad_() 
    
    history = []
    epoch = 0
    err = 1
    while (err > tol and epoch < epochs): 
        cost = 0
        for sample in range(n_samples):
            eps = torch.rand(n, m) 
            P_noisy = mean + std * eps 
            proj = sink_exp(torch.relu(P_noisy) + 1/n)
            cost = cost + loss(proj, g1, g2, loss_type)
        cost = cost/n_samples
        cost.backward()
        
        # Aux.
        s2 = std.data**2
        d  = lr/2 * s2 * std.grad
        
        # Update
        mean_prev = mean.data
        mean.data = mean.data - lr * mean.grad * s2
        std.data  = torch.sqrt(s2 + d) - d   
        
        mean.grad.zero_()
        std.grad.zero_()
        
        # Tracking
        #history.append(cost.item())
        if ((epoch+1) % 10 == 0 and (epoch>50)):

            err = np.linalg.norm(sink(-tau*mean.detach(), tau) - sink(-tau*mean_prev.detach(), tau)) / (n*m)
        epoch = epoch + 1
    
    P = sink(-tau*mean, tau)
    
    P = P.squeeze()
    P = P.detach().numpy()
    
    
    # Convergence plot
#     if verbose:
#         plt.plot(history)
#         plt.show()
    return P


# Tools # ===================================================================================================================
   
    
def torch_invert(x,alpha, ones=False):
    if ones:
        return torch.inverse(x   + 
                             alpha * torch.from_numpy(np.eye(len(x))) + 
                             torch.from_numpy(np.ones([len(x),len(x)])/len(x)))
    else:
        return torch.inverse(x   + 
                             alpha * torch.from_numpy(np.eye(len(x))))

    
def to_torch(x):
    return torch.from_numpy(x.astype(np.float64))



def doubly_stochastic(P, tau, it):
    """ Uses logsumexp for numerical stability. """    
    A = P / tau
    for i in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True) 
    return torch.exp(A)


def rnorm(M):
    r  = M.shape[0] 
    c  = M.shape[1]
    N  = np.zeros((r, c))

    
    for i in range(0,r):
        Mi=np.linalg.norm(M[i,:])
        if Mi!=0:
            N[i,:] = M[i,:] / Mi
        else:
            N[i,:] = M[i,:]
    return N  


