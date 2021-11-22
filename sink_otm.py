import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd

def sink_vect_exp(K, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances

    samples = K.size()[0]
    a = torch.ones((K.size()[1],)) / K.size()[1]
    b = torch.ones(( K.size()[2],)) / K.size()[2]
    b_scalar = b[0]
    
    # init data
    Nini = len(a)
    Nfin = len(b)

    u = torch.ones(samples, Nini, 1) / Nini
    v = torch.ones(samples, Nfin, 1) / Nfin


    Kp = (1 / a).view(-1, 1) * K

    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = torch.matmul(K.transpose(1,2), u)
        v = b_scalar/KtransposeU 
        u = 1. / (Kp @ v)
        
        if cpt % 10 == 0:
            transp = u * (K * v.transpose(1,2))
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1
    return u * K * v.transpose(1,2)



def sink_vect(M, reg, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances


    samples = M.size()[0]
    a = torch.ones((M.size()[1],)) / M.size()[1]
    b = torch.ones(( M.size()[2],)) / M.size()[2]
    b_scalar = b[0]
    
    # init data
    Nini = len(a)
    Nfin = len(b)

    u = torch.ones(samples, Nini, 1) / Nini
    v = torch.ones(samples, Nfin, 1) / Nfin


    K = torch.exp(-M / reg)

    Kp = (1 / a).view(-1, 1) * K

    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = torch.matmul(K.transpose(1,2), u) 
        v = b_scalar/KtransposeU
        u = 1. / (Kp @ v)
        
        if cpt % 10 == 0:
            transp = u * (K * v.transpose(1,2))
            err = (torch.sum(transp) - b).norm(1).pow(2).item() / samples

        cpt += 1
    return u * K * v.transpose(1,2)



def sink_exp(K, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((K.size()[0],)) / K.size()[0]).cuda()
        b = Variable(torch.ones((K.size()[1],)) / K.size()[1]).cuda()
    else:
        a = torch.ones((K.size()[0],)) / K.size()[0]
        b = torch.ones((K.size()[1],)) / K.size()[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = torch.ones(Nini) / Nini
        v = torch.ones(Nfin) / Nfin


    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = K.t() @ u
        v = b / KtransposeU
        u = 1. / (Kp @ v)
        
        if cpt % 10 == 0:
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1
    return u.view((-1, 1)) * K * v.view((1, -1)) 


def sink(M, reg, numItermax=10, stopThr=1e-9, cuda = False):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = torch.ones((M.size()[0],)) / M.size()[0]
        b = torch.ones((M.size()[1],)) / M.size()[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = torch.ones(Nini) / Nini
        v = torch.ones(Nfin) / Nfin

    K = torch.exp(-M / reg)

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax): 
        uprev = u
        vprev = v
        KtransposeU = K.t() @ u
        v = b / KtransposeU
        u = 1. / (Kp @ v)
        
        if cpt % 10 == 0:
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        cpt += 1
    return u.view((-1, 1)) * K * v.view((1, -1)) 


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, cuda=False):

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if cuda:
            alpha, beta = Variable(torch.zeros(na)).cuda(), Variable(torch.zeros(nb)).cuda()
        else:
            alpha, beta = Variable(torch.zeros(na)), Variable(torch.zeros(nb))
    else:
        alpha, beta = warmstart

    if cuda:
        u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
    else:
        u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).data[0] > tau or torch.max(torch.abs(v)).data[0] > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            if cuda:
                u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
            else:
                u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).data[0]

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        #if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        #    # we have reached the machine precision
        #    # come back to previous solution and quit loop
        #    print('Warning: numerical errors at iteration', cpt)
        #    u = uprev
        #    v = vprev
        #    break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v)*M)

def pairwise_distances(x, y, method='l1'):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l1':
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()

def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

    return mm
