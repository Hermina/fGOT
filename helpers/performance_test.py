import numpy as np
from matplotlib import pyplot as plt
from sgd_otm.sgd_otm import *
from sgd_otm.dykstra_otm import satisfies_constraints
import seaborn as sns
import pandas as pd

def nb_different_edges(L1, L2):
    """
    Calculates number of different edges between two graphs defined by 
    their laplacians L1 and L2
    """
    return np.sum(np.triu(L1,1) != np.triu(L2,1))


def matrix_difference(L1, L2, P, Pstar):
    """
    Loss function: L2 Frobenius norm between P and Pstar
    """
    return np.linalg.norm(P-Pstar)


def collapsed_edge_difference(L1, L2, P, Pstar):
    """
    Loss function: Collapses graph L2 according to Pstar, then calculates the number
    of different edges from first to second graph
    """
    coll = Pstar@L2@Pstar.T
    return nb_different_edges(L1, coll)
    
    
def errorbar_plot(xaxis, values, axis=1):
    plt.errorbar(xaxis, np.nanmean(values, axis=axis), np.std(values,axis=axis),
                 fmt='o')
    
def error_plot(x, error_w, error_l2):
    """
    Plots error of Wasserstein and L2 loss for varying values of x as boxplot. 
    """
    def to_pandas(x, y, val):
        _, ncol = y.shape
        out = pd.DataFrame({'y': y.flatten(), 'x': np.repeat(x, ncol), 'Loss': val})
        return out
    
    pd_w = to_pandas(x, error_w, 'w')
    pd_l2 = to_pandas(x, error_l2, 'l2')
    concat = pd.concat([pd_w, pd_l2])
    
    sns.boxplot(x='x', y='y', hue='Loss', data=concat)
    plt.ylabel('Error')


def performance_test(nb_test_iters, testgen, loss_fcn, kmax, 
                     loss_type, verbose=False, 
                     **kwargs):
    """
    Tests performance of the algorithm for randomly generated graphs 
    and permutations, given a set of parameters. 
    
    Parameters
    ----------
    nb_test_iters:
        Number of random graphs generated
    testgen:
        Test generator
        Function with no arguments that returns a tuple (L1, L2, P),
        where L1 and L2 are the Laplacians of the generated graphs
        and P is a permutation or one-to-many matrix
    loss:
        Function with four arguments: L1, L2, P, Pstar, which calculates
        the error given the two graph Laplacians, L1 and L2, the true 
        one-to-many matrix P and the found matrix Pstar. Returns float.
    verbose:
        Prints error and whether constraints are verified at every iteration.
    **kwargs:
        Parameters passed to sgd_otm.
    
    Returns
    -------
    err:
        A numpy.ndarray of errors, storing the error at each iteration
    """
    err = np.empty(nb_test_iters)
    for it in range(nb_test_iters):
        # Generate random graph
        L1, L2, P = testgen()
        
        # Find permutation matrix
        try:
            if loss_type=="w":
                L1_inv = regularise_invert_one(L1, .1, True)
                L2_inv = regularise_invert_one(L2, .1, True)
                Pstar = sgd_otm(L1_inv, L2_inv, kmax, loss_type='w', verbose=verbose, **kwargs)
            else:
                Pstar = sgd_otm(L1, L2, kmax, loss_type=loss_type, verbose=verbose, **kwargs)
            
        except RuntimeError:
            print("Warning: Did not converge.")
            err[it] = np.NaN
            continue
            
        # Calculate error, verify constraints
        err[it] = loss_fcn(L1, L2, P, np.rint(Pstar))
        verif = satisfies_constraints(P, kmax)
        
        if verbose:
            print("It",it+1,"\t Error: ", err[it], 
                  "\tConstraints verified:", verif)
        
    return err


def parameter_optimization(optimize, opt_range, nb_test_iters, testgen, loss_fcn, kmax,
                           loss_type='w', verbose=False, **kwargs):
    """
    Generates random graphs and tries to find the permutation, with one
    parameter varying over a specified range.
    
    Parameters
    ----------
    optimize:
        Parameter to optimize. String, currently supported are:
        "nit", "tau", "std_init", "n_samples", "epochs", "lr"
    opt_range:
        Parameter values to use
    nb_test_iters:
        How many random graphs should be generated for a given parameter
    testgen:
        Test generator
        Function with no arguments that returns a tuple (L1, L2, P),
        where L1 and L2 are the Laplacians of the generated graphs
        and P is a permutation or one-to-many matrix
    loss:
        Function with four arguments: L1, L2, P, Pstar, which calculates
        the error given the two graph Laplacians, L1 and L2, the true 
        one-to-many matrix P and the found matrix Pstar. Returns float.
    **kwargs:
        Passed to performance_test
        
        
    The remaining parameters are passed to sgd_otm.
    
    Returns
    -------
    errs:
        numpy.ndarray of size (len(opt_range), nb_test_iters), that stores
        all errors obtained
    """
    
    # Function that integrates the given nb_test_iters, generator and loss
    def tester(**kwargs):
        return performance_test(nb_test_iters, testgen, loss_fcn,  kmax, loss_type,**kwargs)
    
    errs = np.empty((len(opt_range), nb_test_iters))
    for it, par in enumerate(opt_range):
        if verbose:
            print("\rIteration", it+1, "of", len(opt_range), end="")
            
        # Dictionary from optimization parameter to testing function
        switcher = {
            "nit":       lambda: tester(nit=par, **kwargs),
            "tau":       lambda: tester(tau=par, **kwargs),
            "std_init":  lambda: tester(std_init=par, **kwargs),
            "n_samples": lambda: tester(n_samples=par, **kwargs),
            "epochs":    lambda: tester(epochs=par, **kwargs),
            "lr":        lambda: tester(lr=par, **kwargs)
        }
        fct = switcher.get(optimize)
        errs[it,:] = fct()
    return errs
    