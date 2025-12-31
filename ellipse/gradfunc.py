import numpy as np

def gradfunc(state, S):
    """
    Computes the gradient for the Gaussian.
    """
    
    return -S@state