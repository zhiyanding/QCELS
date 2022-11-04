import numpy as np

def eval_Fejer_kernel(J,x):
    x_avoid = np.abs(x % (2*np.pi))<1e-8
    numer = np.sin(0.5*J*x)**2
    denom = np.sin(0.5*x)**2
    denom += x_avoid
    ret = numer / denom
    ret = (1-x_avoid)*ret + (J**2)*x_avoid
    return ret/J
