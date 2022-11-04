import numpy as np
from numpy.polynomial.chebyshev import chebval
from matplotlib import pyplot as plt


def M_unnormalized(x,d,delta):
    inside = 1.0 + 2.0 * ((np.cos(x)-np.cos(delta))/(1+np.cos(delta)))
#     print(inside)
    c = np.zeros(d+1)
    c[-1] = 1.0
#    return chebval(inside,c) + 1.0
    return chebval(inside,c)
    
    
def M_fourier_coeffs(d,delta):
    M = 2*d + 1 # number of nodes to perform FFT
    x = (2*np.pi/M)*np.arange(M)
    y = M_unnormalized(x,d,delta)
    coeffs_raw = np.fft.fft(y,M)
#     print(coeffs_raw)
    return (1.0/M)*coeffs_raw


def reconstruct_from_fourier(x,fourier_coeffs):
    d = (fourier_coeffs.shape[0]-1)//2
    y = np.zeros(fourier_coeffs.shape)
    k = np.zeros(fourier_coeffs.shape)
    k[:d+1] = np.arange(d+1)
    k[d+1:] = np.arange(-d,0) # k = 0,1,...,d,-d,-d+1,...,-1
    exp_array = np.exp(1.0j*np.tensordot(k,x,axes=0))
    return np.matmul(fourier_coeffs,exp_array)
#     for k in range(d+1):
#         exp_array = np.exp(1.0j*k*x)
        
    
def M_fourier_coeffs_normalized(d,delta):
    coeffs = M_fourier_coeffs(d,delta)
    coeffs = coeffs/(coeffs[0]*2*np.pi)
    return coeffs
    

def H_fourier_coeffs(d):
    H_coeffs = np.zeros(2*d+1,dtype=np.complex128)
    H_coeffs[0] = 0.5
    H_coeffs[1:d+1] = -1.0j/np.arange(1,d+1)/np.pi*(np.arange(1,d+1)%2)
    H_coeffs[d+1:] = -1.0j/np.arange(-d,0)/np.pi*(np.arange(-d,0)%2)
    return H_coeffs
    

def F_fourier_coeffs(d,delta):
    H_coeffs = H_fourier_coeffs(d)
    M_coeffs = M_fourier_coeffs_normalized(d,delta)
    coeffs = 2*np.pi * H_coeffs * M_coeffs
#     coeffs = H_coeffs * M_coeffs
    return coeffs

    
    
def find_max_error(F_coeffs,delta,Nsample=0):
    if Nsample == 0:
        Nsample = F_coeffs.shape[0]*10
    x = np.pi*(2*np.random.rand(Nsample)-1)
    y = reconstruct_from_fourier(x,F_coeffs)
    y_target = x>0
    x_valid = (np.abs(x)>delta) * (np.abs(x+np.pi)>delta) * (np.abs(x-np.pi)>delta)
#     plt.scatter(x,x_valid,s=1)
#     plt.scatter(x,y_target,s=1)
#     plt.scatter(x,y,s=1)
    return np.max(np.abs(y-y_target)*x_valid)
    
    
def compute_total_evolution_time(F_coeffs):
    d = (F_coeffs.shape[0]-1)//2
    T = np.zeros(2*d+1)
    T[:d+1] = np.arange(d+1)
    T[d+1:] = -np.arange(-d,0)
#     plt.plot(T)
    return np.dot(T,np.abs(F_coeffs))
