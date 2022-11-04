import numpy as np
#from numpy.polynomial.chebyshev import chebval
from matplotlib import pyplot as plt

import fourier_filter


#def compute_prob_X_(T,epsilon_list,popu_list):
#    cos_array = np.cos(-np.tensordot(epsilon_list,T,axes=0))
#    return 0.5*np.matmul(popu_list,cos_array)+0.5


#def compute_prob_Y_(T,epsilon_list,popu_list):
#    cos_array = np.sin(-np.tensordot(epsilon_list,T,axes=0))
#    return 0.5*np.matmul(popu_list,cos_array)+0.5
    
    
# --------------- with decay -----------------------
#DECAY = 1.0

def compute_prob_X_(T,epsilon_list,popu_list,decay_rate=0.0):
    cos_array = np.cos(-np.tensordot(epsilon_list,T,axes=0))
    p = np.exp(-decay_rate*np.abs(T))
    return p*0.5*np.matmul(popu_list,cos_array) + 0.5


def compute_prob_Y_(T,epsilon_list,popu_list,decay_rate=0.0):
    sin_array = np.sin(-np.tensordot(epsilon_list,T,axes=0))
    p = np.exp(-decay_rate*np.abs(T))
    return p*0.5*np.matmul(popu_list,sin_array) + 0.5
# -------------------------------------------------------


def draw_with_prob(measure,Nsample):
    L = measure.shape[0]
    cdf_measure = np.cumsum(measure)
    normal_fac = cdf_measure[-1]
    U = np.random.rand(Nsample) * normal_fac
    j = np.searchsorted(cdf_measure,U)
    return j
    
    
def generate_cdf(x, compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch):

    d = (F_coeffs.shape[0]-1)//2
    T_list = np.zeros(2*d + 1)
    T_list[:d+1] = np.arange(d+1)
    T_list[d+1:] = np.arange(-d,0)
    angles = np.angle(F_coeffs)
    phase_fac = np.exp(1.0j*angles)
    F_normal_fac = np.sum(np.abs(F_coeffs))
    
    Nx = x.shape[0]
    y_sum = np.zeros(Nx,dtype=np.complex128)
    for nbatch in range(Nbatch):

        J_list = draw_with_prob(np.abs(F_coeffs),Nsample)
        p_X = compute_prob_X(T_list[J_list])
        p_Y = compute_prob_Y(T_list[J_list])
        # print(outcome)
        # print(np.imag(F_coeffs))
        # reconstructed_signal = 
        U = np.random.rand(Nsample)
        outcome_X = 2*(U<p_X)-1
        U = np.random.rand(Nsample)
        outcome_Y = 2*(U<p_Y)-1

    
        exp_array = np.exp(1.0j*np.tensordot(T_list[J_list],x,axes=0))
        y = np.matmul((outcome_X + 1.0j*outcome_Y)*phase_fac[J_list],exp_array)/Nsample*F_normal_fac

        y_sum += y
    
    y_avg = y_sum/Nbatch
    return y_avg
    
    
def generate_cdf_median(x, compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch, Nbin):

    Nx = x.shape[0]
    y_arr = np.zeros([Nbin,Nx],dtype=np.complex128)
    for ixbin in range(Nbin):
        y_arr[ixbin,:] = generate_cdf(x, compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch)
    y_median = np.median(y_arr,0)
    return y_median
    
    
def sample_XY(compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch):
    
    d = (F_coeffs.shape[0]-1)//2
    T_list = np.zeros(2*d + 1)
    T_list[:d+1] = np.arange(d+1)
    T_list[d+1:] = np.arange(-d,0)
    angles = np.angle(F_coeffs)
    phase_fac = np.exp(1.0j*angles)
    F_normal_fac = np.sum(np.abs(F_coeffs))
    
    outcome_X_arr = np.zeros([Nbatch,Nsample],dtype=np.complex128)
    outcome_Y_arr = np.zeros([Nbatch,Nsample],dtype=np.complex128)
    J_arr = np.zeros([Nbatch,Nsample],dtype=np.int)
    for nbatch in range(Nbatch):
        J_list = draw_with_prob(np.abs(F_coeffs),Nsample)
        J_arr[nbatch,:] = J_list
        p_X = compute_prob_X(T_list[J_list])
        p_Y = compute_prob_Y(T_list[J_list])
        U = np.random.rand(Nsample)
        outcome_X_arr[nbatch,:] = 2*(U<p_X)-1
        U = np.random.rand(Nsample)
        outcome_Y_arr[nbatch,:] = 2*(U<p_Y)-1
    
    return outcome_X_arr, outcome_Y_arr, J_arr
    
    
def sample_XY_median(compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch, Nbin):

    outcome_X_arr_cube = np.zeros([Nbin,Nbatch,Nsample],dtype=np.complex128)
    outcome_Y_arr_cube = np.zeros([Nbin,Nbatch,Nsample],dtype=np.complex128)
    J_arr_cube = np.zeros([Nbin,Nbatch,Nsample],dtype=np.int)
    
    for ixbin in range(Nbin):
        outcome_X_arr, outcome_Y_arr, J_arr = sample_XY(compute_prob_X, 
                        compute_prob_Y, F_coeffs, Nsample, Nbatch)
        outcome_X_arr_cube[ixbin,:,:] = outcome_X_arr
        outcome_Y_arr_cube[ixbin,:,:] = outcome_Y_arr
        J_arr_cube[ixbin,:,:] = J_arr
    
    return outcome_X_arr_cube, outcome_Y_arr_cube, J_arr_cube
    
    
def compute_cdf_from_XY(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs):

    d = (F_coeffs.shape[0]-1)//2
    T_list = np.zeros(2*d + 1)
    T_list[:d+1] = np.arange(d+1)
    T_list[d+1:] = np.arange(-d,0)
    angles = np.angle(F_coeffs)
    phase_fac = np.exp(1.0j*angles)
    F_normal_fac = np.sum(np.abs(F_coeffs))
    
    Nbatch, Nsample = J_arr.shape
    Nx = x.shape[0]
    y_sum = np.zeros(Nx,dtype=np.complex128)
    for nbatch in range(Nbatch):
        J_list = J_arr[nbatch,:]
        exp_array = np.exp(1.0j*np.tensordot(T_list[J_list],x,axes=0))
        y = np.matmul((outcome_X_arr[nbatch,:] + 1.0j*outcome_Y_arr[nbatch,:])*phase_fac[J_list],exp_array)/Nsample*F_normal_fac

        y_sum += y
    
    y_avg = y_sum/Nbatch
    return y_avg


    
def compute_cdf_from_XY_median(x, outcome_X_arr_cube, outcome_Y_arr_cube, J_arr_cube, F_coeffs):

    Nbin, Nbatch, Nsample = J_arr_cube.shape
    
    Nx = x.shape[0]
    y_arr = np.zeros([Nbin,Nx],dtype=np.complex128)
    for ixbin in range(Nbin):
        y_arr[ixbin,:] = compute_cdf_from_XY(x, 
            outcome_X_arr_cube[ixbin,:,:], outcome_Y_arr_cube[ixbin,:,:], 
            J_arr_cube[ixbin,:,:], F_coeffs)
    y_median = np.median(y_arr,0)
    return y_median

def sample_XY_QCELS(compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch,t):
    
    d = (F_coeffs.shape[0]-1)//2
    F_coeffs_new=np.zeros(len(F_coeffs),dtype=np.complex128)
    a=-F_coeffs[d+1:]
    F_coeffs_new[1:d+1]=a[::-1]
    a=-F_coeffs[1:d+1]
    F_coeffs_new[d+1:]=a[::-1]
    F_coeffs_new[0]=1-F_coeffs[0]
    T_list = np.zeros(2*d + 1)
    T_list[:d+1] = np.arange(d+1)
    T_list[d+1:] = np.arange(-d,0)
    angles_new = np.angle(F_coeffs_new)
    phase_fac_new = np.exp(1.0j*angles_new)
    F_normal_fac_new = np.sum(np.abs(F_coeffs_new))
    
    outcome_X_arr = np.zeros([Nbatch,Nsample],dtype=np.complex128)
    outcome_Y_arr = np.zeros([Nbatch,Nsample],dtype=np.complex128)
    J_arr = np.zeros([Nbatch,Nsample],dtype=np.int)
    for nbatch in range(Nbatch):
        J_list = draw_with_prob(np.abs(F_coeffs_new),Nsample)
        J_arr[nbatch,:] = J_list
        p_X = compute_prob_X(T_list[J_list]+t)# - to match the - in compute_prob_X 
        p_Y = compute_prob_Y(T_list[J_list]+t)
        U = np.random.rand(Nsample)
        outcome_X_arr[nbatch,:] = 2*(U<p_X)-1
        U = np.random.rand(Nsample)
        outcome_Y_arr[nbatch,:] = 2*(U<p_Y)-1
    
    return outcome_X_arr, outcome_Y_arr, J_arr

def compute_cdf_from_XY_QCELS(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs):
#data generator
    d = (F_coeffs.shape[0]-1)//2
    F_coeffs_new=np.zeros(len(F_coeffs),dtype=np.complex128)
    a=np.zeros(d,dtype=np.complex128)
    a=-F_coeffs[d+1:]
    F_coeffs_new[1:d+1]=a[::-1]
    a=-F_coeffs[1:d+1]
    F_coeffs_new[d+1:]=a[::-1]
    F_coeffs_new[0]=1-F_coeffs[0]
    T_list = np.zeros(2*d + 1)
    T_list[:d+1] = np.arange(d+1)
    T_list[d+1:] = np.arange(-d,0)
    angles_new = np.angle(F_coeffs_new)
    phase_fac_new = np.exp(1.0j*angles_new)
    F_normal_fac_new = np.sum(np.abs(F_coeffs_new))
    Nbatch, Nsample = J_arr.shape
    y_sum = 0
    for nbatch in range(Nbatch):
        J_list = J_arr[nbatch,:]
        exp_array = np.exp(1.0j*np.tensordot(T_list[J_list],x,axes=0))
        y = np.matmul((outcome_X_arr[nbatch,:] + 1.0j*outcome_Y_arr[nbatch,:])*phase_fac_new[J_list],exp_array)/Nsample*F_normal_fac_new
        y_sum += y
    
    y_avg = y_sum/Nbatch
    return y_avg
    
    
if __name__ == "__main__":

    d = 20000
    delta = 0.001
#    Nsample = 1000
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)
#    x = np.random.randn(Nsample) % np.pi
#    y = fourier_filter.reconstruct_from_fourier(x,F_coeffs)
#    plt.scatter(x,np.real(y),s=1)

    epsilon_list = np.asarray([-0.2*np.pi,0.1*np.pi,0.15*np.pi])
    popu_list = np.asarray([0.6,0.3,0.1])
    compute_prob_X = lambda T: compute_prob_X_(T,epsilon_list,popu_list)
    compute_prob_Y = lambda T: compute_prob_Y_(T,epsilon_list,popu_list)
    
    Nsample = 400
    Nbatch = 10
    Nx = 1000
#    x = (2*np.arange(Nx)/Nx-1)*np.pi
    x = (2*np.arange(Nx)/Nx-1)*np.pi/3
#    y_avg = generate_cdf(x, compute_prob_X, compute_prob_Y, F_coeffs, Nsample, Nbatch)
    outcome_X_arr, outcome_Y_arr, J_arr = sample_XY(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch)
    y_avg = compute_cdf_from_XY(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs)
    
    plt.plot(x,np.real(y_avg))
    plt.plot(x,np.imag(y_avg))
    plt.show()

    
