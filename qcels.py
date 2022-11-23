""" Main routines for QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the ground-state energy with reduced circuit depth. 

Last revision: 11/22/2022
"""

import scipy.io as sio
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erf
import cmath
import fejer_kernel
import fourier_filter
import generate_cdf


def generate_QPE_distribution(spectrum,population,J):
    N = len(spectrum)
    dist = np.zeros(J)
    j_arr = 2*np.pi*np.arange(J)/J - np.pi
    for k in range(N):
        dist += population[k] * fejer_kernel.eval_Fejer_kernel(J,j_arr-spectrum[k])/J
    return dist

def get_estimated_ground_energy_rough(d,delta,spectrum,population,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)

    compute_prob_X = lambda T: generate_cdf.compute_prob_X_(T,spectrum,population)
    compute_prob_Y = lambda T: generate_cdf.compute_prob_Y_(T,spectrum,population)


    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch) #Generate sample to calculate CDF

    total_evolution_time = np.sum(np.abs(J_arr))
    average_evolution_time = total_evolution_time/(Nsample*Nbatch)
    maxi_evolution_time=max(np.abs(J_arr[0,:]))

    Nx = 10
    Lx = np.pi/3
    ground_energy_estimate = 0.0
    count = 0
    #---"binary" search
    while Lx > delta:
        x = (2*np.arange(Nx)/Nx-1)*Lx +  ground_energy_estimate
        y_avg = generate_cdf.compute_cdf_from_XY(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs)#Calculate the value of CDF
        indicator_list = y_avg > (population[0]/2.05)
        ix = np.nonzero(indicator_list)[0][0]
        ground_energy_estimate = x[ix]
        Lx = Lx/2
        count += 1
    
    return ground_energy_estimate, count*total_evolution_time, maxi_evolution_time

def generate_filtered_Z_est(spectrum,population,t,x,d,delta,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)
    compute_prob_X = lambda t_: generate_cdf.compute_prob_X_(t_,spectrum,population)
    compute_prob_Y = lambda t_: generate_cdf.compute_prob_Y_(t_,spectrum,population)
    #Calculate <\psi|F(H)\exp(-itH)|\psi>
    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY_QCELS(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch,t) #Generate samples using Hadmard test
    y_avg = generate_cdf.compute_cdf_from_XY_QCELS(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs) 
    total_time = np.sum(np.abs(J_arr))+t*Nsample*Nbatch
    max_time= max(np.abs(J_arr[0,:]))+t
    return y_avg, total_time, max_time


def generate_Z_est(spectrum,population,t,Nsample):
    Re=0
    Im=0
    z=np.dot(population,np.exp(-1j*spectrum*t))
    Re_true=(1+z.real)/2
    Im_true=(1+z.imag)/2
    #Simulate Hadmard test
    for nt in range(Nsample):
        if np.random.uniform(0,1)<Re_true:
           Re+=1
    for nt2 in range(Nsample):
        if np.random.uniform(0,1)<Im_true:
           Im+=1
    Z_est = complex(2*Re/Nsample-1,2*Im/Nsample-1)
    max_time = t
    total_time = t * Nsample
    return Z_est, total_time, max_time 
       

def generate_spectrum_population(eigenenergies, population, p):

    p = np.array(p)
    spectrum = eigenenergies * 0.25*np.pi/np.max(np.abs(eigenenergies))#normalize the spectrum
    q = population
    print(p)
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = 'complex_')
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt(ts, Z_est, x0, bounds = None, method = 'SLSQP'):

    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res


def qcels_largeoverlap(spectrum, population, T, NT, Nsample, lambda_prior):
    """Multi-level QCELS for systems a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy with large initial overlap

    Args: eigenvalues of Hamiltonian: spectrum; overlaps between initial state and eigenvectors: population; 
    the circuit depth: T; number of data pairs: NT; number of samples: Nsample; initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0; maximal evolution time T_{max}; total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.

    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex_')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_Z_est(
                spectrum,population,ts[i],Nsample) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est, x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
    #Iteration
    for n_QCELS in range(N_level):
        Z_est=np.zeros(NT,dtype = 'complex_')
        tau=T/NT/(2**(N_level-n_QCELS-1)) #generate a sequence of \tau_j
        ts=tau*np.arange(NT)
        for i in range(NT):
            Z_est[i], total_time, max_time=generate_Z_est(
                    spectrum,population,ts[i],Nsample) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
            total_time_all += total_time
            max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
        res = qcels_opt(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 

    return res, total_time_all, max_time_all


def qcels_smalloverlap(spectrum, population, T, NT, d, rel_gap, err_tol_rough, Nsample_rough, Nsample):
    """Multi-level QCELS with a filtered data set for systems with a small initial overlap.

    Description: The codes of using Multi-level QCELS and eigenvalue filter to estimate the ground state energy of 
    systems that have small initial overlap

    Args: eigenvalues of Hamiltonian: spectrum; overlaps between initial state and eigenvectors: population; 
    the circuit depth: T; number of data pairs: NT; number of samples for constructing eigenvalue filter: Nsample_rough; 
    number of samples for generating data set: Nsample; initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0; maximal evolution time T_{max}; total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.

    lambda_prior, total_time_prior, max_time_prior = get_estimated_ground_energy_rough(
            d,err_tol_rough,spectrum,population,Nsample_rough,Nbatch=1) #Get the rough estimation of the ground state energy
    x = lambda_prior + rel_gap/2 #center of the eigenvalue filter
    total_time_all += total_time_prior
    max_time_all = max(max_time_all, max_time_prior)
    
    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex_')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_filtered_Z_est(
                spectrum,population,ts[i],x,d,err_tol_rough,Nsample_rough,Nbatch=1)#Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est, x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau)
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau)
    #Iteration
    for n_QCELS in range(N_level):
        Z_est=np.zeros(NT,dtype = 'complex_')
        tau=T/NT/(2**(N_level-n_QCELS-1))
        ts=tau*np.arange(NT)
        for i in range(NT):
            Z_est[i], total_time, max_time=generate_filtered_Z_est(
                    spectrum,population,ts[i],x,d,err_tol_rough,Nsample,Nbatch=1)#Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
            total_time_all += total_time
            max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max))
        res = qcels_opt(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau)
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau)

    return ground_energy_estimate_QCELS, total_time_all, max_time_all


if __name__ == "__main__":
    import scipy.io as sio
    import numpy as np
    from copy import deepcopy
    from scipy.optimize import minimize
    from matplotlib import pyplot as plt
    from scipy.special import erf
    from mpl_toolkits.mplot3d import Axes3D
    import cmath
    import matplotlib
    import hubbard_1d
    import quspin
    import fejer_kernel
    import fourier_filter
    import generate_cdf
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10

    num_sites = 4
    J = 1.0
    U = 10.0
    U0 = 0.0
    mu = 0.0
    N_up = num_sites // 2
    N_down = num_sites - N_up
    
    num_eigenstates_max = 100
    
    ham0 = hubbard_1d.generate_ham(num_sites, J, U0, mu, N_up, N_down)
    ground_state_0 = ham0.eigsh(k=1,which="SA")[1][:,0]
    
    ham = hubbard_1d.generate_ham(num_sites, J, U, mu, N_up, N_down)
    if( num_eigenstates_max > ham.shape[0] // 2):
        eigenenergies, eigenstates = ham.eigh()
    else:
        eigenenergies, eigenstates = ham.eigsh(k=num_eigenstates_max,which="SA")
    ground_state = eigenstates[:,0]
    
    population_raw = np.abs(np.dot(eigenstates.conj().T, ground_state_0))**2
    
    plt.plot(eigenenergies,population_raw,'b-o');plt.show()

    print("large overlap using multi-level QCELS")
    p0_array = np.array([0.8]) 
    T0 = 100
    N_test_QCELS = 10  #number of QCELS test
    N_QPE = 10  #number of QPE test
    T_list_QCELS = 11+T0/2*(np.arange(N_test_QCELS))
    T_list_QPE = 11+T0/2*(np.arange(N_QPE))
    err_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    err_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    cost_list_avg_QCELS = np.zeros((len(p0_array),len(T_list_QCELS)))
    cost_list_avg_QPE = np.zeros((len(p0_array),len(T_list_QPE)))
    rate_success_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    rate_success_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    max_T_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    Navg = 3 #number of trying
    err_thres_hold=0.01
    err_thres_hold_QPE=0.01
    #-----------------------------    
    for a1 in range(len(p0_array)):
        p0=p0_array[a1]
        n_success_QCELS= np.zeros(len(T_list_QCELS))
        n_success_QPE= np.zeros(len(T_list_QPE))
        for n_test in range(Navg):
            print("For p0=",p0,"For N_test=",n_test+1)
            spectrum, population = generate_spectrum_population(eigenenergies, 
                    population_raw, [p0])
            #------------------QCELS-----------------
            Nsample=50
            for ix in range(len(T_list_QCELS)):
                T = T_list_QCELS[ix]
                NT = 5
                lambda_prior = spectrum[0]
                ground_energy_estimate_QCELS, cost_list_avg_QCELS[a1,ix], max_T_QCELS[a1,ix] = \
                        qcels_largeoverlap(spectrum, population, T, NT,
                                Nsample, lambda_prior)
                err_this_run_QCELS = np.abs(ground_energy_estimate_QCELS - spectrum[0])
                err_QCELS[a1,ix] = err_QCELS[a1,ix]+np.abs(err_this_run_QCELS)
                if np.abs(err_this_run_QCELS)<err_thres_hold:
                    n_success_QCELS[ix]+=1
           
           # ----------------- QPE -----------------------
            N_try_QPE=int(15*np.ceil(1.0/p0))
            for ix in range(len(T_list_QPE)):
                T = int(T_list_QPE[ix])
                discrete_energies = 2*np.pi*np.arange(2*T)/(2*T) - np.pi 
                dist = generate_QPE_distribution(spectrum,population,2*T)
                samp = generate_cdf.draw_with_prob(dist,N_try_QPE)
                j_min = samp.min()
                ground_energy_estimate_QPE = discrete_energies[j_min]
                err_this_run_QPE = np.abs(ground_energy_estimate_QPE-spectrum[0])
                err_QPE[a1,ix] = err_QPE[a1,ix]+np.abs(err_this_run_QPE)
                if np.abs(err_this_run_QPE)<err_thres_hold_QPE:
                    n_success_QPE[ix]+=1
                cost_list_avg_QPE[a1,ix] = T*N_try_QPE
        rate_success_QCELS[a1,:] = n_success_QCELS[:]/Navg
        rate_success_QPE[a1,:] = n_success_QPE[:]/Navg
        err_QCELS[a1,:] = err_QCELS[a1,:]/Navg
        err_QPE[a1,:] = err_QPE[a1,:]/Navg
        cost_list_avg_QCELS[a1,:]=cost_list_avg_QCELS[a1,:]/Navg


    print('QCELS')
    print(rate_success_QCELS)
    print('QPE')
    print(rate_success_QPE)    
    plt.figure(figsize=(12,10))
    plt.plot(T_list_QCELS,err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.8")
    plt.plot(T_list_QPE,err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.8")
    plt.xlabel("$T_{max}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.figure(figsize=(12,10))
    plt.plot(cost_list_avg_QCELS[0,:],err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.8")
    plt.plot(cost_list_avg_QPE[0,:],err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.8")
    plt.xlabel("$T_{total}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35) 
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=25)
    plt.show()


    #
    print("small overlap using multi-level QCELS with filtered data")
    p0_array=np.array([0.1]) #population
    p1_array=np.array([0.025]) #population
    #relative population=0.8
    T0 = 2000
    N_test_QCELS = 10  #number of QCELS test
    N_QPE = 10  #number of QPE test
    T_list_QCELS = 150+T0/4*(np.arange(N_test_QCELS))
    T_list_QPE = 150+T0*7.5*(np.arange(N_QPE))
    err_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    err_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    cost_list_avg_QCELS = np.zeros((len(p0_array),len(T_list_QCELS)))
    cost_list_avg_QPE = np.zeros((len(p0_array),len(T_list_QPE)))
    rate_success_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    rate_success_QPE=np.zeros((len(p0_array),len(T_list_QPE)))
    max_T_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    Navg = 3 #number of trying
    err_thres_hold=0.01
    err_thres_hold_QPE=0.01
    #-----------------------------    
    for a1 in range(len(p0_array)):
        p0=p0_array[a1]
        p1=p1_array[a1]
        n_success_QCELS= np.zeros(len(T_list_QCELS))
        n_success_QPE= np.zeros(len(T_list_QPE))
        for n_test in range(Navg):
            print("For p0=",p0," p1=", p1, "For N_test=",n_test+1)
            spectrum, population = generate_spectrum_population(eigenenergies, 
                    population_raw, [p0,p1])
            #------------------QCELS-----------------
            # heuristic estimate of relative gap
            rel_gap_idx = np.where(population>p0/2)[0][1] 
            rel_gap = spectrum[rel_gap_idx]-spectrum[0]
            d=int(20/rel_gap)
            print("d=", d, "rel_gap = ", rel_gap)
            Nsample_rough=int(500/p0**2*np.log(d))
            Nsample=int(30/p0**2*np.log(d))
            for ix in range(len(T_list_QCELS)):
                T = T_list_QCELS[ix]
                NT = 5
                err_tol_rough=rel_gap/4
                ground_energy_estimate_QCELS, cost_list_avg_QCELS[a1,ix], max_T_QCELS[a1,ix] = \
                        qcels_smalloverlap(spectrum, population, T, NT, d, rel_gap, \
                                        err_tol_rough, Nsample_rough, Nsample)
                err_this_run_QCELS = np.abs(ground_energy_estimate_QCELS - spectrum[0])
                err_QCELS[a1,ix] = err_QCELS[a1,ix]+np.abs(err_this_run_QCELS)
                if np.abs(err_this_run_QCELS)<err_thres_hold:
                    n_success_QCELS[ix]+=1
           
           # ----------------- QPE -----------------------
            N_try_QPE=int(15*np.ceil(1.0/p0))
            for ix in range(len(T_list_QPE)):
                T = int(T_list_QPE[ix])
                discrete_energies = 2*np.pi*np.arange(2*T)/(2*T) - np.pi
                dist = generate_QPE_distribution(spectrum,population,2*T)
                samp = generate_cdf.draw_with_prob(dist,N_try_QPE)
                j_min = samp.min()
                ground_energy_estimate_QPE = discrete_energies[j_min]
                err_this_run_QPE = np.abs(ground_energy_estimate_QPE-spectrum[0])
                err_QPE[a1,ix] = err_QPE[a1,ix]+np.abs(err_this_run_QPE)
                if np.abs(err_this_run_QPE)<err_thres_hold_QPE:
                    n_success_QPE[ix]+=1
                cost_list_avg_QPE[a1,ix] = T*N_try_QPE
        rate_success_QCELS[a1,:] = n_success_QCELS[:]/Navg
        rate_success_QPE[a1,:] = n_success_QPE[:]/Navg
        err_QCELS[a1,:] = err_QCELS[a1,:]/Navg
        err_QPE[a1,:] = err_QPE[a1,:]/Navg
        cost_list_avg_QCELS[a1,:]=cost_list_avg_QCELS[a1,:]/Navg


    print('QCELS')
    print(rate_success_QCELS)
    print('QPE')
    print(rate_success_QPE)    
    plt.figure(figsize=(12,10))
    plt.plot(T_list_QCELS,err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.4,p_r=0.8")
    plt.plot(T_list_QPE,err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.4, p_r=0.8")
    plt.xlabel("$T_{max}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.figure(figsize=(12,10))
    plt.plot(cost_list_avg_QCELS[0,:],err_QCELS[0,:],linestyle="-.",marker="o",label="error of QCELS p_0=0.4, p_r=0.8")
    plt.plot(cost_list_avg_QPE[0,:],err_QPE[0,:],linestyle="-.",marker="*",label="error of QPE p_0=0.4, p_r=0.8")
    plt.xlabel("$T_{total}$",fontsize=35)
    plt.ylabel("error($\epsilon$)",fontsize=35) 
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=25)
    plt.show()
