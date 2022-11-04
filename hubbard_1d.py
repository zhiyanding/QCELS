"""Generate the Hamiltonian for 1D spinful Hubbard.

The implementation uses quspin. The periodic boundary condition is not
enforced. The interaction term is z|z, where z=c^{\dag}c-1/2

Last revision: 11/3/2022
"""
import numpy as np 
import scipy.sparse
import scipy.linalg as la
import scipy.io

from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from quspin.tools.evolution import expm_multiply_parallel


def generate_ham(L,J,U,mu,N_up,N_down, verbose=0):

    basis = spinful_fermion_basis_1d(L,Nf=(N_up,N_down))
    if verbose > 0:
        print(basis)
    
    # define site-coupling lists
    hop_right=[[-J,i,i+1] for i in range(L-1)]
    hop_left= [[+J,i,i+1] for i in range(L-1)]
    pot=[[-mu,i] for i in range(L)] # -\mu \sum_j n_{j \sigma}
    interact=[[U,i,i] for i in range(L)] # U/2 \sum_j n_{j,up} n_{j,down}
    # define static and dynamic lists
    static=[
            ['+-|',hop_left],  # up hops left
            ['-+|',hop_right], # up hops right
            ['|+-',hop_left],  # down hops left
            ['|-+',hop_right], # down hops right
            ['n|',pot],        # up on-site potention
            ['|n',pot],        # down on-site potention
            ['z|z',interact]   # up-down interaction with z=c^{\dag} c-1/2
           ]
    dynamic=[]
    # build Hamiltonian
    if verbose == 0:
        no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
    else: 
        H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

    return H


if __name__ == "__main__":

    ##### define model parameters #####
    L=4 # system size
    J=1.0 # hopping
    U=10.0 # interaction
    mu=0.1 # chemical potential
    
    # define boson basis with 3 states per site L bosons in the lattice
    N_up = L//2 + L % 2 # number of fermions with spin up
    N_down = L//2 # number of fermions with spin down
    
    H = generate_ham(L,J,U,mu,N_up,N_down)

    eigpair = H.eigsh(k=10,which="SA")
    E_GS = eigpair[0][0]
    psi_GS = eigpair[1][:,0]
    print('E_GS = ', E_GS)
