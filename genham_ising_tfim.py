# Generate the Hamiltonian corresponding to the TFIM
#
# Lin Lin
# Last revision: 08/01/2021

import numpy as np 
import scipy.sparse
import scipy.linalg as la
import scipy.io

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel


def generate_ham(L,J,h,verbose=0):

    basis = spin_basis_1d(L=L)
    if verbose > 0:
        print(basis)
    
    # define site-coupling lists
    h_field=[[-h,i] for i in range(L)]
    J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC
    static =[["zz",J_zz],["x",h_field]] # static part of H
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
    L=8 # system size
    
    H_zz = generate_ham(L, J=1.0, h=0.0)
    H_x  = generate_ham(L, J=0.0, h=1.0)

    # initial state can be the eigenstate of H_x
    eigpair = H_x.eigsh(k=1,which="SA")
    E_GS = eigpair[0][0]
    psi_GS = eigpair[1][:,0]
    print('E_GS = ', E_GS)

    fname='tfim_L_'+str(L)+'.npz'
    # scipy.sparse.save_npz('tfim_H_zz.npz', H_zz.tocsr())
    # scipy.sparse.save_npz('tfim_H_x.npz', H_zz.tocsr())
    np.savez(fname, H_zz=H_zz.todense(), 
            H_x=H_x.todense(), psi_GS=psi_GS)

    # load data
    npzfile = np.load(fname)
    print(npzfile['H_x'])
    print(npzfile['H_zz'])
    print(npzfile['psi_GS'])
