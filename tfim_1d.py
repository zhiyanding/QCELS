"""Generate the Hamiltonian with the 1D TFIM model.

The implementation uses quspin. The periodic boundary condition is enforced. 

Last revision: 11/5/2022
"""

import numpy as np 
import scipy.sparse
import scipy.linalg as la
import scipy.io

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel


def generate_ham(L,J,g,verbose=0):

    basis = spin_basis_1d(L=L)
    if verbose > 0:
        print(basis)
    
    # define site-coupling lists
    h_field=[[-g,i] for i in range(L)]
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

    H = generate_ham(L, J=1.0, g=4.0)

    # initial state can be the eigenstate of H_x
    eigpair = H.eigsh(k=1,which="SA")
    E_GS = eigpair[0][0]
    psi_GS = eigpair[1][:,0]
    print('E_GS = ', E_GS)

    fname='Ising_L_'+str(L)+'g_'+str(int(g))+'.mat'
    print('saving results to ', fname)
    H_mat = np.array(H.todense())
    matdic = {'H': H_mat, 'E0': E_GS[0], 'gap': E_GS[1]-E_GS[0], 'psi0': psi_GS[:,0]}
    scipy.io.savemat(fname, matdic)


    # fname='tfim_L_'+str(L)+'.npz'
    # scipy.sparse.save_npz('tfim_H_zz.npz', H_zz.tocsr())
    # scipy.sparse.save_npz('tfim_H_x.npz', H_zz.tocsr())
    # np.savez(fname, H=H.todense(), psi_GS=psi_GS)
    # npzfile = np.load(fname)
    # print(npzfile['H_x'])
    # print(npzfile['H_zz'])
    # print(npzfile['psi_GS'])
