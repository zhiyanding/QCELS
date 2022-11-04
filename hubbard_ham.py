from qutip import *
import numpy as np


create_op = Qobj([[0.0,0],[1,0]])
annihilate_op = create_op.dag()
num_op = create_op*annihilate_op

# Creation and annihilation operators using Jordan-Wigner
# Sites and spins are indexed by 0up, 0down, 1up, 1down, ...
def create_op_JW(site,spin,num_sites):
    # site:      the index of the site on which this operator is applied
    # spin:      the spin
    # num_sites: total number of sites in the system
    # output:    the creation operator with JW transformation as a Qobj
    op_list = [sigmaz() for i in range(2*site)]
    if spin == 0: # spin up
        op_list.append(create_op)
        op_list.append(identity(2))
    elif spin == 1: # spin down
        op_list.append(sigmaz())
        op_list.append(create_op)
    else:
        raise ValueError("Wrong spin type")
    for i in range(2*(num_sites-site-1)):
        op_list.append(identity(2))
#     print(op_list)
    return tensor(op_list)


def num_op_JW(site,spin,num_sites):
    # site:      the index of the site on which this operator is applied
    # spin:      the spin
    # num_sites: total number of sites in the system
    # output:    the number operator n=a*a with JW transformation as a Qobj
    op_list = [identity(2) for i in range(2*site)]
    if spin == 0: # spin up
        op_list.append(num_op)
        op_list.append(identity(2))
    elif spin == 1: # spin down
        op_list.append(identity(2))
        op_list.append(num_op)
    else:
        raise ValueError("Wrong spin type")
    for i in range(2*(num_sites-site-1)):
        op_list.append(identity(2))
#     print(op_list)
    return tensor(op_list)


def hubbard_ham_1d(num_sites,U,chem_pot=0.0): 
    # create the 1D hubbard hamiltonian
    # of the form -sum_{i,sigma} a_{i,sigma}* a_{i+1,sigma} + h.c. 
    #               + U sum_i (n_{i,up}-1/2)(n_{i,down}-1/2) - mu N
    # where N is the total particle number operator
    # when mu = 0 the system is at half-filling
    # 
    # num_sites:    total number of sites in the system
    # U:            interaction strength
    # chem_pot:     the chemical potential mu
    # output:       the Hamiltonian as a Qobj
    H = qzero([2 for j in range(2*num_sites)])
#     print(H.dims)
    for j in range(num_sites-1):
        H += -1.0 * create_op_JW(j,0,num_sites).dag() * create_op_JW(j+1,0,num_sites) # jump spin up
        H += -1.0 * create_op_JW(j,1,num_sites).dag() * create_op_JW(j+1,1,num_sites) # jump spin down
    H += H.dag() # add the Hermitian conj
    for j in range(num_sites):
        H += U * (num_op_JW(j,0,num_sites) * num_op_JW(j,1,num_sites)) - (0.5*U+chem_pot)*(num_op_JW(j,0,num_sites)
                 + num_op_JW(j,1,num_sites)) + 0.25*U*identity([2 for j in range(2*num_sites)]) # on-site interaction and chemical potential
    return H


def num_op_total(num_sites):
    # num_sites:    total number of sites in the system
    # output:       the total particle number operator as a Qobj
    Num_op = qzero([2 for j in range(2*num_sites)])
    for j in range(num_sites):
        Num_op += num_op_JW(j,0,num_sites) + num_op_JW(j,1,num_sites)
    return Num_op


def num_op_up_total(num_sites):
    # num_sites:    total number of sites in the system
    # output:       the total particle number operator as a Qobj
    Num_op = qzero([2 for j in range(2*num_sites)])
    for j in range(num_sites):
        Num_op += num_op_JW(j,0,num_sites)
    return Num_op


def num_op_down_total(num_sites):
    # num_sites:    total number of sites in the system
    # output:       the total particle number operator as a Qobj
    Num_op = qzero([2 for j in range(2*num_sites)])
    for j in range(num_sites):
        Num_op += num_op_JW(j,1,num_sites)
    return Num_op
