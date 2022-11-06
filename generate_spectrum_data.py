import numpy as np 
import scipy.sparse
import scipy.linalg as la
import scipy.io
import argparse

from quspin.operators import hamiltonian

#from matplotlib import pyplot as plt

#import matplotlib
#font = {'size'   : 18}

#matplotlib.rc('font', **font)

import hubbard_1d
import fourier_filter
import generate_cdf


# --------- Parsing ----------------
parser = argparse.ArgumentParser()
parser.add_argument("L",help="Number of sites",type=int)
parser.add_argument("U",help="Interaction strength",type=float)
args = parser.parse_args()
#------------------------------------

#print(args.L)
#print(args.U)
L = args.L
U = args.U

J = 1.0
N_up = L//2 + L % 2 # number of fermions with spin up
N_down = L//2 # number of fermions with spin down
mu = 0.0

H = hubbard_1d.generate_ham(L,J,U,mu,N_up,N_down)
H_sparse = H.tocsr()
ew, ev = la.eigh(H_sparse.toarray())

H0 = hubbard_1d.generate_ham(L,J,0,mu,N_up,N_down)
H0_sparse = H0.tocsr()
ew0, ev0 = la.eigh(H0_sparse.toarray())

inner_prod = np.dot(ev0[:,0],ev)
popu = np.abs(inner_prod)**2



# Read the last experiment number
with open("./experiment_data/log","r") as filelog:
#    print(len(filelog))
    line_count = 0
    for line in filelog:
        line_split = line.split()
        expr_num = int(line_split[0])
        line_count += 1
filelog.closed

if line_count > 0:
    expr_num += 1
else:
    expr_num = 1


# Write the log
with open("./experiment_data/log","a") as filelog:
    line = "{expr_num:0>3d} L={L} U={U}\n"
    filelog.write(line.format(expr_num=expr_num,L=L,U=U))
filelog.closed


# Store the spectrum info
outfile_popu = open("./experiment_data/popu_{expr_num:0>3d}.txt".format(expr_num=expr_num),"w")
np.savetxt(outfile_popu,popu)
outfile_popu.close()

outfile_eigval = open("./experiment_data/eigval_{expr_num:0>3d}.txt".format(expr_num=expr_num),"w")
np.savetxt(outfile_eigval,ew)
outfile_eigval.close()


print("Experiment number:",expr_num)
print("L=",L,"U=",U)
print("Initial overlap:",popu[0])
print("g.s. energy:",ew[0])
