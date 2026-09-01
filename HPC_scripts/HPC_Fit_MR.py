# Fit mouse retina connectome
import numpy as np
import pickle
import sys
sys.path.append('..')
from scripts import weight_adjustment, rewire, basic_components as bcomp

f = open('../Fit to empirical connectomes/Output/inits_MR.pckl', 'rb')
initial_matrix = pickle.load(f)
f.close()

eta = 10.0 # learning rate
steps = 10000 # sample interval
k = int(sys.argv[1]) # index of instance

prand = 0.1 # proportion of random rewiring
tau_rewire = 1.0 # tau_rewire
iters = 60000

pin = 0.15 # proportion of in-link rewiring
tau = 0.05 # tau_reweight
M = 100 # steps of adaptive rewiring that use the same kernel before update
rate = int(tau_rewire / tau) # steps of adaptive weight adjustment  between two successive rewiring steps
reweight_per_iter = int(M*rate) # steps of adaptive weight adjustment that use the same kernel before update

A_matrices = {}
A_matrices[0] = initial_matrix[k].copy()
trunc_num = 0

j = 0
A = A_matrices[j*reweight_per_iter].copy()
n = A.shape[0]

for i in range(j, j+iters):
    u = np.random.choice(n, reweight_per_iter)
    A, trunc_temp = weight_adjustment.weight_adjust_Cout(A, eta, tau, u)
    trunc_num = trunc_num + trunc_temp

    conK = bcomp.consensus_kernel(A, tau_rewire)
    advK = bcomp.advection_kernel(A, tau_rewire)
    R1 = np.random.random_sample(M) # choose incoming or outgoing connections
    R2 = np.random.random_sample(M) # choose random or adaptive rewiring
    flg1 = np.where(R1 < pin, 'in', 'out')

    for l, r2 in enumerate(R2):
        if r2<prand:
            A = rewire.random_rewire(A, n, flg1[l])
        else:
            A = rewire.rewire(A, conK, advK, n, flg1[l])

    if (i+1)%steps == 0:
        print(str(i+1),flush=True)
        A_matrices[(1+i)*reweight_per_iter] = A.copy()

f = open('../Fit to empirical connectomes/Output/MR_Cout_pin_'+str(pin)+'_prand_'+str(prand)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
pickle.dump([A, trunc_num], f)
f.close()
f = open('../Fit to empirical connectomes/Output_long/MR_Cout_pin_'+str(pin)+'_prand_'+str(prand)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
pickle.dump([A_matrices, trunc_num], f)
f.close()
