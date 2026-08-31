# Adaptive weight adjustment with fixed connectivity
import numpy as np
import pickle
import sys
sys.path.append('..')
from scripts import weight_adjustment

f = open('../Basic analysis/Output/inits.pckl', 'rb')
initial_matrix = pickle.load(f)
f.close()

n = 100 # number of nodes
eta = 10.0 # learning rate
steps = 10000 # sample interval
k = int(sys.argv[1]) # index of instance
tau_ls = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # tau_reweight
tau_stationary = [0.05, 0.1, 0.5] # tau_reweight selected for stationary check
blocks = 20 # run 'blocks * steps' steps of weight adjustment
# A-in
for tau in tau_ls:
    A_matrices = {}
    A_matrices[0] = initial_matrix[k].copy()
    trunc_num = 0

    A = A_matrices[0].copy()
    for i in range(blocks):
        print(i, flush=True)
        u_series = np.random.choice(n, steps)
        for u in u_series:
            A, trunc_temp = weight_adjustment.weight_adjust_Ain(A, eta, tau, u)
            trunc_num = trunc_num + trunc_temp
            A_matrices[(1+i)*steps] = A.copy()

    f = open('../Basic analysis/Output/wAdjust_alone/A_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
    pickle.dump([A, trunc_num], f)
    f.close()
    if tau in tau_stationary:
        f = open('../Basic analysis/Output_long/wAdjust_alone/A_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
        pickle.dump([A_matrices, trunc_num], f)
        f.close()

# C-out
for tau in tau_ls:
    A_matrices = {}
    A_matrices[0] = initial_matrix[k].copy()
    trunc_num = 0

    A = A_matrices[0].copy()
    for i in range(blocks):
        print(i, flush=True)
        u_series = np.random.choice(n, steps)
        for u in u_series:
            A, trunc_temp = weight_adjustment.weight_adjust_Cout(A, eta, tau, u)
            trunc_num = trunc_num + trunc_temp
            A_matrices[(1+i)*steps] = A.copy()
            
    f = open('../Basic analysis/Output/wAdjust_alone/C_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
    pickle.dump([A, trunc_num], f)
    f.close()
    if tau in tau_stationary:
        f = open('../Basic analysis/Output_long/wAdjust_alone/C_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
        pickle.dump([A_matrices, trunc_num], f)
        f.close()




