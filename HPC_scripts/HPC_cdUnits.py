# Generate networks for cd-units
import numpy as np
import pickle
import sys
sys.path.append('..')
from scripts import weight_adjustment, rewire, basic_components as bcomp

f = open('../Basic analysis/Output/inits.pckl', 'rb')
initial_matrix = pickle.load(f)
f.close()

n = 100 # number of nodes
eta = 10.0 # learning rate
steps = 10000 # sample interval
k = int(sys.argv[1]) # index of instance

prand = 0.1 # proportion of random rewiring
tau_rewire = 1.0 # tau_rewire
tau_ls = [0.05, 0.1, 0.2, 0.5, 1.0] # tau_reweight
tau_stationary = [0.05, 0.1, 0.5] # tau_reweight selected for stationary check
blocks = 50 # run 'blocks * steps' steps of weight adjustment

# A-in
for tau in tau_ls:
    A_matrices = {}
    A_matrices[0] = initial_matrix[k].copy()
    trunc_num = 0

    rate = int(tau_rewire / tau) # steps of adaptive weight adjustment  between two successive rewiring steps
    A = A_matrices[0].copy()
    for i in range(blocks*steps):
        # weight adjustment
        u = np.random.choice(n)
        A, trunc_temp = weight_adjustment.weight_adjust_Ain(A, eta, tau, u)
        trunc_num = trunc_num + trunc_temp

        # rewiring
        if (i+1)%rate==0:
            r1 = np.random.random_sample() # choose incoming or outgoing connections
            if r1<0.5:
                flag = 'in'
            else:
                flag = 'out'

            r2 = np.random.random_sample() # choose random or adaptive rewiring
            if r2<prand:
                A = rewire.random_rewire(A, n, flag)
            else:
                conK = bcomp.consensus_kernel(A, tau_rewire)
                advK = bcomp.advection_kernel(A, tau_rewire)
                A = rewire.rewire(A, conK, advK, n, flag)
        
        if (i+1)%steps == 0:
            print(str(i+1),flush=True)
            A_matrices[1+i] = A.copy()

    f = open('../Basic analysis/Output/cdUnits/A_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
    pickle.dump([A, trunc_num], f)
    f.close()
    if tau in tau_stationary:
        f = open('../Basic analysis/Output_long/cdUnits/A_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
        pickle.dump([A_matrices, trunc_num], f)
        f.close()

# C-out
for tau in tau_ls:
    A_matrices = {}
    A_matrices[0] = initial_matrix[k].copy()
    trunc_num = 0

    rate = int(tau_rewire / tau) # steps of adaptive weight adjustment  between two successive rewiring steps
    A = A_matrices[0].copy()
    for i in range(blocks*steps):
        # weight adjustment
        u = np.random.choice(n)
        A, trunc_temp = weight_adjustment.weight_adjust_Cout(A, eta, tau, u)
        trunc_num = trunc_num + trunc_temp
        
        # rewiring
        if (i+1)%rate==0:
            r1 = np.random.random_sample() # choose incoming or outgoing connections
            if r1<0.5:
                flag = 'in'
            else:
                flag = 'out'

            r2 = np.random.random_sample() # choose random or adaptive rewiring
            if r2<prand:
                A = rewire.random_rewire(A, n, flag)
            else:
                conK = bcomp.consensus_kernel(A, tau_rewire)
                advK = bcomp.advection_kernel(A, tau_rewire)
                A = rewire.rewire(A, conK, advK, n, flag)
        
        if (i+1)%steps == 0:
            print(str(i+1),flush=True)
            A_matrices[1+i] = A.copy()

    f = open('../Basic analysis/Output/cdUnits/C_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
    pickle.dump([A, trunc_num], f)
    f.close()
    if tau in tau_stationary:
        f = open('../Basic analysis/Output_long/cdUnits/C_eta_'+str(eta)+'_tau_'+str(tau)+'_'+str(k)+'.pckl', 'wb')
        pickle.dump([A_matrices, trunc_num], f)
        f.close()
