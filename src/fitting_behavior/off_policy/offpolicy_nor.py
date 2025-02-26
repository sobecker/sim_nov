import numpy as np
import pandas as pd
import scipy
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

from src.models.mb_agent.mb_surnor import prioritized_sweeping, compute_hnov2, compute_hnov3, update_hnov_fixedrate, update_hnov_varrate
import src.models.mf_agent.ac as ac

def offpolicy_nor(params,data,rec_counts=False,rec_qvals=False,rec_ll=False,rec_weights=False):

    # Set optional parameters
    if not 'k' in params.keys(): params['k']=0
    if not 'ntype' in params.keys(): params['ntype']='N'
    if not 'k_alph' in params.keys(): params['k_alph']=1

    if params['ntype']=='hN': rec_counts = False
    if params['ntype']=='N':  rec_weights = False
    
    # Create environment for the experiment
    S = params['S']
    A = params['A']
    P = params['P']
    R = params['R']

    # Create environment (fixed across subjects) 
    T       = (params['T'] if 'T' in params.keys() else np.array([]))
    t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)
    env     = ac.env(S,list(P),list(R),T,t_deact)
    sg          = env.getGoal()
    t_deact     = env.getTDeact()

    # Extract params
    eps     = params['epsilon']
    lamR    = params['lambda_R']
    lamN    = params['lambda_N']
    Tps     = params['T_PS']
    beta    = params['beta_1']
    k       = params['k']
    k_alph  = params['k_alph'] # leakiness of counts
    if params['ntype']=='hN':
        params['h']['k_alph'] = k_alph

    # Init loglikelihood
    if rec_ll: ll = 0
    
    ## Compute sequence of Q-values ##      
    # Initialize the model variables
    betaN = params['beta_N1']
    alph  = eps*np.ones((S,A,S))
                
    # Initialize  novelty variables
    ntype = params['ntype']
    if ntype=='hN':
        # Set hnov type
        if 'hnov_type' in params['h'].keys():
            hnov_type = params['h']['hnov_type']
        elif 'hnov_type' in params.keys():
            hnov_type = params['hnov_type']
        else: 
            hnov_type = 2

        if hnov_type==2:
            compute_hnov = compute_hnov2
        elif hnov_type==3:
            compute_hnov = compute_hnov3

        # Set update type (fixed/variable learning rate for novelty signal)
        if 'update_type' in params['h'].keys():
            update_type = params['h']['update_type']
        elif 'update_type' in params.keys():
            update_type = params['update_type']
        else:
            update_type = 'var'

        if update_type=='fix':
            update_hnov = update_hnov_fixedrate
        elif update_type=='var':
            update_hnov = update_hnov_varrate
                    
        # Set remaining hnov params
        w       = params['w']
        h       = params['h']       
        h_w     = h['h_w']              # kernel mixture weights
        kmat    = h['kmat']             # kernel function matrix (list of matrices |S|xlen(av))
        h_eps   = [h['k_alph']]*len(h_w) if update_type=='fix' else h['eps'] # prior or fixed learning rate for novelty
        if h_eps==None: h_eps = [1/(len(h_w[i])**2) for i in range(len(h_w))] 
    else:
        c      = np.zeros(S)
                
    # Initialize recording
    all_t           = []
    all_tt          = []
    all_s           = []
    all_s_new       = []
    all_nov_s       = []
    all_nov_s_new   = []
    if rec_counts:  
        all_c_s     = []
        all_c_s_new = []
    if rec_weights: all_hw = []
    if rec_qvals:   all_qvals = []
    if rec_ll:      all_ll = []

    # Compute initial novelty
    if ntype=='hN': 
        N0_vec, N0 = compute_hnov(h_w,kmat,k,w)
        N0 = N0/(1-lamN) 
    else: 
        N0 = np.log(S)/(1-lamN)*np.ones(S)
    uN = N0
    qN = N0.reshape(-1,1)*np.ones(np.shape(env.P))

    # Set to nan where applicable (i.e. where no transitions)
    a_nan = np.isnan(env.P).nonzero()
    qN[a_nan] = np.NaN
    for i,j in zip(a_nan[0],a_nan[1]):
        alph[i,j,:] = np.NaN
                
    # Init agent to first state of observed data
    env.setAgentLoc(data['state'].values[0])
    s        = env.agentLoc
    t        = 1  # absolute time
    tt       = 1  # leaky time integrator

    # Update novelty counts / weights
    if ntype=='hN':
        h_w,_ = update_hnov(h_w,kmat,h_eps,s,t)
    else:
        c[s] += 1 

    # Compute initial novelty
    if ntype=='hN':
        Nvec, N = compute_hnov(h_w,kmat,k,w)
    else:
        N = np.log((tt+S)/(c+1))

    # Record initial variables
    all_t.append(t-1)
    all_tt.append(tt)
    all_s.append(s)
    all_s_new.append(s)
    all_nov_s.append(N[s])
    all_nov_s_new.append(N[s])
    if rec_counts:
        all_c_s.append(c[s])
        all_c_s_new.append(c[s])
    if rec_weights: all_hw.append(h_w)
    if rec_qvals:   all_qvals.append(qN)
    if rec_ll:      all_ll.append(0)

    # Compute components of loglikelihood
    for i in range(len(data)):
        #print(f'{i}')
        # Compute Q-values and softmax distribution
        q = betaN*qN                                                       #q = (1-betaN)*qR + betaN*qN
        q_notnan = list((~np.isnan(q[s])).nonzero()[0])
        p_softmax = scipy.special.softmax(beta*q[s][q_notnan])   
        if beta==0 or np.isnan(p_softmax).any():
            print(f"{q[s][:]}, {beta}, {p_softmax}\n")

        # Compute loglikelihood of current action from observed data
        a   = int(data['action'].values[i])
        if rec_ll:
            lpa = np.log(p_softmax[q_notnan.index(a)])
            ll  += lpa    
        s_new_env, _    = env.evalAction(a,int(s))
        s_new           = data['next_state'].values[i]
        if s_new_env!=s_new:
            print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')
        #print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')

        # Update time
        tt = tt*k_alph + 1
        t  = t + 1

        # Update novelty variables and recompute novelty
        if ntype=='hN':
            h_w,_   = update_hnov(h_w,kmat,h_eps,s,t)
            Nvec,N  = compute_hnov(h_w,kmat,k,w)
        else:
            c *= k_alph
            c[s_new] += 1
            N = np.log((tt+S)/(c+1))  

        # Run mbNoR update step
        # alph[s][a][:] = k_leak*alph[s][a][:] + (1-k_leak)*eps
        alph[s][a][s_new]+=1
        theta = alph/np.expand_dims(np.sum(alph,axis=2),axis=2)
        qN, uN = prioritized_sweeping(qN,uN,N,lamN,theta,Tps)

        # Recording of data
        all_t.append(t-1)
        all_tt.append(tt)
        all_s.append(s)
        all_s_new.append(s_new)
        all_nov_s.append(N[s])
        all_nov_s_new.append(N[s_new])
        if rec_counts:  
            all_c_s.append(c[s])
            all_c_s_new.append(c[s_new])
        if rec_weights: all_hw.append(h_w)
        if rec_qvals:   all_qvals.append(q) 
        if rec_ll:      all_ll.append(lpa)
        
        # Update stats for next iteration
        s = s_new
    
    rec = {'time': all_t,
           'leaky_time': all_tt,
           'state': all_s,
           'next_state': all_s_new,
           'nov_s': all_nov_s,
           'nov_s_new': all_nov_s_new
           }
    if rec_counts: 
        rec['counts_s'] = all_c_s
        rec['counts_s_new'] = all_c_s_new
    if rec_ll: rec['LL'] = all_ll
    if rec_weights: rec['weights'] = all_hw
    rec = pd.DataFrame(rec)

    if not rec_qvals: all_qvals = None
        
    return rec, all_qvals