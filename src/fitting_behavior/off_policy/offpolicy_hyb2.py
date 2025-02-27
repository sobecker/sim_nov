import numpy as np
import pandas as pd
import scipy
from scipy.special import softmax

import sys
import models.mf_agent.ac as ac
import models.mb_agent.mb_surnor as nor
import models.hybrid_agent.hybrid_ac_nor as hyb


def offpolicy_hyb2(params,data,rec_counts=False,rec_qvals=False,rec_ll=False):

    # Create environment ######################################################################
    S = params['S']; A = params['A']; P = params['P']; R = params['R']
    T       = (params['T'] if 'T' in params.keys() else np.array([]))   # Set terminal states
    t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)  # Set reward deactivation
    env     = ac.env(S,list(P),list(R),T,t_deact)
    sg      = env.getGoal()
    t_deact = env.getTDeact()
    term    = env.getTerminal()
    term    = env.getTerminal()

    # Save params #############################################################################
    # Save params MB
    mb_eps     = params['epsilon']
    mb_lamR    = params['lambda_R']
    mb_lamN    = params['lambda_N']
    mb_Tps     = params['T_PS']
    mb_beta    = params['beta_1']
    mb_k_leak  = params['k_leak']
    if not 'mb_ntype' in params.keys():  params['mb_ntype'] = 'N'
    if not 'mb_k' in params.keys():      params['mb_k']     = 0
    if not 'mb_k_alph' in params.keys(): params['mb_k_alph'] = 1
    mb_ntype   = params['mb_ntype']
    mb_k       = params['mb_k']
    mb_k_alph  = params['mb_k_alph']

    # Save params MF
    mf_temp    = params['temp']

    # Save params hybrid
    w_mf = params['w_mf']
    w_mb = 1-params['w_mf']

    # Set hierarchical
    mf_hierarchical = ('n' in params['agent_types'] and params['mf_ntype']=='hN')
    mb_hierarchical = (mb_ntype=='hN')

    # Initialize recording ####################################################################
    all_t_mb                = []
    all_tt_mb               = []
    all_t_mf                = []
    all_tt_mf               = []
    all_s                   = []
    all_s_new               = []

    all_nov_s_mb            = []
    all_nov_s_new_mb        = []
    all_nov_s_pre_mf        = []
    all_nov_s_new_pre_mf    = []
    all_nov_s_post_mf       = []
    all_nov_s_new_post_mf   = []

    if rec_counts:  
        all_c_s_mb          = []
        all_c_s_new_mb      = []
        all_c_s_pre_mf      = []
        all_c_s_new_pre_mf  = []
        all_c_s_post_mf     = []
        all_c_s_new_post_mf = []

    if rec_qvals:   
        all_qvals_mb = []
        all_qvals_mf = []

    if rec_ll:   
        ll = 0   
        all_ll = []

    # Initialize agents #######################################################################
    # Initialize MF agent
    params_mf = params.copy(); overlap = ['mf_rec_type','mf_k','mf_ntype','mf_k_alph','mf_h','mf_w']
    for i in range(len(overlap)):
        if overlap[i] in params_mf.keys(): params_mf[f'{overlap[i].replace("mf_","")}'] = params_mf.pop(overlap[i])
    mf_agent    = ac.agent(params_mf)           

    # Initialize MB agent
    mb_uR     = np.zeros(S)
    mb_qR     = np.zeros(np.shape(env.P))
    mb_alph   = mb_eps*np.ones((S,A,S))

    # Initialize  novelty variables
    if mb_hierarchical:  
        params_mb = params.copy(); overlap = ['mb_rec_type','mb_k','mb_ntype','mb_k_alph','mb_h','mb_w']
        for i in range(len(overlap)):
            if overlap[i] in params_mb.keys(): params_mb[f'{overlap[i].replace("mb_","")}'] = params_mb.pop(overlap[i])                
        mb_hnov_type, mb_compute_hnov, mb_update_type, mb_update_hnov = hyb.set_hnov(params_mb)
        mb_w       = params_mb['w']
        mb_h       = params_mb['h']                      
        mb_h_w     = mb_h['h_w']              # kernel mixture weights
        mb_kmat    = mb_h['kmat']             # kernel function matrix (list of matrices |S|xlen(av))
        mb_h_eps   = mb_h['k_alph'] if mb_update_type=='fix' else mb_h['eps']
        if mb_h_eps==None: mb_h_eps = [1/(len(mb_h_w[i])**2) for i in range(len(mb_h_w))]  
    else:
        mb_c      = np.zeros(S)

    # Initialize Q-values with initial novelty
    if mb_hierarchical: 
        _, mb_N0 = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
        mb_N0 = mb_N0/(1-mb_lamN) 
    else: 
        mb_N0 = np.log(S)/(1-mb_lamN)*np.ones(S)
    mb_uN = mb_N0
    mb_qN = mb_N0.reshape(-1,1)*np.ones(np.shape(env.P))

    # Initialize beliefs
    mb_a_nan = np.isnan(env.P).nonzero()
    mb_qR[mb_a_nan] = np.NaN  # set to nan where no transitions
    mb_qN[mb_a_nan] = np.NaN
    for i,j in zip(mb_a_nan[0],mb_a_nan[1]):
        mb_alph[i,j,:] = np.NaN   

    # Compute initial theta values (just for recording purposes)
    mb_theta = mb_alph/np.expand_dims(np.sum(mb_alph,axis=2),axis=2)
    #print(f'sum theta over s:{np.sum(theta,axis=2)}\n') # check whether sum is equal 

    # Run episodes ############################################################################
    for e in range(params['number_epi']):
        # Set beta_N1
        mb_betaN = params['beta_N1'] if e==0 else 0
        #print(f"betaN:{betaN}\n")

        # Reset goal states (to active) and initialize list of deactivated goal states and deactivation counters
        for s in sg:
            env.activateGoal(s)
        g_deact = []
        it_deact = []

        # Initialize the state variables and agent location
        s             = data.iloc[0]['state']
        Tmax          = params['max_it']
        foundTerminal = False
        mb_t          = 1  # absolute time
        mb_tt         = 1  # leaky time integrator

        # Record initial variables (MF, pre)
        m0_s,_,_,_ = mf_agent.evalMod(s,0)
        c0_s = mf_agent.critics[0].pc.counts[s]
        all_nov_s_pre_mf.append(m0_s[0])
        all_nov_s_new_pre_mf.append(m0_s[0])
        if rec_counts:
            all_c_s_pre_mf.append(c0_s)
            all_c_s_new_pre_mf.append(c0_s)
    
        # Update state + novelty MF agent
        env.setAgentLoc(s)
        mf_agent.giveStateInput(s)
        mf_agent.updateMod(s)
        mf_agent.resetTraces()  
        m1_s,_,_,_ = mf_agent.evalMod(s,0)
        c1_s = mf_agent.critics[0].pc.counts[s]          

        # Update novelty MB agent
        if mb_hierarchical:     
            mb_h_w,_ = mb_update_hnov(mb_h_w,mb_kmat,mb_h_eps,s,mb_t)
            mb_Nvec, mb_N = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
        else:             
            mb_c[s] +=1
            mb_N = np.log((mb_tt+S)/(mb_c+1)) 

        # Record initial variables
        all_t_mb.append(mb_t-1)
        all_tt_mb.append(mb_tt)
        all_t_mf.append(0)
        all_tt_mf.append(0)
        all_s.append(s)
        all_s_new.append(s)

        all_nov_s_mb.append(mb_N[s])
        all_nov_s_new_mb.append(mb_N[s])
        all_nov_s_post_mf.append(m1_s[0])
        all_nov_s_new_post_mf.append(m1_s[0])
        if rec_counts:
            all_c_s_mb.append(mb_c[s])
            all_c_s_new_mb.append(mb_c[s])
            all_c_s_post_mf.append(c1_s)
            all_c_s_new_post_mf.append(c1_s)
        if rec_qvals:   
            all_qvals_mb.append(mb_qN)
            all_qvals_mf.append(mf_agent.actors[0].w)
        if rec_ll:      
            all_ll.append(0)

        # Simulate steps until goal found or maximum number of steps reached
        for it in range(len(data)):
            # Active goal states if their deactivation time is over
            it_act = (((np.array(it_deact)-t_deact)>0).nonzero()[0])
            for i in it_act:
                env.activateGoal(g_deact)
                g_deact.pop(i)
                it_deact.pop(i)
            
            # Get MF action preferences 
            mf_agent.actors[0].giveStateInput(s)
            mf_out_rates = np.array([mf_agent.actors[0].a[i].computeRate() if ~np.isnan(mf_agent.actors[0].w[s][i]) else np.NaN for i in range(len(mf_agent.actors[0].a))])
            not_nan = (~np.isnan(mf_out_rates)).nonzero()
            mf_p_softmax = softmax(mf_out_rates[list(not_nan[0])]/mf_temp,axis=0)
            if mf_temp==0 or np.isnan(mf_p_softmax).any():
                print(f"{mf_out_rates}, {mf_temp}, {mf_p_softmax}\n")

            # Get MB ation preferences
            mb_q = mb_qR + mb_betaN*mb_qN #q = (1-betaN)*qR + betaN*qN
            mb_p_softmax = scipy.special.softmax(mb_beta*mb_q[s][list(not_nan[0])])   
            if mb_beta==0 or np.isnan(mb_p_softmax).any():
                print(f"{mb_q[s][:]}, {mb_beta}, {mb_p_softmax}\n")
                    
            # Combine action preferences and take next action
            p_softmax = w_mb*mb_p_softmax + w_mf*mf_p_softmax

            # Compute loglikelihood of current action from observed data
            a    = int(data['action'].values[it]) # get action from data
            if rec_ll:
                lpa  = np.log(p_softmax[list(not_nan[0]).index(a)]) 
                all_ll.append(lpa)
            s_new_env, r    = env.evalAction(a,s)
            s_new           = data['next_state'].values[it]
            if s_new_env!=s_new:
                print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')

            # Record time and transition variables
            all_t_mf.append(it)
            all_tt_mf.append(mf_agent.critics[0].pc.t)
            all_s.append(s)
            all_s_new.append(s_new)

            # Record novelty and counts before seeing s_new (used as learning signal)
            mf_m_current, _, _, _ = mf_agent.evalMod(s,r)
            mf_m, _, _, _  = mf_agent.evalMod(s_new,r)  
            all_nov_s_pre_mf.append(mf_m_current[0])
            all_nov_s_new_pre_mf.append(mf_m[0])
            if rec_counts:
                all_c_s_pre_mf.append(mf_agent.critics[0].pc.counts[s])
                all_c_s_new_pre_mf.append(mf_agent.critics[0].pc.counts[s_new])

            # Update novelty + values (MF)
            mf_agent.updateMod(s_new)                      # update state of modulator (e.g. state count, time count for novelty)
            mf_tds, mf_new_wcs, mf_new_ecs, mf_new_was, mf_new_eas = mf_agent.learn(s,s_new,a,mf_m) # learn from modulator signal

            # Record novelty and counts after seeing s_new (for comparison with nor)
            mf_m_current, _, _, _ = mf_agent.evalMod(s,r)
            mf_m, _, _, _  = mf_agent.evalMod(s_new_env,r)  
            all_nov_s_post_mf.append(mf_m_current[0])
            all_nov_s_new_post_mf.append(mf_m[0])
            if rec_counts:
                all_c_s_post_mf.append(mf_agent.critics[0].pc.counts[s])
                all_c_s_new_post_mf.append(mf_agent.critics[0].pc.counts[s_new])

            # Record q-values after learning from N(s_new)
            if rec_qvals:   
                all_qvals_mf.append(np.array(mf_new_was)) 

            # Update time (MB)
            mb_t  = mb_t + 1
            mb_tt = mb_tt*mb_k_alph + 1

            # Update novelty (MB)
            if mb_hierarchical:
                mb_h_w,_   = mb_update_hnov(mb_h_w,mb_kmat,mb_h_eps,s_new,mb_t)
                mb_Nvec,mb_N  = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
            else:
                mb_c *= mb_k_alph
                mb_c[s_new]+=1
                mb_N = np.log((mb_tt+S)/(mb_c+1))  

            # Update values (MB)
            mb_alph[s][a][:] = mb_k_leak*mb_alph[s][a][:] + (1-mb_k_leak)*mb_eps
            mb_alph[s][a][s_new]+=1
            mb_theta = mb_alph/np.expand_dims(np.sum(mb_alph,axis=2),axis=2)
            #print(f'sum theta over s:{np.sum(mb_theta,axis=2)}\n') # check whether sum is equal 1
            mb_qN, mb_uN = nor.prioritized_sweeping(mb_qN,mb_uN,mb_N,mb_lamN,mb_theta,mb_Tps)
            if e==0 and (not s in sg):
                mb_uR[:] = 0   # maybe unnecessary
                mb_qR[:] = 0   # maybe unnecessary
            else:
                mb_qR, mb_uR = nor.prioritized_sweeping(mb_qR,mb_uR,r,mb_lamR,mb_theta,mb_Tps)
            
            # Recording of data
            all_t_mb.append(mb_t-1)
            all_tt_mb.append(mb_tt)
            all_nov_s_mb.append(mb_N[s])
            all_nov_s_new_mb.append(mb_N[s_new])
            if rec_counts:  
                all_c_s_mb.append(mb_c[s])
                all_c_s_new_mb.append(mb_c[s_new])
            if rec_qvals:   
                all_qvals_mb.append(mb_q) 
            if rec_ll:      
                all_ll.append(lpa)
                
            # Check whether agent has reached the goal / terminal state
            foundGoal = False
            if s_new in np.nonzero(env.R)[0]:
                foundGoal = True
            if s_new == term:
                foundTerminal = True

            # Deactivate reward for some time
            if foundGoal:
                env.deactivateGoal(env.agentLoc)
                g_deact.append(env.agentLoc)
                it_deact.append(0)
            it_deact = [it_deact[i]+1 for i in range(len(it_deact))]

            # Update state for next iteration
            s = s_new

    # Format recording
    rec = {'mb_time': all_t_mb,
           'mb_leaky_time': all_tt_mb,
           'mf_time': all_t_mf,
           'mf_leaky_time': all_tt_mf,
           'state': all_s,
           'next_state': all_s_new,
           'mb_nov_s': all_nov_s_mb,
           'mb_nov_s_new': all_nov_s_new_mb,
           'mf_nov_s_pre': all_nov_s_pre_mf,
           'mf_nov_s_new_pre': all_nov_s_new_pre_mf,
           'mf_nov_s_post': all_nov_s_post_mf,
           'mf_nov_s_new_post': all_nov_s_new_post_mf
           }
    if rec_counts: 
        rec['mb_counts_s'] = all_c_s_mb
        rec['mb_counts_s_new'] = all_c_s_new_mb
        rec['mf_counts_s_pre'] = all_c_s_pre_mf
        rec['mf_counts_s_new_pre'] = all_c_s_new_pre_mf
        rec['mf_counts_s_post'] = all_c_s_post_mf
        rec['mf_counts_s_new_post'] = all_c_s_new_post_mf
    if rec_ll: rec['LL'] = all_ll
    rec = pd.DataFrame(rec)

    if not rec_qvals: 
        all_qvals = None
    
    return rec, all_qvals