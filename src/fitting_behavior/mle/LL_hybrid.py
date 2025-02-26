import numpy as np
import scipy
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.models.hybrid_agent.hybrid_ac_nor as hyb

import numpy as np
import scipy
from scipy.special import softmax

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.models.mf_agent.ac as ac
import src.models.mb_agent.mb_surnor as nor
import src.utils.saveload as sl

def ll_hybrid(params,data,mb_qvals=[],mf_qvals=[],verbose=False):

    # Create environment
    if verbose: print('Creating environment.\n')
    S = params['S']; A = params['A']; P = params['P']; R = params['R']
    T       = (params['T'] if 'T' in params.keys() else np.array([]))   # Set terminal states
    t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)  # Set reward deactivation
    env     = ac.env(S,list(P),list(R),T,t_deact)
    sg      = env.getGoal()
    t_deact = env.getTDeact()
    term    = env.getTerminal()
    term    = env.getTerminal()

    # Save params MB
    mb_eps     = params['epsilon']
    mb_lamR    = params['lambda_R']
    mb_lamN    = params['lambda_N']
    mb_Tps     = params['T_PS']
    mb_beta    = params['beta_1']
    mb_k_leak  = params['k_leak']
    if not 'mb_ntype' in params.keys(): params['mb_ntype'] = 'N'
    if not 'mb_k' in params.keys():     params['mb_k']     = 0
    mb_ntype   = params['mb_ntype']
    mb_k       = params['mb_k']

    # Save params MF
    mf_temp    = params['temp']

    # Save params hybrid
    w_mf = params['w_mf']
    w_mb = params['w_mb']

    # Set hierarchical
    mf_hierarchical = ('n' in params['agent_types'] and params['mf_ntype']=='hN')
    mb_hierarchical = (mb_ntype=='hN')

    # Init loglikelihood
    ll = 0
    if len(mb_qvals)>0: all_q_mb_equal = True
    else:               all_q_mb_equal = None  
    if len(mf_qvals)>0: all_q_mf_equal = True
    else:               all_q_mf_equal = None  

    ##########################################################################
    
    # Run trials
    for trial in range(params['number_trials']):
        if verbose: print(f"Start running trial {trial}.\n")
        np.random.seed(params['seeds'][trial])  # Set random seed
        params_mf = params.copy(); overlap = ['mf_rec_type','mf_k','mf_ntype','mf_h','mf_w']
        for i in range(len(overlap)):
            if overlap[i] in params_mf.keys(): params_mf[f'{overlap[i].replace("mf_","")}'] = params_mf.pop(overlap[i])
        mf_agent    = ac.agent(params_mf)           # Initialize MF agent

        # Initialize MB agent
        mb_uR     = np.zeros(S)
        mb_qR     = np.zeros(np.shape(env.P))
        mb_alph   = mb_eps*np.ones((S,A,S))
        # Initialize  novelty variables
        if mb_hierarchical: 
            params_mb = params.copy(); overlap = ['mb_rec_type','mb_k','mb_ntype','mb_h','mb_w']
            for i in range(len(overlap)):
                if overlap[i] in params_mb.keys(): params_mb[f'{overlap[i].replace("mb_","")}'] = params_mb.pop(overlap[i])                
            mb_hnov_type, mb_compute_hnov, mb_update_type, mb_update_hnov = hyb.set_hnov(params_mb)
            mb_w       = params_mb['w']
            mb_h       = params_mb['h']       
            mb_h_w     = mb_h['h_w']              # kernel mixture weights
            mb_kmat    = mb_h['kmat']             # kernel function matrix (list of matrices |S|xlen(av))
            mb_h_eps   = mb_h['eps']
            if mb_h_eps==None: mb_h_eps = [1/(len(mb_h_w[i])**2) for i in range(len(mb_h_w))]  
        else:
            mb_c      = np.zeros(S)
        # Compute initial novelty
        if mb_hierarchical: 
            _, mb_N0 = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
            mb_N0 = mb_N0/(1-mb_lamN) 
        else: 
            mb_N0 = np.log(S)/(1-mb_lamN)*np.ones(S)
        mb_uN = mb_N0
        mb_qN = mb_N0.reshape(-1,1)*np.ones(np.shape(env.P))
        # Set to nan where applicable (i.e. where no transitions)
        mb_a_nan = np.isnan(env.P).nonzero()
        mb_qR[mb_a_nan] = np.NaN  
        mb_qN[mb_a_nan] = np.NaN
        for i,j in zip(mb_a_nan[0],mb_a_nan[1]):
            mb_alph[i,j,:] = np.NaN     
        # Compute initial theta values (just for recording purposes)
        mb_theta = mb_alph/np.expand_dims(np.sum(mb_alph,axis=2),axis=2)
        #print(f'sum theta over s:{np.sum(theta,axis=2)}\n') # check whether sum is equal 

        # Run episodes
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
            env.setAgentLoc(params['x0'][trial][e])
            Tmax          = params['max_it']
            foundTerminal = False
            t             = 1
            s             = env.agentLoc

            # Update state + novelty MF agent
            mf_agent.giveStateInput(s)
            mf_agent.updateMod(s)
            mf_agent.resetTraces()            

            # Update novelty MB agent
            if mb_hierarchical:     
                mb_h_w,_ = mb_update_hnov(mb_h_w,mb_kmat,mb_h_eps,s,t)
                mb_Nvec, mb_N = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
            else:                   
                mb_c[s] +=1
                mb_N = np.log((t+S)/(mb_c+1)) 

            # Simulate steps until goal found or maximum number of steps reached
            for t in range(1,len(data.loc[data.epi==e])):
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
                mb_q = mb_qR + mb_betaN*mb_qN                                                       #q = (1-betaN)*qR + betaN*qN
                mb_p_softmax = scipy.special.softmax(mb_beta*mb_q[s][list(not_nan[0])])   
                if mb_beta==0 or np.isnan(mb_p_softmax).any():
                    print(f"{mb_q[s][:]}, {mb_beta}, {mb_p_softmax}\n")
                # Combine action preferences and take next action
                p_softmax = (w_mb*mb_p_softmax + w_mf*mf_p_softmax)/(w_mb+w_mf)

                # Compute loglikelihood of current action from observed data
                a    = int(data['action'].values[t-1]) # get action from data
                lpa  = np.log(p_softmax[list(not_nan[0]).index(a)]) 
                #print(f'a={a}, LL(a)={lpa}\n')
                #print(f'pMF = {mf_p_softmax}')
                #print(f'pMB = {mb_p_softmax}')
                ll  += lpa  
                s_new_env, r    = env.evalAction(a,s)
                s_new           = data['next_state'].values[t-1]
                #print(f'Observed next state: {s_new}, computed next state: {s_new_env}.\n')

                # Update novelty + values (MF)
                mf_m, mf_mh, mf_mw, mf_mg  = mf_agent.evalMod(s_new,r)     # compute modulator signal 
                mf_agent.updateMod(s_new)                      # update state of modulator (e.g. state count, time count for novelty)
                mf_tds, mf_new_wcs, mf_new_ecs, mf_new_was, mf_new_eas = mf_agent.learn(s,s_new,a,mf_m) # learn from modulator signal

                # Update novelty (MB)
                if mb_hierarchical:
                    mb_h_w,_   = mb_update_hnov(mb_h_w,mb_kmat,mb_h_eps,s_new,t)
                    mb_Nvec,mb_N  = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
                else:
                    mb_c[s_new]+=1
                    mb_N = np.log((t+S)/(mb_c+1))  

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
                
                if len(mb_qvals)>0:
                    qvals_notnan = (~np.isnan(mb_qN)).nonzero()
                    if not (mb_qN[qvals_notnan]==mb_qvals['qvals'].values[t-1][qvals_notnan]).all():
                        all_q_mb_equal = False
                if len(mf_qvals)>0:
                    qvals_notnan = (~np.isnan(mf_new_was)).nonzero()
                    if not (np.array(mf_new_was)[qvals_notnan]==mf_qvals[t-1]).all():
                        all_q_mf_equal = False
                
                # Check whether agent has reached the goal / terminal state
                foundGoal = False
                if s_new in np.nonzero(env.R)[0]:
                    foundGoal = True
                    if verbose: print(f"Found goal state after {t} iterations.\n") 
                if s_new == term:
                    foundTerminal = True
                    if verbose: print(f"Episode ended in terminal state after {t} iterations.\n")

                # Deactivate reward for some time
                if foundGoal:
                    env.deactivateGoal(env.agentLoc)
                    g_deact.append(env.agentLoc)
                    it_deact.append(0)
                it_deact = [it_deact[i]+1 for i in range(len(it_deact))]

                # Update state and time for next iteration
                s = s_new
                t +=1

    return ll, [all_q_mb_equal, all_q_mf_equal]