import numpy as np
import pandas as pd
import multiprocessing as mp
import scipy
import os
import timeit
import datetime
from scipy.special import softmax

import sys

import models.mf_agent.ac as ac
import models.mf_agent.experiment as exp
import models.mb_agent.mb_surnor as nor
import utils.saveload as sl

def run_hybrid_exp(params_exp,params_mb, params_mf,verbose=True,saveData=False,returnData=False,dirData='',rec_init=False):
    # Create folder to save data
    dataFolder = None
    if saveData:
        if len(str(dirData))==0:
            dirData = sl.get_rootpath()
            print(f"Directory to save data not specified. Data is saved in current directory:\n{dirData}\n")
        if verbose: print(f"Start making folder to save data.\n")
        dataFolder = dirData / f'{datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}_{params_exp['sim_name']}'
        sl.make_long_dir(dataFolder)

    # Start timer
    if verbose: print(f"Start running experiment.\n")
    start_exp = timeit.default_timer()

    # Run trials in parallel 
    num_trial = params_exp['number_trials']
    num_pool = min(mp.cpu_count(),num_trial)
    pool = mp.Pool(num_pool)
    all_recs1 = [pool.apply_async(run_trial,args=(trial,params_exp,params_mb,params_mf,verbose,rec_init)) for trial in range(num_trial)]
    pool.close()
    pool.join()

    all_recs = [all_recs1[i].get() for i in range(len(all_recs1))]

    all_mb_recs = [all_recs[i][0] for i in range(len(all_recs))]
    all_mf_recs = [all_recs[i][1] for i in range(len(all_recs))]
    
    # End timer
    end_exp = timeit.default_timer()
    exp_duration = end_exp - start_exp
    if verbose: print(f"Simulated experiment ({num_trial} agents) in {exp_duration} s, parallelized across {num_pool} kernels.\n")

    if returnData or saveData:
        if verbose: print("Start reading out data.\n")
        start_data = timeit.default_timer()  
        
        # Read out MB data
        all_mb_basic = []
        if params_exp['rec_type']=='advanced': all_mb_q = []
        for i in range(len(all_mb_recs)):
            all_mb_basic.append(all_mb_recs[i].readoutData_basic())
            if params_exp['rec_type']=='advanced':
                all_mb_q.append(all_mb_recs[i].readoutData_advanced())      
        mb_data_basic = pd.concat(all_mb_basic)
        mb_all_data = [mb_data_basic]
        if params_exp['rec_type']=='advanced':  
            mb_q = pd.concatenate(all_mb_q)
            mb_all_data.append(mb_q) 
        end_data = timeit.default_timer()

        # Read out MF data
        all_mf_basic = []
        if params_mf['rec_type']=='advanced2':
            all_mf_wc = []; all_mf_ec = []; all_mf_wa = []; all_mf_ea = []
        for i in range(len(all_mf_recs)):
            all_mf_basic.append(all_mf_recs[i].readoutData_basic())
            if params_mf['rec_type']=='advanced2':
                mf_wc,mf_ec,mf_wa,mf_ea = all_mf_recs[i].readoutData_advanced2()
                all_mf_wc.append(mf_wc)
                all_mf_ec.append(mf_ec)
                all_mf_wa.append(mf_wa)
                all_mf_ea.append(mf_ea)
        mf_data_basic = pd.concat(all_mf_basic)
        mf_exp_data = [mf_data_basic]
        if params_mf['rec_type']=='advanced2':
            mf_exp_data.append(pd.concat(all_mf_wc))
            mf_exp_data.append(pd.concat(all_mf_ec))
            mf_exp_data.append(pd.concat(all_mf_wa))
            mf_exp_data.append(pd.concat(all_mf_ea))
        if verbose: print(f"Read out data in {end_data-start_data} s.\n")

        # Save data
        if saveData:
            if verbose: print(f"Start saving data into {dataFolder}.\n")
            start_data = timeit.default_timer()
            sl.saveCodeVersion(dataFolder) # save code version
            
            # Save MB params + data
            all_params_mb = {k:v for d in (params_exp,params_mb) for k,v in d.items()}
            all_mb_recs[0].saveParams(all_params_mb,dataFolder,'dict',params_name='mb_params')
            all_mb_recs[0].saveParams(all_params_mb,dataFolder,'csv',params_name='mb_params')
            saveData_mb(dataFolder,mb_all_data,format_data='df',data_name='mb_data_basic')
            saveData_mb(dataFolder,mb_all_data,format_data='csv',data_name='mb_data_basic')

            # Save MF params + data
            all_mf_recs[0].saveParams(params_mf,dataFolder,'dict',params_name='mf_params')
            all_mf_recs[0].saveParams(params_mf,dataFolder,'csv',params_name='mf_params')
            saveData_mf(dataFolder,mf_exp_data,'df',data_name='mf_data_basic')
            saveData_mf(dataFolder,mf_exp_data,'csv',data_name='mf_data_basic')
            end_data = timeit.default_timer()
            if verbose: print(f"Saved data in {end_data-start_data} s.\n")
        
    return mb_all_data, all_params_mb, mf_exp_data, params_mf, exp_duration, dataFolder


def run_trial(trial,params_exp,params_mb, params_mf,verbose=True, rec_init=False):
    # Create recorder MB + MF
    if verbose: print('Creating recorder.\n')
    all_params_mb = {k:v for d in (params_exp,params_mb) for k,v in d.items()}
    mb_rec = nor.surnor_recorder(all_params_mb,params_exp['rec_type']) 
    mf_rec = exp.recorder(params_mf,params_mf['rec_type'])  

    # Create environment
    if verbose: print('Creating environment.\n')
    S = params_exp['S']; A = params_exp['A']; P = params_exp['P']; R = params_exp['R']
    T       = (params_exp['T'] if 'T' in params_exp.keys() else np.array([]))   # Set terminal states
    t_deact = (params_exp['t_deact'] if 't_deact' in params_exp.keys() else 0)  # Set reward deactivation
    env     = ac.env(S,list(P),list(R),T,t_deact)
    sg      = env.getGoal()
    t_deact = env.getTDeact()
    term    = env.getTerminal()
    term    = env.getTerminal()

    # Save params MB
    mb_eps     = params_mb['epsilon']
    mb_lamR    = params_mb['lambda_R']
    mb_lamN    = params_mb['lambda_N']
    mb_Tps     = params_mb['T_PS']
    mb_beta    = params_mb['beta_1']
    mb_k_leak  = params_mb['k_leak']
    if not 'ntype' in params_exp.keys(): params_exp['ntype'] = 'N'
    if not 'k' in params_exp.keys():     params_exp['k']     = 0
    mb_ntype   = params_exp['ntype']
    mb_k       = params_exp['k']

    # Save params MF
    mf_temp    = params_mf['temp']

    # Save params hybrid
    w_mf = params_exp['w_mf']
    w_mb = params_exp['w_mb']

    # Set hierarchical
    mf_hierarchical = ('n' in params_mf['agent_types'] and params_mf['ntype']=='hN')
    mb_hierarchical = (mb_ntype=='hN')
    if verbose: print(f"Start running trial {trial}.\n")
    np.random.seed(params_exp['seeds'][trial])  # Set random seed
    start_trial = timeit.default_timer()        # Start timer
    mf_agent    = ac.agent(params_mf)           # Initialize MF agent

    # Initialize MB agent
    mb_uR     = np.zeros(S)
    mb_qR     = np.zeros(np.shape(env.P))
    mb_alph   = mb_eps*np.ones((S,A,S))
    # Initialize  novelty variables
    if mb_hierarchical:                 
        mb_hnov_type, mb_compute_hnov, mb_update_type, mb_update_hnov = set_hnov(params_exp)
        mb_w       = params_exp['w']
        mb_h       = params_exp['h']       
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
    for e in range(params_exp['number_epi']):
        if verbose: print(f"Start running episode {e}.\n")
        start_epi = timeit.default_timer()

        # Set beta_N1
        mb_betaN = params_mb['beta_N1'] if e==0 else 0
        #print(f"betaN:{betaN}\n")

        # Reset goal states (to active) and initialize list of deactivated goal states and deactivation counters
        for s in sg:
            env.activateGoal(s)
        g_deact = []
        it_deact = []

        # Initialize the state variables and agent location
        env.setAgentLoc(params_exp['x0'][trial][e])
        Tmax          = params_exp['max_it']
        foundTerminal = False
        t             = 1
        s             = env.agentLoc

        # Update novelty MF agent
        mf_agent.giveStateInput(s)
        mf_agent.updateMod(s)
        mf_agent.resetTraces()  
        mf_m, mf_mh, mf_mw, mf_mg  = mf_agent.evalMod(s,0)          

        # Update novelty MB agent
        if mb_hierarchical:     
            mb_h_w,_ = mb_update_hnov(mb_h_w,mb_kmat,mb_h_eps,s,t)
            mb_Nvec, mb_N = mb_compute_hnov(mb_h_w,mb_kmat,mb_k,mb_w)
        else:                   
            mb_c[s] +=1
            mb_N = np.log((t+S)/(mb_c+1)) 

        if rec_init:
            # Record initial variables MB (after s0 = s_init)
            if mb_hierarchical:     rec_basic = [trial,0,t-1,s,np.NaN,s,0,mb_N,mb_Nvec,(s in sg)]
            else:                   rec_basic = [trial,0,t-1,s,np.NaN,s,0,mb_N,(s in sg)]
            if mb_rec.rec_type=='basic':        mb_rec.recordData(rec_basic)
            elif mb_rec.rec_type=='advanced':   mb_rec.recordData(rec_basic,mb_qN) # rec.recordData(rec_basic,theta,qN)
            # Record initial variables MF (after s0 = s_init)
            rec_list = [trial,0,t-1,s,np.NaN,s,0,0,(s in sg)]
            if mf_rec.rec_type == 'advanced1' or mf_rec.rec_type == 'advanced2':
                if mf_hierarchical:    rec_list = rec_list + mf_m + [0] + mf_mh + mf_mw + mf_mg
                else:                  rec_list = rec_list + mf_m + [0] 
            if mf_rec.rec_type == 'advanced2':
                wc_notnan = (~np.isnan(mf_agent.critics[0].w)).nonzero()[0]
                wa_notnan = (~np.isnan(mf_agent.actors[0].w.flatten())).nonzero()[0]
                rec_wc = [mf_agent.critics[0].w[i] for i in wc_notnan]
                rec_ec = [mf_agent.critics[0].e[i] for i in wc_notnan] 
                rec_wa = [mf_agent.actors[0].w.flatten()[i] for i in wa_notnan] 
                rec_ea = [mf_agent.actors[0].e.flatten()[i] for i in wa_notnan]
                mf_rec.recordData(rec_list,rec_wc,rec_ec,rec_wa,rec_ea)
            else:
                mf_rec.recordData(rec_list)

        # Simulate steps until goal found or maximum number of steps reached
        while not foundTerminal and t<Tmax:
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
            #print(mf_p_softmax)
            # Get MB ation preferences
            mb_q = mb_qR + mb_betaN*mb_qN                                                       #q = (1-betaN)*qR + betaN*qN
            mb_q_notnan = (~np.isnan(mb_q[s])).nonzero()
            mb_p_softmax = scipy.special.softmax(mb_beta*mb_q[s][list(not_nan[0])])   
            if mb_beta==0 or np.isnan(mb_p_softmax).any():
                print(f"{mb_q[s][:]}, {mb_beta}, {mb_p_softmax}\n")
            # Combine action preferences and take next action
            p_softmax = (w_mb*mb_p_softmax + w_mf*mf_p_softmax)/(w_mb+w_mf)
            a = np.random.choice(np.arange(A)[list(not_nan[0])], p=p_softmax)
            #print(a)
            s_new, r = env.evalAction(a,s) # take action in env

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

            # Record variables (MB)
            if mb_hierarchical:     rec_basic = [trial,e,t,s,a,s_new,r,mb_N,mb_Nvec,(s_new in sg)]
            else:                   rec_basic = [trial,e,t,s,a,s_new,r,mb_N,(s_new in sg)]
            if mb_rec.rec_type=='basic':
                mb_rec.recordData(rec_basic)
            elif mb_rec.rec_type=='advanced':
                # rec.recordData(rec_basic,theta,qN)
                mb_rec.recordData(rec_basic,mb_qN)
                
            # Check whether agent has reached the goal / terminal state
            foundGoal = False
            if s_new in np.nonzero(env.R)[0]:
                foundGoal = True
                if verbose: print(f"Found goal state after {t} iterations.\n") 
            if s_new == term:
                foundTerminal = True
                if verbose: print(f"Episode ended in terminal state after {t} iterations.\n")

            # Record step (MF)
            rec_list = [trial,e,t,s,a,s_new,foundGoal,foundTerminal,(r!=0)]
            if mf_rec.rec_type == 'advanced1' or mf_rec.rec_type == 'advanced2':
                if mf_hierarchical:    rec_list = rec_list + mf_m + mf_tds + mf_mh + mf_mw + mf_mg
                else:               rec_list = rec_list + mf_m + mf_tds 
            if mf_rec.rec_type == 'advanced2':
                wc_notnan = (~np.isnan(mf_new_wcs)).nonzero()[0]
                wa_notnan = (~np.isnan(mf_new_was)).nonzero()[0]
                rec_wc = [mf_new_wcs[i] for i in wc_notnan]
                rec_ec = [mf_new_ecs[i] for i in wc_notnan] 
                rec_wa = [mf_new_was[i] for i in wa_notnan] 
                rec_ea = [mf_new_eas[i] for i in wa_notnan]
                mf_rec.recordData(rec_list,rec_wc,rec_ec,rec_wa,rec_ea)
            else:
                mf_rec.recordData(rec_list)

            # Deactivate reward for some time
            if foundGoal:
                env.deactivateGoal(env.agentLoc)
                g_deact.append(env.agentLoc)
                it_deact.append(0)
            it_deact = [it_deact[i]+1 for i in range(len(it_deact))]

            # Update state and time for next iteration
            s = s_new
            t +=1

        # # Record final step of episode (MB)
        # if mb_hierarchical:  rec_basic = [trial,e,t,s,a,s_new,r,mb_N,mb_Nvec,(s in sg)]
        # else:               rec_basic = [trial,e,t,s,a,s_new,r,mb_N,(s in sg)]
        # if mb_rec.rec_type=='basic':
        #     mb_rec.recordData(rec_basic)
        # elif mb_rec.rec_type=='advanced':
        #     # rec.recordData(rec_basic,theta,qN)
        #     mb_rec.recordData(rec_basic,mb_qN)

        end_epi = timeit.default_timer()
        if verbose: print(f"Simulated episode {e} in {end_epi-start_epi} s.\n")

    end_trial = timeit.default_timer()
    if verbose: print(f"Simulated agent {trial} in {end_trial-start_trial} s.\n")

    return mb_rec, mf_rec

def set_hnov(params):
    # Set hnov type 
    if 'hnov_type' in params['h'].keys():   hnov_type = params['h']['hnov_type']
    elif 'hnov_type' in params.keys():      hnov_type = params['hnov_type']
    else:                                       hnov_type = 2

    if hnov_type==2:    compute_hnov = nor.compute_hnov2
    elif hnov_type==3:  compute_hnov = nor.compute_hnov3

    # Set update type (fixed/variable learning rate for novelty signal)
    if 'update_type' in params['h'].keys():     update_type = params['h']['update_type']
    elif 'update_type' in params.keys():        update_type = params['update_type']
    else:                                           update_type = 'var'

    if update_type=='fix':      update_hnov = nor.update_hnov_fixedrate
    elif update_type=='var':    update_hnov = nor.update_hnov_varrate

    return hnov_type, compute_hnov, update_type, update_hnov


def saveData_mf(dir_data,data_basic,q=[],format_data='df',data_name='data_basic'):
    if not os.path.isdir(dir_data):
        os.mkdir(dir_data)
        
    if format_data=='df':
        data_basic.to_pickle(dir_data / f'{data_name}.pickle')  
    elif format_data=='csv':
        data_basic.to_csv(dir_data / f'{data_name}.csv',sep='\t')

    if len(q)>0:
        if format_data=='df':
            q.to_pickle(dir_data / 'qvals.pickle')  
        elif format_data=='csv':
            q.to_csv(dir_data / 'qvals.csv',sep='\t')

def saveData_mb(dir_data,all_data,format_data='df',data_name='data_basic'):
    if not os.path.isdir(dir_data):
        os.mkdir(dir_data)

    data_basic = all_data[0]
    if len(all_data)==2:    q = all_data[1]  
    else:                   q = []
             
    if format_data=='df':
        data_basic.to_pickle(dir_data / f'{data_name}.pickle')  
    elif format_data=='csv':
        data_basic.to_csv(dir_data / f'{data_name}.csv',sep='\t')

    if len(q)>0:
        if format_data=='df':
            q.to_pickle(dir_data / 'qvals.pickle')  
        elif format_data=='csv':
            q.to_csv(dir_data / 'qvals.csv',sep='\t')

def saveData_mf(dir_data,all_data,format_data='df',data_name='data_basic'):
        data_basic = all_data[0]
        if len(all_data)==5:
            wc = data_basic[1]
            ec = data_basic[2]
            wa = data_basic[3]
            ea = data_basic[4]
        else: wc=[];ec=[];wa=[];ea=[]
        if format_data=='df':
            data_basic.to_pickle(dir_data / f'{data_name}.pickle')  
        elif format_data=='csv':
            data_basic.to_csv(dir_data / f'{data_name}.csv',sep='\t')

        if format_data=='df':
            if len(wc)>0: wc.to_pickle(dir_data / 'wc.pickle')
            if len(ec)>0: ec.to_pickle(dir_data / 'ec.pickle')  
            if len(wa)>0: wa.to_pickle(dir_data / 'wa.pickle')  
            if len(ea)>0: ea.to_pickle(dir_data / 'ea.pickle')    
        elif format_data=='csv':
            if len(wc)>0: wc.to_csv(dir_data / 'wc.csv',sep='\t')
            if len(ec)>0: ec.to_csv(dir_data / 'ec.csv',sep='\t')
            if len(wa)>0: wa.to_csv(dir_data / 'wa.csv',sep='\t')
            if len(ea)>0: ea.to_csv(dir_data / 'ea.csv',sep='\t')

        