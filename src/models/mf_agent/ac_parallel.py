import numpy as np
import datetime
import timeit
import multiprocessing as mp

import sys

import models.mf_agent.ac as ac
import utils.saveload as sl 
import models.mf_agent.experiment as exp

def run_ac_exp(params,verbose=False,saveData=False,dirData='',returnData=False,rec_init=False):
    # Create folder to save data
    dataFolder = None
    timestamp  = None
    if saveData:
        if len(dirData)==0:
            dirData = sl.get_datapath()
            print(f"Directory to save data not specified. Data is saved in current directory:\n{dirData}\n")
        if verbose: print(f"Start making folder to save data.\n")
        timestamp = datetime.datetime.now()
        dataFolder = dirData / f'{timestamp.strftime("%Y_%m_%d_%H-%M-%S")}_{params["sim_name"]}'
        sl.make_long_dir(dataFolder)
        print(f"Simulation data will be saved in: \n{dataFolder}\n")

    # Start timer
    if verbose: print(f"Start running experiment.\n")
    start_exp = timeit.default_timer()

    # Creating recorder
    if verbose: print('Creating recorder.\n')
    rec = exp.recorder(params,params['rec_type'])  

    # Run trials in parallel 
    num_trial = params['number_trials']
    num_pool = min(mp.cpu_count(),num_trial)
    pool = mp.Pool(num_pool)
    for trial in range(num_trial):
        pool.apply_async(run_trial,args=(trial,params,rec,verbose,rec_init))
    pool.close()
    pool.join()
    
    end_exp = timeit.default_timer()
    exp_duration = end_exp - start_exp
    if verbose: print(f"Simulated experiment ({num_trial} agents) in {exp_duration} s, parallelized across {num_pool} kernels.\n")

    if saveData:
        if verbose: print(f"Start saving data into folder {dataFolder}.\n")
        start_data = timeit.default_timer()  
        rec.saveParams(params,dataFolder,'dict')
        rec.saveParams(params,dataFolder,'csv')
        rec.saveCodeVersion(dataFolder)
        rec.saveData(dataFolder,'df')
        rec.saveData(dataFolder,'csv')
        end_data = timeit.default_timer()
        if verbose: print(f'Done saving data, time elapsed: {end_data-start_data} sec.\n') 
        
    if returnData:
        if verbose: print("Start reading out data.\n")
        start_data = timeit.default_timer()  
        data_basic = rec.readoutData_basic()
        exp_data = [data_basic]
        if rec.rec_type=='advanced2':
            wc,ec,wa,ea = rec.readoutData_advanced2()
            exp_data.append(wc)
            exp_data.append(ec)            
            exp_data.append(wa)
            exp_data.append(ea) 
        end_data = timeit.default_timer()
        if verbose: print(f'Done reading data, time elapsed: {end_data-start_data} sec.\n') 
           
    else:
        exp_data = None
            
    return exp_data, params, timestamp, exp_duration, dataFolder
        

def run_trial(trial,params,rec,verbose=False,rec_init=False):
    # Create environment
    if verbose: print('Creating environment.\n')
    S = params['S']; P = params['P']; R = params['R']
    T       = (params['T'] if 'T' in params.keys() else np.array([]))   # Set terminal states
    t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)  # Set reward deactivation
    ac_env     = ac.env(S,list(P),list(R),T,t_deact)
    sg      = ac_env.getGoal()
    t_deact = ac_env.getTDeact()
    term    = ac_env.getTerminal()
    hierarchical = ('n' in params['agent_types'] and params['ntype']=='hN')

    # Run trial
    if verbose: print(f"Start running trial {trial}.\n")
    np.random.seed(params['seeds'][trial]) # Set random seed
    start_trial = timeit.default_timer()   # Start timer
    ac_agent    = ac.agent(params)         # Create agent
        
    # Run episodes
    for e in range(params['number_epi']):
        if verbose: print(f"Start running episode {e}.\n")
        start_epi = timeit.default_timer()

        # Reset goal states (to active) and initialize list of deactivated goal states and deactivation counters
        for s in sg:
            ac_env.activateGoal(s)
        g_deact = []
        it_deact = []
            
        # Initialize the state variables
        Tmax          = params['max_it']
        foundTerminal = False
        it            = 1 
            
        # Set initial state of agent
        ac_env.setAgentLoc(params['x0'][trial][e])
        ac_agent.giveStateInput(params['x0'][trial][e])
        ac_agent.updateMod(params['x0'][trial][e])
        ac_agent.resetTraces()
        mf_m, mf_mh, mf_mw, mf_mg  = ac_agent.evalMod(s,0)   

        if rec_init:
            # Record initial variables MF (after s0 = s_init)
            rec_list = [trial,0,it-1,s,np.NaN,s,0,0,(s in sg)]
            if rec.rec_type == 'advanced1' or rec.rec_type == 'advanced2':
                if hierarchical:    rec_list = rec_list + mf_m + [0] + mf_mh + mf_mw + mf_mg
                else:               rec_list = rec_list + mf_m + [0] 
            if rec.rec_type == 'advanced2':
                wc_notnan = (~np.isnan(ac_agent.critics[0].w)).nonzero()[0]
                wa_notnan = (~np.isnan(ac_agent.actors[0].w.flatten())).nonzero()[0]
                rec_wc = [ac_agent.critics[0].w[i] for i in wc_notnan]
                rec_ec = [ac_agent.critics[0].e[i] for i in wc_notnan] 
                rec_wa = [ac_agent.actors[0].w.flatten()[i] for i in wa_notnan] 
                rec_ea = [ac_agent.actors[0].e.flatten()[i] for i in wa_notnan]
                rec.recordData(rec_list,rec_wc,rec_ec,rec_wa,rec_ea)
            else:
                rec.recordData(rec_list)
        
        # Simulate steps until goal found or maximum number of steps reached
        while not foundTerminal and it<Tmax:
            # Active goal states if their deactivation time is over
            it_act = (((np.array(it_deact)-t_deact)>0).nonzero()[0])
            for i in it_act:
                ac_env.activateGoal(g_deact)
                g_deact.pop(i)
                it_deact.pop(i)
            
            # Take next action + update values
            s_current      = ac_env.agentLoc
            a              = ac_agent.act(s_current,e)          # decide on action
            s_next, r      = ac_env.evalAction(a,s_current)     # take action in environment
            m, mh, mw, mg  = ac_agent.evalMod(s_next,r)         # compute modulator signal 
            ac_agent.updateMod(s_next)                          # update state of modulator (e.g. state count, time count for novelty)
        
            tds, new_wcs, new_ecs, new_was, new_eas = ac_agent.learn(s_current,s_next,a,m) # learn from modulator signal

            # Check whether agent has reached the goal / terminal state
            foundGoal = False
            if s_next in np.nonzero(ac_env.R)[0]:
                foundGoal = True
                if verbose: print(f"Found goal state after {it} iterations.\n") 
            if s_next == term:
                foundTerminal = True
                if verbose: print(f"Episode ended in terminal state after {it} iterations.\n")
                
            # Record step
            rec_list = [trial,e,it,s_current,a,s_next,foundGoal,foundTerminal,(r!=0)]
            if rec.rec_type == 'advanced1' or rec.rec_type == 'advanced2':
                if hierarchical:    rec_list = rec_list + m + tds + mh + mw + mg
                else:               rec_list = rec_list + m + tds 
            if rec.rec_type == 'advanced2':
                wc_notnan = (~np.isnan(new_wcs)).nonzero()[0]
                wa_notnan = (~np.isnan(new_was)).nonzero()[0]
                rec_wc = [new_wcs[i] for i in wc_notnan]
                rec_ec = [new_ecs[i] for i in wc_notnan] 
                rec_wa = [new_was[i] for i in wa_notnan] 
                rec_ea = [new_eas[i] for i in wa_notnan]
                rec.recordData(rec_list,rec_wc,rec_ec,rec_wa,rec_ea)
            else:
                rec.recordData(rec_list)

            # Deactivate reward for some time
            if foundGoal:
                ac_env.deactivateGoal(ac_env.agentLoc)
                g_deact.append(ac_env.agentLoc)
                it_deact.append(0)
            it += 1
            it_deact = [it_deact[i]+1 for i in range(len(it_deact))]

        end_epi = timeit.default_timer()
        if verbose: print(f"Simulated episode {e} in {end_epi-start_epi} s.\n")

    end_trial = timeit.default_timer()
    if verbose: print(f"Simulated agent {trial} in {end_trial-start_trial} s.\n")

                

        
        
