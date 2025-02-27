import numpy as np
import scipy
import timeit
import datetime
import multiprocessing as mp

import sys

import models.mf_agent.ac as ac
import models.mb_agent.mb_surnor as nor_seq
import utils.saveload as sl

def run_surnor_exp(params_exp,params_model,verbose=True,saveData=False,returnData=False,dirData='',rec_init=False):
    # Create folder to save data
    dataFolder = None
    if saveData:
        if len(dirData)==0:
            dirData = sl.get_datapath()
            print(f"Directory to save data not specified. Data is saved in current directory:\n{dirData}\n")
        if verbose: print(f"Start making folder to save data.\n")
        dataFolder = sl.make_long_dir(dirData,datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')+'_'+params_exp['sim_name'])

    # Start timer
    if verbose: print(f"Start running experiment.\n")
    start_exp = timeit.default_timer()

    # Create recorder (fixed across subjects)
    if verbose: print('Creating recorder.\n')
    all_params = {k:v for d in (params_exp,params_model) for k,v in d.items()}
    rec = nor_seq.surnor_recorder(all_params,params_exp['rec_type']) 
    
    # Run trials in parallel 
    num_trial = params_exp['number_trials']
    num_pool = min(mp.cpu_count(),num_trial)
    pool = mp.Pool(num_pool)
    for trial in range(num_trial):
        pool.apply_async(run_trial,args=(trial,params_exp,params_model,rec,verbose,rec_init))
    pool.close()
    pool.join()

    end_exp = timeit.default_timer()
    exp_duration = end_exp - start_exp
    if verbose: print(f"Simulated experiment ({num_trial} agents) in {exp_duration} s, parallelized across {num_pool} kernels.\n")

    if returnData:
        if verbose: print("Start reading out data.\n")
        start_data = timeit.default_timer()  
        data_basic = rec.readoutData_basic()
        all_data = [data_basic]
        if rec.rec_type=='advanced':
            # b,q = rec.readoutData_advanced()
            q = rec.readoutData_advanced()
            #all_data.append(b)
            all_data.append(q)
        end_data = timeit.default_timer()
        if verbose: print(f"Read out data in {end_data-start_data} s.\n")
    else:
        all_data = None

    if saveData:
        if verbose: print(f"Start saving data into {dataFolder}.\n")
        start_data = timeit.default_timer()
        # if 'src/mbnor' in os.getcwd() or 'src/optimization' in os.getcwd():
        #     dataFolder = '../../data/'+datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')+'_'+params_exp['sim_name']
        # else:
        #     dataFolder = './data/'+datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')+'_'+params_exp['sim_name']
        rec.saveParams(all_params,dataFolder,'dict')
        rec.saveParams(all_params,dataFolder,'csv')
        rec.saveCodeVersion(dataFolder)
        # rec.saveParams(all_params,dataFolder,'json')
        rec.saveData(dataFolder,format_data='df')
        rec.saveData(dataFolder,format_data='csv')
        end_data = timeit.default_timer()
        if verbose: print(f"Saved data in {end_data-start_data} s.\n")
        
    return all_data, all_params, exp_duration, dataFolder

def run_trial(trial,params_exp,params_model,rec,verbose=False,rec_init=False):
    # Create environment
    if verbose: print('Creating environment.\n')
    S = params_exp['S']; A = params_exp['A']; P = params_exp['P']; R = params_exp['R']
    T       = (params_exp['T'] if 'T' in params_exp.keys() else np.array([]))   # Set terminal states
    t_deact = (params_exp['t_deact'] if 't_deact' in params_exp.keys() else 0)  # Set reward deactivation
    env     = ac.env(S,list(P),list(R),T,t_deact)
    sg      = env.getGoal()
    t_deact = env.getTDeact()

    # Save params
    eps     = params_model['epsilon']
    lamR    = params_model['lambda_R']
    lamN    = params_model['lambda_N']
    Tps     = params_model['T_PS']
    beta    = params_model['beta_1']
    k_leak  = params_model['k_leak']
    if not 'ntype' in params_exp.keys(): params_exp['ntype'] = 'N'
    if not 'k' in params_exp.keys():     params_exp['k']     = 0
    ntype   = params_exp['ntype']
    k       = params_exp['k']
    #if verbose: print(f"eps:{eps}\n lamR:{lamR}\n lamN:{lamN}\n beta:{beta}\n")
    
    # Run trial
    if verbose: print(f"Start running trial {trial}.\n")
    np.random.seed(params_exp['seeds'][trial])  # Set random seed
    start_trial = timeit.default_timer()        # Start timer
        
    # Run episodes
    for e in range(params_exp['number_epi']):
        if verbose: print(f"Start running episode {e}.\n")
        start_epi = timeit.default_timer()
        
        # Initialize the model variables
        betaN = params_model['beta_N1'] if e==0 else 0
        #print(f"betaN:{betaN}\n")
        if e==0:
            uR     = np.zeros(S)
            qR     = np.zeros(np.shape(env.P))
            alph   = eps*np.ones((S,A,S))
                
            # Initialize  novelty variables
            if ntype=='hN':
                # Set hnov type
                if 'hnov_type' in params_exp['h'].keys():
                    hnov_type = params_exp['h']['hnov_type']
                elif 'hnov_type' in params_exp.keys():
                    hnov_type = params_exp['hnov_type']
                else: 
                    hnov_type = 2

                if hnov_type==2:
                    compute_hnov = nor_seq.compute_hnov2
                elif hnov_type==3:
                    compute_hnov = nor_seq.compute_hnov3

                # Set update type (fixed/variable learning rate for novelty signal)
                if 'update_type' in params_exp['h'].keys():
                    update_type = params_exp['h']['update_type']
                elif 'update_type' in params_exp.keys():
                    update_type = params_exp['update_type']
                else:
                    update_type = 'var'

                if update_type=='fix':
                    update_hnov = nor_seq.update_hnov_fixedrate
                elif update_type=='var':
                    update_hnov = nor_seq.update_hnov_varrate
                    
                # Set remaining hnov params
                w       = params_exp['w']
                h       = params_exp['h']       
                h_w     = h['h_w']              # kernel mixture weights
                kmat    = h['kmat']             # kernel function matrix (list of matrices |S|xlen(av))
                h_eps   = h['eps']
                if h_eps==None: h_eps = [1/(len(h_w[i])**2) for i in range(len(h_w))] 
            else:
                c      = np.zeros(S)
                
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
            qR[a_nan] = np.NaN  
            qN[a_nan] = np.NaN
            for i,j in zip(a_nan[0],a_nan[1]):
                alph[i,j,:] = np.NaN
                
            # Compute initial theta values (just for recording purposes)
            theta = alph/np.expand_dims(np.sum(alph,axis=2),axis=2)
            #print(f'sum theta over s:{np.sum(theta,axis=2)}\n') # check whether sum is equal 1

        # Initialize the state variables and agent location
        env.setAgentLoc(params_exp['x0'][trial][e])
        s        = env.agentLoc
        Tmax     = params_exp['max_it']
        t        = 1

        # Update novelty counts / weights
        if ntype=='hN':     h_w,_ = update_hnov(h_w,kmat,h_eps,s,t)
        else:               c[s] +=1

        # Compute initial novelty
        if ntype=='hN':     Nvec, N = compute_hnov(h_w,kmat,k,w)
        else:               N = np.log((t+S)/(c+1)) 

        # Record initial variables
        if rec_init:
            if ntype=='hN':     rec_basic = [trial,e,t-1,s,np.NaN,s,0,N,Nvec,(s in sg)]
            else:               rec_basic = [trial,e,t-1,s,np.NaN,s,0,N,(s in sg)]
        if rec.rec_type=='basic':       rec.recordData(rec_basic)
        elif rec.rec_type=='advanced':  rec.recordData(rec_basic,qN)

        # Run steps until goal found
        while (not s in sg) and t<Tmax:
            # Take action
            q = qR + betaN*qN                                                       #q = (1-betaN)*qR + betaN*qN
            q_notnan = (~np.isnan(q[s])).nonzero()
            p_softmax = scipy.special.softmax(beta*q[s][list(q_notnan[0])])   
            if beta==0 or np.isnan(p_softmax).any():
                print(f"{q[s][:]}, {beta}, {p_softmax}\n")
            a = np.random.choice(np.arange(A)[list(q_notnan[0])], p=p_softmax)
            s_new, r = env.evalAction(a,s)

            # Update novelty variables and recompute novelty
            if ntype=='hN':
                h_w,_   = update_hnov(h_w,kmat,h_eps,s_new,t)
                Nvec,N  = compute_hnov(h_w,kmat,k,w)
            else:
                c[s_new]+=1
                N = np.log((t+S)/(c+1))  

            # Run mbNoR update step
            alph[s][a][:] = k_leak*alph[s][a][:] + (1-k_leak)*eps
            alph[s][a][s_new]+=1
            theta = alph/np.expand_dims(np.sum(alph,axis=2),axis=2)
            #print(f'sum theta over s:{np.sum(theta,axis=2)}\n') # check whether sum is equal 1
            qN, uN = nor_seq.prioritized_sweeping(qN,uN,N,lamN,theta,Tps)
            if e==0 and (not s in sg):
                uR[:] = 0   # maybe unnecessary
                qR[:] = 0   # maybe unnecessary
            else:
                qR, uR = nor_seq.prioritized_sweeping(qR,uR,r,lamR,theta,Tps)

            # Record variables
            if ntype=='hN':
                rec_basic = [trial,e,t,s,a,s_new,r,N,Nvec,(s_new in sg)]
            else:
                rec_basic = [trial,e,t,s,a,s_new,r,N,(s_new in sg)]
                
            if rec.rec_type=='basic':
                rec.recordData(rec_basic)
            elif rec.rec_type=='advanced':
                # rec.recordData(rec_basic,theta,qN)
                rec.recordData(rec_basic,qN)
                
            # Update state and time for next iteration
            s = s_new
            t+=1

        # # Record variables in final step of episode
        # if ntype=='hN':     rec_basic = [trial,e,t,s,a,s_new,r,N,Nvec,(s in sg)]
        # else:               rec_basic = [trial,e,t,s,a,s_new,r,N,(s in sg)]
        # if rec.rec_type=='basic':
        #     rec.recordData(rec_basic)
        # elif rec.rec_type=='advanced':
        #     # rec.recordData(rec_basic,theta,qN)
        #     rec.recordData(rec_basic,qN)

        end_epi = timeit.default_timer()
        if verbose: print(f"Simulated episode {e} in {end_epi-start_epi} s.\n")

    end_trial = timeit.default_timer()
    if verbose: print(f"Simulated agent {trial} in {end_trial-start_trial} s.\n")


