import pandas as pd
import numpy as np
import scipy
import os
import timeit
import datetime
import pickle
import csv
import subprocess

import models.mf_agent.ac as ac
import utils.visualization as vis
import utils.saveload as sl

### Helper functions ########################################################################################
def all_zero_x0(trials,epi):
    return np.zeros((trials,epi),dtype=int).tolist()

def seq_per_trial_x0(seq,trials):
    return np.array(seq*trials).reshape((trials,len(seq))).tolist()

def auto_seeds(trials):
    return list(range(trials))

def import_exploration_params_surnor(path=''):
    if not path:
        full_path = sl.get_rootpath() / 'src' / 'models' / 'mb_agent' / 'fittedparams_mbnor.csv'
    else:
        full_path = path / 'fittedparams_mbnor.csv'

    params_df = pd.read_csv(full_path,sep=';')

    params_df['Value'] = params_df['Value'].str.replace(',','.')
    params_df['Error'] = params_df['Error'].str.replace(',','.')
    params_df['Value'] = pd.to_numeric(params_df['Value'],downcast='float')
    params_df['Error'] = pd.to_numeric(params_df['Error'],downcast='float')
    params_dict = dict(zip(params_df['Parameter'],params_df['Value']))

    params_dict['beta_N1']=1; params_dict['lambda_R']=0

    return params_dict

def import_params_surnor(path=''):
    if not path:
        full_path = sl.get_rootpath() / 'src' / 'models' / 'mb_agent' / 'fittedparams_mbnor.csv'
    else:
        full_path = path / 'fittedparams_mbnor.csv'

    params_df = pd.read_csv(full_path,sep=';')

    params_df['Value'] = params_df['Value'].str.replace(',','.')
    params_df['Error'] = params_df['Error'].str.replace(',','.')
    params_df['Value'] = pd.to_numeric(params_df['Value'],downcast='float')
    params_df['Error'] = pd.to_numeric(params_df['Error'],downcast='float')
    params_dict = dict(zip(params_df['Parameter'],params_df['Value']))

    return params_dict

def import_params_surnor_witherr():
    params_df = pd.read_csv('./src/mbnor/fittedparams_mbnor.csv',sep=';')
    params_df['Value'] = params_df['Value'].str.replace(',','.')
    params_df['Error'] = params_df['Error'].str.replace(',','.')
    params_df['Value'] = pd.to_numeric(params_df['Value'],downcast='float')
    params_df['Error'] = pd.to_numeric(params_df['Error'],downcast='float')
    
    return params_df

def beta1_to_beta1r(beta1):
    return np.round(beta1/(beta1+1),4)

def beta1r_to_beta1(beta1r):
    return np.round(beta1r/(1-beta1r),4)

# def update_params_surnor(new_val):
#     # new_val should be list of format: Parameter;Value;Error
#     with open('./src/mbnor/fittedparams_mbnor.csv', 'a') as f:
#         w = csv.writer(f)
#         w.writerow(new_val)
#         f.close()

### Recorder ################################################################################################
class surnor_recorder():

    def __init__(self,params,rec_type='basic'):
        self.rec_type = rec_type
        self.data_basic = []
        self.cols_basic = ['subID','epi','it','state','action','next_state','reward']
        if params['h']['update_type'] == 'leaky':
            self.cols_basic += ['leaky_time']
        
        if params['ntype']=='hN':
            self.cols_basic += ['novelty','foundGoal'] #['novelty','novelty_vector','foundGoal']
        else:
            self.cols_basic += ['novelty','foundGoal'] #['counts','novelty','foundGoal']
        
        if self.rec_type=='advanced':
            # self.beliefs = []
            self.qvals = []
            self.cols_qvals = ['subID','epi','it','qvals']
                
    # def recordData(self,d_basic,d_beliefs=None,d_qvals=None):
    def recordData(self,d_basic,d_qvals=None):
        self.data_basic.append(d_basic.copy())
        if self.rec_type=='advanced':
            # self.beliefs.append(d_beliefs.copy())
            self.qvals.append([d_basic[0],d_basic[1],d_basic[2],d_qvals.copy()])
        
    def readoutData_basic(self):
        return pd.DataFrame(self.data_basic,columns=self.cols_basic)

    def readoutData_advanced(self):
        # beliefs = np.array(self.beliefs)
        # qvals = np.array(self.qvals)
        # return beliefs, qvals
        qvals = pd.DataFrame(self.qvals,columns=self.cols_qvals)
        return qvals
    
    def saveData(self,dir_data,format_data='df',data_name='data_basic'):
        if not os.path.isdir(dir_data):
            os.mkdir(dir_data)
        
        start_save = timeit.default_timer()
        
        data_basic = self.readoutData_basic()
        if format_data=='df':
            data_basic.to_pickle(dir_data / f'{data_name}.pickle')  
        elif format_data=='csv':
            data_basic.to_csv(dir_data / f'{data_name}.csv',sep='\t')

        if self.rec_type=='advanced':
            # b,q = self.readoutData_advanced()
            q = self.readoutData_advanced()
            if format_data=='df':
                #np.save(dir_data+'/beliefs.npy',b)
                # np.save(dir_data+'/qvals.npy',q)
                q.to_pickle(dir_data / 'qvals.pickle')  
            elif format_data=='csv':
                #np.savetxt(dir_data+'/beliefs.csv',b.reshape(len(b[:,0,0,0]),-1),delimiter='\t')
                # np.savetxt(dir_data+'/qvals.csv',q.reshape(len(q[:,0,0]),-1),delimiter='\t')
                q.to_csv(dir_data / 'qvals.csv',sep='\t')
        
        end_save = timeit.default_timer()
        
    def saveParams(self,params,dir_params,format_params='dict',params_name='params'):
        if not os.path.isdir(dir_params):
            os.mkdir(dir_params)
        
        if format_params=='dict':
            with open(dir_params / f'{params_name}.pickle', 'wb') as f:
                pickle.dump(params,f)   
        elif format_params=='csv':
            with open(dir_params / f'{params_name}.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=params.keys())
                writer.writeheader()
                writer.writerow(params)
        elif format_params=='txt':
            with open(dir_params / f'{params_name}.txt', 'w') as f: 
                for key, value in params.items(): 
                    f.write('%s:%s\n' % (key, value))
        # elif format_params=='json':
        #     with open(dir_params+'/params.json', 'w') as f: 
        #         json.dump(params,f,sort_keys=True,indent=2)
    
    def saveCodeVersion(self,dir_cv):
        if not os.path.isdir(dir_cv):
            os.mkdir(dir_cv)
        
        with open(dir_cv / 'code_version.txt','w') as f:
            cv = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
            f.write(cv)
        
### Experiment ##############################################################################################
def run_surnor_exp(params_exp,params_model,verbose=True,saveData=False,returnData=False,dirData=''):
    # Create folder to save data
    dataFolder = None
    if saveData:
        if len(str(dirData))==0:
            dirData = sl.get_rootpath() / 'data' / 'ppc'
            print(f"Directory to save data not specified. Data is saved in current directory:\n{dirData}\n")
        if verbose: 
            print(f"Start making folder to save data.\n")
        dataFolder = dirData / f"{datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')}_{params_exp['sim_name']}"

    # Start timer
    if verbose: print(f"Start running experiment.\n")
    start_exp = timeit.default_timer()
    
    # Create environment 
    if verbose: print('Creating environment.\n')
    S       = params_exp['S'] 
    A       = params_exp['A']
    P       = params_exp['P']
    R       = params_exp['R']
    T       = (params_exp['T'] if 'T' in params_exp.keys() else np.array([]))   # Set terminal states
    t_deact = (params_exp['t_deact'] if 't_deact' in params_exp.keys() else 0)  # Set reward deactivation
    env     = ac.env(S,list(P),list(R),T,t_deact)
    sg      = env.getGoal()
    t_deact = env.getTDeact()

    # Extract parameters
    if not 'ntype' in params_exp.keys():  
        params_exp['ntype'] = 'N'
    if not 'k' in params_exp.keys():      
        params_exp['k']     = 0

    eps     = params_model['epsilon']
    lamR    = params_model['lambda_R']
    lamN    = params_model['lambda_N']
    Tps     = params_model['T_PS']
    beta    = params_model['beta_1']
    k_leak  = params_model['k_leak']
    ntype   = params_exp['ntype']
    k       = params_exp['k']

    # if not 'k_alph' in params_exp.keys(): params_exp['k_alph'] = 1
    # k_alph  = params_exp['k_alph'] # leakiness of counts; k_alph = 1 means no leakiness
    #if verbose: print(f"eps:{eps}\n lamR:{lamR}\n lamN:{lamN}\n beta:{beta}\n")

    # Create recorder (fixed across subjects)
    if verbose: print('Creating recorder.\n')
    all_params = {k:v for d in (params_exp,params_model) for k,v in d.items()}
    rec = surnor_recorder(all_params,params_exp['rec_type']) 
    
    # Run trials
    for trial in range(params_exp['number_trials']):
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
                    # Set hnov type and update function (only matters when using hierarchical novelty)
                    # type 2: apply -log on each level of the hierachy, then compute weighted sum of novelty per level
                    # type 3: compute weighted sum of familiarity per level, then apply -log on the sum    
                    hnov_type    = params_exp['h']['hnov_type'] if 'hnov_type' in params_exp['h'].keys() else params_exp['hnov_type'] if 'hnov_type' in params_exp.keys() else 2
                    compute_hnov = eval(f'compute_hnov{hnov_type}')
            
                    # Set update type (fixed/variable learning rate for novelty signal)
                    update_type = params_exp['h']['update_type'] if 'update_type' in params_exp['h'].keys() else params_exp['update_type'] if 'update_type' in params_exp.keys() else 'var'
                    update_hnov = eval(f'update_hnov_{update_type}rate')  # update_hnov_fixedrate, update_hnov_varrate, update_hnov_leakyrate
                    
                    # Set remaining hnov params
                    w       = params_exp['w']
                    h       = params_exp['h']       
                    h_w     = h['h_w']              # kernel mixture weights
                    kmat    = h['kmat']             # kernel function matrix (list of matrices |S|xlen(av))

                    if update_type=='fixed':
                        h_eps       = None  
                        h_alph_leak = None
                        k_alph      = h['k_alph'] if 'k_alph' in h.keys() else 0.1
                        if not isinstance(k_alph, list):
                            k_alph = [k_alph]
                        if len(k_alph)!=len(h_w):
                            k_alph = k_alph*len(h_w)
                    
                    elif update_type=='leaky':
                        h_eps       = h['eps_leak'] if 'eps_leak' in h.keys() else [1]
                        if not isinstance(h_eps, list):
                            h_eps = [h_eps]
                        if len(h_eps)!=len(h_w):
                            h_eps = h_eps*len(h_w)
                        h_eps       = [h_eps[i] * h['eps'][i] for i in range(len(h_eps))] # scale epsilon with initial weights
                        h_alph_leak = h['alph_leak'] if 'alph_leak' in h.keys() else [0]
                        if not isinstance(h_alph_leak, list):
                            h_alph_leak = [h_alph_leak]
                        if len(h_alph_leak)!=len(h_w):
                            h_alph_leak = h_alph_leak*len(h_w)
                        k_alph      = None
                        gg          = [np.zeros(h_w[i].shape) for i in range(len(h_w))]

                    elif update_type=='var':
                        h_eps       = h['eps'] if 'eps' in h.keys() else [1]
                        if not isinstance(h_eps, list):
                            h_eps = [h_eps]
                        if len(h_eps)!=len(h_w):
                            h_eps = h_eps*len(h_w)   
                        h_alph_leak = None
                        k_alph      = None
                    
                else:
                    # Set update type (fixed/variable learning rate for novelty signal)
                    update_type  = params_exp['h']['update_type'] if 'update_type' in params_exp['h'].keys() else 'var'
                    update_nov   = eval(f'update_nov_{update_type}rate')  # update_nov_fixedrate, update_nov_varrate, update_nov_leakyrate
                    compute_nov  = eval(f'compute_nov_{update_type}rate')
                    
                    # Set remaining params
                    if update_type=='fixed':
                        c_eps = None
                        c_alph_leak = None
                        k_alph = params_exp['h']['k_alph'] if 'k_alph' in params_exp['h'].keys() else 0.1
                        # p = 1/S * np.ones(S)
                        p = np.log(1/S) * np.ones(S)
                    elif update_type=='leaky':
                        c_eps = params_exp['h']['eps_leak'][0] if 'eps_leak' in params_exp['h'].keys() else 1
                        c_alph_leak = params_exp['h']['alph_leak'] if 'alph_leak' in params_exp['h'].keys() else None
                        k_alph = None
                        c = np.zeros(S)
                    elif update_type=='var':
                        c_eps = 1
                        c_alph_leak = None
                        k_alph = None
                        c = np.zeros(S)

                # Compute initial novelty
                if ntype=='hN': 
                    _, N0 = compute_hnov(h_w,kmat,k,w)
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
            t        = 1  # absolute time
            if update_type=='leaky':
                if ntype=='hN':
                    tt   = [0]*len(h_w)
                else:
                    tt   = 0  # leaky time integrator

            # Update novelty counts / weights
            if ntype=='hN':   
                if update_type=='fixed':
                    h_w, _ = update_hnov(h_w,kmat,s,k_alph)  
                elif update_type=='leaky':
                    h_w, gg, tt = update_hnov(h_w,kmat,s,h_eps,tt,gg,h_alph_leak)
                elif update_type=='var':
                    h_w, _ = update_hnov(h_w,kmat,s,h_eps,t)
            else:   
                if update_type=='fixed':
                    p = update_nov(p,s,k_alph)
                elif update_type=='leaky':
                    c, tt = update_nov(c,s,tt,c_alph_leak)
                elif update_type=='var':
                    c = update_nov(c,s)            

            # Compute initial novelty
            if ntype=='hN':     
                Nvec, N = compute_hnov(h_w,kmat,k,w)
            else:  
                if update_type=='fixed':
                    N = compute_nov(p)
                elif update_type=='leaky':
                    N = compute_nov(c,tt,S,c_eps)
                elif update_type=='var':
                    N = compute_nov(c,tt,S)                  

            # Record initial variables
            if update_type=='leaky':
                rec_basic = [trial,e,t-1,s,np.NaN,s,0,tt]
            else:
                rec_basic = [trial,e,t-1,s,np.NaN,s,0]
            if ntype=='hN':     
                rec_basic = rec_basic + [Nvec[s,:],(s in sg)] #[N,Nvec,(s in sg)]
            else:               
                rec_basic = rec_basic + [N[s],(s in sg)] #[c,N,(s in sg)]

            if rec.rec_type=='basic':
                rec.recordData(rec_basic)
            elif rec.rec_type=='advanced':
                # rec.recordData(rec_basic,theta,qN)
                rec.recordData(rec_basic,qN)    
  
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

                # Update time
                t  = t + 1

                # Update novelty variables and recompute novelty
                if ntype=='hN':  
                    # Update novelty weights 
                    if update_type=='fixed':
                        h_w, _ = update_hnov(h_w,kmat,s_new,k_alph)   #s_new??
                    elif update_type=='leaky':
                        h_w, gg, tt = update_hnov(h_w,kmat,s_new,h_eps,tt,gg,h_alph_leak)
                    elif update_type=='var':
                        h_w, _ = update_hnov(h_w,kmat,s_new,h_eps,t)
                    # Compute novelty
                    Nvec, N = compute_hnov(h_w,kmat,k,w)
                else:   
                    if update_type=='fixed':
                        p = update_nov(p,s_new,k_alph)
                        N = compute_nov(p)
                    elif update_type=='leaky':
                        c, tt = update_nov(c,s_new,tt,c_alph_leak)
                        N = compute_nov(c,tt,S,c_eps)
                    elif update_type=='var':
                        c = update_nov(c,s_new)    
                        N = compute_nov(c,tt,S)  
                
                # Run mbNoR update step
                alph[s][a][:] = k_leak*alph[s][a][:] + (1-k_leak)*eps
                alph[s][a][s_new]+=1
                theta = alph/np.expand_dims(np.sum(alph,axis=2),axis=2)
                #print(f'sum theta over s:{np.sum(theta,axis=2)}\n') # check whether sum is equal 1
                qN, uN = prioritized_sweeping(qN,uN,N,lamN,theta,Tps)
                if e==0 and (not s in sg):
                    uR[:] = 0   # maybe unnecessary
                    qR[:] = 0   # maybe unnecessary
                else:
                    qR, uR = prioritized_sweeping(qR,uR,r,lamR,theta,Tps)

                # Record variables
                if update_type=='leaky':
                    rec_basic = [trial,e,t-1,s,a,s_new,r,tt]
                else:
                    rec_basic = [trial,e,t-1,s,a,s_new,r]
                if ntype=='hN':     
                    rec_basic = rec_basic + [Nvec[s_new,:], (s_new in sg)] #[N,Nvec,(s_new in sg)]
                else:   
                    if update_type=='fixed':  
                        rec_basic = rec_basic + [N[s_new], (s_new in sg)] #[p,N,(s_new in sg)]   
                    else:       
                        rec_basic = rec_basic + [N[s_new], (s_new in sg)] #[c,N,(s_new in sg)]

                if rec.rec_type=='basic':
                    rec.recordData(rec_basic)
                elif rec.rec_type=='advanced':
                    # rec.recordData(rec_basic,theta,qN)
                    rec.recordData(rec_basic,qN)    

                # Update state for next iteration
                s = s_new

            end_epi = timeit.default_timer()
            if verbose: print(f"Simulated episode {e} in {end_epi-start_epi} s.\n")

        end_trial = timeit.default_timer()
        if verbose: print(f"Simulated agent {trial} in {end_trial-start_trial} s.\n")

    end_exp = timeit.default_timer()
    exp_duration = end_exp - start_exp
    if verbose: print(f"Simulated agent {trial} in {exp_duration} s.\n")

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


## Prioritized sweeping #####################################################################################  
def prioritized_sweeping(qM,uM,M,lamM,theta,Tps):
    start_ps = timeit.default_timer()
    # Apply effect of latest observation of Q-values using previous U-values
    #vec             = np.ones((1,1,len(S)))
    vec = M + lamM*uM
    qM = np.sum(theta*vec,axis=2)

    # Priority queue of U-values
    prio = np.abs(uM - np.nanmax(qM,axis=1))

    # Update U-values up to Tps steps       
    for t_update in range(int(Tps)):
        i_update = np.argmax(prio)
        dV = np.nanmax(qM[i_update][:]) - uM[i_update]
        uM[i_update] = np.nanmax(qM[i_update][:])

        # Apply effect of U-value update to Q-values
        qM = qM + lamM*dV*theta[...,i_update]

        # Update priority queue
        prio = np.abs(uM - np.nanmax(qM,axis=1))

    end_ps = timeit.default_timer()
    #print(f"time taken for prioritized sweeping: {end_ps-start_ps}\n")
    return qM, uM


## Computation of hierarchical novelty ########################################################################
def compute_hnov2(h_w,kmat,k,w):
    nov_vec = np.zeros((np.size(kmat[-1],axis=0),len(kmat)))
    nov = np.zeros(np.size(kmat[-1],axis=0))
    for i in range(len(h_w)):       # for each level i in the hierarchy
        nov_vec[:,i] = -np.log(np.sum(kmat[i]*h_w[i],axis=1))-k 
        nov += w[i]*nov_vec[:,i]      # summed novelty 
    hnov = nov_vec
    #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(np.sum(kmat[i]*h_w[i],axis=1)),4) for i in range(len(h_w))]}.\n")
    return hnov, nov

def compute_hnov3(h_w,kmat,k,w):
    nov_vec = np.zeros((np.size(kmat[-1],axis=0),len(kmat)))
    nov = np.zeros(np.size(kmat[-1],axis=0))
    for i in range(len(h_w)):           # for each level i in the hierarchy
        nov_vec[:,i] = np.sum(kmat[i]*h_w[i],axis=1)  
        nov += w[i]*nov_vec[:,i]      # summed novelty (before log)
    hnov = nov_vec
    nov = -np.log(nov)-k
    #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(nov_vec[:,i]),4) for i in range(len(h_w))]}.\n")
    return hnov, nov

def compute_hnov2_log(h_w,kmat,k,w): # kmat, hw in logspace

    # Initialize
    nov_vec = np.zeros((np.size(kmat[-1],axis=0),len(kmat)))
    nov     = np.zeros(np.size(kmat[-1],axis=0))

    # Compute novelty for each level in the hierachy + sum (log space)
    for i in range(len(h_w)):      
        nov_vec[:,i] = -np.log(np.sum(np.exp(kmat[i] + h_w[i]),axis=1))-k 
        nov += w[i]*nov_vec[:,i] 

    hnov = nov_vec
    #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(np.sum(self.kmat[i]*self.h_w[i],axis=1)),4) for i in range(len(self.h_w))]}.\n")
    return hnov, nov
    
def compute_hnov3_log(h_w,kmat,k,w): # kmat, hw in logspace

    # Initialize
    nov_vec = np.zeros((np.size(kmat[-1],axis=0),len(kmat)))
    nov     = np.zeros(np.size(kmat[-1],axis=0))

    # Compute familiarity for each level in the hierachy + sum (log space)
    for i in range(len(h_w)): 
        nov_vec[:,i] = np.sum(np.exp(kmat[i] + h_w[i]),axis=1)  
        nov += w[i]*nov_vec[:,i] 

    hnov = nov_vec
    nov  = -np.log(nov)-k
    #print(f"H-Nov check: probability sums for each level are {[np.round(np.sum(nov_vec[:,i]),4) for i in range(len(self.h_w))]}.\n")
    return hnov, nov

def compute_nov_varrate(c,tt,S):
    return np.log((tt+S)/(c+1))  

# def compute_nov_fixedrate(p):
#     return -np.log(p)

def compute_nov_fixedrate(plog):
    return -plog

def compute_nov_leakyrate(c,tt,S,eps):
    return np.log((tt+S*eps)/(c+eps))

def update_hnov_varrate(h_w,kmat,s,eps,t):
    h_w_new = []
    gamma_new = []
    for i in range(len(h_w)):
        # Compute new responsibilities
        gamma_i_nom = h_w[i]*kmat[i][s,:]
        gamma_i_denom = np.sum(gamma_i_nom)
        gamma_i = gamma_i_nom/gamma_i_denom
        gamma_new.append(gamma_i.copy())
        
        # Update weights (incremental update rule with prior)
        h_w_i = h_w[i] + 1/(t+len(h_w[i])*eps[i])*(gamma_i-h_w[i])
        #h_w_i = h_w[i] + 1/(t+len(kmat[0])*eps[i])*(gamma_i-h_w[i])
        h_w_new.append(h_w_i)
    #print(f"H-Nov update check: sum of new weights for each level are {[np.round(np.sum(h_w_new[i]),4) for i in range(len(h_w_new))]}.\n") 
    return h_w_new, gamma_new

def update_nov_varrate(c,s):
    c[s] += 1
    return c

def update_hnov_leakyrate(h_w,kmat,s,eps,t_cum,gamma_cum,alph_leak):
    t_cum_new       = []
    h_w_new         = []
    gamma_cum_new   = []
    for i in range(len(h_w)):

        # Update cumulative time steps
        t_cum_new.append((1-alph_leak[i])*t_cum[i] + 1)

        # Update cumulative responsibilities
        gamma_i_nom     = h_w[i]*kmat[i][s,:]
        gamma_i_denom   = np.sum(gamma_i_nom)
        gamma_i         = gamma_i_nom/gamma_i_denom
        gamma_cum_i     = (1-alph_leak[i])*gamma_cum[i] + gamma_i
        gamma_cum_new.append(gamma_cum_i.copy())
        
        # Compute new weights (based on integrated responsibilities and time)
        h_w_i = (gamma_cum_i + eps[i])/(t_cum_new[i]+np.sum(eps[i]))
        # h_w_i = (gamma_cum_i + eps[i])/(t_cum+len(h_w[i])*eps[i])
        h_w_new.append(h_w_i)
    #print(f"H-Nov update check: sum of new weights for each level are {[np.round(np.sum(h_w_new[i]),4) for i in range(len(h_w_new))]}.\n") 
    return h_w_new, gamma_cum_new, t_cum_new

def update_nov_leakyrate(c,s,tt,alph_leak):
    c = c*(1-alph_leak)
    c[s] += 1
    tt = tt*(1-alph_leak) + 1
    return c, tt

# def update_hnov_fixedrate(h_w,kmat,s,alph):

#     h_w_new = []
#     gamma_new = []
#     for i in range(len(h_w)):

#         # Update the responsibilities
#         gamma_i_nom = h_w[i]*kmat[i][s,:]
#         gamma_i_denom = np.sum(gamma_i_nom)
#         gamma_i = gamma_i_nom/gamma_i_denom
#         gamma_new.append(gamma_i.copy())
        
#         # Update weights (incremental update rule with prior)
#         h_w_i = h_w[i] + alph*(gamma_i-h_w[i]) 
#         h_w_new.append(h_w_i)
#     #print(f"H-Nov update check: sum of new weights for each level are {[np.round(np.sum(h_w_new[i]),4) for i in range(len(h_w_new))]}.\n") 
#     return h_w_new, gamma_new

def update_hnov_fixedrate(h_w,kmat,s,alph): # kmat, h_w and gamma in logspace

        h_w_new       = []
        gamma_new     = []
        for i in range(len(h_w)):

            # Update the responsibilities
            gamma_i_nom     = h_w[i] + kmat[i][s,:]     # gamma_i = h_w[i] * kmat[s,:] ==> in logspace: gamma_i = log(h_w[i]) + log(kmat[s,:])
            if (np.exp(gamma_i_nom)==0).all(): # if all entries are zero, assume equal responsibilities across all (?)
                gamma_i = np.log(1/len(gamma_i_nom))*np.ones(len(gamma_i_nom)) # if all entries are zero, set gamma to zero (in logspace: -infinity)
            else:
                gamma_i_denom   = np.log(np.sum(np.exp(gamma_i_nom))) # gamma_sum = sum(gamma_i) ==> in logspace: log(sum(exp(gamma_i)))
                gamma_i         = gamma_i_nom - gamma_i_denom           # gamma_i = gamma_i / gamma_sum ==> in logspace: gamma_i - gamma_sum
            gamma_new.append(gamma_i.copy())
            
            # Update weights (incremental update rule with prior)
            h_w_i = np.log(np.exp(h_w[i] + np.log(1-alph[i])) + np.exp(gamma_i + np.log(alph[i])))   # h_w[i] = h_w[i] + k_alph*(gamma_i - h_w[i]) ==> in logspace: log(h_w[i]) + log(1-k_alph), then: log(exp(log(h_w[i])) + k_alph*exp(gamma_i)))
            h_w_i = np.clip(h_w_i, -sys.float_info.max, None) # clip weights (lower bound: 0 ==> in logspace: - infinity)
            h_w_new.append(h_w_i)

        #print(f"H-Nov update check: sum of new weights for each level are {[np.round(np.sum(h_w_new[i]),4) for i in range(len(h_w_new))]}.\n") 
        return h_w_new, gamma_new

# def update_nov_fixedrate(p,s,k_alph):
#     p = (1-k_alph)*p
#     p[s] += k_alph
#     return p

def update_nov_fixedrate(plog,s,k_alph):
    plog = plog + np.log(1-k_alph)
    plog[s] = np.log(np.exp(plog[s]) + k_alph)
    return plog

#############################################################################################################
if __name__ == "__main__":
    params_surnor = import_params_surnor()

    params_exp = {'sim_name':'mbNoR_debug',
                  'rec_type':'basic',
                  'number_trials':1,
                  'number_epi':5,
                  'max_it':10000,
                  'seeds':list(range(1)),
                  'x0':seq_per_trial_x0([5,8,3,4,7],1), #+[8, 6, 5, 4, 6, 2, 1, 2, 8, 8, 4, 6, 2, 4, 4, 5, 5, 5, 7, 3],12), #all_zero_x0(12,25),
                  'S':11,
                  'A':4,
                  'R':np.array([0,0,0,0,0,0,0,0,0,0,1]), 
                  'P':np.array([[0, 1, 7, 8],      # transition matrix: rows = states, cols = actions; each matrix entry P_ij shows the state that results from taking action j in state i
                                [1, 2, 8, 9],
                                [2, 3, 7, 9],
                                [3, 4, 7, 8],
                                [4, 5, 8, 9],
                                [5, 6, 7, 9],
                                [6, 10,7, 8],
                                [7, 8, 9, 0],
                                [8, 7, 9, 0],
                                [9, 7, 8, 0],
                                [10,10,10,10]]),
                    }

   # Run experiment
    exp_data, _, _, dir_data = run_surnor_exp(params_exp,params_surnor,saveData=True)







