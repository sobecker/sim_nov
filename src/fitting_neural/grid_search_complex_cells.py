import os
import sys
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import spearmanr
from scipy.special import binom
import itertools

import json
from argparse import ArgumentParser

import src.utils.saveload as sl
import src.fitting_neural.simulate_data as sd
import src.models.snov.run_gabor_knov as gknov
import src.models.snov.run_gabor_knov_complex as gknovc
import src.fitting_neural.create_homann_input as h_in

## This script runs a grid search over the parameters of the simple cell kernel novelty model, simulating several samples of the Homann experiments for each parameter combination. ##

############################################################################################################
#               Function to load experimental data for Homann experiments                                  #
############################################################################################################
def load_exp_data2(h_type,filter_emerge=True,cluster=True,type='mean'):
    # Load experimental data (for fitting measure computation)
    print(f'Current path:{os.getcwd()}')
    if cluster:
        path_cluster = '/lcncluster/becker/RL_reward_novelty/ext_data/Homann2022'
    else:
        path_cluster = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Homann2022'
    if h_type=='tau_memory':
        # Load novelty responses
        path1  = os.path.join(path_cluster,f'Homann2022_{h_type}_{type}.csv')
        edata1 = pd.read_csv(path1)
        edata1 = edata1.sort_values('x')
        edx    = list(map(lambda x: int(np.round(x)),edata1['x']))
        edy1   = np.array(list(map(lambda x: np.round(x,4),edata1[' y'])))
        # Load steady state responses
        path2  = os.path.join(path_cluster,f'Homann2022_steadystate_{type}.csv')
        edata2 = pd.read_csv(path2)
        edata2 = edata2.rename(columns={'y':' y'})
        edy2   = np.array(list(map(lambda x: np.round(x,4),edata2[' y'])))
        # Combine
        edy    = [edy1,edy2]
    else:
        # Load novelty responses
        path = os.path.join(path_cluster,f'Homann2022_{h_type}_{type}.csv')
        edata = pd.read_csv(path)
        edata = edata.sort_values('x')
        if h_type=='tau_emerge' and filter_emerge:
            edata = edata.iloc[::2]
        edx   = list(map(lambda x: int(np.round(x)),edata['x']))
        edy   = np.array(list(map(lambda x: np.round(x,4),edata[' y'])))
    return edx,edy

def load_exp_homann(cluster=True,type='mean'):
    htypes = ['tau_emerge','tau_recovery','tau_memory']
    all_data = []
    for i in range(len(htypes)):
        edx,edy = load_exp_data2(htypes[i],cluster=cluster,type=type)
        if htypes[i] == 'tau_memory':   
            all_data.append((edx,edy[0]))
            all_data.append((edx,edy[1]))
        else:
            all_data.append((edx,edy)) 
    return all_data

############################################################################################################
#               Function to simulate single L-experiment (tau emerge)                                      #
############################################################################################################
def sim_tau_emerge(ivec,iseed,itype,k_params,n_fam,idx=True,norm=True,flr=False,verbose=False,kwargs={}):
    debug = kwargs['debug'] if 'debug' in kwargs else False
    # Simulate model
    all_data = []
    for j in range(len(ivec)):
        stim_type = itype[j] if debug else None
        if flr:
            idata,_,_ = gknovc.run_gabor_knov_withparams_flr(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
        else:
            idata,_,_ = gknovc.run_gabor_knov_withparams(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
        idata['n_fam'] = [n_fam[j]]*len(idata) 
        idata['seed']  = [iseed]*len(idata)  
        idata['stim_type'] = itype[j]   
        all_data.append(idata)
        if verbose: print(f'Done with experimental condition {j}/{len(n_fam)}.')
    data = pd.concat(all_data)

    if 'savepath_sim_data' in kwargs.keys() and 'savename_sim_data' in kwargs.keys():
        if os.path.exists(os.path.join(kwargs['savepath_sim_data'],kwargs['savename_sim_data']+'_l.csv')):
            header=False
        else:
            header=True
        data.to_csv(os.path.join(kwargs['savepath_sim_data'],kwargs['savename_sim_data']+'_l.csv'),mode='a',header=header)

    # Process simulation data
    data_nov, _ = sd.get_nov_response(data,'n_fam',get_stats=False)          

    return data_nov[['n_fam','nt_norm']]

############################################################################################################
#               Function to simulate single M-experiment (tau memory)                                      #
############################################################################################################
def sim_tau_memory(ivec,iseed,itype,k_params,n_images,idx=True,flr=False,flip=False,kmat_seq_flipped=None,verbose=False,kwargs={}):
    debug = kwargs['debug'] if 'debug' in kwargs else False
    # Simulate model
    all_data = []
    for j in range(len(ivec)):
        if flip and (kmat_seq_flipped is not None):     kmatj = kmat_seq_flipped[j]
        else:                                           kmatj = None
        stim_type = itype[j] if debug else None
        if flr:
            idata,_,_ = gknovc.run_gabor_knov_withparams_flr(ivec[j],k_params,idx=idx,flip=flip,kmat_seq_flipped=kmatj,stim_type=stim_type,kwargs=kwargs)
        else:
            idata,_,_ = gknovc.run_gabor_knov_withparams(ivec[j],k_params,idx=idx,flip=flip,kmat_seq_flipped=kmatj,stim_type=stim_type,kwargs=kwargs)
        idata['n_im'] = [n_images[j]]*len(idata) 
        idata['seed']  = [iseed]*len(idata)  
        idata['stim_type'] = itype[j]  
        all_data.append(idata)
        if verbose: print(f'Done with experimental condition {j}/{len(n_images)}.')
    data = pd.concat(all_data)

    if 'savepath_sim_data' in kwargs.keys() and 'savename_sim_data' in kwargs.keys():
        if os.path.exists(os.path.join(kwargs['savepath_sim_data'],kwargs['savename_sim_data']+'_m.csv')):
            header=False
        else:
            header=True
        data.to_csv(os.path.join(kwargs['savepath_sim_data'],kwargs['savename_sim_data']+'_m.csv'),mode='a',header=header)

    # Process simulation data
    data_nov,_ = sd.get_nov_response(data,'n_im',get_stats=False,vary_with_m=True,cycle_m=5)          

    return data_nov[['n_im','nt_norm']], data_nov[['n_im','steady']]

############################################################################################################
#               Function to simulate single L'-experiment (tau recovery)                                   #
############################################################################################################
def sim_tau_recovery(ivec,iseed,itype,k_params,dN,idx=True,flr=False,verbose=False,kwargs={}):
    debug = kwargs['debug'] if 'debug' in kwargs else False
    # Simulate model
    all_data = []
    for j in range(len(ivec)):
        stim_type = itype[j] if debug else None
        if flr:
            idata,_,_ = gknovc.run_gabor_knov_withparams_flr(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
        else:
            idata,_,_ = gknovc.run_gabor_knov_withparams(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
        idata['dN'] = [dN[j]]*len(idata) 
        idata['seed']  = [iseed]*len(idata)  
        idata['stim_type'] = itype[j]   
        all_data.append(idata)
        if verbose: print(f'Done with experimental condition {j}/{len(dN)}.')
    data = pd.concat(all_data)

    if 'savepath_sim_data' in kwargs.keys() and 'savename_sim_data' in kwargs.keys():
        if os.path.exists(os.path.join(kwargs['savepath_sim_data'],kwargs['savename_sim_data']+'_lp.csv')):
            header=False
        else:
            header=True
        data.to_csv(os.path.join(kwargs['savepath_sim_data'],kwargs['savename_sim_data']+'_lp.csv'),mode='a',header=header)

    # Process simulation data
    data_nov,_ = sd.get_trans_response(data,'dN',get_stats=False)     # why window_size=9?   

    return data_nov[['dN','tr_norm']]

############################################################################################################
#               Function to simulate a single experiment run (same input per trial)                        #
############################################################################################################
def run_homann_exp(k_params,params_input,inputs,sample_id,append_id=True,kwargs={}):

    data_nov_l = sim_tau_emerge(inputs[0][0],inputs[0][1],inputs[0][2],k_params,params_input['n_fam'],flr=k_params['flr'],kwargs=kwargs)
    if append_id: data_nov_l['sample_id'] = [sample_id]*len(data_nov_l)

    data_nov_lp = sim_tau_recovery(inputs[1][0],inputs[1][1],inputs[1][2],k_params,params_input['dN'],flr=k_params['flr'],kwargs=kwargs)
    if append_id: data_nov_lp['sample_id'] = [sample_id]*len(data_nov_lp)

    data_nov_m, data_steady_m = sim_tau_memory(inputs[2][0],inputs[2][1],inputs[2][2],k_params,params_input['n_im'],flr=k_params['flr'],kwargs=kwargs)
    if append_id: 
        data_nov_m['sample_id'] = [sample_id]*len(data_nov_m)
        data_steady_m['sample_id'] = [sample_id]*len(data_steady_m)

    print('Done with sample:',sample_id)

    return sample_id, data_nov_l, data_nov_lp, data_nov_m, data_steady_m

############################################################################################################
#               Function to run several experiment runs + save data                                        #
############################################################################################################
def run_homann_exp_n(k_params,params_input,inputs,parallel=False,save_path='',kwargs={}):
    # print(k_params)

    # Initialize Gabor kernel novelty model 
    k_params_init = gknovc.init_gabor_knov(gnum_complex=k_params['knum'],
                                    k_type=k_params['k_type'],
                                    ksig=k_params['ksig'],
                                    kcenter=k_params['kcenter'],
                                    seed=k_params['gabor_seed'],
                                    rng=None,
                                    sampling=k_params['gabor_sampling'],
                                    conv=k_params['conv'],
                                    cdens=k_params['cdens'],
                                    alph_k=k_params['k_alph'],
                                    parallel=False,
                                    adj_w=params_input['adj_w'],
                                    adj_f=params_input['adj_f'],
                                    alph_adj=params_input['alph_adj'])
    k_params_init['flr'] = k_params['flr']

    append_mode = kwargs['append_mode'] if 'append_mode' in kwargs else False
    start_id    = kwargs['start_id'] if 'start_id' in kwargs else 0
    
    # Run Homann experiments
    if parallel:
        num_pool = mp.cpu_count()
        pool = mp.Pool(num_pool)
        jobs = [pool.apply_async(run_homann_exp,args=(k_params_init,params_input,si,start_id+i,True,kwargs)) for i, si in enumerate(inputs)]
        data = [r.get() for r in jobs]
        pool.close()
        pool.join() 
    else:
        data = [run_homann_exp(k_params_init,params_input,si,start_id+i,kwargs=kwargs) for i, si in enumerate(inputs)]
        data.sort(key=lambda tup: tup[0])

    # Stack data
    stats_l = []; stats_m = []; stats_steady_m = []; stats_lp = []
    for i in range(len(data)):
        stats_l.append(data[i][1])
        stats_lp.append(data[i][2])
        stats_m.append(data[i][3])
        stats_steady_m.append(data[i][4])
    s_l = pd.concat(stats_l)
    s_lp = pd.concat(stats_lp)
    s_m = pd.concat(stats_m)
    ss_m = pd.concat(stats_steady_m)
    data_all = [s_l, s_lp, s_m, ss_m]
    data_var = ['n_fam','dN','n_im','n_im']
    data_val = ['nt_norm','tr_norm','nt_norm','steady']

    # Compute average response per experimental condition
    sim_data = []
    for i in range(len(data_all)):
        sim_data_i = data_all[i].groupby(data_var[i]).mean().reset_index()
        sim_data.append((sim_data_i[data_var[i]].values,sim_data_i[data_val[i]].values))  
    
    # Save data
    if len(save_path)>0:
        data_names = ['l','lp','m','m_steady']
        for i,n in zip(range(len(data_all)),data_names):
            if append_mode:
                data_all[i].to_csv(save_path+f'data_all_{n}.csv',index=False,mode='a',header=False)
                pd.DataFrame({data_var[i]:sim_data[i][0],data_val[i]:sim_data[i][1]}).to_csv(save_path+f'stats_{n}.csv',index=False,mode='a',header=False)
            else:
                data_all[i].to_csv(save_path+f'data_all_{n}.csv',index=False)
                pd.DataFrame({data_var[i]:sim_data[i][0],data_val[i]:sim_data[i][1]}).to_csv(save_path+f'stats_{n}.csv',index=False)
    return sim_data

############################################################################################################
#               MSE and score loss function                                                                #
############################################################################################################
mse_loss    = lambda y_true, y_pred: np.mean((y_true-y_pred)**2)
score_loss  = lambda y_true, y_pred: mse_loss(y_true,y_pred)/mse_loss(y_true,np.mean(y_true)*np.ones(y_true.shape))

############################################################################################################
#               Function to fit simulated data to experimental data                                        #
############################################################################################################
def fit_homann_exp(sim_data,homann_data,coef_steady=True,regr_meas='score',save_path='',save_name=''):
    ## Fit linear regression ##
    # Dimensions for fitting of multiplicative regression factor
    if coef_steady: x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],sim_data[3][1]]).reshape((-1,1)) 
    else:           x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],np.zeros(len(sim_data[3][1]))]).reshape((-1,1)) 

    # Dimension for fitting of shift (to steady state features)
    x2   = np.concatenate([np.zeros(len(sim_data[0][1])+len(sim_data[1][1])+len(sim_data[2][1])),np.ones(len(sim_data[3][1]))]).reshape((-1,1)) 
    x    = np.concatenate([x2,x1],axis=1)
    y    = np.concatenate([homann_data[0][1],homann_data[1][1],homann_data[2][1],homann_data[3][1]]).reshape((-1,1)) 

    # Fit coefficients using least squares estimation
    fit     = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y).flatten()
    coef    = fit[1]
    shift   = fit[0]
    
    ## Compute MSE between simulated (fitted) data and experiment ##
    # Combined MSE (including all experiments)
    comp_meas  = eval(f'{regr_meas}_loss')
    ypred      = np.dot(x,fit.reshape((-1,1))).flatten()
    mse_comb   = comp_meas(y,ypred)

    # MSE for each experiment
    cc = 0
    pred_data = []
    for i in range(len(sim_data)):
        pred_data.append((sim_data[i][0],ypred[cc:cc+len(sim_data[i][1])]))
        cc += len(sim_data[i][1])
    mse_tem    = comp_meas(homann_data[0][1],pred_data[0][1])
    mse_trec   = comp_meas(homann_data[1][1],pred_data[1][1])
    mse_tmem   = comp_meas(homann_data[2][1],pred_data[2][1])
    mse_steady = comp_meas(homann_data[3][1],pred_data[3][1])

    # Save fitting results
    if len(save_path)>0:
        sl.make_long_dir(save_path)
        mse_df = pd.DataFrame({'mse_comb':mse_comb,
                               'mse_tem':mse_tem,
                               'mse_trec':mse_trec,
                               'mse_tmem':mse_tmem,
                               'mse_steady':mse_steady},
                               index=[0])
        mse_df.to_csv(os.path.join(save_path,f'mse_fit{save_name}.csv'),index=False)
        coef_df = pd.DataFrame({'coef':coef,
                                'shift':shift},
                                index=[0])
        coef_df.to_csv(os.path.join(save_path,f'coef_fit{save_name}.csv'),index=False)
        data_names = ['l','lp','m','m_steady']
        data_var = ['n_fam','dN','n_im','n_im']
        data_val = ['nt_norm','tr_norm','nt_norm','steady']
        for i,n in zip(range(len(pred_data)),data_names):
            pd.DataFrame({data_var[i]:pred_data[i][0],data_val[i]:pred_data[i][1]}).to_csv(os.path.join(save_path,f'pred_{n}{save_name}.csv'),index=False)

    return pred_data, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady]

############################################################################################################
#               Function to compute correlation between simulated and experimental data                    #
############################################################################################################
def corr_homann_exp(sim_data,homann_data,regr_meas='score',corr_type='pearson',save_path=''):
    ## Compute correlation between simulated data and experiment ##
    if 'spearman' in corr_type:
        corr_fun = lambda x,y: spearmanr(x,y).statistic 
    else:
        corr_fun = lambda x,y: np.corrcoef(x,y)[0,1]
    comp_meas  = eval(f'{regr_meas}_loss')
    
    corr_ll  = []
    if 'mse' in corr_type: 
        corr_exp_ll = []; corr_sim_ll = []
    for i in range(len(sim_data)):
        if 'mse' in corr_type:
            corr_exp = corr_fun(homann_data[i][0],homann_data[i][1].flatten())      # correlation of experimental data with experimental variables
            corr_sim = corr_fun(homann_data[i][0],sim_data[i][1])                   # correlation of simulated data with experimental variables
            corr_exp_ll.extend(corr_exp); corr_sim_ll.extend(corr_sim)
            corr_i = comp_meas(corr_exp,corr_sim)                                    # MSE of correlations (experimental vs. simulated data) per experiment
        else:
            corr_i = 1-corr_fun(homann_data[i][1],sim_data[i][1])                     # 1-correlation between experimental and simulated data 
        corr_ll.append(corr_i)
    if 'mse' in corr_type: 
        corr_comb = comp_meas(np.array(corr_exp_ll),np.array(corr_sim_ll)) # MSE of correlations (experimental vs. simulated data) overall
    else:
        corr_comb = np.mean(np.array(corr_i)) # average correlation between experimental and simulated data (across experiments)

    if len(save_path)>0:
        corr_df = pd.DataFrame({'corr_comb':corr_comb,
                                'corr_tem':corr_ll[0],
                                'corr_trec':corr_ll[1],
                                'corr_tmem':corr_ll[2],
                                'corr_steady':corr_ll[3]},
                                index=[0])
        corr_df.to_csv(save_path+'corr.csv',index=False)

    return corr_comb, corr_ll

############################################################################################################
#               Function to generate inputs for simulation                                                 #
############################################################################################################
def generate_input(params_input,num_sim,init_seed=12345,input_corrected=True,input_sequence_mode='sep'):
    # Specify parameters input
    seed_input = sl.get_random_seed(5,num_sim,init_seed=init_seed)

    if params_input is None:
        params_input = {'num_gabor': 40,
                        'adj_w':     True,
                        'adj_f':     False,
                        'alph_adj':  3,
                        'n_fam':     [1,3,8,18,38],
                        'n_im':      [3,6,9,12],
                        'dN':        list(np.array([0,22,44,66,88,110,143])/0.3), # number of recovery images presented, computed based read out from Homann graphs #[0,70,140,210,280,360,480]
                        'idx':       True
                        }

    # Create input sequences for experiments
    inputs = []
    if input_corrected:
        for i in range(len(seed_input)):
            ivec_l, iseed_l, itype_l, _ = h_in.create_tau_emerge_input_gabor(n_fam=params_input['n_fam'],
                                                                                    len_fam=3,
                                                                                    num_gabor=params_input['num_gabor'],
                                                                                    idx=params_input['idx'],
                                                                                    adj_w=params_input['adj_w'],
                                                                                    adj_f=params_input['adj_f'],
                                                                                    alph_adj=params_input['alph_adj'],
                                                                                    seed=seed_input[i],
                                                                                    sequence_mode=input_sequence_mode)
            ivec_lp, iseed_lp, itype_lp, _ = h_in.create_tau_recovery_input_gabor(n_fam=22,
                                                                                    len_fam=3,
                                                                                    dN=params_input['dN'],
                                                                                    num_gabor=params_input['num_gabor'],
                                                                                    idx=params_input['idx'],
                                                                                    adj_w=params_input['adj_w'],
                                                                                    adj_f=params_input['adj_f'],
                                                                                    alph_adj=params_input['alph_adj'],
                                                                                    seed=seed_input[i],
                                                                                    sequence_mode=input_sequence_mode)
            ivec_m, iseed_m, itype_m, _ = h_in.create_tau_memory_input_gabor(n_fam=17,
                                                                                    len_fam=params_input['n_im'],
                                                                                    num_gabor=params_input['num_gabor'],
                                                                                    idx=params_input['idx'],
                                                                                    adj_w=params_input['adj_w'],
                                                                                    adj_f=params_input['adj_f'],
                                                                                    alph_adj=params_input['alph_adj'],
                                                                                    seed=seed_input[i],
                                                                                    sequence_mode=input_sequence_mode)
            inputs.append([(ivec_l,iseed_l,itype_l),(ivec_lp,iseed_lp,itype_lp),(ivec_m,iseed_m,itype_m)])

    else:
        for i in range(len(seed_input)):
            ivec_l, iseed_l, itype_l, _ = gknov.create_tau_emerge_input_gabor(n_fam=params_input['n_fam'],
                                                                                    len_fam=3,
                                                                                    num_gabor=params_input['num_gabor'],
                                                                                    idx=params_input['idx'],
                                                                                    adj_w=params_input['adj_w'],
                                                                                    adj_f=params_input['adj_f'],
                                                                                    alph_adj=params_input['alph_adj'],
                                                                                    seed=seed_input[i])
            ivec_lp, iseed_lp, itype_lp, _ = gknov.create_tau_recovery_input_gabor(n_fam=22,
                                                                                    len_fam=3,
                                                                                    dN=params_input['dN'],
                                                                                    num_gabor=params_input['num_gabor'],
                                                                                    idx=params_input['idx'],
                                                                                    adj_w=params_input['adj_w'],
                                                                                    adj_f=params_input['adj_f'],
                                                                                    alph_adj=params_input['alph_adj'],
                                                                                    seed=seed_input[i])
            ivec_m, iseed_m, itype_m, _ = gknov.create_tau_memory_input_gabor(n_fam=17,
                                                                                    len_fam=params_input['n_im'],
                                                                                    num_gabor=params_input['num_gabor'],
                                                                                    idx=params_input['idx'],
                                                                                    adj_w=params_input['adj_w'],
                                                                                    adj_f=params_input['adj_f'],
                                                                                    alph_adj=params_input['alph_adj'],
                                                                                    seed=seed_input[i])
            inputs.append([(ivec_l,iseed_l,itype_l),(ivec_lp,iseed_lp,itype_lp),(ivec_m,iseed_m,itype_m)])

    return seed_input, inputs, params_input

############################################################################################################
#              Function to run grid search over parameters of the simple cell kernel novelty model        #
############################################################################################################
def simulate_grid_point(i,k_params_ext,k_params_def,grid_df,grid_keys,save_path,params_input,inputs,parallel_exp,comp_fit,comp_corr,cluster,kwargs_def,kwargs_ext):
    k_params = k_params_def.copy()
    kwargs = kwargs_def.copy()

    # Build k_params for homann experiment  
    if len(k_params_ext)>0:
        k_params.update(k_params_ext)
    k_params_i = grid_df.loc[grid_df.grid_id==i,grid_keys].iloc[0].to_dict()  
    k_params.update(k_params_i)
    k_params['cdens'] = int(k_params['cdens'])
    k_params['knum'] = int(k_params['knum'])
    
    if len(kwargs_ext)>0:
        kwargs.update(kwargs_ext)
    kwargs_i = grid_df.loc[grid_df.grid_id==i,grid_keys].iloc[0].to_dict()  
    kwargs.update(kwargs_i)
    if 'type_complex' in kwargs.keys() and isinstance(kwargs['type_complex'],float):
        kwargs['type_complex'] = int(kwargs['type_complex'])

    # Create save_path for homann experiment
    if len(save_path)>0:
        save_path_i = save_path+f'grid_{i}/'
        sl.make_long_dir(save_path_i)
    else:
        save_path_i = ''
    grid_df.loc[grid_df.grid_id==i,'save_path'] = save_path_i

    # Run Homann experiments
    sim_data = run_homann_exp_n(k_params,params_input,inputs,parallel=parallel_exp,save_path=save_path_i,kwargs=kwargs)

    if comp_fit or comp_corr:
        # Load experimental data
        homann_data = load_exp_homann(cluster=cluster)

        if comp_fit:
            # Fit simulated data to experimental data
            _, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = fit_homann_exp(sim_data,homann_data,coef_steady=True,regr_meas='score',save_path=save_path_i)
            grid_df.loc[grid_df.grid_id==i,'mse_comb'] = mse_comb
            grid_df.loc[grid_df.grid_id==i,'mse_tem'] = mse_tem
            grid_df.loc[grid_df.grid_id==i,'mse_trec'] = mse_trec
            grid_df.loc[grid_df.grid_id==i,'mse_tmem'] = mse_tmem
            grid_df.loc[grid_df.grid_id==i,'mse_steady'] = mse_steady
            grid_df.loc[grid_df.grid_id==i,'coef'] = coef
            grid_df.loc[grid_df.grid_id==i,'shift'] = shift

        if comp_corr:
            # Compute correlation between simulated and experimental data
            corr_comb, [corr_tem, corr_trec, corr_tmem, corr_steady] = corr_homann_exp(sim_data,homann_data,regr_meas='score',corr_type='pearson',save_path=save_path_i)
            grid_df.loc[grid_df.grid_id==i,'corr_comb'] = corr_comb
            grid_df.loc[grid_df.grid_id==i,'corr_tem'] = corr_tem
            grid_df.loc[grid_df.grid_id==i,'corr_trec'] = corr_trec
            grid_df.loc[grid_df.grid_id==i,'corr_tmem'] = corr_tmem
            grid_df.loc[grid_df.grid_id==i,'corr_steady'] = corr_steady
        
        del homann_data
        
    # Save + print progress
    print(f'Done with grid point {i}/{max(grid_df.grid_id.values)}.')
    print(f'Parameters: {grid_df.loc[grid_df.grid_id==i].iloc[0].to_dict()}.')
    print(f'Results saved in {save_path_i}.')

    # Delete variables
    del sim_data
    
    return i

############################################################################################################
#               Function to run grid search over parameters of the simple cell kernel novelty model        #
############################################################################################################
def simulate_grid(grid,k_params_ext={},num_sim=50,params_input=None,init_seed=12345,parallel_exp=False,parallel_grid=False,save_path='',comp_fit=False,comp_corr=False,cluster=False,kwargs_ext={},resume=False,input_corrected=True,input_sequence_mode='sep',num_cpu=1): # grid is a dictionary with parameter ranges

    if parallel_grid and parallel_exp:
        raise ValueError('Cannot run grid search and experiments in parallel.')

    # Generate inputs for all simulations
    seed_input, inputs, params_input = generate_input(params_input,num_sim,init_seed,input_corrected=input_corrected,input_sequence_mode=input_sequence_mode)

    # Default kernel parameters
    k_params = {'knum':             8, 
                'cdens':            8,
                'k_alph':           0.1,
                'flr':              True,
                'k_type':           'triangle',
                'ksig':             1,
                'kcenter':          1,
                'conv':             True,
                'gabor_seed':       12345,
                'gabor_sampling':   'equidist'
                }
    
    kwargs = {'no_simple_cells': False,
              'no_complex_cells': False,
              'mode_complex': 'sum', # 'sum' or 'mean'
              'type_complex': [4],
              'num_complex': 1, #
              'debug': False}

    # Create dataframe to store results
    grid_ll = list(itertools.product(*[grid[k] for k in grid.keys()]))
    grid_df = pd.DataFrame(grid_ll,columns=grid.keys())
    
    if 'ratio_complex' in grid.keys() and 'type_complex' in grid.keys():
        # Compute the maximum number of complex cells
        max_complex = [np.sum(np.array([binom(n,k) for k in range(2,n+1)])) for n in grid['type_complex']]
        max_complex = dict(zip(grid['type_complex'],max_complex))

        # Adjust ratio if not feasible
        grid_df['num_complex'] = grid_df.ratio_complex *grid_df.type_complex
        grid_df['num_complex'] = [min(grid_df.num_complex[i],max_complex[grid_df.type_complex[i]]) for i in range(len(grid_df))]
        grid_df.drop(labels='ratio_complex',axis=1,inplace=True)
        grid_df.drop_duplicates(inplace=True)
        
    grid_df['grid_id'] = np.arange(len(grid_df))
    grid_df['save_path'] = [''] * len(grid_df)
    grid_keys = list(grid.keys())
    if 'ratio_complex' in grid_keys: grid_keys.remove('ratio_complex'); grid_keys.append('num_complex')
    if comp_fit:
        grid_df['mse_comb'] = [0] * len(grid_df)
        grid_df['mse_tem'] = [0] * len(grid_df)
        grid_df['mse_trec'] = [0] * len(grid_df)
        grid_df['mse_tmem'] = [0] * len(grid_df)
        grid_df['mse_steady'] = [0] * len(grid_df)
        grid_df['coef'] = [0] * len(grid_df)
        grid_df['shift'] = [0] * len(grid_df)
    if comp_corr:
        grid_df['corr_comb'] = [0] * len(grid_df)
        grid_df['corr_tem'] = [0] * len(grid_df)
        grid_df['corr_trec'] = [0] * len(grid_df)
        grid_df['corr_tmem'] = [0] * len(grid_df)
        grid_df['corr_steady'] = [0] * len(grid_df)

    sl.make_long_dir(save_path)
    list_done = []

    if resume and os.path.exists(save_path+'grid.csv'):
        # Check if data for grid points already exists
        for i in grid_df.grid_id.values:
            if os.path.exists(save_path+f'grid_{i}/data_all_l.csv') and os.path.exists(save_path+f'grid_{i}/data_all_lp.csv') and os.path.exists(save_path+f'grid_{i}/data_all_m.csv') and os.path.exists(save_path+f'grid_{i}/data_all_m_steady.csv'):
                list_done.append(i)
        # Reduce grid to new grid points
        new_ids = [i for i in grid_df.grid_id.values if i not in list_done]
        grid_df = grid_df[grid_df.grid_id.isin(new_ids)]
    else:
        file_done = save_path+'done.txt'    
        grid_df.to_csv(save_path+'grid.csv',index=False)

    if parallel_grid:
        pool = mp.Pool(num_cpu)
        jobs = [pool.apply_async(simulate_grid_point,args=(i,k_params_ext,k_params,grid_df,grid_keys,save_path,params_input,inputs,parallel_exp,comp_fit,comp_corr,cluster,kwargs,kwargs_ext)) for i in grid_df.grid_id.values]
        list_done = [r.get() for r in jobs]
        pool.close()
        pool.join()
        with open(file_done,'a') as f:
            for i in list_done:
                f.write(f'{i}\n')
    
    else:
        for i in grid_df.grid_id.values:
            i = simulate_grid_point(i,k_params_ext,k_params,grid_df,grid_keys,save_path,params_input,inputs,parallel_exp,comp_fit,comp_corr,cluster,kwargs,kwargs_ext)
            list_done.append(i)
            with open(file_done,'a') as f:
                f.write(f'{i}\n')
    
    # Save grid dataframe (with paths)
    if resume and os.path.exists(save_path+'grid.csv'):
        grid_old = pd.read_csv(save_path+'grid.csv')
        grid_df = pd.concat([grid_old[grid_old.grid_id.isin(list_done)],grid_df])
    grid_df.to_csv(save_path+'grid.csv',index=False)
        
    return grid_df

############################################################################################################
#               Function to run grid search based on config file                                           #
############################################################################################################
def run_grid_search_config(config):

    grid            = config['grid']
    k_params_ext    = config['k_params_ext'] if 'k_params_ext' in config else {}
    num_sim         = config['num_sim'] if 'num_sim' in config else 50
    params_input    = config['params_input'] if 'params_input' in config else None
    init_seed       = config['init_seed'] if 'init_seed' in config else 12345
    parallel_exp    = config['parallel_exp'] if 'parallel_exp' in config else False
    parallel_grid   = config['parallel_grid'] if 'parallel_grid' in config else False
    save_path       = config['save_path'] if 'save_path' in config else ''
    comp_fit        = config['comp_fit'] if 'comp_fit' in config else False
    comp_corr       = config['comp_corr'] if 'comp_corr' in config else False
    cluster         = config['cluster'] if 'cluster' in config else False
    kwargs          = config['kwargs'] if 'kwargs' in config else {}
    resume          = config['resume'] if 'resume' in config else False
    input_corrected     = config['input_corrected'] if 'input_corrected' in config else True
    input_sequence_mode = config['input_sequence_mode'] if 'input_sequence_mode' in config else 'sep'
    num_cpu         = config['num_cpu'] if 'num_cpu' in config else 1

    simulate_grid(grid,
                  k_params_ext=k_params_ext,
                  num_sim=num_sim,
                  params_input=params_input,
                  init_seed=init_seed,
                  parallel_exp=parallel_exp,
                  parallel_grid=parallel_grid,
                  save_path=save_path,
                  comp_fit=comp_fit,
                  comp_corr=comp_corr,
                  cluster=cluster,
                  kwargs_ext=kwargs,
                  resume=resume,
                  input_corrected=input_corrected,
                  input_sequence_mode=input_sequence_mode,
                  num_cpu=num_cpu)
    
    print('Done with grid search.')


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument(
            '-c',
            '--config',
            dest='config_file',
            type=str,
            default=None,
            help='config file',
        )
    args = parser.parse_args()
    print('Successfully loaded config file.\n')

    if args.config_file:
        config = json.load(open(args.config_file))
    else: 
        config = json.load(open('./src/scripts/gabor_kernels/grid_search_manual/configs_set1_complex_cells/complex_cells_triangle_adj_w3_G-40.json'))
    print(config)

    run_grid_search_config(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json


        

    

                     




