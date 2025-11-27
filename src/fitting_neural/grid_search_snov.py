import os

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.special import binom
import itertools

import json
from argparse import ArgumentParser

import utils.saveload as sl
import fitting_neural.simulate_data as sd
import models.snov.run_gabor_knov as gknov
import models.snov.run_gabor_knov_complex as gknovc
import fitting_neural.create_homann_input as h_in
import fitting_neural.load_homann_data as load_homann
import fitting_neural.fit_homann as fit_homann


## This script runs a grid search over the parameters of the simple cell kernel novelty model, simulating several samples of the Homann experiments for each parameter combination. ##

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
            idata,_,_,_ = gknovc.run_gabor_knov_withparams_flr2(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
        else:
            idata,_,_,_,_,_ = gknovc.run_gabor_knov_withparams_leaky2(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
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
    record = kwargs['record'] if 'record' in kwargs else 'minimal'
    # Simulate model
    all_data = []
    for j in range(len(ivec)):
        if flip and (kmat_seq_flipped is not None):     kmatj = kmat_seq_flipped[j]
        else:                                           kmatj = None
        stim_type = itype[j] if debug else None
        if flr:
            idata,_,_,_ = gknovc.run_gabor_knov_withparams_flr2(ivec[j],k_params,idx=idx,flip=flip,kmat_seq_flipped=kmatj,stim_type=stim_type,kwargs=kwargs)
        else:
            idata,icomponents,_,iweights,iresp,itemp = gknovc.run_gabor_knov_withparams_leaky2(ivec[j],k_params,idx=idx,flip=flip,kmat_seq_flipped=kmatj,stim_type=stim_type,kwargs=kwargs)
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
            idata,_,_,_ = gknovc.run_gabor_knov_withparams_flr2(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
        else:
            idata,_,_,_,_,_ = gknovc.run_gabor_knov_withparams_leaky2(ivec[j],k_params,idx=idx,stim_type=stim_type,kwargs=kwargs)
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
    k_params_components = gknovc.init_gabor_knov(gnum_complex=k_params['knum'],       # complex cells: number of different preferred orientations  
                                                    ctype=kwargs['type_complex'],       # complex cells: how many simple cells are pooled
                                                    dfreq=1.5,                           # complex cells: factor between simple cell frequencies (fixed at default)
                                                    cvar='frequency',                    # complex cells: feature of simple cells to be varied (fixed at default)
                                                    k_type=k_params['k_type'],           # simple cells: component shape
                                                    ksig=k_params['ksig'],               # simple cells: component width
                                                    kcenter=k_params['kcenter'],         # simple cells: component center (in similarity space)
                                                    seed=k_params['gabor_seed'],         # simple cells: seed for component generation 
                                                    rng=None,
                                                    mask=False,                          # simple cells: Gaussian mask if True (fixed at default False)
                                                    sampling=k_params['gabor_sampling'], # simple cells: sampling of preferred features (e.g. equidistant, random, ...)
                                                    adj_w=params_input['adj_w'],
                                                    adj_f=params_input['adj_f'],
                                                    alph_adj=params_input['alph_adj'],
                                                    conv=k_params['conv'],               # simple cells: convolutional simple cells if True
                                                    cdens=k_params['cdens'],             # simple cells: convolutional stride length 
                                                    parallel=False,                      # simple cells: parallel computation of activation matrix if True   
                                                    debug=kwargs['debug'] if 'debug' in kwargs else False
                                                )
    k_params.update(k_params_components)
    if len(kwargs)>0 and 'flr' in kwargs.keys():
        k_params['flr'] = kwargs['flr']

    append_mode = kwargs['append_mode'] if 'append_mode' in kwargs else False
    start_id    = kwargs['start_id'] if 'start_id' in kwargs else 0
    
    # Run Homann experiments
    if parallel:
        num_pool = mp.cpu_count()
        pool = mp.Pool(num_pool)
        jobs = [pool.apply_async(run_homann_exp,args=(k_params,params_input,si,start_id+i,True,kwargs)) for i, si in enumerate(inputs)]
        data = [r.get() for r in jobs]
        pool.close()
        pool.join() 
    else:
        data = [run_homann_exp(k_params,params_input,si,start_id+i,kwargs=kwargs) for i, si in enumerate(inputs)]
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
    kwargs['cdens'] = int(kwargs['cdens'])
    kwargs['knum'] = int(kwargs['knum'])
    if 'type_complex' in kwargs.keys() and isinstance(kwargs['type_complex'],float):
        kwargs['type_complex'] = [int(kwargs['type_complex'])]
    if 'type_complex' in k_params.keys() and isinstance(k_params['type_complex'],float):
        k_params['type_complex'] = [int(k_params['type_complex'])]

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
        homann_data = load_homann.load_exp_homann(cluster=cluster)

        if comp_fit:
            # Fit simulated data to experimental data
            _, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = fit_homann.fit_homann_exp(sim_data,homann_data,coef_steady=True,regr_meas='score',save_path=save_path_i)
            grid_df.loc[grid_df.grid_id==i,'mse_comb'] = mse_comb
            grid_df.loc[grid_df.grid_id==i,'mse_mean'] = np.mean([mse_tem, mse_trec, mse_tmem, mse_steady])
            grid_df.loc[grid_df.grid_id==i,'mse_tem'] = mse_tem
            grid_df.loc[grid_df.grid_id==i,'mse_trec'] = mse_trec
            grid_df.loc[grid_df.grid_id==i,'mse_tmem'] = mse_tmem
            grid_df.loc[grid_df.grid_id==i,'mse_steady'] = mse_steady
            grid_df.loc[grid_df.grid_id==i,'coef'] = coef
            grid_df.loc[grid_df.grid_id==i,'shift'] = shift

        if comp_corr:
            # Compute correlation between simulated and experimental data
            corr_comb, [corr_tem, corr_trec, corr_tmem, corr_steady] = fit_homann.corr_homann_exp(sim_data,homann_data,regr_meas='score',corr_type='pearson',save_path=save_path_i)
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
                'alph_leak':        0.1,
                'eps':              1,
                'flr':              True,
                'k_type':           'triangle',
                'ksig':             1,
                'kcenter':          1,
                'conv':             True,
                'gabor_seed':       12345, # controls randomness in choice of Gabor receptive fields
                'gabor_sampling':   'equidist_fixed'
                }
    
    kwargs = {'no_simple_cells': False,
              'no_complex_cells': False,
              'mode_complex': 'sum', # 'sum' or 'mean'
              'type_complex': [4],
              'num_complex': 1, #
              'complex_seed': 12345, # controls randomness in choice of simple to complex cell pooling
              'debug': False}

    # Create dataframe to store results
    # grid_ll = list(itertools.product(*[grid[k] for k in grid.keys()]))
    # grid_df = pd.DataFrame(grid_ll,columns=grid.keys())
    grid_df = pd.DataFrame(grid)
    
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
        grid_df['mse_mean'] = [0] * len(grid_df)
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
        # config = json.load(open('./src/scripts/gabor_kernels/grid_search_new/configs/snov-complex-fr_set2_sep/job-0_test.json'))
        # config = json.load(open('./src/scripts/gabor_kernels/grid_search_new/configs/snov-complex-fr_set2_sep/job-0_test2.json'))
        # config = json.load(open('./src/scripts/gabor_kernels/grid_search_new/configs/2025_07_robustness_shape_width/snov-complex-fr_set1_sep/job-0-test.json'))
        config = json.load(open('./src/scripts/gabor_kernels/grid_search_new/configs/snov-simple-leaky_set1_sep/job-0-test.json'))

    print(config)

    run_grid_search_config(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json


        

    

                     




