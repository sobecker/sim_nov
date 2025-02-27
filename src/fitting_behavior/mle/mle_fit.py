from argparse import ArgumentParser
import json
import os
import numpy as np
import pandas as pd
import timeit

import sys
import utils.saveload as sl
import fitting_behavior.optimization.base_params_opt as bpo
from models.mb_agent.mb_surnor import import_params_surnor
from fitting_behavior.mle.mle_fit_sequential import mle_fit, preprocess_micedata
from fitting_behavior.mle.mle_fit_parallel import mle_fit_parallel

### Run test function for different test cases ###
def run_mle_fit(config):

    start = timeit.default_timer()
    
    data_type       = config['data_type']
    data_folder     = config['data_folder']
    data_path_type  = config['data_path_type']
    comb_type       = config['comb_type']
    var_name        = config['var_name']
    kwargs          = config['kwargs'] 
    alg_type_sim    = config['alg_type_sim'] if 'alg_type_sim' in config.keys() else config['alg_type']
    alg_type_fit    = config['alg_type_fit'] if 'alg_type_fig' in config.keys() else config['alg_type']
    save_folder     = config['save_name']+'_'+kwargs['opt_method']
    verbose         = config['verbose']
    # Optional parameters
    rand_start      = config['rand_start'] if 'rand_start' in config.keys() else 0  # number of random starts of mle (with random initial values)
    seed            = config['seed'] if 'seed' in config.keys() else 12345          # seed for random starts of mle (with random initial values)
    save_path       = config['save_path'] if 'save_path' in config.keys() else ''   # optional path to save mle results (e.g. used for parameter recovery)
    level           = config['level'] if 'level' in config.keys() else ''           # level of granularity/hierarchy (for hnor and hnac models)
    parallel        = config['parallel'] if 'parallel' in config.keys() else False
    overwrite       = config['overwrite'] if 'overwrite' in config.keys() else True
    local           = config['local'] if 'local' in config.keys() else True

    if parallel:    mle_fun = mle_fit_parallel
    else:           mle_fun = mle_fit

    # Set directory to save data
    if save_path=='': dir_save = f'MLE_results/Fits/{"MultiStart/" if rand_start>0 else "SingleRun/"}{save_folder}/'
    else:             dir_save = f'{save_path}{"" if save_path[-1]=="/" else "/"}{save_folder}/'

    # Set overall path
    if data_path_type=='auto':
        if local:  
            base_path = '/Users/sbecker/Projects/RL_reward_novelty/' 
            dir_save = f'{base_path}data/{dir_save}'
        else:   
            base_path = '/lcncluster/becker/RL_reward_novelty/'    
            dir_save = f'{base_path}data/{dir_save}'

    # Check whether file already exists and set bools (which cases to fit)
    fit_sep = True
    fit_app = True
    if os.path.isdir(dir_save) and not overwrite:
        if os.path.isfile(dir_save / f'{save_folder}_sep.csv'): fit_sep = False
        if os.path.isfile(dir_save / f'{save_folder}_app.csv'): fit_app = False
    if comb_type=='' and not fit_sep and fit_app:   comb_type = 'app'
    elif comb_type=='' and not fit_app and fit_sep: comb_type = 'sep'
    # Check if we need to fit any
    fit_any = False
    if comb_type=='' and fit_sep and fit_app:   fit_any = True
    elif comb_type=='sep' and fit_sep:          fit_any = True
    elif comb_type=='app' and fit_app:          fit_any = True

    if fit_any: 
        sl.make_long_dir(dir_save)
        log_file = open(dir_save+f'log{"_" if len(comb_type)>0 else ""}{comb_type}.txt', 'w') # create log file
        with open(dir_save+f'config{"_" if len(comb_type)>0 else ""}{comb_type}.json', 'w') as fc: json.dump(config, fc) # save config file for reproducibility
        # sl.saveCodeVersion(dir_save,file_cv=f'code_version{"_" if len(comb_type)>0 else ""}{comb_type}.txt') # save code version

        # Load base parameters
        update_type = 'fix' if 'leaky' in alg_type_fit else None
        notrace = 'notrace' in alg_type_fit
        center  = 'center' in alg_type_fit
        center_type = alg_type_fit.split('center-')[1].split('_')[0].split('-')[0] if center else ''
        if 'nor' in alg_type_fit:
            if 'hnor' in alg_type_fit:
                params = bpo.baseparams_h1mbnor_eps1([level],notrace=notrace,center=center,center_type=center_type,update_type=update_type)
            else:
                params = bpo.base_params_mbnortree_exp.copy()
            params_surnor = import_params_surnor(path=f'{base_path}src/mbnor/')
            params.update(params_surnor)
        elif 'nac' in alg_type_fit:
            if 'hnac' in alg_type_fit:
                params = bpo.baseparams_h1nac_eps1([level],notrace=notrace,center=center,center_type=center_type,update_type=update_type)
            else: 
                params = bpo.base_params_nACtree.copy()
            if 'gv' in alg_type_fit:
                params['agent_types'] = ['gn']
            if 'goi' in alg_type_fit: 
                params['ntype'] = 'hN-k'
            if 'kpop' in alg_type_fit:      params['ntype'] = 'N-kpop'
            elif 'kmix' in alg_type_fit:    params['ntype'] = 'N-kmix'
            elif 'oi-only' in alg_type_fit: params['agent_type'] = ['oi']
        elif 'hybrid' in alg_type_fit:
            if 'hhybrid' in alg_type_fit:
                mb_type = config['hyb_type'][0] if 'hyb_type' in config.keys() else 'hnor'
                mf_type = config['hyb_type'][1] if 'hyb_type' in config.keys() else 'hnac-gn'
                params = bpo.baseparams_all_hhybrid_comb(mb_type,mf_type,levels=[level],notrace=notrace,center=center,center_type=center_type,path_surnor=f'{base_path}src/mbnor/',update_type=update_type)
            else:
                params = bpo.baseparams_hybrid_comb()
        # Load mice/sim data (to be fitted)      
        if data_type=='mice':
            UnrewNames  = ['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
            RewNames    = ['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
            AllNames    = RewNames+UnrewNames
            P = params['P']
            d = []
            for i in range(len(AllNames)):
                # dir = sl.get_datapath().replace('data','ext_data')+'Rosenberg2021/'
                dir = f'{base_path}ext_data/Rosenberg2021/'
                file=f'{AllNames[i]}-stateseq_UntilG.pickle'
                df_i = preprocess_micedata(dir,file,P,subID=AllNames[i],epi=0)
                d.append(df_i)
            all_data = pd.concat(d,ignore_index=True)
        else:
            if data_path_type=='auto':  
                # dir_data = sl.get_datapath()+data_folder    
                dir_data = f'{base_path}{data_folder}'    
            else:                       
                dir_data = data_folder
            # if 'hnac' in alg_type_sim:
            #     params = bpo.baseparams_h1nac_eps1([level])
            #     params1 = sl.load_sim_params(dir_data)
            #     params.update(params1)
            # elif 'hybrid' in alg_type_sim:
            #     params_mb = sl.load_sim_params(dir_data,file_params='mb_params.pickle')
            #     params_mf = sl.load_sim_params(dir_data,file_params='mf_params.pickle')
            #     params = bpo.comb_params(params_mb,params_mb,params_mf)
            # else:
            #     params      = sl.load_sim_params(dir_data)
            # Load simulation data (to be fitted)
            if 'hybrid' in alg_type_sim: all_data = sl.load_sim_data(dir_data,file_data='mf_data_basic.pickle') # note: only states and actions are used for fitting, these are the same for mf_data and mb_data
            else:                        all_data = sl.load_sim_data(dir_data)
            # subID_lim = 2                                       # only for testing; comment for real run
            # all_data = all_data.loc[all_data.subID<subID_lim]   # only for testing; comment for real run
        all_data = all_data.loc[all_data.epi==0]
        print(f'Number of data samples fitted: {len(all_data)}.')

        # Compute MLE with multiple random starts
        if not 'x0' in kwargs.keys(): raise ValueError('Please specify initial values for the parameter estimates.')
        if not 'bounds' in kwargs.keys(): raise ValueError('Please specify bounds for the parameter estimates.')
        if rand_start>0:
            # Generate random initial values for all variables
            rng     = np.random.default_rng(seed)
            lows    = np.array([kwargs['bounds'][i][0] for i in range(len(kwargs['bounds']))])
            highs   = np.array([kwargs['bounds'][i][1] for i in range(len(kwargs['bounds']))])
            x0_arr  = rng.uniform(low=lows,high=highs,size=(rand_start,len(lows))) 
        if comb_type=='':
            # Compute both sep. and app. MLE estimates
            if rand_start>0: # with random starts for initial parameters x0
                res_all     = []
                resapp_all  = []
                for r in range(rand_start):  
                    print(f'Starting random initial condition {r+1}/{rand_start}.')
                    sl.write_log(log_file,f'Starting random initial condition {r+1}/{rand_start}.')
                    x0_list      = list(x0_arr[r,:])
                    kwargs['x0'] = x0_list
                    res     = mle_fun(all_data,params,var_name,alg_type_fit,'sep',kwargs,verbose=verbose,log_file=log_file)
                    resapp  = mle_fun(all_data,params,var_name,alg_type_fit,'app',kwargs,verbose=verbose,log_file=log_file)
                    res['rand_it']    = r*np.ones(len(res),dtype=int)
                    resapp['rand_it'] = r*np.ones(len(resapp),dtype=int)
                    res_all.append(res)
                    resapp_all.append(resapp)
                res    = pd.concat(res_all,ignore_index=True)
                resapp = pd.concat(resapp_all,ignore_index=True)
                # Save initial values
                x0_df = pd.DataFrame(x0_arr,columns=var_name)
                x0_df.to_csv(dir_save+'x0.csv')
                x0_df.to_pickle(dir_save+'x0.pickle')
            else: # with user-defined x0
                res    = mle_fun(all_data,params,var_name,alg_type_fit,'sep',kwargs,verbose=verbose,log_file=log_file)
                resapp = mle_fun(all_data,params,var_name,alg_type_fit,'app',kwargs,verbose=verbose,log_file=log_file)
            save_name = f'{save_folder}_sep'
            print(dir_save)
            res.to_csv(dir_save+save_name+'.csv')
            res.to_pickle(dir_save+save_name+'.pickle')
            save_name = f'{save_folder}_app'
            resapp.to_csv(dir_save+save_name+'.csv')
            resapp.to_pickle(dir_save+save_name+'.pickle')
        else:
            # Compute MLE estimate with user-defined comb_type (sep. or app.)
            if rand_start>0: # with random starts for initial parameters x0
                res_all     = []
                for r in range(rand_start): 
                    print(f'Starting random initial condition {r+1}/{rand_start}.')
                    sl.write_log(log_file,f'Starting random initial condition {r+1}/{rand_start}.') 
                    x0_list      = list(x0_arr[r,:])
                    kwargs['x0'] = x0_list
                    res = mle_fun(all_data,params,var_name,alg_type_fit,comb_type,kwargs,verbose=verbose,log_file=log_file)
                    res['rand_it'] = r*np.ones(len(res),dtype=int)
                    res_all.append(res)
                res = pd.concat(res_all,ignore_index=True)
                # Save initial values
                x0_df = pd.DataFrame(x0_arr,columns=var_name)
                x0_df.to_csv(dir_save+'x0.csv')
                x0_df.to_pickle(dir_save+'x0.pickle')
            else: # with user-defined x0
                res = mle_fun(all_data,params,var_name,alg_type_fit,comb_type,kwargs,verbose=verbose,log_file=log_file)
            save_name = f'{save_folder}_{comb_type}'
            res.to_csv(dir_save+save_name+'.csv')
            res.to_pickle(dir_save+save_name+'.pickle')

        end = timeit.default_timer()
        print(f'Finished running MLE fit (mice data) in {end-start} s.\n')
        sl.write_log(log_file,f'Finished running MLE fit (mice data) in {end-start} s.\n')
        log_file.close()
    else:
        print('No data fitted since overwrite option disabled.\n')

### Run MLE fit for input arguments ### 
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
        # config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_nor-opt_Nelder-Mead-sep_local.json'))
        # config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_nor-opt_Nelder-Mead-app_local.json'))
        # config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_nor-mice_Nelder-Mead-sep_local.json'))
        #config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_nor-mice_Nelder-Mead-app_local.json'))
        #config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_nac-opt_Nelder-Mead-sep.json'))
        # config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_nac-mice_Nelder-Mead-app.json'))
        #config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle-maxit_hhybrid-l1-mice_Nelder-Mead.json'))
        config = json.load(open('./src/scripts/MLE/mle_fit_configs/mle_leaky_hnor_center-triangle-l5-mice_Nelder-Mead.json'))
    print(type(config))
    print(config)

    run_mle_fit(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json

    #For tests of multistart (cluster):
    #python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-opt_Nelder-Mead-app_multi.json
    #python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nac-opt_Nelder-Mead-sep_multi.json

