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


def run_mle_fit(config):

    start = timeit.default_timer()
    
    # Load config file + set params ##################################################
    data_type       = config['data_type']
    comb_type       = config['comb_type']
    var_name        = config['var_name']
    kwargs          = config['kwargs'] 
    alg_type_sim    = config['alg_type_sim'] if 'alg_type_sim' in config.keys() else config['alg_type']
    alg_type_fit    = config['alg_type_fit'] if 'alg_type_fig' in config.keys() else config['alg_type']
    save_name       = config['save_name']
    verbose         = config['verbose']

    # Optional parameters
    data_folder     = config['data_folder'] if 'data_folder' in config.keys() else ''   # optional path to data that is fitted (e.g. for model recovery)
    save_path       = config['save_path'] if 'save_path' in config.keys() else ''       # optional path to save mle results (e.g. used for parameter recovery)
    overwrite       = config['overwrite'] if 'overwrite' in config.keys() else True     # overwrite existing files
    rand_start      = config['rand_start'] if 'rand_start' in config.keys() else 0      # number of random starts of mle (with random initial values)
    seed            = config['seed'] if 'seed' in config.keys() else 12345              # seed for random starts of mle (with random initial values)
    level           = config['level'] if 'level' in config.keys() else ''               # level of granularity/hierarchy (for hnor and hnac models)
    parallel        = config['parallel'] if 'parallel' in config.keys() else False

    # Set mle function
    if parallel:    mle_fun = mle_fit_parallel
    else:           mle_fun = mle_fit

    # Set paths
    base_path = sl.get_rootpath() 

    if data_folder=='':
        if data_type=='mice':
            dir_data  = base_path / 'ext_data' / 'Rosenberg2021'
        else:
            raise ValueError('Please specify a data folder for artificial data.')
    else:
        dir_data  = base_path / data_folder

    if save_path=='':
        dir_save = base_path / 'data' / 'mle_results' / 'fits' / {"multistart" if rand_start>0 else "singlerun"} / save_name
    else:
        dir_save  = base_path / 'data' / save_path / save_name 
    sl.make_long_dir(dir_save)

    # Check whether file already exists and set bools (which cases to fit)
    fit_sep = True
    fit_app = True
    if os.path.isdir(dir_save) and not overwrite:
        if os.path.isfile(dir_save / f'{save_name}_sep.csv'): fit_sep = False
        if os.path.isfile(dir_save / f'{save_name}_app.csv'): fit_app = False
    if comb_type=='' and not fit_sep and fit_app:   comb_type = 'app'
    elif comb_type=='' and not fit_app and fit_sep: comb_type = 'sep'

    # Check if we need to fit any
    fit_any = False
    if comb_type=='' and fit_sep and fit_app:   fit_any = True
    elif comb_type=='sep' and fit_sep:          fit_any = True
    elif comb_type=='app' and fit_app:          fit_any = True


    # Fit data #######################################################################
    if fit_any: 
        # Save config file, log file, and code version
        log_file = open(dir_save / f'log{"_" if len(comb_type)>0 else ""}{comb_type}.txt', 'w') # create log file
        with open(dir_save / f'config{"_" if len(comb_type)>0 else ""}{comb_type}.json', 'w') as fc: 
            json.dump(config, fc) # save config file for reproducibility
        sl.saveCodeVersion(dir_save,file_cv=f'code_version{"_" if len(comb_type)>0 else ""}{comb_type}.txt') # save code version

        # Initialize model parameters to default values
        update_type = 'fix' if 'leaky' in alg_type_fit else None
        notrace     = 'notrace' in alg_type_fit
        center      = 'center' in alg_type_fit
        center_type = alg_type_fit.split('center-')[1].split('_')[0].split('-')[0] if center else ''
        
        if 'nor' in alg_type_fit:
            if 'hnor' in alg_type_fit:  # MB agent with snov
                params = bpo.baseparams_h1mbnor_eps1([level],notrace=notrace,center=center,center_type=center_type,update_type=update_type)
            else:                       # MB agent with cnov
                params = bpo.base_params_mbnortree_exp.copy()
            params_surnor = import_params_surnor(path = base_path / 'src' / 'models' / 'mb_agent')
            params.update(params_surnor)

        elif 'nac' in alg_type_fit:
            if 'hnac' in alg_type_fit:      # MF agent with snov
                params = bpo.baseparams_h1nac_eps1([level],notrace=notrace,center=center,center_type=center_type,update_type=update_type)
            else:                           # MF agent with cnov
                params = bpo.base_params_nACtree.copy()

            if 'gv' in alg_type_fit:        # Use granular critic (otherwise: use normal critic; agent_types=='n')
                params['agent_types'] = ['gn']
            elif 'oi-only' in alg_type_fit: # Use OI critic
                params['agent_type'] = ['oi'] 

            if 'goi' in alg_type_fit:       # Use modified novelty signal: subtract mean across all component novelties from currently active component novelty
                params['ntype'] = 'hN-k' 
            if 'kpop' in alg_type_fit:      # Use modified novelty signal: subtract moving mean of most recent novelty signals (component-specific) from novelty  
                params['ntype'] = 'N-kpop'
            elif 'kmix' in alg_type_fit:    # Use modified novelty signal: subtract moving mean of most recent novelty signals (across all components) from novelty
                params['ntype'] = 'N-kmix'

        elif 'hybrid' in alg_type_fit:
            if 'hhybrid' in alg_type_fit:   # Hybrid agent with snov
                mb_type = config['hyb_type'][0] if 'hyb_type' in config.keys() else 'hnor'
                mf_type = config['hyb_type'][1] if 'hyb_type' in config.keys() else 'hnac-gn'
                params = bpo.baseparams_all_hhybrid_comb(mb_type,mf_type,levels=[level],notrace=notrace,center=center,center_type=center_type,path_surnor= base_path / 'src' / 'models' / 'mb_agent',update_type=update_type)
            else:                           # Hybrid agent with cnov
                params = bpo.baseparams_hybrid_comb()

        # Load mice or artificial data to be fitted      
        if data_type=='mice':   # Mice data
            UnrewNames  = ['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
            RewNames    = ['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
            AllNames    = RewNames+UnrewNames
            P = params['P']
            d = []
            for i in range(len(AllNames)):
                file = f'{AllNames[i]}-stateseq_UntilG.pickle'
                df_i = preprocess_micedata(dir_data,file,P,subID=AllNames[i],epi=0)
                d.append(df_i)
            all_data = pd.concat(d,ignore_index=True)
        else:                   # Artificial data
            if 'hybrid' in alg_type_sim: 
                all_data = sl.load_sim_data(dir_data,file_data='mf_data_basic.pickle') # note: only states and actions are used for fitting, these are the same for mf_data and mb_data
            else:                        
                all_data = sl.load_sim_data(dir_data)
            # subID_lim = 2                                       # only for testing; comment for real run
            # all_data = all_data.loc[all_data.subID<subID_lim]   # only for testing; comment for real run
        all_data = all_data.loc[all_data.epi==0]
        print(f'Number of data samples fitted: {len(all_data)}.')

        # Check that all parameters for fitting are set
        if not 'x0' in kwargs.keys(): 
            raise ValueError('Please specify initial values for the parameter estimates.')
        if not 'bounds' in kwargs.keys(): 
            raise ValueError('Please specify bounds for the parameter estimates.')
        
        # Generate initial conditions if running with multiple random starts
        if rand_start>0: 
            rng     = np.random.default_rng(seed)
            lows    = np.array([kwargs['bounds'][i][0] for i in range(len(kwargs['bounds']))])
            highs   = np.array([kwargs['bounds'][i][1] for i in range(len(kwargs['bounds']))])
            x0_arr  = rng.uniform(low=lows,high=highs,size=(rand_start,len(lows))) 

        # Compute MLE 
        if comb_type=='':       # for both sep. and app. fitting
            if rand_start>0:        # with random starts for initial parameters x0
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
                x0_df = pd.DataFrame(x0_arr,columns=var_name)
                x0_df.to_csv(dir_save+'x0.csv')
                x0_df.to_pickle(dir_save+'x0.pickle')
            else:                   # with user-defined x0
                res    = mle_fun(all_data,params,var_name,alg_type_fit,'sep',kwargs,verbose=verbose,log_file=log_file)
                resapp = mle_fun(all_data,params,var_name,alg_type_fit,'app',kwargs,verbose=verbose,log_file=log_file)
            print(dir_save)
            res.to_csv(dir_save / f'{save_name}_sep.csv')
            res.to_pickle(dir_save / f'{save_name}_sep.pickle')
            resapp.to_csv(dir_save / f'{save_name}_app.csv')
            resapp.to_pickle(dir_save / f'{save_name}_app.pickle')
        else:                   # for either sep. or app. fitting (user-defined)
            if rand_start>0:        # with random starts for initial parameters x0
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
                x0_df = pd.DataFrame(x0_arr,columns=var_name)
                x0_df.to_csv(dir_save+'x0.csv')
                x0_df.to_pickle(dir_save+'x0.pickle')
            else:                   # with user-defined x0
                res = mle_fun(all_data,params,var_name,alg_type_fit,comb_type,kwargs,verbose=verbose,log_file=log_file)
            res.to_csv(dir_save / f'{save_name}_{comb_type}.csv')
            res.to_pickle(dir_save / f'{save_name}_{comb_type}.pickle')
        
        # Stop timer and save log file
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

