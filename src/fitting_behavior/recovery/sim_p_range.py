import json
from argparse import ArgumentParser
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import fitting_behavior.optimization.auxiliary_opt as auxo
import fitting_behavior.optimization.base_params_opt as bpo
import utils.saveload as sl
from models.mb_agent.mb_surnor import import_params_surnor

# Create random parameters within range (app) or use all sep param combinations
def get_uni_params(comb_type, num_sim, seed, var_name, kwargs, dir_save, res, batchID=1):
    if comb_type=='sep':
        p_rand = res.mle_var.values.reshape((-1,len(var_name)))
    elif comb_type=='app':
        p_rand = np.ones((num_sim,1))*res.mle_var.values.reshape((1,len(var_name)))
    p_rand_df           = pd.DataFrame(p_rand,columns=var_name)
    p_rand_df['simID']  = np.arange(num_sim)

    p_rand_df.to_csv(dir_save / f'params_{comb_type}_overview{batchID}.csv')
    p_rand_df.to_pickle(dir_save / f'params_{comb_type}_overview{batchID}.pickle')
    return p_rand_df

# Create random parameters within range (app) or use all sep param combinations
def get_rand_params(comb_type, num_sim, seed, var_name, kwargs, dir_save, res):
    if comb_type=='sep':
        p_rand = res.mle_var.values.reshape((-1,len(var_name)))
    elif comb_type=='app':
        p_mle = res.mle_var.values
        rng     = np.random.default_rng(seed)
        if 'bounds' in kwargs.keys():
            lows    = np.array([kwargs['bounds'][i][0] for i in range(len(kwargs['bounds']))])
            highs   = np.array([kwargs['bounds'][i][1] for i in range(len(kwargs['bounds']))])
        elif 'range' in kwargs.keys():
            lows1    = np.array([p_mle[i]-kwargs['range'][i] for i in range(len(kwargs['range']))])
            highs1   = np.array([p_mle[i]+kwargs['range'][i] for i in range(len(kwargs['range']))])
            lows     = [kwargs['abs_bounds'][i][0] if lows1[i]<kwargs['abs_bounds'][i][0] else lows1[i] for i in range(len(lows1))]
            highs    = [kwargs['abs_bounds'][i][1] if highs1[i]>kwargs['abs_bounds'][i][1] else highs1[i] for i in range(len(highs1))]
        p_rand  = rng.uniform(low=lows,high=highs,size=(num_sim,len(lows))) 
    p_rand_df           = pd.DataFrame(p_rand,columns=var_name)
    p_rand_df['simID']  = np.arange(num_sim)
    p_rand_df.to_csv(dir_save / f'params_{comb_type}_overview.csv')
    p_rand_df.to_pickle(dir_save / f'params_{comb_type}_overview.pickle')
    return p_rand_df

# Get baseparams for sim
def get_sim_setup(alg_type,level=0,no_rew=False):
    notrace = 'notrace' in alg_type
    center  = 'center' in alg_type
    center_type = alg_type.split('center-')[1].split('_')[0].split('-')[0] if center else ''

    if 'nac' in alg_type:
        sim_fun     = auxo.run_optsim_nac_tree
        sim_fun_str = 'auxo.run_optsim_nac_tree'
        rec_type    = 'advanced1'
        if alg_type=='nac':                 # MF agent with cnov
            base_params = bpo.base_params_nACtree.copy()
        elif 'hnac' in alg_type:            # MF agent with snov
            base_params = bpo.get_baseparams_all_hnac(alg_type,[level],True,notrace=notrace,center=center,center_type=center_type)

    elif 'nor' in alg_type:
        sim_fun     = auxo.run_opt_sim_mbnor
        sim_fun_str = 'auxo.run_opt_sim_mbnor'
        rec_type    = 'basic'
        if alg_type=='nor':                 # MB agent with cnov
            base_params = bpo.base_params_mbnortree_exp.copy()
        elif 'hnor' in alg_type:            # MB agent with snov
            base_params = bpo.get_baseparams_all_hnor(alg_type,[level],True,notrace=notrace,center=center,center_type=center_type)
        params_surnor = import_params_surnor(path = sl.get_rootpath() / 'src' / 'models' / 'mb_agent')
        base_params.update(params_surnor)

    elif 'hybrid' in alg_type:
        sim_fun     = auxo.run_optsim_hybrid_tree
        sim_fun_str = 'auxo.run_optsim_hybrid_tree'
        rec_type    = 'basic'
        if 'hhybrid' in alg_type:           # Hybrid agent with snov
            base_params = bpo.baseparams_all_hhybrid_comb('hnor','hnac-gn',[level],True,notrace=notrace,center=center,center_type=center_type,path_surnor= sl.get_rootpath() / 'src' / 'models' / 'mb_agent')
        elif 'hybrid' in alg_type:          # Hybrid agent with cnov
            base_params = bpo.baseparams_hybrid_comb(path_surnor= sl.get_rootpath() / 'src' / 'models' / 'mb_agent')

    if no_rew:
        T = base_params['T'] 
        T[:] = 0
        base_params['T'] = T

    return sim_fun,sim_fun_str,rec_type,base_params

# Run simulation for each parameter set
def run_sim_setup_parallel(i,startID,start_seed,alg_type,params_i,sim_fun_str,agent_num,epi_num,base_params,rec_type,dir_save):
    sim_name = f'{alg_type}_sim{startID+i}'
    seeds_i = list(range(start_seed+i*(agent_num),start_seed+(i+1)*agent_num))
    eval(sim_fun_str)(params_i,agent_num,epi_num,base_params,sim_name=sim_name,rec_type=rec_type,dir_data=dir_save,seeds=seeds_i)
    return i

# Run simulation for each parameter set
def run_sim_setup(startID,start_seed,num_sim,log_file,alg_type,p_rand_df,sim_fun,sim_fun_str,agent_num,epi_num,base_params,rec_type,dir_save,parallel=False):
    print(f"Starting {num_sim} simulations for {alg_type}, mode: {'parallel' if parallel else 'sequential'}.")
    sl.write_log(log_file,f"Starting {num_sim} simulations for {alg_type}, mode: {'parallel' if parallel else 'sequential'}.")
    if parallel:
        params_all = [p_rand_df.iloc[i].to_dict() for i in range(num_sim)]
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply_async(run_sim_setup_parallel, args=(i,startID,start_seed,alg_type,params_all[i],sim_fun_str,agent_num,epi_num,base_params,rec_type,dir_save)) for i in range(num_sim)]
        done = [p.get() for p in results]
        pool.close()
        pool.join()
        print(f'Finished simulations: {done}.')
    else:
        for i in range(num_sim):
            if i%10==0:
                print(f'Starting simulation with parameter set {i}/{num_sim-1}.')
                sl.write_log(log_file,f'Starting simulation with parameter set {i+1}/{num_sim-1}.')
            sim_name = f'{alg_type}_sim{startID+i}'
            params_i    = p_rand_df.iloc[i].to_dict()
            seeds_i = list(range(start_seed+i*(agent_num),start_seed+(i+1)*agent_num))
            sim_fun(params_i,agent_num,epi_num,base_params,sim_name=sim_name,rec_type=rec_type,dir_data=dir_save,seeds=seeds_i)

# Run multiple simulations (app/sep) for different alg types
def run_multisim_range(config):
    comb_type   = config['comb_type'] # 'sep','app'
    num_sim     = config['num_sim'] # 1000
    startID     = config['startID'] if 'startID' in config.keys() else 0
    start_seed  = config['start_seed'] if 'start_seed' in config.keys() else 0
    agent_num   = config['agent_num'] # 20
    seed        = config['seed'] if 'seed' in config.keys() else 12345

    alg_type    = config['alg_type'] # 'nac','nor','hnac-gn','hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    maxit       = config['maxit'] if 'maxit' in config.keys() else False
    if 'hnor' in alg_type or 'hnac' in alg_type or 'hhybrid' in alg_type: 
        levels  = config['levels']
    var_name    = config['var_name'] 
    kwargs      = config['kwargs']
    epi_num     = config['epi'] if 'epi' in config.keys() else 1
    no_rew      = config['no_rew'] if 'no_rew' in config.keys() else False
    parallel    = config['parallel'] if 'parallel' in config.keys() else False  
    data_type   = 'mice'
    opt_type    = 'Nelder-Mead'

    # Set whether parameter recovery simulations are done with uniform (fitted) params or randomly drawn params
    uniparam    = config['uniparam'] if 'uniparam' in config.keys() else False
    if uniparam:    get_params = get_uni_params
    else:           get_params = get_rand_params
    
    # Prepare folder for saving
    base_path = sl.get_rootpath()
    dir_save  = base_path / 'data' / 'recovery' / f'simdata{"_uniparam" if uniparam else ""}' / f'{alg_type}_{comb_type}{"_norew" if no_rew else ""}'
    sl.make_long_dir(dir_save)

    # Get batch ID
    batchID = 1
    while os.path.exists(dir_save / f'log{batchID}.txt') and batchID<100:
        batchID += 1

    # Save metadata
    log_file = open(dir_save / f'log{batchID}.txt', 'w')                # create log file
    with open(dir_save / f'config{batchID}.json', 'w') as fc: 
        json.dump(config, fc)                                           # save config file for reproducibility
    sl.saveCodeVersion(dir_save,base_path,file_cv=f'code_version{batchID}.txt')   # save code version

    # Generate + save parameter sets tp be simulated
    dir_load = base_path / 'data'

    if 'hnor' in alg_type or 'hnac' in alg_type or 'hhybrid' in alg_type:       # for similarity-based agents
        res = []
        for i in range(len(levels)):
            dir_save_i = dir_save / f'{alg_type}_{comb_type}_l{levels[i]}'
            sl.make_long_dir(dir_save_i)

            # Load fitted data sets
            dir_i   = dir_load / 'mle_results' / 'fits' / 'singlerun' / f'mle{"-maxit" if maxit else ""}_{alg_type}-l{levels[i]}-{data_type}_{opt_type}' 
            file_i  = f'mle{"-maxit" if maxit else ""}_{alg_type}-l{levels[i]}-{data_type}_{opt_type}_{comb_type}.csv'
            res_i = sl.load_sim_data(dir_i,file_data=file_i)       
            res.append(res_i)

            # Generate parameter sets for simulation and simulate
            p_rand_df = get_params(comb_type,num_sim,seed,var_name,kwargs,dir_save_i,res_i,batchID=batchID)
            sim_fun, sim_fun_str, rec_type, base_params = get_sim_setup(alg_type,level=levels[i],no_rew=no_rew)
            run_sim_setup(startID,start_seed,num_sim,log_file,alg_type,p_rand_df,sim_fun,sim_fun_str,agent_num,epi_num,base_params,rec_type,dir_save_i,parallel=parallel)

    else:                                                                   # for count-based agents
        # Load fitted data sets                                                                    
        dir_i  = dir_load / 'mle_results' / 'fits' / 'singlerun' / f'mle{"-maxit" if maxit else ""}_{alg_type}-{data_type}_{opt_type}'
        file_i = f'mle{"-maxit" if maxit else ""}_{alg_type}-{data_type}_{opt_type}_{comb_type}.csv'
        res = sl.load_sim_data(dir_i,file_data=file_i)

        # Generate parameter sets for simulation and simulate
        p_rand_df = get_params(comb_type, num_sim, seed, var_name, kwargs, dir_save, res, batchID=batchID)
        sim_fun, sim_fun_str, rec_type, base_params = get_sim_setup(alg_type,no_rew=no_rew)
        run_sim_setup(startID,start_seed,num_sim,log_file,alg_type,p_rand_df,sim_fun,sim_fun_str,agent_num,epi_num,base_params,rec_type,dir_save,parallel=parallel)

    print(f'Finished simulation of {num_sim-1} parameter sets.')
    sl.write_log(log_file,f'Finished simulation of {num_sim-1} parameter sets.')
    log_file.close()

### Run multisim for input arguments ### 
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
        config = json.load(open('./src/fitting_behavior/recovery/param_recov_range_configs/multisim-hybrid2_app_uniparam_seed-12345.json'))
    print(type(config))
    print(config)

    run_multisim_range(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json
