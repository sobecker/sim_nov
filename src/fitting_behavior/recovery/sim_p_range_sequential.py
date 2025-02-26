import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import sys
import os
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.fitting_behavior.optimization.auxiliary_opt as auxo
import src.fitting_behavior.optimization.base_params_opt as bpo
import src.utils.saveload as sl

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
    p_rand_df.to_csv(os.path.join(dir_save,f'params_{comb_type}_overview.csv'))
    p_rand_df.to_pickle(os.path.join(dir_save,f'params_{comb_type}_overview.pickle'))
    return p_rand_df

# Get baseparams for sim
def get_sim_setup(alg_type,level=0):
    if 'nac' in alg_type:
        sim_fun     = auxo.run_optsim_nac_tree
        rec_type    = 'advanced1'
        if alg_type=='nac':
            base_params = bpo.base_params_nACtree.copy()
        elif 'hnac' in alg_type:
            base_params = bpo.get_baseparams_all_hnac(alg_type,[level],True)
    elif 'nor' in alg_type:
        sim_fun     = auxo.run_opt_sim_mbnor
        rec_type    = 'basic'
        if alg_type=='nor':     base_params = bpo.base_params_mbnortree_exp.copy()
        elif alg_type=='hnor':  base_params = bpo.get_baseparams_all_hnor(alg_type,[level],True)
    elif 'hybrid' in alg_type:
        sim_fun     = auxo.run_optsim_hybrid_tree
        rec_type    = 'basic'
        if alg_type=='hybrid':      base_params = bpo.baseparams_hybrid()
        elif alg_type=='hhybrid':   base_params = bpo.baseparams_all_hhybrid('hnor','hnac-gn',[level],True)
    return sim_fun,rec_type,base_params

# Run simulation for each parameter set
def run_sim_setup(num_sim,log_file,alg_type,p_rand_df,sim_fun,agent_num,epi_num,base_params,rec_type,dir_save):
    for i in range(num_sim):
        if i%10==0:
            print(f'Starting simulation with random parameter set {i}/{num_sim-1}.')
            sl.write_log(log_file,f'Starting simulation with random parameter set {i+1}/{num_sim-1}.')
        sim_name = f'{alg_type}_sim{i}'
        params_i    = p_rand_df.iloc[i].to_dict()
        sim_fun(params_i,agent_num,epi_num,base_params,sim_name=sim_name,rec_type=rec_type,dir_data=dir_save)

# Run multiple simulations (app/sep) for different alg types
def run_multisim_range(config):
    comb_type   = config['comb_type'] # 'sep','app'
    num_sim     = config['num_sim'] # 1000
    agent_num   = config['agent_num'] # 20
    seed        = config['seed'] if 'seed' in config.keys() else 12345

    alg_type    = config['alg_type'] # 'nac','nor','hnac-gn','hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    if 'hnor' in alg_type or 'hnac' in alg_type or 'hhybrid' in alg_type: 
        levels  = config['levels']
    var_name    = config['var_name'] 
    kwargs      = config['kwargs']
    epi_num     = 1
    data_type   = 'mice'
    opt_type    = 'Nelder-Mead'
    
    # Prepare folder and save metadata
    dir_save = sl.get_datapath()+f'ParameterRecovery/SimData/{alg_type}_{comb_type}/'
    #dir_save = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData/{alg_type}_{comb_type}/'
    sl.make_long_dir(dir_save)
    log_file = open(dir_save+'log.txt', 'w') # create log file
    with open(dir_save+'config.json', 'w') as fc: json.dump(config, fc) # save config file for reproducibility
    sl.saveCodeVersion(dir_save,file_cv='code_version.txt') # save code version

    # Generate + save random parameter sets
    # dir_load = '/Volumes/lcncluster/becker/RL_reward_novelty/data/'
    dir_load = sl.get_datapath() 
    if 'hnor' in alg_type or 'hnac' in alg_type or 'hhybrid' in alg_type:
        res = []
        for i in range(len(levels)):
            dir_save_i = os.path.join(dir_save,f'{alg_type}_{comb_type}_l{levels[i]}')
            sl.make_long_dir(dir_save_i)
            print()
            if 'hhybrid'==alg_type:
                dir_i = os.path.join(dir_load,f'MLE_results/Fits/SingleRun/mle-maxit_{alg_type}-{data_type}_{opt_type}/mle_{alg_type}-l{levels[i]}-{data_type}_{opt_type}') 
                file_i = f'mle-maxit_{alg_type}-l{levels[i]}-{data_type}_{opt_type}_{comb_type}.csv'
            else:
                dir_i = os.path.join(dir_load,f'MLE_results/Fits/SingleRun/mle_{alg_type}-{data_type}_{opt_type}/mle_{alg_type}-l{levels[i]}-{data_type}_{opt_type}') 
                file_i = f'mle_{alg_type}-l{levels[i]}-{data_type}_{opt_type}_{comb_type}.csv'
            print(dir_i)
            print(file_i)
            res_i = sl.load_sim_data(dir_i,file_data=file_i)       
            res.append(res_i)
            p_rand_df = get_rand_params(comb_type,num_sim,seed,var_name,kwargs,dir_save_i,res_i)
            sim_fun, rec_type, base_params = get_sim_setup(alg_type,level=levels[i])
            run_sim_setup(num_sim,log_file,alg_type,p_rand_df,sim_fun,agent_num,epi_num,base_params,rec_type,dir_save_i)
    else:
        if 'hybrid'==alg_type:
            res = sl.load_sim_data(os.path.join(dir_load,f'MLE_results/Fits/SingleRun/mle-maxit_{alg_type}-{data_type}_{opt_type}'),file_data=f'mle-maxit_{alg_type}-{data_type}_{opt_type}_{comb_type}.csv')
        else:
            res = sl.load_sim_data(os.path.join(dir_load,f'MLE_results/Fits/SingleRun/mle_{alg_type}-{data_type}_{opt_type}'),file_data=f'mle_{alg_type}-{data_type}_{opt_type}_{comb_type}.csv')
        p_rand_df = get_rand_params(comb_type, num_sim, seed, var_name, kwargs, dir_save, res)
        sim_fun, rec_type, base_params = get_sim_setup(alg_type)
        run_sim_setup(num_sim,log_file,alg_type,p_rand_df,sim_fun,agent_num,epi_num,base_params,rec_type,dir_save)

    print(f'Finished simulation of {num_sim-1} randomly generated parameter sets.')
    sl.write_log(log_file,f'Finished simulation of {num_sim-1} randomly generated parameter sets.')
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
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multisim-nac_sep_fixrange_seed-12345.json'))
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multisim-nac_app_fixrange_seed-12345.json'))
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multisim-nor_sep_fixrange_seed-12345.json'))
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multisim-hnac-gn-gv-goi_sep_fixrange_seed-12345.json'))
        config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multisim-hhybrid_app_fixrange_seed-12345.json'))
    print(type(config))
    print(config)

    run_multisim_range(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json
