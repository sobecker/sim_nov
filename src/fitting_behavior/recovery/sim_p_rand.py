import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.fitting_behavior.optimization.auxiliary_opt as auxo
import src.fitting_behavior.optimization.base_params_opt as bpo
import src.utils.saveload as sl

def run_multisim(config):
    seed        = config['seed'] # 12345
    num_sim     = config['num_sim'] # 1000
    agent_num   = config['agent_num'] # 20
    alg_type    = config['alg_type'] # nac, nor
    var_name    = config['var_name'] 
    kwargs      = config['kwargs']
    epi_num     = 1
    
    # Prepare folder and save metadata
    dir_save = sl.get_datapath()+f'ParameterRecovery/SimData/{alg_type}_allparams/'
    sl.make_long_dir(dir_save)
    log_file = open(dir_save+'log.txt', 'w') # create log file
    with open(dir_save+'config.json', 'w') as fc: json.dump(config, fc) # save config file for reproducibility
    sl.saveCodeVersion(dir_save,file_cv='code_version.txt') # save code version

    # Generate + save random parameter sets
    rng     = np.random.default_rng(seed)
    lows    = np.array([kwargs['bounds'][i][0] for i in range(len(kwargs['bounds']))])
    highs   = np.array([kwargs['bounds'][i][1] for i in range(len(kwargs['bounds']))])
    p_rand  = rng.uniform(low=lows,high=highs,size=(num_sim,len(lows))) 
    p_rand_df           = pd.DataFrame(p_rand,columns=var_name)
    p_rand_df['simID']  = np.arange(num_sim)
    p_rand_df.to_csv(dir_save+'params_overview.csv')
    p_rand_df.to_pickle(dir_save+'params_overview.pickle')

    # Set simulation method
    if alg_type=='nac':
        sim_fun     = auxo.run_optsim_nac_tree
        rec_type    = 'advanced1'
        base_params = params = bpo.base_params_nACtree.copy()
    elif alg_type=='nor':
        sim_fun     = auxo.run_opt_sim_mbnor
        rec_type    = 'basic'
        base_params = bpo.base_params_mbnortree_exp.copy()

    # Run simulation for each parameter set
    for i in range(num_sim):
        if i%10==0:
            print(f'Starting simulation with random parameter set {i}/{num_sim-1}.')
            sl.write_log(log_file,f'Starting simulation with random parameter set {i+1}/{num_sim-1}.')
        sim_name = f'{alg_type}_sim{i}'
        params_i    = p_rand_df.iloc[i].to_dict()
        sim_fun(params_i,agent_num,epi_num,base_params,sim_name=sim_name,rec_type=rec_type,dir_data=dir_save)

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
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_configs/multisim-nac_seed-12345.json'))
        config = json.load(open('./src/scripts/ParameterRecovery/param_recov_configs/multisim-nor_seed-12345.json'))
    print(type(config))
    print(config)

    run_multisim(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json
