import json
import glob
from argparse import ArgumentParser
import multiprocessing as mp
import os
import sys
import fitting_behavior.optimization.auxiliary_opt as auxo
import fitting_behavior.optimization.base_params_opt as bpo
import utils.saveload as sl
from fitting_behavior.mle.mle_fit import run_mle_fit

def run_single_parallel(config,dataset):
    config['data_folder'] = dataset # set path to sim data (to be fitted)    
    config['save_name'] = f'mle-recov_{config["alg_type"]}_sim{dataset.split("_sim")[-1]}' 
    run_mle_fit(config)

def run_multifit_single_parallel(config):
    print('starting parallel fit')
    data_path_type  = config['data_path_type']
    data_path       = config['data_path']
    num_fit         = config['num_fit']

    if data_path_type=='auto':
        #data_path = '/Volumes/lcncluster/becker/RL_reward_novelty/data/'+data_path
        data_path = sl.get_datapath()+config['data_path']
    regex = '*_sim*'
    datasets = glob.glob(data_path / regex)
    datasets.sort(key=lambda x: int(x.split('_sim')[-1]))

    config_new = config.copy()
    config_new['data_type']       = 'recov'
    #config_new['save_path']      = '/Volumes/lcncluster/becker/RL_reward_novelty/data/'+config['save_path']  
    config_new['save_path']       = sl.get_datapath()+config['save_path']      
    config_new['data_path_type']  = 'manual' # or 'auto' depending on how we reduce the full path

    # Adjust configs and fit all data sets
    pool = mp.Pool(mp.cpu_count())
    for i in range(min(num_fit,len(datasets))):
        pool.apply_async(run_single_parallel,args=(config_new,datasets[i]))
    pool.close()
    pool.join()        
        
# Load list of data sets to be fitted
def run_multifit_single_sequential(config):
    print('starting fit')
    data_path_type  = config['data_path_type']
    data_path       = config['data_path']
    num_fit         = config['num_fit']

    if data_path_type=='auto':
        #data_path = '/Volumes/lcncluster/becker/RL_reward_novelty/data/'+data_path
        data_path = sl.get_datapath()+config['data_path']
    regex = '*_sim*'
    datasets = glob.glob(data_path / regex)
    datasets.sort(key=lambda x: int(x.split('_sim')[-1]))

    startID = config['startID'] if 'startID' in config.keys() else 0

    # Adjust configs and fit all data sets
    for i in range(startID,min(startID+num_fit,len(datasets))):
        config_i = config.copy()
        config_i['data_type']       = 'recov'
        config_i['data_folder']     = datasets[i] # set path to sim data (to be fitted)
        #config_i['save_path']       = '/Volumes/lcncluster/becker/RL_reward_novelty/data/'+config['save_path']  
        config_i['save_path']       = sl.get_datapath()+config['save_path']      
        config_i['data_path_type']  = 'manual' # or 'auto' depending on how we reduce the full path
        config_i['save_name']       = f'mle-recov_{config["alg_type"]}_sim{datasets[i].split("_sim")[-1]}' 

        run_mle_fit(config_i)
        # make_overview_fits(config['save_path']) # write later

def run_multifit(config):
    alg_type_sim    = config['alg_type'] if 'alg_type' in config.keys() else config['alg_type_sim']
    alg_type_fit    = config['alg_type'] if 'alg_type' in config.keys() else config['alg_type_fit']
    comb_type   = config['comb_type']
    opt_method  = config['kwargs']['opt_method'] 

    parallel_sim = config['parallel_sim'] if 'parallel_sim' in config.keys() else False
    if parallel_sim: 
        config['parallel'] = False 
        run_multifit_single = run_multifit_single_parallel 
    else:
        run_multifit_single = run_multifit_single_sequential 

    if 'hnor' in alg_type_fit or 'hnac' in alg_type_fit or 'hhybrid' in alg_type_fit:
        levels_sim   = config['levels_sim'] if 'levels_sim' in config.keys() else config['levels']
        levels_fit   = config['levels_fit'] if 'levels_fit' in config.keys() else config['levels']
        for i,j in zip(range(len(levels_sim)),range(len(levels_fit))):
            config_i = config.copy()
            config_i['data_path'] = os.path.join(config_i['data_path'],f'{alg_type_sim}_{comb_type}_l{levels_sim[i]}')
            if levels_sim[i]==levels_fit[j]:
                config_i['save_path'] = os.path.join(config_i['save_path'],f'{config["save_name"]}_l{levels_fit[j]}')
            else:
                config_i['save_path'] = os.path.join(config_i['save_path'],f'sim-{os.path.normpath(config_i["data_path"]).split(os.sep)[-1].replace("_","-")}_{alg_type_fit}-{comb_type}-l{levels_fit[j]}_{opt_method}')
            config_i['alg_type']  = alg_type_fit
            config_i['level']     = levels_fit[j]
            run_multifit_single(config_i)
    else:
        run_multifit_single(config)

### Run multifit for input arguments ### 
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
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multifit-nor_app_Nelder-Mead.json'))
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multifit-hnor_app_Nelder-Mead.json')) 
        # config = json.load(open('./src/scripts/ParameterRecovery/param_recov_range_configs/multifit-hhybrid_app_Nelder-Mead.json'))
        #config = json.load(open('./src/scripts/ParameterRecovery/model_recov_configs/multifit_sim-hhybrid-l1_fit-hhybrid-l1_app_Nelder-Mead.json'))
        config = json.load(open('./src/scripts/ParameterRecovery/model_recov_configs/multifit_sim-hnor-l1_fit-hnor-l5_app_Nelder-Mead_local.json'))
        # config = json.load(open('/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/sim-hnor-app_fit-hnor-app_Nelder-Mead/sim-hnor-app-l1_hnor-app-l5_Nelder-Mead/mle-recov_hnor_sim0_Nelder-Mead'))
    print(type(config))
    print(config)

    run_multifit(config)

    # run from terminal:  
    # python -u ./src/scripts/MLE/mle_fit.py --config ./src/scripts/MLE/mle_fit_configs/mle_nor-naive_lambda_N.json
