import json
import numpy as np

params = {'test_grid': True,
            'test_NelderMead': True,
            'test_LBFGSB': True,
            'test_SLSQP': True,
            'data_folder': '2022_11_17_10-57-08_nAC_debug', #'/Volumes/lcncluster/becker/RL_reward_novelty/data/bintree_archive/sim_opt/2022_08_16_11-23-13_gpopt_nAC-N-expl_OI', 
            'data_path_type': 'auto', #'manual'
            'var_name': 'gamma',
            'kwargs': {'var_range':list(np.linspace(0.,0.999,10)),"bounds":((0,0.999),)},
            'alg_type': 'nac',
            'save_name': 'mle_nac-naive', #'mle_nac-opt',
            'verbose': True}

path = './src/scripts/MLE/singleparam_test_configs/'
name = f'{params["save_name"]}_{params["var_name"]}'

with open(path+name+'.json', 'w') as fp:
    json.dump(params, fp)
