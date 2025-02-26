import json
import numpy as np

params = {'test_grid': True,
            'test_NelderMead': True,
            'test_LBFGSB': True,
            'test_SLSQP': True,
            #'data_folder':'2022_12_09_13-31-14_sim_nor-tree_naive-nov', 
            'data_folder':'/Volumes/lcncluster/becker/RL_reward_novelty/data/nor_tree/sim_opt/2022_10_07_19-46-06_sim_mbnor_tree-nov-beta1r',
            #'data_path_type': 'auto', 
            'data_path_type': 'manual',
            'var_name': 'lambda_N',
            'kwargs': {'var_range':list(np.linspace(0.,0.999,10)),"bounds":((0,0.999),)},
            'alg_type': 'nor',
            #'save_name': 'mle_nor-naive',
            'save_name': 'mle_nor-opt',
            'verbose': True}

path = './src/scripts/MLE/singleparam_test_configs/'
name = f'{params["save_name"]}_{params["var_name"]}'

with open(path+name+'.json', 'w') as fp:
    json.dump(params, fp)
