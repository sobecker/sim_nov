import json
import numpy as np

# Create config files to run singleparam tests (MLE) for all free variables of NoR
l_var   = ['lambda_N','beta_1','epsilon','k_leak']
l_ub    = [0.999,30,1,0.999]
l_lb    = [0.,0.1,0.0001,0.001]
type    = 'opt' #'naive'
path    = './src/scripts/MLE/singleparam_test_configs/'

if type=='opt': 
    params = {'test_grid': True,
                'test_NelderMead': True,
                'test_LBFGSB': True,
                'test_SLSQP': True,
                # 'data_folder':'nor_tree/sim_opt/2022_10_07_19-46-06_sim_mbnor_tree-nov-beta1r', # add data path if running locally: /Volumes/lcncluster/becker/RL_reward_novelty/data/
                'data_folder':'nor_tree/sim_opt/2023_01_11_16-59-13_sim_mbnor_tree-allparams', # add data path if running locally: /Volumes/lcncluster/becker/RL_reward_novelty/data/
                'data_path_type': 'auto', # set to 'manual' when running locally
                'var_name': 'lambda_N',
                'kwargs': {'var_range':list(np.linspace(0.,0.999,10)),"bounds":((0.,0.999),)},
                'alg_type': 'nor',
                'save_name': 'mle_nor-opt',
                'verbose': True}
elif type=='naive':
    params = {'test_grid': True,
                'test_NelderMead': True,
                'test_LBFGSB': True,
                'test_SLSQP': True,
                'data_folder':'2022_12_09_13-31-14_sim_nor-tree_naive-nov', 
                'data_path_type': 'auto', 
                'var_name': 'lambda_N',
                'kwargs': {'var_range':list(np.linspace(0.,0.999,10)),"bounds":((0.,0.999),)},
                'alg_type': 'nor',
                'save_name': 'mle_nor-naive',
                'verbose': True}

for i in range(len(l_var)):
    params['var_name'] = l_var[i]
    params['kwargs']['var_range'] = list(np.linspace(l_lb[i],l_ub[i],10))
    params['kwargs']['bounds'] = ((l_lb[i],l_ub[i]),)

    name = f'{params["save_name"]}_{params["var_name"]}'

    with open(path+name+'.json', 'w') as fp:
        json.dump(params, fp)
