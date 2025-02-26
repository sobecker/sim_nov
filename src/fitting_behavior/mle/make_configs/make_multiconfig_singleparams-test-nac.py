import json
import numpy as np

# Create config files to run singleparam tests (MLE) for all free variables of NoR
l_var   = ['gamma', 'c_alph',   'a_alph',    'c_lam',    'a_lam',   'temp', 'c_w0', 'a_w0']
l_ub    = [0.999,    0.5,        0.5,        0.999,      0.999,     1,      100,    100]
l_lb    = [0,        0.001,      0.001,      0,          0,         0.001,  -100,   -100]
type    = 'opt' #'naive'
path    = './src/scripts/MLE/singleparam_test_configs/'

if type=='opt': 
    params = {'test_grid': True,
                'test_NelderMead': True,
                'test_LBFGSB': True,
                'test_SLSQP': True,
                'data_folder': 'bintree_archive/sim_opt/2022_08_16_11-23-13_gpopt_nAC-N-expl_OI',  # when running locally add '/Volumes/lcncluster/becker/RL_reward_novelty/data/' in front of path
                'data_path_type': 'auto', #set to 'manual' when running locally
                'var_name': 'gamma',
                'kwargs': {'var_range':list(np.linspace(0.,0.999,10)),"bounds":((0.,0.999),)},
                'alg_type': 'nac',
                'save_name': 'mle_nac-opt',
                'verbose': True}
elif type=='naive':
    params = {'test_grid': True,
                'test_NelderMead': True,
                'test_LBFGSB': True,
                'test_SLSQP': True,
                'data_folder':'2022_11_17_10-57-08_nAC_debug', 
                'data_path_type': 'auto', 
                'var_name': 'gamma',
                'kwargs': {'var_range':list(np.linspace(0.,0.999,10)),"bounds":((0.,0.999),)},
                'alg_type': 'nac',
                'save_name': 'mle_nac-naive',
                'verbose': True}

for i in range(len(l_var)):
    params['var_name'] = l_var[i]
    params['kwargs']['var_range'] = list(np.linspace(l_lb[i],l_ub[i],10))
    params['kwargs']['bounds'] = ((l_lb[i],l_ub[i]),)

    name = f'{params["save_name"]}_{params["var_name"]}'

    with open(path+name+'.json', 'w') as fp:
        json.dump(params, fp)
