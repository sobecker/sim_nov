import json
import numpy as np

import sys
import utils.saveload as sl

config_sim = True
config_fit = True
lim_range  = True

alg_type    = ['nac','nor']
path        = './src/scripts/ParameterRecovery/param_recov_configs/'
opt_method  = ['Nelder-Mead']    # 'SLSQP','L-BFGS-B'

l_var_nor       = ['lambda_N','beta_1','epsilon','k_leak']
l_x0_nor        = [0.5,5,0.0002,0.5]
l_bounds_nor    = [[0.,0.999],
                    [0.1,30],
                    [0.001,1],
                    [0.001,0.999]]

l_var_nac       = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0']
l_x0_nac        = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0]
l_bounds_nac    = [[0.,0.999],      #gamma
                    [0.001,0.5],    #c_alph
                    [0.001,0.5],    #a_alph
                    [0.,0.999],     #c_lam
                    [0.,0.999],     #a_lam
                    [0.001,1.],     #temp
                    [-100,100],     #c_w0
                    [-100,100]]     #a_w0

# Make configs for simulation (recovery data)
if config_sim:
    for i in range(len(alg_type)):
        if alg_type[i]=='nor':
            l_var       = l_var_nor
            l_x0        = l_x0_nor
            l_bounds    = l_bounds_nor
        elif alg_type[i]=='nac':
            l_var       = l_var_nac
            l_x0        = l_x0_nac
            l_bounds    = l_bounds_nac
        if lim_range:
            data_path = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/MLE_results/Fits/SingleRun/mle_{alg_type}-mice_{opt_method}/'
            data_file = f'mle' ## finish here
            # Set x0 to MLE estimates
            # Set bounds to error around MLE estimate (sep)
        params = {'seed': 12345,
                    'num_sim': 50, #1000
                    'agent_num': 20, #20
                    'var_name': l_var,
                    'kwargs': {"x0":l_x0,"bounds":l_bounds},
                    'alg_type': alg_type[i]}
        name = f'multisim-{params["alg_type"]}_seed-{params["seed"]}'
        with open(path+name+'.json', 'w') as fp:
            json.dump(params, fp)


# Make configs for MLE fit (parameter recovery)
if config_fit:
    for i in range(len(alg_type)):
        for j in range(len(opt_method)):
            if alg_type[i]=='nor':
                l_var       = l_var_nor
                l_x0        = l_x0_nor
                l_bounds    = l_bounds_nor
            elif alg_type[i]=='nac':
                l_var       = l_var_nac
                l_x0        = l_x0_nac
                l_bounds    = l_bounds_nac
            save_path = f'ParameterRecovery/FitData/{alg_type[i]}_{opt_method[j]}/'
            #sl.make_long_dir(save_path)
            params = {'num_fit': 50,
                        'data_path': f'ParameterRecovery/SimData/{alg_type[i]}/', # path to sim data (to be fitted)
                        'data_path_type': 'auto',
                        'comb_type': '',
                        'var_name': l_var,
                        'kwargs': {"x0":l_x0,
                                    "bounds":l_bounds,
                                    "opt_method":opt_method[j]},
                        'alg_type': alg_type[i],
                        'save_name': f'mle_{alg_type[i]}-recov',
                        'verbose': True,
                        'save_path': save_path   # path where the fitting results are saved
                        }
            name = f'multifit-{params["alg_type"]}_{params["kwargs"]["opt_method"]}'
            with open(path+name+'.json', 'w') as fp:
                json.dump(params, fp)

    

    
        