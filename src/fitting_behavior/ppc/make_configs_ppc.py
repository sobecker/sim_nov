import numpy as np
import json
import os

import utils.saveload as sl

## Transform functions for initial conditions ###########################################
softplus_inv = lambda x: np.log(np.exp(x)-1)
softplus     = lambda x: np.log(np.exp(x)+1)
logit        = lambda x: np.log(x/(1-x))
sigmoid      = lambda x: 1/(1+np.exp(-x))

## Specify models to be simulated #######################################################
leakiness_type_all  = 'leaky'
runai_name_combined = f'{leakiness_type_all}_multinov'

model_type_list     = ['multinov-eps_hnor'] * 10 + ['hnor']*12
kernel_type_list    = ['notrace']*22
level_list          = [[1,6],[1,6],[2,6],[2,6],[3,6],[3,6],[4,6],[4,6],[5,6],[5,6]] +[[1],[1],[2],[2],[3],[3],[4],[4],[5],[5],[6],[6]] 
single_run_id_list  = ['_2','_3','_2','_3','_2','_3','_1','_3','_1','_3'] + ['_1','_5','_1','_5','_4','_5','_1','_5','_1','_5','_2','_5'] 
   
leakiness_type_list = [f'{leakiness_type_all}'] * len(model_type_list)
comb_type           = 'app'
parallel            = True  # True
run_local           = False # False
num_cpu             = 20 #20 
opt_method          = 'Nelder-Mead'
transformed         = True # use transformed optimization (e.g. log, softplus)

## Specify options for simulation #######################################################
uniparam        = True      # only simulate a single parameter set (but with different random seeds)
no_rew          = True      # simulate without stopping at rewarded state

if not uniparam:
    fixed_range     = True      # only used if uniparam = False
    range_perc      = 0.2       # only used if uniparam = False and fixed_range = True
    seed            = 12345     # only used if uniparam = False and fixed_range = True
else:
    fixed_range     = None
    range_perc      = None
    seed            = None

start_seed      = 0
startID         = 0
num_sim         = 20 # 20
agent_num       = 20  # 20

## Set paths ############################################################################
path_config = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'ppc' / 'configs_ppc'
path_exp    = sl.get_rootpath() / 'exp' / 'ppc' 
sl.make_long_dir(path_config)
sl.make_long_dir(path_exp)

## Create config files and exps files ###################################################
for i, (model_type, level, kernel_type, leakiness_type, sr_id) in enumerate(zip(model_type_list, level_list, kernel_type_list, leakiness_type_list, single_run_id_list)):

    # Construct full model type
    if level is not None:
        if isinstance(level, list):
            level_str = '_l' + '-'.join([str(l) for l in level])
        else:
            level_str = f'_l{level}'
        full_model_type = f'{leakiness_type}_{model_type}_{kernel_type}'
    else:
        full_model_type = f'{leakiness_type}_{model_type}'
        level_str       = ''

    # Set kwargs to specify model features (for multinov models)
    if 'nor' in model_type and 'multinov' in model_type: 

        notrace_i       = False if 'triangle' in kernel_type else True
        center_i        = True if 'center' in kernel_type else False
        center_type_i   = kernel_type.split('_center-')[-1].split('_')[0] if 'center' in kernel_type else 'box' # 'box','triangle'
        notrace_list_i  = [notrace_i, True] 
        center_list_i   = [center_i, False]

        kwargs_i = {'notrace': notrace_list_i,
                    'center': center_list_i,
                    'center_type': [center_type_i, 'box']}
    else:                                  
        kwargs_i = {}
            
    # Create config file
    params = {'alg_type': full_model_type,
                'single_run_id': [sr_id],
                'levels': [level],
                'kwargs': kwargs_i,
                'comb_type':comb_type,
                'seed': seed,
                'start_seed': start_seed,
                'startID': startID,
                'num_sim': num_sim,
                'agent_num': agent_num,
                'uniparam': uniparam,
                'no_rew': no_rew,
                'parallel': parallel,
                'run_local': run_local
                }

    param_str   = '_uniparam' if uniparam else ('_fixrange' if fixed_range else '_varrange')
    seed_str    =  f'seed-{seed}' if not uniparam else ''

    name = f'multisim-{full_model_type}{sr_id}_{comb_type}{param_str}{seed_str}{level_str}'
    with open(os.path.join(path_config, f'{name}.json'), 'w') as fp:
        json.dump(params, fp)

    # Create exps file
    with open(os.path.join(path_exp, f'{name}.sh'), 'w') as rsh:
        rsh.write(f'''\
    #!/bin/bash
    echo "creating directory"
    log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_multisim-{full_model_type}{sr_id}"
    base_path="/lcncluster/becker/sim_nov"
    echo "folder name: ${{log_folder}}"
    mkdir -p ${{base_path}}/logs/ppc
    mkdir -p ${{base_path}}/logs/ppc/sim_{param_str}/
    mkdir -p ${{base_path}}/logs/ppc/sim_{param_str}/${{log_folder}}

    echo "activating conda environment"
    source activate rlnet_cluster

    echo "build multisim-nor"
    python -u -b ${{base_path}}/src/fitting_behavior/ppc/sim_ppc.py -c ${{base_path}}/src/fitting_behavior/ppc/configs_ppc/multisim-{full_model_type}{sr_id}_{comb_type}{param_str}{seed_str}{level_str}.json | tee ${{base_path}}/logs/ppc/sim_{param_str}/${{log_folder}}/log{level_str}.txt
    ''')