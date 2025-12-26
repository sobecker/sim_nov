import json
import numpy as np
import pandas as pd
import utils.saveload as sl
import os

#############################################################################
#                 Set grid search hyper-parameters and paths                #
#############################################################################
cluster         = True # set True if running on cluster
input_seq_mode  = 'sep' # 'sep': run Homann exp. with the same image sets for each parameter value (e.g. different values of M), independently; 'app': appended format of the experiment, with different image sets per parameter value.
num_sim         = 50        # number of homann simulations for each trial (i.e. in each optimization step)
start_id        = 0
num_cpu         = 24
resume          = False # set True if resuming a previous run
append_mode     = False
init_seed       = 98765
parallel_exp    = False
parallel_grid   = True

#############################################################################
#                 Specify model for which to run grid search                #
#############################################################################
name_proj       = 'gridsearch_robustness_width' # project name
type_cells      = 'complex' # 'complex', 'simple'
type_update     = 'leaky' # 'fr', 'leaky'
name_model      = f'snov-{type_cells}-{type_update}'

# Parameters for input stimuli (fixed)
params_input = {'num_gabor': 40,
                'adj_w':     True,
                'adj_f':     False,
                'alph_adj':  3,
                'n_fam':     [1,3,8,18,38],
                'n_im':      [3,6,9,12],
                'dN':        list(np.array([0,22,44,66,88,110,143])/0.3), #[0,70,140,210,280,360,480],
                'idx':       True
                }

# Model and grid search parameters
set_num         = 1
name_set        = f'{name_model}_set{set_num}_{input_seq_mode}' 

if set_num in [1, 2]: 
    # set 1: load optimal parameters for triangle components, systematically vary component width (ksig)
    # set 2: load optimal parameters for triangle components, systematically vary component width (ksig) and timescale parameters (alph_leak, eps_leak)

    # Fixed model parameters
    gabor_sampling = 'equidist_fixed' 
    k_type         = 'triangle' 
    k_params_ext   = {'k_type': k_type,
                      'gabor_sampling': gabor_sampling} 
    
    kwargs_ext = {'no_simple_cells': False,
                    'no_complex_cells': type_cells=='simple',
                    'mode_complex': 'sum', # 'sum' or 'mean'
                    'debug': False,
                    'append_mode': append_mode,
                    'start_id': start_id}
   
    if set_num==1: # No refitting of timescale parameters, load best parameters
        # Load fitted model parameters
        path_opt = sl.get_rootpath() / 'data' / 'grid_search_results'

        measure_fit       = 'train_mse'
        sampling_type_fit = 'normal'
        drop_type_fit     = 'none' 
        weighting_fit     = 'equal-samples' # 'equal-samples', 'equal-exp', 'none' (not available for bootstrap)

        file_opt = f'best_params_{measure_fit}_{sampling_type_fit}'
        if 'jackknife' in sampling_type_fit or 'loo' in sampling_type_fit:
            file_opt += f'-drop-{drop_type_fit}'
        if weighting_fit!='none':
            file_opt += f'-{weighting_fit}'
        file_opt += f'-{type_update}.json'

        best_params_raw = json.load(open(os.path.join(path_opt, f'data_for_figures/{file_opt}')))
        best_params = [bp for bp in best_params_raw if bp[0]==name_model][0][1]

        # Model parameters to be varied
        grid = {'ksig':           [0.5, 0.6, 0.7, 0.25, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.3, 1.4, 1.5],
                #[0.5, 0.55, 0.6, 0.65, 1, 0.98, 0.96, 0.94, 0.93, 0.92, 0.91, 0.875, 0.825, 0.775, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5], #[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99],
                'cdens':          [best_params['cdens']], 
                'knum':           [best_params['knum']]}
    
    elif set_num==2: # Refitting of timescale parameters
        # Model parameters to be varied
        grid = {'cdens':          [8], #[4,8,16,32],
                'knum':           [2], #[2,4,6,8,10,20,40]
                'ksig':           [0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975, 1.025,1.05,1.075,1.1,1.125,1.15,1.175,1.2] # [0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15],
                }
    
    if type_cells=='complex':
        grid['type_complex']  = [8]         # number of simple cells per complex cell - rerun with different shifts instead of different frequencies?
        grid['ratio_complex'] = [1/3]       # ratio of complex to simple cells - taken from biological data

    if set_num==1: # No refitting of timescale parameters, load best parameters
        if type_update=='fr':
            grid['k_alph']      = [best_params['k_alph']] # fixed learning rate (fr model)
            k_params_ext['flr'] = True # fixed learning rate (fr model)

        elif type_update=='leaky':
            grid['alph_leak']   = [best_params['alph_leak']] # leakiness (leaky model): 0 = no leak, 1 = full leak
            grid['eps']         = [best_params['eps']] # prior (leaky model)
            k_params_ext['flr']   = False # fixed learning rate (leaky model)
    elif set_num==2: # Refitting of timescale parameters
        if type_update=='fr':
            grid['k_alph']      = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999] # fixed learning rate (fr model)
            k_params_ext['flr'] = True # fixed learning rate (fr model)

        elif type_update=='leaky':
            grid['alph_leak']   = [0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            grid['eps']         = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6] # prior (leaky model)
            k_params_ext['flr']   = False # fixed learning rate (leaky model)

else:
    assert False, "Set number not recognized. Use set_num=1 for basic grid search."

#############################################################################
#                Split grid search into parallel jobs                       #
#############################################################################
gridpoints_per_job = 8

# Create separate dataframes for each grid variables
df_gridvars = []
for k, v in grid.items():
    df_v = pd.DataFrame(grid[k], columns=[k])
    df_gridvars.append(df_v)

# Cross join all grid variables to create dataframe for grid search
df_grid = df_gridvars[0]
for i in range(1,len(df_gridvars)):
    df_grid = df_grid.merge(df_gridvars[i], how='cross', on=None)

# Create dictionaries for each job
grid_dicts = []
c_start = 0
while c_start<len(df_grid):
    grid_dicts.append(df_grid.iloc[c_start:c_start+gridpoints_per_job].to_dict(orient='list'))
    c_start += gridpoints_per_job

#############################################################################
#                 Set paths                                                 #
#############################################################################
path_config_summary = sl.get_rootpath() / 'src' / 'fitting_neural' / 'configs_robustness' / f'{name_proj}'
path_config         = path_config_summary / f'{name_set}'
path_exp            = sl.get_rootpath() / 'exp' / f'{name_proj}' / f'{name_set}' 
path_results        = sl.get_rootpath() / 'data' / f'{name_proj}' / f'{name_set}'

sl.make_long_dir(path_config)
sl.make_long_dir(path_exp)  

for i in range(len(grid_dicts)):
    # Build config file #################################################################################################################
    job_name   = f'job-{i}'

    config = {'job_name': job_name,
              'grid': grid_dicts[i],
              'k_params_ext': k_params_ext,
              'num_sim': num_sim,  
              'params_input': params_input,
              'init_seed': init_seed,
              'parallel_exp': parallel_exp,
              'parallel_grid': parallel_grid,
              'save_path': f'{path_results}/{job_name}/',
              'comp_fit': False,
              'comp_corr': False,
              'cluster': cluster,
              'resume': resume,
              'input_corrected': True,
              'input_sequence_mode': input_seq_mode,
              'kwargs': kwargs_ext,
              'num_cpu': num_cpu
              }

    name_config = job_name

    with open(os.path.join(path_config,f'{name_config}.json'), 'w') as fp:
        json.dump(config, fp)
    print(f'Config file saved as {os.path.join(path_config,f"{name_config}.json")}')

    # Build exp file ####################################################################################################################
    name_exp = job_name
    
    with open (os.path.join(path_exp,f'{name_exp}.sh'), 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{name_exp}"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/{name_proj}
mkdir -p ${{base_path}}/logs/{name_proj}/{name_set}
mkdir -p ${{base_path}}/logs/{name_proj}/{name_set}/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {name_exp}"
python -u -b ${{base_path}}/src/fitting_neural/grid_search_snov.py -c ${{base_path}}/src/fitting_neural/configs_robustness/{name_proj}/{name_set}/{name_config}.json | tee ${{base_path}}/logs/{name_proj}/{name_set}/${{log_folder}}/log.txt
''')
        
    print(f'Exp file saved as {os.path.join(path_exp,f"{name_exp}.sh")}')

# Save dataframe with simulation info ###############################################################################################
info = {'study_name': name_set,
        'model_name': name_model,
        'model_k_type': k_type,
        'grid_set_num': set_num,
        'grid_num_sim': num_sim,
        'grid_init_seed': init_seed,
        'path_results': str(path_results),
        'path_config': str(path_config),
        'path_exp': str(path_exp)
        }

with open(f'{path_config_summary}summary_{name_set}.json', 'w') as fp:
    json.dump(info,fp)
