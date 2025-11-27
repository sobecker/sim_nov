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
name_proj       = '2025_05_grid_search_new'
type_cells      = 'simple' # 'complex', 'simple'
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

if set_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: 
    # set 1: basic grid search - fr / leaky
    # set 2: basic grid search (less params) - fr / leaky
    # set 3: small epsilon - leaky
    # set 4: more small epsilon - leaky
    # set 6: missing values of cdens, knum for small epsilon as in set 3 and 4 - leaky
    # set 7: grid search with equidist_fixed sampling, first 50 samples - fr (full grid, input seed = 98765) / leaky (partial grid, input seed = 98765)
    # set 8: grid search with equidist_fixed sampling, second 50 samples - fr (full grid, input seed = 23456) / leaky (missing grid points, input seed = 98765)
    # set 9: grid search with equidist_fixed sampling - leaky (combined set 7 and 8, input seed = 98765)
    # set 10: grid search with equidist_fixed sampling, second 50 samples - leaky (full grid, input seed = 23456)

    # Fixed model parameters
    k_type         = 'triangle' # 'box', 'triangle'
    gabor_sampling = 'equidist_fixed' # 'equidist', 'equidist_fixed'
    k_params_ext   = {'k_type': k_type,
                      'gabor_sampling': gabor_sampling} 
    
    kwargs_ext = {'no_simple_cells': False,
                    'no_complex_cells': type_cells=='simple',
                    'mode_complex': 'sum', # 'sum' or 'mean'
                    'debug': False,
                    'append_mode': append_mode,
                    'start_id': start_id
                    }

    # Model parameters to be varied
    grid = {'cdens':          [4,8,16,32], #[4,8,16,32],
            'knum':           [2,4,6,8,10] #[2,4,6,8,10,20,40]
            }
    
    if type_cells=='complex':
        grid['type_complex']  = [8]         # number of simple cells per complex cell - rerun with different shifts instead of different frequencies?
        grid['ratio_complex'] = [1/3]       # ratio of complex to simple cells - taken from biological data

    if type_update=='fr':
        grid['k_alph']      = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999] # fixed learning rate (fr model)
        k_params_ext['flr'] = True # fixed learning rate (fr model)

    elif type_update=='leaky':
        grid['alph_leak']   = [0, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        # grid['alph_leak']   = [0.3, 0.4, 0.6, 0.7] #[0, 0.01,0.1,0.2,0.5,0.8,0.9] # leakiness (leaky model): 0 = no leak, 1 = full leak
        # grid['eps']         = [0.1,1,10] # prior (leaky model)
        grid['eps']         = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6] # prior (leaky model)
        # grid['alph_leak']   = [0, 0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99, 0.999] # leakiness (leaky model): 0 = no leak, 1 = full leak
        # grid['eps']         = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,20,50] # prior (leaky model)
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
path_config_summary = './src/scripts/gabor_kernels/grid_search_new/configs/'
path_config         = f'{path_config_summary}{name_set}/'
path_exp            = f'/Volumes/lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/' 
path_runai          = f'/Users/sbecker/runai_cli_files/rlnet/{name_proj}/{name_set}/'
path_results        = ('/lcncluster/becker/RL_reward_novelty/data/' if cluster else '/Users/sbecker/Projects/RL_reward_novelty/data/') + f'{name_proj}/{name_set}/'

sl.make_long_dir(path_config)
sl.make_long_dir(path_exp)  
sl.make_long_dir(path_runai)

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
              'save_path': f'{path_results}{job_name}/',
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
base_path="/lcncluster/becker/RL_reward_novelty"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/{name_proj}
mkdir -p ${{base_path}}/logs/{name_proj}/{name_set}
mkdir -p ${{base_path}}/logs/{name_proj}/{name_set}/${{log_folder}}

echo "activating conda environment"
source activate rlnet_cluster

echo "build {name_exp}"
python -u -b ${{base_path}}/src/scripts/gabor_kernels/grid_search_new/grid_search_snov.py -c ${{base_path}}/src/scripts/gabor_kernels/grid_search_new/configs/{name_set}/{name_config}.json | tee ${{base_path}}/logs/{name_proj}/{name_set}/${{log_folder}}/log.txt
''')
        
    print(f'Exp file saved as {os.path.join(path_exp,f"{name_exp}.sh")}')

    # Build runai client file ##########################################################################################################
    name_runai_job = name_set.replace('_','-').lower()+f'-{job_name}'
    if len(name_runai_job)>62:
        name_runai_job = name_runai_job[-62:]

    with open (os.path.join(path_runai,f'submit_{name_exp}.sh'), 'w') as rsh:
        rsh.write(f'''\
runai submit \
  --name {name_runai_job} \
  --image nvcr.io/nvidia/pytorch:25.03-py3 \
  --gpu 0 \
  --cpu {num_cpu} \
  --cpu-limit {num_cpu} \
  --memory 40Gi \
  --memory-limit 80Gi \
  --large-shm \
  --node-pools default,h100 \
  --environment HOME="/lcncluster/becker/.caas_HOME" \
  --run-as-uid 229361 \
  --run-as-gid 20184 \
  --run-as-user \
  --existing-pvc claimname=lcn1-lcncluster,path=/lcncluster \
  --existing-pvc claimname=lcn1-scratch,path=/scratch \
  --working-dir /lcncluster \
  --command \
  -- /bin/bash /lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/{name_exp}.sh
''')
        
    print(f'Exp file saved as {os.path.join(path_runai,f"submit_{name_exp}.sh")}')

# Save dataframe with simulation info ###############################################################################################
info = {'study_name': name_set,
        'model_name': name_model,
        'model_k_type': k_type,
        'grid_set_num': set_num,
        'grid_num_sim': num_sim,
        'grid_init_seed': init_seed,
        'path_results': path_results,
        'path_config': path_config,
        'path_exp': path_exp,
        'path_runai': path_runai
        }

with open(f'{path_config_summary}summary_{name_set}.json', 'w') as fp:
    json.dump(info,fp)
