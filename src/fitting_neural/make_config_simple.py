import json
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl
import os

# Path variables
cluster         = True # set True if running on cluster
input_corr      = True # set True if running with corrected Homann input (corrected exp. protocol + novelty response readout)
input_seq_mode  = 'sep' # 'sep': run Homann exp. with the same image sets for each parameter value (e.g. different values of M), independently; 'app': appended format of the experiment, with different image sets per parameter value.

set_num         = 28
name_set        = f'set{set_num}_simple_cells' + '_seq' + input_seq_mode
name_proj       = '2024-08_grid_search_manual' + ('_corr' if input_corr else '')

path_config_summary = './src/scripts/gabor_kernels/grid_search_manual/'
path_config         = f'{path_config_summary}configs_{name_set}/'
path_exp            = f'/Volumes/lcncluster/becker/RL_reward_novelty/exps/{name_proj}/{name_set}/' 
path_yaml           = f'/Users/sbecker/yaml_files/yaml_rlnet/{name_proj}/{name_set}/'

sl.make_long_dir(path_config)
sl.make_long_dir(path_exp)  
sl.make_long_dir(path_yaml)

# Model parameters to be varied
adj_w       = [True, True]
adj_f       = [(not i) for i in adj_w]
alph_adj    = [3, 3]
num_gabor   = [40, 40]
k_type      = ['box', 'triangle']
# adj_w       = [True]
# adj_f       = [(not i) for i in adj_w]
# alph_adj    = [3]
# num_gabor   = [40]
# k_type      = ['box']
num_sim     = 50        # number of homann simulations for each trial (i.e. in each optimization step)
num_cpu     = 40
init_seed   = 12345
parallel_exp    = False
parallel_grid   = True

# Initialize info dataframe
info_unique = ('name_set',
               'adj_w',
               'adj_f',
               'alph_adj',
               'num_gabor',
               'num_sim',
               'init_seed',
               'save_path',
               'study_name'
               )
info_columns = list(info_unique) + ['path_config', 'path_exp', 'path_yaml']
df = pd.DataFrame(columns=info_columns)

for i in range(len(adj_w)):
    # Build config file #################################################################################################################
    params_input = {'num_gabor': num_gabor[i],
                    'adj_w':     adj_w[i],
                    'adj_f':     adj_f[i],
                    'alph_adj':  alph_adj[i],
                    'n_fam':     [1,3,8,18,38],
                    'n_im':      [3,6,9,12],
                    'dN':        list(np.array([0,22,44,66,88,110,143])/0.3), #[0,70,140,210,280,360,480],
                    'idx':       True
                    }
    
    if set_num == 21:
      grid = {'k_alph':[0.2,0.3,0.4],
              'cdens':[4,8],
              'knum':[2,4,6,8]
              }
    elif set_num == 22:
      grid = {'k_alph':[0.6,0.7,0.8],
              'cdens':[4,8],
              'knum':[2,4,6,8]
              }
      
    elif set_num == 23:
      grid = {'k_alph':[0.2,0.3,0.4],
              'cdens':[4,8],
              'knum':[10,20,40]
              }
    elif set_num == 24:
      grid = {'k_alph':[0.6,0.7,0.8],
              'cdens':[4,8],
              'knum':[10,20,40]
              }
      
    elif set_num == 25:
      grid = {'k_alph':[0.2,0.3,0.4],
              'cdens':[16,32],
              'knum':[2,4,6,8]
              }
    elif set_num == 26:
      grid = {'k_alph':[0.6,0.7,0.8],
              'cdens':[16,32],
              'knum':[2,4,6,8]
              }
      
    elif set_num == 27:
      grid = {'k_alph':[0.2,0.3,0.4],
              'cdens':[16,32],
              'knum':[10,20,40]
              }
    elif set_num == 28:
      grid = {'k_alph':[0.6,0.7,0.8],
              'cdens':[16,32],
              'knum':[10,20,40]
              }
      
    elif set_num == 2:
      grid = {'k_alph':[0.01,0.1,0.5,0.9],
              'cdens':[4,8,16,32],
              'knum':[2,4,6,8,10,20,40]
              }
      
    k_params_ext = {'k_type': k_type[i],
                    'ksig': 1
                    }

    study_name = f'simple_cells{"-corr" if input_corr else ""}_{k_type[i]}' + (f'_adj_w{alph_adj[i]}' if adj_w[i] else f'_adj_f{alph_adj[i]}') + f'_G-{num_gabor[i]}'
    save_path  = ('/lcncluster/becker/RL_reward_novelty/data/' if cluster else '/Users/sbecker/Projects/RL_reward_novelty/data/') + f'{name_proj}/{name_set}/{study_name}/'

    config = {'grid': grid,
              'k_params_ext': k_params_ext,
              'num_sim': num_sim,  
              'params_input': params_input,
              'init_seed': init_seed,
              'parallel_exp': parallel_exp,
              'parallel_grid': parallel_grid,
              'num_cpu': num_cpu,
              'save_path': save_path,
              'comp_fit': True,
              'comp_corr': True,
              'cluster': cluster,
              'input_corrected': input_corr,
              'input_sequence_mode': input_seq_mode,
              'study_name': study_name
              }

    name_config = study_name

    with open(os.path.join(path_config,f'{name_config}.json'), 'w') as fp:
        json.dump(config, fp)
    print(f'Config file saved as {os.path.join(path_config,f"{name_config}.json")}')

    # Build exp file ####################################################################################################################
    name_exp = study_name
    
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
python -u -b ${{base_path}}/src/scripts/gabor_kernels/grid_search_manual/grid_search_simple_cells.py -c ${{base_path}}/src/scripts/gabor_kernels/grid_search_manual/configs_{name_set}/{name_config}.json | tee ${{base_path}}/logs/{name_proj}/{name_set}/${{log_folder}}/log.txt
''')
        
    print(f'Exp file saved as {os.path.join(path_exp,f"{name_exp}.sh")}')

    # Build yaml file ###################################################################################################################
    name_yaml = study_name
    if len(name_yaml)>62:
        name_yaml_job = name_yaml[:62]
    else:
        name_yaml_job = name_yaml
    
    with open (os.path.join(path_yaml,f'{name_yaml}.yaml'), 'w') as rsh:
        rsh.write(f'''\
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata:
  name: {name_yaml_job.replace('_','-').lower()}-{set_num} # e.g. training-pod
spec:
  image:
    value: nvcr.io/nvidia/pytorch:20.03-py3
  name:
    value: {name_yaml_job.replace('_','-').lower()}-{set_num} # e.g. training-pod
  command:
    value: "/bin/bash" # bash commands as args below, e.g. using a custom conda installation on the lcncluster
  arguments: 
    value: {os.path.join(path_exp.replace('/Volumes',''),f'{name_exp}.sh')}
  environment:
    items:
      HOME:
        value: /lcncluster/becker/.caas_HOME # PATH to HOME e.g. /lcncluster/user/.caas_HOME
  runAsUser:
    value: true # system automatically fetches and applies UID and GID
  cpu:
    value: "{num_cpu}"
  cpuLimit:
    value: "{num_cpu}"
  gpu:
    value: "0" # up to 4, 0 for CPU only
  memory:
    value: 80Gi
  memoryLimit:
    value: 80Gi
  pvcs:
    items:
      pvc--0: # First is "pvc--0", second is "pvc--1", etc.
        value: 
          claimName: runai-lcn1-sbecker-lcncluster
          existingPvc: true
          path: /lcncluster
  nodePools:
    value: "default" # S8 nodes are now named default, if using GPUs, put "g10 g9"
''')
        
    print(f'Yaml file saved as {os.path.join(path_yaml,f"{name_yaml}.yaml")}')

    # Save dataframe with simulation info ###############################################################################################
    params_df = config.copy()
    params_df.update(params_input)
    params_df['name_set']       = name_set
    params_df['path_config']    = os.path.join(path_config,f'{name_config}.yaml')
    params_df['path_exp']       = os.path.join(path_exp,f'{name_exp}.yaml')
    params_df['path_yaml']      = os.path.join(path_yaml,f'{name_yaml}.yaml')
    params_df1 = {k: params_df[k] for k in info_unique}
    
    df = df.append(params_df1, ignore_index=True)

# Add info dataframe to overview file ###############################################################################################
if os.path.isfile(f'{path_config_summary}config_{name_set}.csv'):
    df_old = pd.read_csv(f'{path_config_summary}config_{name_set}.csv')
    df = pd.concat([df_old, df], ignore_index=True)
    df = df.drop_duplicates(subset=info_unique,keep='last')
df.to_csv(f'{path_config_summary}config_{name_set}.csv', index=False)