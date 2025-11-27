import os
import numpy as np
import pandas as pd
import utils.saveload as sl

model_name = 'snov-complex-leaky' #'snov-complex-leaky'
set_name = 1
# base_path_data = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2025_07_robustness_shape_width/{model_name}_set{set_name}_sep/'
base_path_data = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2025_07_robustness2_width/{model_name}_set{set_name}_sep/'

# job_names   = np.array(list(np.arange(57)) + list(np.arange(58,63)) + list(np.arange(64,93)))
job_names   = np.array(list(np.arange(94,129)) + list(np.arange(130,132))) #np.array([0,2,3,5,6,7])
# job_names = [57,63,93,129]
set_paths   = [os.path.join(base_path_data,f'job-{jn}/') for jn in job_names]
target_path = os.path.join(base_path_data,f'combined_data/')
sl.make_long_dir(target_path)

# Prepare folders for combining
grid_all = []
done_all = []
grid_cc  = 728

# Check if target folder already has data
# if os.path.exists(os.path.join(target_path,'grid.csv')):
#     grid_init = pd.read_csv(os.path.join(target_path,'grid.csv'))
#     unnamed = [col for col in grid_init.columns if 'Unnamed' in col]
#     if len(unnamed) > 0:
#         grid_init = grid_init.drop(columns=unnamed)
#     grid_cc   = grid_init['grid_id'].max() + 1
#     grid_all.append(grid_init)
#     with open(os.path.join(target_path,'done.txt'),'r') as f:
#         done_init = np.array([int(line.replace('\n','')) for line in f])
#     done_init = np.unique(done_init)
#     done_all.append(done_init)

for ii, jn in enumerate(job_names):

    # Load grid and shift grid_ids
    grid_ii = pd.read_csv(os.path.join(set_paths[ii],'grid.csv'))
    min_idx = grid_ii['grid_id'].min()
    max_idx = grid_ii['grid_id'].max()
    grid_ii['grid_id_old'] = grid_ii['grid_id'].copy()
    grid_ii['grid_id'] = grid_ii['grid_id'] - min_idx + grid_cc
    grid_all.append(grid_ii)

    # Load done file and shift grid_ids
    with open(os.path.join(set_paths[ii],'done.txt'),'r') as f:
        done_ii = np.array([int(line.replace('\n','')) for line in f])
    done_ii = np.unique(done_ii)
    done_ii = done_ii - min_idx + grid_cc
    done_all.append(done_ii)

    # Rename and move all individual grid folders
    for jj in grid_ii['grid_id_old'].values:
        os.rename(os.path.join(set_paths[ii],f'grid_{jj}'),os.path.join(target_path,f'grid_{jj-min_idx+grid_cc}'))
    
    # Update grid_cc
    grid_cc = grid_ii['grid_id'].max() + 1

# Combine grid and done files and save in target location
grid = pd.concat(grid_all)
grid.to_csv(os.path.join(target_path,'grid.csv'))
done = np.concatenate(done_all)
with open(os.path.join(target_path,'done.txt'), 'w') as f:
    [f.write(f'{gi}\n') for gi in done]

print(f'Combined {len(grid)} grids and {len(done)} done files.')



