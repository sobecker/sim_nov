import os
import sys

import numpy as np
import pandas as pd
import utils.saveload as sl

base_path_data = '/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual_corr/'

set_names   = [4,41,42,43,44,45,46,47,48]
target_name = 4
k_type      = 'triangle'
c_type      = 'complex'

set_paths = [os.path.join(base_path_data,f'set{sn}_{c_type}_cells_seqsep/{c_type}_cells-corr_{k_type}_adj_w3_G-40/') for sn in set_names]
target_path = os.path.join(base_path_data,f'set{target_name}_{c_type}_cells_seqsep/{c_type}_cells-corr_{k_type}_adj_w3_G-40/')
sl.make_long_dir(target_path)

# Prepare folders for combining
grid_all = []
done_all = []
grid_cc  = 0
for ii, sn in enumerate(set_names):

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



