import os
import json
import utils.saveload as sl

path_configs = '/Users/sbecker/Projects/RL_reward_novelty/src/scripts/gabor_kernels/grid_search_new/configs_fitting/'
sl.make_long_dir(path_configs)

models          = ['snov-complex-fr', 'snov-complex-fr', 'snov-complex-fr']
set_num         = [[7,8], [7,8], [7,8]]
robustness      = [False] * len(set_num)
weighting       = ['equal-exp'] * len(set_num)
drop_type       = ['exp'] * len(set_num)
sampling_type   = ['jackknife-loo', 'outerjack-cv', 'normal'] 

for i, (m,sn,rb,w,d,st) in enumerate(zip(models, set_num, robustness, weighting, drop_type, sampling_type)):
    # Create config
    config = {'model':          m,  
              'set_num':        sn,  
              'robustness':     rb,             
              'comp_type':      'fit',             
              'weighting':      w,   
              'drop_type':      d,            
              'sampling_type':  st, 
              'run_from_cluster': True  
             }
    
    # Create config name
    add_name    = f'{st}'
    if 'jackknife' in st or 'loo' in st or 'outerjack' in st or 'outerboot' in st:
        add_name += f'-drop-{d}'
    if w!='none':
        add_name += f'-{w}'
    if isinstance(sn, list):
        sn_str = '-'.join([str(s) for s in sn])
    else:
        sn_str = str(sn)
    config_name = f'{m}_set{sn_str}{"_rob" if rb else ""}_{add_name}.json'

    # Save config
    with open(os.path.join(path_configs, config_name), 'w') as fp:
        json.dump(config,fp)



