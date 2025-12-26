import os
import json
import utils.saveload as sl

################################################################################################################################################
# This scripts creates the config files for run_fit_withconfig.py for four different types of models:
# 1. Leaky count-based novelty (model='cnov_leaky')
# 2. Fixed learning rate count-based novelty (model='cnov_fr')
# 3. Leaky similarity-based novelty (model='snov-complex-leaky')
# 4. Fixed learning rate similarity-based novelty (model='snov-complex-fr')
################################################################################################################################################

path_configs = sl.get_rootpath() / 'src' / 'fitting_neural' / 'configs_fitting'
sl.make_long_dir(path_configs)

# Specify models and settings for which to create configs -- EDIT HERE TO CREATE DIFFERENT CONFIGS --
# Case 1: Full fitting for leaky count-based novelty
# models          = ['cnov_leaky', 'cnov_leaky', 'cnov_leaky']
# set_num         = ['', '', '']

# Case 2: Full fitting for fixed learning rate count-based novelty
# models          = ['cnov_fr', 'cnov_fr', 'cnov_fr']
# set_num         = ['', '', '']

# Case 3: Full fitting for leaky similarity-based novelty
models          = ['snov-complex-leaky', 'snov-complex-leaky', 'snov-complex-leaky']
set_num         = [1, 1, 1]

# Case 4: Full fitting for fixed learning rate similarity-based novelty
# models          = ['snov-complex-fr', 'snov-complex-fr', 'snov-complex-fr']
# set_num         = [1, 1, 1]

# Three different fits: 
# (i) normal fitting (MSE), 
# (ii) cross-validation (jackknife resampling), 
# (iii) jackknife resampling of the cross-validation fit
sampling_type   = ['normal', 'jackknife-loo', 'outerjack-cv'] 
robustness      = [False] * len(set_num)                # set to TRUE for robustness control
weighting       = ['equal-samples'] * len(set_num)
drop_type       = ['sample'] * len(set_num)
run_from_cluster = False  # whether to run the fitting from cluster (True) or locally (False)

################################################################################################################################################
for i, (m,sn,rb,w,d,st) in enumerate(zip(models, set_num, robustness, weighting, drop_type, sampling_type)):
    # Create config
    config = {'model':          m,  
              'set_num':        sn,  
              'robustness':     rb,             
              'comp_type':      'fit',             
              'weighting':      w,   
              'drop_type':      d,            
              'sampling_type':  st, 
              'run_from_cluster': run_from_cluster  
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



