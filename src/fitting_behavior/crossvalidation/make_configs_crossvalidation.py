import numpy as np
import json
import utils.saveload as sl

softplus_inv = lambda x: np.log(np.exp(x)-1)
softplus     = lambda x: np.log(np.exp(x)+1)
logit        = lambda x: np.log(x/(1-x))
sigmoid      = lambda x: 1/(1+np.exp(-x))

leakiness_type_all  = 'leaky'
runai_name_combined = f'{leakiness_type_all}_multinov' # For paper results, use '{leakiness_type_all}_multinov'

if runai_name_combined==f'{leakiness_type_all}_cnov':
    model_type_list     = ['nor','nac','hybrid2']
    level_list          = [None, None, None]
    kernel_type_list    = [None, None, None]

elif runai_name_combined==f'{leakiness_type_all}_triangle':
    model_type_list     = ['hnor', 'hhybrid2']*4
    level_list          = [3]*2 + [4]*2 + [5]*2 + [6]*2
    kernel_type_list    = ['triangle']*8

elif runai_name_combined==f'{leakiness_type_all}_center-triangle':
    model_type_list     = ['hnor', 'hhybrid2']*4
    level_list          = [3]*2 + [4]*2 + [5]*2 + [6]*2
    kernel_type_list    = ['center-triangle']*8

elif runai_name_combined==f'{leakiness_type_all}_notrace':
    model_type_list     = ['hnor', 'hhybrid2']*4
    level_list          = [3]*2 + [4]*2 + [5]*2 + [6]*2
    kernel_type_list    = ['notrace']*8

elif runai_name_combined==f'{leakiness_type_all}_notrace_center-box':
    model_type_list     = ['hnor', 'hhybrid2']*4
    level_list          = [3]*2 + [4]*2 + [5]*2 + [6]*2
    kernel_type_list    = ['notrace_center-box']*8

elif runai_name_combined==f'{leakiness_type_all}_all':
    model_type_list     = ['nor','nac','hybrid2'] + ['hnor', 'hhybrid2']*4 + ['hnor', 'hhybrid2']*4 + ['hnor', 'hhybrid2']*4 + ['hnor', 'hhybrid2']*4
    level_list          = [None, None, None] + [3]*2 + [4]*2 + [5]*2 + [6]*2 + [3]*2 + [4]*2 + [5]*2 + [6]*2 + [3]*2 + [4]*2 + [5]*2 + [6]*2 + [3]*2 + [4]*2 + [5]*2 + [6]*2
    kernel_type_list    = [None, None, None] + ['triangle']*8 + ['center-triangle']*8 + ['notrace']*8 + ['notrace_center-box']*8

elif runai_name_combined==f'{leakiness_type_all}_multinov':
    # Note: rerun crossvalidation only for best parameter initializations of the optimization procedure.
    model_type_list     = ['multinov-eps_hnor'] * 10 + ['hnor'] * 12
    kernel_type_list    = ['notrace'] * 22
    level_list          = [[1,6],[1,6],[2,6],[2,6],[3,6],[3,6],[4,6],[4,6],[5,6],[5,6]] + [[1],[1],[2],[2],[3],[3],[4],[4],[5],[5],[6],[6]] 
    single_run_id_list  = ['_2','_3','_2','_3','_2','_3','_1','_3','_1','_3'] + ['_1','_5','_1','_5','_4','_5','_1','_5','_1','_5','_2','_5'] 

# Shared parameters
leakiness_type_list = [f'{leakiness_type_all}'] * len(model_type_list)
comb_type           = 'app'
parallel            = True  # True
run_local           = False # False
num_cpu             = 24 # 12  / 20 
opt_method          = 'Nelder-Mead'
transformed         = True # use transformed optimization (e.g. log, softplus) --> use version = 4!

split_type          = 'mouseid'     # 'mouseid' or 'pathlen'
kfold               = 5     # only used when split_type=='mouseid'
seed_crossval       = 42957 # only used when split_type=='mouseid' 98765 / 62530 / 42957 / ...
test_order          = 'testset_last'    # only used when split_type=='pathlen'
test_ratio_list     = [0.25, 0.5, 0.75]             # only used when split_type=='pathlen'

# Set paths
path_config = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'crossvalidation' / f'configs_cv-{split_type}'
path_exp = sl.get_rootpath() / 'exp' / 'crossvalidation' / f'cv-{split_type}'
sl.make_long_dir(path_config)
sl.make_long_dir(path_exp)

for i, (model_type, level, kernel_type, leakiness_type) in enumerate(zip(model_type_list, level_list, kernel_type_list, leakiness_type_list)):

    single_run_id = single_run_id_list[i] if 'single_run_id_list' in locals() else ''

    if level is not None:
        if isinstance(level, list):
            level_str = 'l' + '-'.join([str(l) for l in level])
        else:
            level_str = f'l{level}'
        base_name = f'{leakiness_type}_{model_type}_{kernel_type}-{level_str}_{comb_type}{single_run_id}'
    else:
        base_name = f'{leakiness_type}_{model_type}_{comb_type}{single_run_id}'

    # Set fitting parameters
    if 'nor' in model_type:

        if 'multinov' in model_type:            # multiple novelty distributions with separate sets of components
            notrace_i       = False if 'triangle' in kernel_type else True
            center_i        = True if 'center' in kernel_type else False
            center_type_i   = kernel_type.split('_center-')[-1].split('_')[0] if 'center' in kernel_type else 'box' # 'box','triangle'
            notrace_list_i  = [notrace_i, True] 
            center_list_i   = [center_i, False]

            kwargs_i = {'notrace': notrace_list_i,
                        'center': center_list_i,
                        'center_type': [center_type_i, 'box']}
        else:                             # single novelty distribution with single set of components                   
            kwargs_i = {}

        # Shared parameters for all 'nor' models
        l_var       = ['lambda_N','beta_1','epsilon','k_leak']
        if single_run_id=='':
            l_x0    = [0.5, 5, 0.0002, 0.5]
        elif single_run_id=='_1': # good init
            l_x0    = [0.3, 10, 0.2, 0.1]
        elif single_run_id=='_2':
            l_x0    = [0.7, 2, 0.02, 0.9]
        elif single_run_id=='_3': # good init
            l_x0    = [0.4, 50, 0.02, 0.9]
        elif single_run_id=='_4':
            l_x0    = [0.3, 100, 0.0002, 0.95]
        elif single_run_id=='_5':
            l_x0    = [0.2, 2, 0.02, 0.99]

        if not transformed:
            l_bounds = [[0.,0.999],          #lambda_N - update rate of novelty Q-values during prioritized sweeping update
                        [0.1,30],           #beta_1   - inverse temperature of softmax policy
                        [0.000001,10],      #epsilon  - prior belief about novelty values (uniform across states)
                        [0.001,0.999]]      #k_leak   - leakiness of beliefs about novelty values 
        else:
            l_transfun      = ['sigmoid', 'softplus', 'softplus', 'sigmoid']
            l_transfun_inv  = ['logit', 'softplus_inv', 'softplus_inv', 'logit']
            
    else:
        raise ValueError(f"Unknown model type: {model_type}.")

    # Parameters depending on leakiness type
    if 'nonleaky_eps1' in leakiness_type:
        print('Using nonleaky_eps1 model')

    elif 'leaky_eps1' in leakiness_type:
        l_var.extend(['alph_leak1', 'alph_leak2'] if ('multinov' in model_type and '-alph' in model_type) else ['alph_leak'])
        if single_run_id=='':
            l_x0.extend([0.5]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.5])
        elif single_run_id=='_1':
            l_x0.extend([0.1]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.1])
        elif single_run_id=='_2':
            l_x0.extend([0.9]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.9])
        if not transformed:
            l_bounds.extend([[0.001, 0.999]]*2 if ('multinov' in model_type and '-alph' in model_type) else [[0.001,0.999]])
        else:
            l_transfun.extend(['sigmoid']*2 if ('multinov' in model_type and '-alph' in model_type) else ['sigmoid'])
            l_transfun_inv.extend(['logit']*2 if ('multinov' in model_type and '-alph' in model_type) else ['logit'])

    elif 'nonleaky' in leakiness_type:
        l_var.extend(['eps_leak1', 'eps_leak2'] if ('multinov' in model_type and '-eps' in model_type) else ['eps_leak'])
        if single_run_id=='':
            l_x0.extend([1]*2 if ('multinov' in model_type and '-eps' in model_type) else [1])
        elif single_run_id=='_1':
            l_x0.extend([0.1]*2 if ('multinov' in model_type and '-eps' in model_type) else [0.1])
        elif single_run_id=='_2':
            l_x0.extend([0.001]*2 if ('multinov' in model_type and '-eps' in model_type) else [0.001])
        if not transformed:
            l_bounds.extend([[0.000001,10]]*2 if ('multinov' in model_type and '-eps' in model_type) else [[0.000001,10]])
        else:
            l_transfun.extend(['softplus']*2 if ('multinov' in model_type and '-eps' in model_type) else ['softplus'])
            l_transfun_inv.extend(['softplus_inv']*2 if ('multinov' in model_type and '-eps' in model_type) else ['softplus_inv'])

    elif 'leaky' in leakiness_type:
        l_var.extend(['alph_leak1', 'alph_leak2'] if ('multinov' in model_type and '-alph' in model_type) else ['alph_leak'])
        l_var.extend(['eps_leak1', 'eps_leak2'] if ('multinov' in model_type and '-eps' in model_type) else ['eps_leak'])
        if single_run_id=='':
            l_x0.extend([0.5]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.5])
            l_x0.extend([1]*2 if ('multinov' in model_type and '-eps' in model_type) else [1])
        elif single_run_id=='_1':
            l_x0.extend([0.1]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.1])
            l_x0.extend([0.1]*2 if ('multinov' in model_type and '-eps' in model_type) else [0.1])
        elif single_run_id=='_2':
            l_x0.extend([0.9]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.9])
            l_x0.extend([0.001]*2 if ('multinov' in model_type and '-eps' in model_type) else [0.001])
        elif single_run_id=='_3':
            l_x0.extend([0.5]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.5])
            l_x0.extend([10,0.1] if ('multinov' in model_type and '-eps' in model_type) else [10])
        elif single_run_id=='_4':
            l_x0.extend([0.6]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.6])
            l_x0.extend([10,10] if ('multinov' in model_type and '-eps' in model_type) else [0.001])
        elif single_run_id=='_5':
            l_x0.extend([0.5]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.5])
            l_x0.extend([0.1,10] if ('multinov' in model_type and '-eps' in model_type) else [0.001])

        if not transformed:
            l_bounds.extend([[0.001,0.999]]*2 if ('multinov' in model_type and '-alph' in model_type) else [[0.001,0.999]])
            l_bounds.extend([[0.000001,10]]*2 if ('multinov' in model_type and '-eps' in model_type) else [[0.000001,10]])
        else:
            l_transfun.extend(['sigmoid']*2 if ('multinov' in model_type and '-alph' in model_type) else ['sigmoid'])
            l_transfun_inv.extend(['logit']*2 if ('multinov' in model_type and '-alph' in model_type) else ['logit'])
            l_transfun.extend(['softplus']*2 if ('multinov' in model_type and '-eps' in model_type) else ['softplus'])
            l_transfun_inv.extend(['softplus_inv']*2 if ('multinov' in model_type and '-eps' in model_type) else ['softplus_inv'])

    elif 'fixed' in leakiness_type:
        l_var.extend(['k_alph1', 'k_alph2'] if ('multinov' in model_type and '-alph' in model_type) else ['k_alph'])
        l_x0.extend([0.5]*2 if ('multinov' in model_type and '-alph' in model_type) else [0.5])

        if not transformed:
            l_bounds.extend([[0.001,0.999]]*2 if ('multinov' in model_type and '-alph' in model_type) else [[0.001,0.999]])
        else:
            l_transfun.extend(['sigmoid']*2 if ('multinov' in model_type and '-alph' in model_type) else ['sigmoid'])
            l_transfun_inv.extend(['logit']*2 if ('multinov' in model_type and '-alph' in model_type) else ['logit'])

    # Extra parameters for multinov models
    if 'multinov' in model_type:
        l_var.append('w_cnov')
        if single_run_id=='':
            l_x0.append(0.5)
        elif single_run_id=='_1':
            l_x0.append(0.1)
        elif single_run_id=='_2':
            l_x0.append(0.9)
        elif single_run_id=='_3':
            l_x0.append(0.1)
        elif single_run_id=='_4':
            l_x0.append(0.001)
        elif single_run_id=='_5':
            l_x0.append(0.9)

        if not transformed:
            l_bounds.append([0.001,0.999])
        else:
            l_transfun.append('sigmoid')
            l_transfun_inv.append('logit')
    
    l_transformed = [transformed]*len(l_var)
    if transformed:
        l_x0            = [eval(fun_inv)(x0) for x0, fun_inv in zip(l_x0, l_transfun_inv)] # transform x0 into transformed space
        l_bounds        = None # no bounds needed for transformed optimization
    else:
        l_transfun      = [None]*len(l_var)
        l_transfun_inv  = [None]*len(l_var)

    # Set kwargs
    kwargs_i['x0'] = l_x0
    kwargs_i['bounds'] = l_bounds
    kwargs_i['opt_method'] = opt_method
    kwargs_i['transformed'] = l_transformed
    kwargs_i['transfun'] = l_transfun
    kwargs_i['transfun_inv'] = l_transfun_inv

    # Create config file
    config = {
                "leakiness_type":   leakiness_type,
                "model_type":       model_type,
                "level":            level,
                "kernel_type":      kernel_type,
                "comb_type":        comb_type,
                "single_run_id":    single_run_id,
                "run_local":        run_local,
                "split_type":       split_type,
                "kfold":            kfold,
                "seed_crossval":    seed_crossval,
                "test_order":       test_order,
                "test_ratio_list":  test_ratio_list,
                "var_name":         l_var,
                "parallel":         parallel,
                "kwargs":           kwargs_i,
            }
    
    with open(path_config / f'{base_name}.json', 'w') as fp:
            json.dump(config, fp)

    # Create exps file
    with open(path_exp / f'{base_name}.sh', 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash
echo "creating directory"
log_folder="$(date +'%Y-%m-%d_%H-%M-%S')_{base_name}"
base_path="/lcncluster/becker/sim_nov"
echo "folder name: ${{log_folder}}"
mkdir -p ${{base_path}}/logs/crossvalidation/
mkdir -p ${{base_path}}/logs/crossvalidation/cv-{split_type}/
mkdir -p ${{base_path}}/logs/crossvalidation/cv-{split_type}/${{log_folder}}

echo "build {base_name}"
python -u -b ${{base_path}}/src/fitting_behavior/crossvalidation/LL_crossvalidation.py -c ${{base_path}}/src/fitting_behavior/crossvalidation/configs_cv-{split_type}/{base_name}.json | tee ${{base_path}}/logs/crossvalidation/cv-{split_type}/${{log_folder}}/log_{base_name}.txt
''')