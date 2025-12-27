import json
import numpy as np
import utils.saveload as sl

softplus_inv = lambda x: np.log(np.exp(x)-1)
softplus     = lambda x: np.log(np.exp(x)+1)
logit        = lambda x: np.log(x/(1-x))
sigmoid      = lambda x: 1/(1+np.exp(-x))

leakiness_type = 'leaky'
alg_type       = [f'{leakiness_type}_hnor_notrace', 
                    # f'{leakiness_type}_hnor_center-triangle',
                    # f'{leakiness_type}_hnor_notrace_center-box',
                    # f'{leakiness_type}_hnor_notrace'
                    ] 
levels         = [1,2,3,4,5,6] # level 1: component center is the second branching point, level 6: component centers are the leaf nodes
single_run_id  = '_5' # '', '_1', '_2', '_3', '_4', '_5' to distinguish the five different parameter initializations for the optimization 
transformed = True # use transformed optimization (e.g. log, softplus) 
data_type   = 'mice'        # 'mice', deprecated: 'naive', 'opt' 
opt_method  = 'Nelder-Mead' # 'Nelder-Mead', 'L-BFGS-B', 'SLSQP'
comb_type   = 'app'         # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False         # set to False for single run with user-specified x0
local       = False         # running on local machine
parallel    = True          # running parallelized

for i in range(len(alg_type)):
    l_var       = ['lambda_N','beta_1','epsilon','k_leak']
    if single_run_id=='':
        l_x0    = [0.5, 5, 0.0002, 0.5]
    elif single_run_id=='_1':
        l_x0    = [0.3, 10, 0.2, 0.1]
    elif single_run_id=='_2':
        l_x0    = [0.7, 2, 0.02, 0.9]
    elif single_run_id=='_3':
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

    if 'nonleaky_eps1' in alg_type[i]:
        print('Using nonleaky_eps1 model')

    elif 'leaky_eps1' in alg_type[i]:
        l_var.append('alph_leak')
        if single_run_id=='':
            l_x0.append(0.5)
        elif single_run_id=='_1':
            l_x0.append(0.1)
        elif single_run_id=='_2':
            l_x0.append(0.9)

        if not transformed:
            l_bounds.append([0.001,0.999])
        else:
            l_transfun.append('sigmoid')
            l_transfun_inv.append('logit')
    
    elif 'nonleaky' in alg_type[i]:
        l_var.append('eps_leak')
        if single_run_id=='':
            l_x0.append(1)
        elif single_run_id=='_1':
            l_x0.append(0.1)
        elif single_run_id=='_2':
            l_x0.append(0.001)

        if not transformed:
            l_bounds.append([0.000001,10])
        else:
            l_transfun.append('softplus')
            l_transfun_inv.append('softplus_inv')

    elif 'leaky' in alg_type[i]:
        l_var.append('alph_leak')
        l_var.append('eps_leak')
        if single_run_id=='':
            l_x0.append(0.5)
            l_x0.append(1)
        elif single_run_id=='_1':
            l_x0.append(0.1)
            l_x0.append(0.1)
        elif single_run_id=='_2':
            l_x0.append(0.9)
            l_x0.append(0.001)
        elif single_run_id=='_3':
            l_x0.append(0.5)
            l_x0.append(10)
        elif single_run_id=='_4':
            l_x0.append(0.6)
            l_x0.append(0.001)
        elif single_run_id=='_5':
            l_x0.append(0.1)
            l_x0.append(10)

        if not transformed:
            l_bounds.append([0.001,0.999])
            l_bounds.append([0.000001,10])
        else:
            l_transfun.append('sigmoid')
            l_transfun_inv.append('logit')
            l_transfun.append('softplus')
            l_transfun_inv.append('softplus_inv')
        
    elif 'fixed' in alg_type[i]:
        l_var.append('k_alph')
        l_x0.append(0.5)

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

    for j in range(len(levels)):
        params = {'data_type':          data_type,
                    'data_folder':      str(sl.get_rootpath() / 'ext_data' / 'Rosenberg2021'),
                    'data_path_type':   'manual',
                    'comb_type':        comb_type,
                    'var_name':         l_var,
                    'kwargs':           {"x0":l_x0, "bounds":l_bounds, "transformed":l_transformed, "transfun":l_transfun, "transfun_inv":l_transfun_inv, "opt_method":opt_method},
                    'alg_type':         alg_type[i],
                    'save_name':        f'mle_{alg_type[i]}-l{levels[j]}-{data_type}',
                    'verbose':          True,
                    'debug':            False,
                    'parallel':         parallel,
                    'local':            local,
                    'level':            levels[j],
                    'save_path':        str(sl.get_rootpath() / 'data' / 'mle_results' / f'fits_{"multi-start/" if randstart else "single-run"}' / f'mle_{alg_type[i]}-{data_type}_{opt_method}')}

        path = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'mle' / 'mle_fit_configs'
        sl.make_long_dir(path)
        name = f'{params["save_name"]}_{opt_method}{("-" if len(comb_type)>0 else "")}{comb_type}'

        if randstart:
            params["seed"] = 12345      
            params["rand_start"] = 10 
            name = name+'_multi'  
            
        with open(path / f'{name}.json', 'w') as fp:
            json.dump(params, fp)
