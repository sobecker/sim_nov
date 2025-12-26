import json
import numpy as np
import utils.saveload as sl

softplus_inv = lambda x: np.log(np.exp(x)-1)
softplus     = lambda x: np.log(np.exp(x)+1)
logit        = lambda x: np.log(x/(1-x))
sigmoid      = lambda x: 1/(1+np.exp(-x))

leakiness_type = 'leaky'
multinov_type  = '_multinov-eps' # '_multinov', '_multinov-eps', '_multinov-alph', 'multinov-eps-alph'
alg_type = [
            # f'{leakiness_type}{multinov_type}_hnor_triangle',
            # f'{leakiness_type}_hnor_center-triangle',
            # f'{leakiness_type}{multinov_type}_hnor_notrace_center-box',
            f'{leakiness_type}{multinov_type}_hnor_notrace'
            ] 
levels   = [[1,6], [2,6], [3,6], [4,6], [5,6]] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

# alg_type = [f'{leakiness_type}{multinov_type}_hnor_placefields']
# levels   = [[1], [2], [3], [4], [5], [1,3,5], [1,2,3,4,5]] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

single_run_id = '_8' # '', '_1', '_2', ... to distinguish multiple single runs with different initial conditions
transformed     = True # use transformed optimization (e.g. log, softplus) --> use version = 4!
version         = 4 if transformed else 5 

data_type   = 'mice'        # 'mice', deprecated: 'naive', 'opt' 
opt_method  = 'Nelder-Mead' # 'Nelder-Mead', 'L-BFGS-B', 'SLSQP'
comb_type   = 'app'         # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False         # set to False for single run with user-specified x0
local       = False         # running on local machine
parallel    = True          # running parallelized

for i in range(len(alg_type)):
    if 'multinov' in alg_type[i]:
        notrace_i       = False if 'triangle' in alg_type[i] else True
        center_i        = True if 'center' in alg_type[i] else False
        center_type_i   = alg_type[i].split('_center-')[-1].split('_')[0] if 'center' in alg_type[i] else 'box' # 'box','triangle'
        notrace_list_i  = [notrace_i, True] 
        center_list_i   = [center_i, False]

        kwargs_i = {'notrace': notrace_list_i,
                    'center': center_list_i,
                    'center_type': [center_type_i, 'box']}
    else:
        kwargs_i = {}

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
    elif single_run_id=='_6':
        l_x0    = [0.35, 5, 0.01, 0.7]
    elif single_run_id=='_7':
        l_x0    = [0.35, 10, 0.01, 0.7]
    elif single_run_id=='_8':
        l_x0    = [0.35, 20, 0.01, 0.7]

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
        l_var.extend(['alph_leak1', 'alph_leak2'] if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['alph_leak'])
        if single_run_id=='':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
        elif single_run_id=='_1':
            l_x0.extend([0.1]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.1])
        elif single_run_id=='_2':
            l_x0.extend([0.9]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.9])
        if not transformed:
            l_bounds.extend([[0.001, 0.999]]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [[0.001,0.999]])
        else:
            l_transfun.extend(['sigmoid']*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['sigmoid'])
            l_transfun_inv.extend(['logit']*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['logit'])

    elif 'nonleaky' in alg_type[i]:
        l_var.extend(['eps_leak1', 'eps_leak2'] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else ['eps_leak'])
        if single_run_id=='':
            l_x0.extend([1]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [1])
        elif single_run_id=='_1':
            l_x0.extend([0.1]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [0.1])
        elif single_run_id=='_2':
            l_x0.extend([0.001]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [0.001])
        if not transformed:
            l_bounds.extend([[0.000001,10]]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [[0.000001,10]])
        else:
            l_transfun.extend(['softplus']*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else ['softplus'])
            l_transfun_inv.extend(['softplus_inv']*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else ['softplus_inv'])


    elif 'leaky' in alg_type[i]:
        l_var.extend(['alph_leak1', 'alph_leak2'] if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['alph_leak'])
        l_var.extend(['eps_leak1', 'eps_leak2'] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else ['eps_leak'])
        if single_run_id=='':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
            l_x0.extend([1]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [1])
        elif single_run_id=='_1':
            l_x0.extend([0.1]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.1])
            l_x0.extend([0.1]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [0.1])
        elif single_run_id=='_2':
            l_x0.extend([0.9]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.9])
            l_x0.extend([0.001]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [0.001])
        elif single_run_id=='_3':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
            l_x0.extend([10,0.1] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [10])
        elif single_run_id=='_4':
            l_x0.extend([0.6]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.6])
            l_x0.extend([10,10] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [0.001])
        elif single_run_id=='_5':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
            l_x0.extend([0.1,10] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [0.001])
        elif single_run_id=='_6':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
            l_x0.extend([1,1] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [1])
        elif single_run_id=='_7':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
            l_x0.extend([5,5] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [5])
        elif single_run_id=='_8':
            l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])
            l_x0.extend([1,1] if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [1])

        if not transformed:
            l_bounds.extend([[0.001,0.999]]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [[0.001,0.999]])
            l_bounds.extend([[0.000001,10]]*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else [[0.000001,10]])
        else:
            l_transfun.extend(['sigmoid']*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['sigmoid'])
            l_transfun_inv.extend(['logit']*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['logit'])
            l_transfun.extend(['softplus']*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else ['softplus'])
            l_transfun_inv.extend(['softplus_inv']*2 if ('multinov' in alg_type[i] and '-eps' in alg_type[i]) else ['softplus_inv'])

    elif 'fixed' in alg_type[i]:
        l_var.extend(['k_alph1', 'k_alph2'] if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['k_alph'])
        l_x0.extend([0.5]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [0.5])

        if not transformed:
            l_bounds.extend([[0.001,0.999]]*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else [[0.001,0.999]])
        else:
            l_transfun.extend(['sigmoid']*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['sigmoid'])
            l_transfun_inv.extend(['logit']*2 if ('multinov' in alg_type[i] and '-alph' in alg_type[i]) else ['logit'])

    if 'multinov' in alg_type[i]:
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
        elif single_run_id=='_6':
            l_x0.append(0.1)
        elif single_run_id=='_7':
            l_x0.append(0.1)
        elif single_run_id=='_8':
            l_x0.append(0.1)

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

    kwargs_i['x0'] = l_x0
    kwargs_i['bounds'] = l_bounds
    kwargs_i['opt_method'] = opt_method
    kwargs_i['transformed'] = l_transformed
    kwargs_i['transfun'] = l_transfun
    kwargs_i['transfun_inv'] = l_transfun_inv
    # kwargs_i['transformed'] = l_transformed
    # kwargs_i['transfun'] = transfun

    assert len(l_var)==len(l_x0), 'Number of variables and initial values do not match!'
    assert len(l_var)==len(l_transformed), 'Number of variables and transformed flags do not match!'
    if not transformed:
        assert len(l_var)==len(l_bounds) or l_bounds is None, 'Number of variables and bounds do not match!'
    else:
        assert len(l_var)==len(l_transfun), 'Number of variables and transformation functions do not match!'
        assert len(l_var)==len(l_transfun_inv), 'Number of variables and inverse transformation functions do not match!'    

    for j in range(len(levels)):
        if 'placefields' in alg_type[i]:
            kwargs_i['placefields'] = [True]*len(levels[j])

        params = {'data_type':data_type,
                    'data_folder':'',
                    'data_path_type': '',
                    'comb_type': comb_type,
                    'var_name': l_var,
                    'kwargs': kwargs_i,
                    'alg_type': alg_type[i],
                    'save_name': f'mle_{alg_type[i]}-l{"-".join(map(str,levels[j]))}',
                    'verbose': True,
                    'parallel': parallel,
                    'local': local,
                    'level': levels[j],
                    'save_path':f'MLE{version}_results/Fits/{"MultiStart/" if randstart else "SingleRun/"}mle_{alg_type[i]}{single_run_id}'}

        # path = './src/scripts/MLE2/mle_fit_configs/'
        path = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'mle' / 'mle_fit_configs'
        sl.make_long_dir(path)
        name = f'{params["save_name"]}{("-" if len(comb_type)>0 else "")}{comb_type}'

        if randstart:
            params["seed"] = 12345      
            params["rand_start"] = 10 
            name = name+'_multi'  

        if data_type=='naive':
            params['data_folder']       = '2022_12_09_13-31-14_sim_nor-tree_naive-nov'
            params['data_path_type']    = 'auto'

        elif data_type=='opt':
            if local:
                params['data_folder']       = '/Volumes/lcncluster/becker/RL_reward_novelty/data/nor_tree/sim_opt/2022_10_07_19-46-06_sim_mbnor_tree-nov-beta1r' 
                params['data_path_type']    = 'manual'
                name = name+'_local'
            else:
                params['data_folder']       = 'nor_tree/sim_opt/2022_10_07_19-46-06_sim_mbnor_tree-nov-beta1r' 
                params['data_path_type']    = 'auto'

        elif data_type=='mice':
            if local:
                params['data_folder']       = '/Volumes/lcncluster/becker/RL_reward_novelty/ext_data/Rosenberg2021/' 
                params['data_path_type']    = 'manual'
                params['save_path']         = f'/Users/sbecker/RL_reward_novelty/data/MLE2_results/Fits/{"MultiStart/" if randstart else "SingleRun/"}mle_{alg_type}-{data_type}_{opt_method}'
            else:
                params['data_folder']       = ''    
                params['data_path_type']    = 'auto'
            
        with open(path / f'{name}.json', 'w') as fp:
            json.dump(params, fp)
