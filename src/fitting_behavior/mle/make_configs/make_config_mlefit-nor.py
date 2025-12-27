import json
import utils.saveload as sl

data_type   = 'mice'         # 'naive', 'opt', 'mice' 
opt_method  = 'Nelder-Mead' # 'Nelder-Mead', 'L-BFGS-B', 'SLSQP'
comb_type   = 'sep'            # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False          # set to False for single run with user-specified x0
local       = False         # running on local machine 
alg_name    = 'leaky_nor'    
# leaky:            optimize both epsilon and alpha jointly            - decaying counts model
# leaky_eps1:       optimize only alpha (eps=1 fixed)                  - decaying counts model
# nonleaky:         optimize only epsilon (alpha=0 fixed)              - decaying counts model
# nonleaky_eps1:    optimize only RL parameters (eps=1, alpha=0 fixed) - decaying counts model
# fixed:            optimize fixed learning rate model

l_var   = ['lambda_N',
           'beta_1',
           'epsilon',
           'k_leak']
l_x0        = [0.5,
               5,
               -8,
               0.5]
l_bounds    = [[0.,0.999],
                [-5,15],
                [-15,15],
                [0.001,0.999]]
l_transformed = [False,
                 True,
                 True,
                 False]
l_transfun      = ['sigmoid', 'softplus', 'softplus', 'sigmoid']
l_transfun_inv  = ['logit', 'softplus_inv', 'softplus_inv', 'logit']

# Non-transformed optimization (included for completeness / comparison)
# l_x0        = [0.5,
#                5,
#                0.0002,
#                0.5]
# l_bounds    = [[0.,0.999],
#                 [0.1,30],
#                 [0.0001,1],
#                 [0.001,0.999]]
# l_transformed = [False,
#                  False,
#                  False,
#                  False]
# transfun = None # softplus, log

if 'nonleaky_eps1' in alg_name:
    print('Using nonleaky_eps1 model')

elif 'leaky_eps1' in alg_name:
    l_var.append('alph_leak')
    l_x0.append()
    l_bounds.append([0.001,0.999])
    l_transformed.append(False)

elif 'nonleaky' in alg_name:

    l_var.append('eps_leak')
    l_x0.append(0.5)
    l_bounds.append([-15,15])
    l_transformed.append(True)

    # Non-transformed version
    # l_var.append('eps_leak')
    # l_x0.append(1)
    # l_bounds.append([0.001,10])
    # l_transformed.append(False)

elif 'leaky' in alg_name:

    l_var.append('alph_leak')
    l_x0.append(0.5)
    l_bounds.append([0.001,0.999])
    l_transformed.append(False)

    l_var.append('eps_leak')
    l_x0.append(0.5)
    l_bounds.append([-15,15])
    l_transformed.append(True)

    # Non-transformed version
    # l_var.append('eps_leak')
    # l_x0.append(1)
    # l_bounds.append([0.001,10])
    # l_transformed.append(False)

elif 'fixed' in alg_name:
    l_var.append('k_alph')
    l_x0.append(0.5)
    l_bounds.append([0.001,0.999])
    l_transformed.append(False)

params = {'data_type':data_type,
            'data_folder':'',
            'data_path_type': '',
            'comb_type': comb_type,
            'var_name': l_var,
            'kwargs': {"x0":l_x0, "bounds":l_bounds, "transformed":l_transformed, "transfun":transfun, "opt_method":opt_method},
            'alg_type': alg_name,
            'save_name': f'mle_{alg_name}-{data_type}',
            'verbose': True}

# if local:
#     path = '/Users/sbecker/Projects/RL_reward_novelty/src/scripts/MLE4/mle_fit_configs/'
# else:
#     path = '/Volumes/lcncluster/becker/RL_reward_novelty/src/scripts/MLE4/mle_fit_configs/'
path = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'mle' / 'mle_fit_configs'
sl.make_long_dir(path)
name = f'{params["save_name"]}_{opt_method}{("-" if len(comb_type)>0 else "")}{comb_type}'

if randstart:
    params["seed"] = 12345      
    params["rand_start"] = 10 
    name = name+'_multi'  

if data_type=='mice':
    params['data_folder']       = str(sl.get_rootpath() / 'ext_data' / 'Rosenberg2021')    
    params['data_path_type']    = 'manual'
else:
    raise ValueError('data_type not recognized')

with open(path / f'{name}.json', 'w') as fp:
    json.dump(params, fp)
