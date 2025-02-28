import json
import utils.saveload as sl
from pathlib import Path

levels = [0,1,2,3,4,5,6] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

alg_type    = 'hhybrid2'    # 'hhybrid': fitting both w_mf and w_mb, 'hhybrid2', 'leaky_hhybrid2'
alg_mf      = 'hnac-gn'
alg_mb      = 'hnor'
data_type   = 'mice'        # 'mice', deprecated: 'naive', 'opt'
opt_method  = 'Nelder-Mead' # L-BFGS-B', 'Nelder-Mead', 'SLSQP'
comb_type   = 'app'         # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False         # set to False for single run with user-specified x0
local       = False         # running on local machine 
parallel    = True          # running parallelized

if 'hybrid2' in alg_type:
    l_var   = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0','lambda_N','beta_1','epsilon','k_leak','w_mf']
    l_x0    = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0,    0.5,       5,       0.0002,    0.5,    0.5]
    l_bounds    = [[0.,0.999],      #gamma
                    [0.001,0.5],    #c_alph
                    [0.001,0.5],    #a_alph
                    [0.,0.999],     #c_lam
                    [0.,0.999],     #a_lam
                    [0.001,1.],     #temp
                    [-100,100],     #c_w0
                    [-100,100],     #a_w0
                    [0.,0.999],     #lambda_N
                    [0.1,30],       #beta_1
                    [0.0001,1],     #epsilon
                    [0.001,0.999],  #k_leak
                    [0,1]]          #w_mf
else:
    l_var   = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0','lambda_N','beta_1','epsilon','k_leak','w_mf','w_mb']
    l_x0    = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0,    0.5,       5,       0.0002,    0.5,    0.5,    0.5]
    l_bounds    = [[0.,0.999],      #gamma
                    [0.001,0.5],    #c_alph
                    [0.001,0.5],    #a_alph
                    [0.,0.999],     #c_lam
                    [0.,0.999],     #a_lam
                    [0.001,1.],     #temp
                    [-100,100],     #c_w0
                    [-100,100],     #a_w0
                    [0.,0.999],     #lambda_N
                    [0.1,30],       #beta_1
                    [0.0001,1],     #epsilon
                    [0.001,0.999],  #k_leak
                    [0,1],          #w_mf
                    [0,30]]         #w_mb

if 'leaky' in alg_type:
    l_var.append('k_alph')
    l_x0.append(0.5)
    l_bounds.append([0.001,0.999])

for j in range(len(levels)):
    params = {'data_type': data_type,
            'data_folder': '',
            'comb_type': comb_type,
            'var_name': l_var,
            'kwargs': {"x0":l_x0,"bounds":l_bounds,"opt_method":opt_method},
            'alg_type': alg_type,
            'hyb_type': [alg_mb,alg_mf],
            'verbose': True,
            'parallel': parallel,
            'level':levels[j],
            'save_name': f'mle_{alg_type}-l{levels[j]}-{data_type}_{opt_method}'
            }   

    path = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'mle' / 'mle_fit_configs'
    sl.make_long_dir(path)
    name = f'{params["save_name"]}_{opt_method}{("-" if len(comb_type)>0 else "")}{comb_type}'

    if randstart:
        params["seed"] = 12345      
        params["rand_start"] = 10 
        name = name+'_multi'  

    if data_type=='naive': # deprecated
        params['data_folder'] = '2022_11_17_10-57-08_nAC_debug'
    elif data_type=='opt': # deprecated
        if local:
            params['data_folder'] = '/Volumes/lcncluster/becker/RL_reward_novelty/data/bintree_archive/sim_opt/2022_08_16_11-23-13_gpopt_nAC-N-expl_OI'
            name = name+'_local'
        else:
            params['data_folder'] = 'bintree_archive/sim_opt/2022_08_16_11-23-13_gpopt_nAC-N-expl_OI'
    elif data_type=='mice':
        params['data_folder'] = str(sl.get_rootpath() / 'ext_data' / 'Rosenberg2021') 

    with open(path / f'{name}.json', 'w') as fp:
        json.dump(params, fp)
