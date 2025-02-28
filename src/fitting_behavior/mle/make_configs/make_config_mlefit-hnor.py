import json
import utils.saveload as sl

levels = [0,1,2,3,4,5,6] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

alg_type    = 'hnor' #'leaky_hnor_center-triangle', 'hnor', 'hnor_notrace' 
data_type   = 'mice'        # 'mice', deprecated: 'naive', 'opt' 
opt_method  = 'Nelder-Mead' # 'Nelder-Mead', 'L-BFGS-B', 'SLSQP'
comb_type   = 'app'         # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False         # set to False for single run with user-specified x0
local       = False         # running on local machine
parallel    = True          # running parallelized

l_var       = ['lambda_N','beta_1','epsilon','k_leak']
l_x0        = [0.5,5,0.0002,0.5]
l_bounds    = [[0.,0.999],
                [0.1,30],
                [0.0001,1],
                [0.001,0.999]]
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
                'verbose': True,
                'parallel': parallel,
                'local': local,
                'level':levels[j],
                'save_name': f'mle_{alg_type}-l{levels[j]}-{data_type}'
                }

    path = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'mle' / 'mle_fit_configs'
    sl.make_long_dir(path)    
    name = f'{params["save_name"]}_{opt_method}{("-" if len(comb_type)>0 else "")}{comb_type}'

    if randstart:
        params["seed"] = 12345      
        params["rand_start"] = 10 
        name = name+'_multi'  

    if data_type=='naive': # deprecated
        params['data_folder'] = '2022_12_09_13-31-14_sim_nor-tree_naive-nov'
    elif data_type=='opt': # deprecated
        if local:
            params['data_folder'] = '/Volumes/lcncluster/becker/RL_reward_novelty/data/nor_tree/sim_opt/2022_10_07_19-46-06_sim_mbnor_tree-nov-beta1r' 
            name = name+'_local'
        else:
            params['data_folder'] = 'nor_tree/sim_opt/2022_10_07_19-46-06_sim_mbnor_tree-nov-beta1r'

    with open(path / f'{name}.json', 'w') as fp:
        json.dump(params, fp)
