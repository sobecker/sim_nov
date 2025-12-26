import json
import pandas as pd
import utils.saveload as sl

leakiness_type = 'nonleaky_eps1'
alg_type = [f'{leakiness_type}_hnac-gn_triangle', 
            f'{leakiness_type}_hnac-gn_center-triangle',
            f'{leakiness_type}_hnac-gn_notrace_center-box',
            f'{leakiness_type}_hnac-gn_notrace'] #['hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi']
levels   = [1,2,3,4,5,6]

data_type   = 'mice'        # 'naive', 'opt', 'mice' 
opt_method  = 'Nelder-Mead' # L-BFGS-B', 'Nelder-Mead', 'SLSQP'
comb_type   = 'app'            # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False          # set to False for single run with user-specified x0
local       = False         # running on local machine 
parallel    = True         # running parallelized

for i in range(len(alg_type)):
    l_var   = ['gamma',
                'c_alph',
                'a_alph',
                'c_lam',
                'a_lam',
                # 'temp',
                'temp', # inverse temperature
                'c_w0',
                'a_w0']
    l_x0    = [0.5,
                0.1,
                0.1,
                0.5,
                0.5,
                # 0.5,
                5,
                0,
                0]
    l_bounds    = [[0.,0.999],      #gamma - discount factor
                    [0.001,0.5],    #c_alph - learning rate for critic
                    [0.001,0.5],    #a_alph - learning rate for actor
                    [0.,0.999],     #c_lam - eligibility trace for critic
                    [0.,0.999],     #a_lam - eligibility trace for actor
                    # [0.001,1.],     #temp - softmax temperature
                    [0.1,30],       #inv_temp - inverse softmax temperature
                    [-100,100],     #c_w0 - initial weights for critic
                    [-100,100]]     #a_w0 - initial weights for actor

    if 'nonleaky_eps1' in alg_type[i]:
        print('Using nonleaky_eps1 model')

    elif 'leaky_eps1' in alg_type[i]:
        l_var.append('alph_leak')
        l_x0.append(0.5)
        l_bounds.append([0.001,0.999])
    
    elif 'nonleaky' in alg_type[i]:
        l_var.append('eps_leak')
        l_x0.append(1)
        l_bounds.append([0.001,10])

    elif 'leaky' in alg_type[i]:
        l_var.append('alph_leak')
        l_x0.append(0.5)
        l_bounds.append([0.001,0.999])
        l_var.append('eps_leak')
        l_x0.append(1)
        l_bounds.append([0.001,10])

    elif 'fixed' in alg_type[i]:
        l_var.append('k_alph')
        l_x0.append(0.1)
        l_bounds.append([0.001,0.999])
        
    l = pd.DataFrame({'var_name':l_var,'x0':l_x0,'bounds':l_bounds})

    for j in range(len(levels)):

        if 'goi' in alg_type[i]:
            var_i = [0,1,2,3,4,5,7]
        else:
            var_i = [0,1,2,3,4,5,6,7]

        if 'nonleaky_eps1' in alg_type[i]:
            pass
        elif 'leaky_eps1' in alg_type[i]:
            var_i.append(8)
        elif 'nonleaky' in alg_type[i]:
            var_i.append(8)
        elif 'leaky' in alg_type[i]:
            var_i.append(8)
            var_i.append(9)
        elif 'fixed' in alg_type[i]:
            var_i.append(8)
            
        li = l.iloc[var_i]
        l_var_i = list(li['var_name'])
        l_x0_i  = list(li['x0'])
        l_bounds_i = list(li['bounds'])

        params = {'data_type': data_type,
                    'data_folder': '',
                    'data_path_type': '',
                    'comb_type': comb_type,
                    'var_name': l_var_i,
                    'kwargs': {"x0":l_x0_i,"bounds":l_bounds_i,"opt_method":opt_method},
                    'alg_type': alg_type[i],
                    'save_name': f'mle_{alg_type[i]}-l{levels[j]}-{data_type}',
                    'verbose': True,
                    'parallel': parallel,
                    'local': local,
                    'level':levels[j],
                    'save_path':f'MLE3_results/Fits/{"MultiStart/" if randstart else "SingleRun/"}mle_{alg_type[i]}-{data_type}_{opt_method}'}

        # path = './src/scripts/MLE2/mle_fit_configs/'
        path = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'mle' / 'mle_fit_configs'
        sl.make_long_dir(path)
        name = f'{params["save_name"]}_{opt_method}{("-" if len(comb_type)>0 else "")}{comb_type}'

        if randstart:
            params["seed"] = 12345      
            params["rand_start"] = 10 
            name = name+'_multi'  

        if data_type=='naive':
            params['data_folder']       = '2022_11_17_10-57-08_nAC_debug'
            params['data_path_type']    = 'auto'
        elif data_type=='opt':
            if local:
                params['data_folder']       = '/Volumes/lcncluster/becker/RL_reward_novelty/data/bintree_archive/sim_opt/2022_08_16_11-23-13_gpopt_nAC-N-expl_OI'
                params['data_path_type']    = 'manual'
                name = name+'_local'
            else:
                params['data_folder']       = 'bintree_archive/sim_opt/2022_08_16_11-23-13_gpopt_nAC-N-expl_OI'
                params['data_path_type']    = 'auto'
        elif data_type=='mice':
            params['data_folder']       = sl.get_datapath().replace('data','ext_data')+'Rosenberg2021/'    
            params['data_path_type']    = 'auto'

        with open(path / f'{name}.json', 'w') as fp:
            json.dump(params, fp)
