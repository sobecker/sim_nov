import json
import pandas as pd
import sys
import utils.saveload as sl

alg_type = ['leaky_hnac-gn'] #hnac-gn_center-triangle', 'hnac-gn_notrace_center-box','hnac-gn-gv_notrace','hnac-gn-goi_notrace','hnac-gn-gv-goi_notrace'] #['hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi']
levels   = [0,1,2,3,4,5,6]

data_type   = 'mice'        # 'naive', 'opt', 'mice' 
opt_method  = 'Nelder-Mead' # L-BFGS-B', 'Nelder-Mead', 'SLSQP'
comb_type   = ''            # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False          # set to False for single run with user-specified x0
local       = False         # running on local machine 
parallel    = True         # running parallelized

l_var   = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0']
l_x0    = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0]
l_bounds    = [[0.,0.999],      #gamma
                [0.001,0.5],    #c_alph
                [0.001,0.5],    #a_alph
                [0.,0.999],     #c_lam
                [0.,0.999],     #a_lam
                [0.001,1.],     #temp
                [-100,100],     #c_w0
                [-100,100]]     #a_w0
if 'leaky' in alg_type:
    l_var.append('k_alph')
    l_x0.append(0.5)
    l_bounds.append([0.001,0.999])
    
l = pd.DataFrame({'var_name':l_var,'x0':l_x0,'bounds':l_bounds})

for i in range(len(alg_type)):
    for j in range(len(levels)):

        if 'goi' in alg_type[i]:
            var_i = [0,1,2,3,4,5,7]
        else:
            var_i = [0,1,2,3,4,5,6,7]
        
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
                    'level':levels[j],
                    'save_name': f'mle_{alg_type[i]}-l{levels[j]}-{data_type}',
                    'verbose': True,
                    'parallel': parallel,
                    'save_path':f'MLE_results/Fits/{"MultiStart/" if randstart else "SingleRun/"}mle_{alg_type[i]}-{data_type}_{opt_method}'}

        path = './src/scripts/MLE/mle_fit_configs/'
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

        with open(path+name+'.json', 'w') as fp:
            json.dump(params, fp)
