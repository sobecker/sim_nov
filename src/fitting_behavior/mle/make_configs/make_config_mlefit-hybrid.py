import json
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl

data_type   = 'mice'         # 'naive', 'opt', 'mice' 
opt_method  = 'Nelder-Mead' # L-BFGS-B', 'Nelder-Mead', 'SLSQP'
comb_type   = ''         # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False          # set to False for single run with user-specified x0
local       = False         # running on local machine 
leaky       = True         # leaky or non-leaky model
alg_name    = ('leaky_' if leaky else '')+'hybrid'   

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
if leaky:
    l_var.append('k_alph')
    l_x0.append(0.5)
    l_bounds.append([0.001,0.999])

params = {'data_type': data_type,
            'data_folder': '',
            'data_path_type': '',
            'comb_type': comb_type,
            'var_name': l_var,
            'kwargs': {"x0":l_x0,"bounds":l_bounds,"opt_method":opt_method},
            'alg_type': alg_name,
            'save_name': f'mle-maxit_{alg_name}-{data_type}',
            'verbose': True}

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
