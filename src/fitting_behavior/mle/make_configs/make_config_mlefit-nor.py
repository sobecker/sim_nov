import json
import sys
import utils.saveload as sl

data_type   = 'mice'        
opt_method  = 'Nelder-Mead' # 'Nelder-Mead', 'L-BFGS-B', 'SLSQP'
comb_type   = ''            # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = True          # set to False for single run with user-specified x0
local       = False         # running on local machine 
leaky       = True         # leaky or non-leaky model
alg_name    = ('leaky_' if leaky else '')+'nor'    

l_var   = ['lambda_N','beta_1','epsilon','k_leak']
l_x0        = [0.5,5,0.0002,0.5]
l_bounds    = [[0.,0.999],
                [0.1,30],
                [0.0001,1],
                [0.001,0.999]]
if leaky:
    l_var.append('k_alph')
    l_x0.append(0.5)
    l_bounds.append([0.001,0.999])

params = {'data_type':data_type,
            'data_folder':'',
            'data_path_type': '',
            'comb_type': comb_type,
            'var_name': l_var,
            'kwargs': {"x0":l_x0,"bounds":l_bounds,"opt_method":opt_method},
            'alg_type': alg_name,
            'save_name': f'mle_{alg_name}-{data_type}',
            'verbose': True}

params['data_folder']       = sl.get_datapath().replace('data','ext_data')+'Rosenberg2021/'    
params['data_path_type']    = 'manual'

path = './src/fitting_behavior/mle/mle_fit_configs/'
name = f'{params["save_name"]}_{opt_method}{("-" if len(comb_type)>0 else "")}{comb_type}'

if randstart:
    params["seed"] = 12345      
    params["rand_start"] = 10 
    name = name+'_multi'  

with open(path+name+'.json', 'w') as fp:
    json.dump(params, fp)
