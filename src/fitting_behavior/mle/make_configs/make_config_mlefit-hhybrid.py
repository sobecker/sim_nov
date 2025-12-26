import json
import utils.saveload as sl

leakiness_type = 'leaky_eps1'
alg_type = [f'{leakiness_type}_hhybrid2_triangle_nocompnorm', 
            # f'{leakiness_type}_hhybrid2_center-triangle_nocompnorm',
            # f'{leakiness_type}_hhybrid2_notrace_center-box_nocompnorm',
            f'{leakiness_type}_hhybrid2_notrace_nocompnorm'] 
#['hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi']
levels   = [1,2,3,4,5,6] # level 0: component center is the first branching point, level 6: component centers are the leaf nodes

# alg_type    = f'{leakiness_type}_hhybrid2_center-triangle' # 'hhybrid': fitting both w_mf and w_mb, 'hhybrid2'
# alg_mf      = f'{leakiness_type}_hnac-gn_center-triangle'
# alg_mb      = f'{leakiness_type}_hnor_center-triangle'
data_type   = 'mice'         # 'naive', 'opt', 'mice' 
opt_method  = 'Nelder-Mead' # L-BFGS-B', 'Nelder-Mead', 'SLSQP'
comb_type   = 'app'         # 'sep', 'app', '' (for '' both sep and app are computed)
randstart   = False          # set to False for single run with user-specified x0
local       = False         # running on local machine 
parallel    = True          # running parallelized

for i in range(len(alg_type)):
    alg_mf      = alg_type[i].replace('hhybrid2','hnac-gn')
    alg_mb      = alg_type[i].replace('hhybrid2','hnor')

    if 'hybrid2' in alg_type[i]:
        l_var   = ['gamma',
                   'c_alph',
                   'a_alph',
                   'c_lam',
                   'a_lam',
                #    'temp',
                   'temp',
                   'c_w0',
                   'a_w0',
                   'lambda_N',
                   'beta_1',
                   'epsilon',
                   'k_leak',
                   'w_mf']
        l_x0    = [0.5,
                    0.1,
                    0.1,
                    0.5,
                    0.5,
                    # 0.5,
                    5, 
                    0,
                    0,
                    0.5,
                    5,
                    0.0002,
                    0.5,
                    0.5]
        l_bounds    = [[0.,0.999],      #gamma - discount factor
                        [0.001,0.5],    #c_alph - learning rate for critic
                        [0.001,0.5],    #a_alph - learning rate for actor
                        [0.,0.999],     #c_lam - eligibility trace for critic
                        [0.,0.999],     #a_lam - eligibility trace for actor
                        # [0.001,1.],     #temp - softmax temperature
                        [0.1,30],       #inv_temp - inverse softmax temperature
                        [-100,100],     #c_w0 - initial weights for critic
                        [-100,100],     #a_w0 - initial weights for actor
                        [0.,0.999],     #lambda_N - update rate of novelty Q-values during prioritized sweeping update
                        [0.1,30],       #beta_1 - inverse temperature of softmax policy
                        [0.0001,1],     #epsilon - prior belief about novelty values (uniform across states)
                        [0.001,0.999],  #k_leak - leakiness of beliefs about novelty values
                        [0,1]]          #w_mf - weight for model free system
    else:
        l_var   = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0','lambda_N','beta_1','epsilon','k_leak','w_mf','w_mb']
        l_x0    = [0.5,
                    0.1,
                    0.1,
                    0.5,
                    0.5,
                    # 0.5,
                    5,
                    0,
                    0,
                    0.5,
                    5,
                    0.0002,
                    0.5,
                    0.5,
                    0.5]
        l_bounds    = [[0.,0.999],      #gamma
                        [0.001,0.5],    #c_alph
                        [0.001,0.5],    #a_alph
                        [0.,0.999],     #c_lam
                        [0.,0.999],     #a_lam
                        # [0.001,1.],     #temp
                        [0.1,30],       #inv_temp
                        [-100,100],     #c_w0
                        [-100,100],     #a_w0
                        [0.,0.999],     #lambda_N
                        [0.1,30],       #beta_1
                        [0.0001,1],     #epsilon
                        [0.001,0.999],  #k_leak
                        [0,1],          #w_mf
                        [0,30]]         #w_mb

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

    for j in range(len(levels)):
        params = {'data_type': data_type,
                'data_folder': '',
                'data_path_type': '',
                'comb_type': comb_type,
                'var_name': l_var,
                'kwargs': {"x0":l_x0,"bounds":l_bounds,"opt_method":opt_method},
                'alg_type': alg_type[i],
                'hyb_type': [alg_mb,alg_mf],
                'save_name': f'mle_{alg_type[i]}-l{levels[j]}-{data_type}',
                'verbose': True,
                'parallel': parallel,
                'local': local,
                'level':levels[j],
                'save_path':f'MLE2_results/Fits/{"MultiStart/" if randstart else "SingleRun/"}mle_{alg_type[i]}-{data_type}_{opt_method}'}          

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
