import json
import numpy as np
import utils.saveload as sl

### Script to create config files for simulation and/or fitting of parameter recovery data ###

config_sim      = True      # create config for simulation
config_fit      = False     # create config for recovery fitting

# Options for sim/fit config ##################################################################
fixed_range     = True      # simulate model within fixed parameter range
range_perc      = 0.2       # percentage of fixed range to cover in simulation
uniparam        = True      # only simulate a single parameter set (but with different random seeds)
no_rew          = True      # simulate without stopping at rewarded state

# Options for fit config ######################################################################
parallel        = True      # run fitting in parallel
maxit           = 200       # maximum number of iterations for fitting optimizer
overwrite       = False     # overwrite existing fit results

# Models to simulate and/or fit ###############################################################
comb_type       = 'app' # 'app','sep'
alg_type        = ['hnor','hhybrid2','hnac-gn','nor','nac','hybrid2']
maxit           = [False, False, False, False, False, True]
# ['hnor_center-triangle','hnac-gn_center-triangle','hhybrid2_center-triangle','hnor_notrace_center-box','hnac-gn_notrace_center-box','hhybrid2_notrace_center-box'] 
# ['hnor_notrace','hnac-gn_notrace','hhybrid2_notrace']
# ['hnor','hhybrid2','hnac-gn','nor','nac','hybrid2'] 
# other models: 'hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
alg_type_sim    = alg_type
alg_type_fit    = alg_type
path            = sl.get_rootpath() / 'src' / 'fitting_behavior' / 'recovery' / 'param_recov_range_configs'
sl.make_long_dir(path)
opt_method      = 'Nelder-Mead' # 'SLSQP','L-BFGS-B'
levels          = list(range(1,7))

# Define parameter ranges for simulation and fitting ###########################################
# MB agents
l_var_nor       = ['lambda_N','beta_1','epsilon','k_leak']
l_x0_nor        = [0.5,5,0.0002,0.5] 
l_bounds_nor    = [[0.,0.999],
                    [0.1,30],
                    [0.0001,1],
                    [0.001,0.999]]
l_limbounds_nor = [[0.,0.999],
                    [0.1,30],
                    [0.001,1],
                    [0.001,0.999]]
l_fixrange_nor  = [np.round(range_perc*(l_bounds_nor[i][1]-l_bounds_nor[i][0]),4) for i in range(len(l_bounds_nor))]

# MF agents
l_var_nac       = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0']
l_x0_nac        = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0] 
l_bounds_nac    = [[0.,0.999],      #gamma
                    [0.001,0.5],    #c_alph
                    [0.001,0.5],    #a_alph
                    [0.,0.999],     #c_lam
                    [0.,0.999],     #a_lam
                    [0.001,1.],     #temp
                    [-100,100],     #c_w0
                    [-100,100]]     #a_w0
l_limbounds_nac = [[0.,0.999],      #gamma
                    [0.001,0.5],    #c_alph
                    [0.001,0.5],    #a_alph
                    [0.,0.999],     #c_lam
                    [0.,0.999],     #a_lam
                    [0.001,1.],     #temp
                    [-100,100],     #c_w0
                    [-100,100]]     #a_w0
l_fixrange_nac  = [np.round(range_perc*(l_bounds_nac[i][1]-l_bounds_nac[i][0]),4) for i in range(len(l_bounds_nac))]

# Hybrid agents
l_var_hyb       = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0','lambda_N','beta_1','epsilon','k_leak','w_mf','w_mb']
l_x0_hyb        = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0,    0.5,       5,       0.0002,    0.5,    0.5,    0.5]
l_bounds_hyb    = [[0.,0.999],      #gamma
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
l_limbounds_hyb = [[0.,0.999],      #gamma
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
l_fixrange_hyb  = [np.round(range_perc*(l_bounds_hyb[i][1]-l_bounds_hyb[i][0]),4) for i in range(len(l_bounds_hyb))]

l_var_hyb2      = ['gamma','c_alph','a_alph','c_lam','a_lam','temp','c_w0','a_w0','lambda_N','beta_1','epsilon','k_leak','w_mf']
l_x0_hyb2       = [0.5,     0.1,     0.1,     0.5,    0.5,    0.5,   0,     0,    0.5,       5,       0.0002,    0.5,    0.5]
l_bounds_hyb2   = [[0.,0.999],      #gamma
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
l_limbounds_hyb2= [[0.,0.999],      #gamma
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
l_fixrange_hyb2 = [np.round(range_perc*(l_bounds_hyb[i][1]-l_bounds_hyb[i][0]),4) for i in range(len(l_bounds_hyb))]

# Make configs for simulation (recovery data) ##################################################
if config_sim:
    for i in range(len(alg_type)):
        if 'nor' in alg_type[i]:
            l_var       = l_var_nor.copy()
            l_x0        = l_x0_nor.copy()
            l_bounds    = l_bounds_nor.copy()
            l_limbounds = l_limbounds_nor.copy()
            l_fixrange  = l_fixrange_nor.copy()
        elif 'nac' in alg_type[i]:
            l_var       = l_var_nac.copy()
            l_x0        = l_x0_nac.copy()
            l_bounds    = l_bounds_nac.copy()
            l_limbounds = l_limbounds_nac.copy()
            l_fixrange  = l_fixrange_nac.copy()
            if 'goi' in alg_type[i]:
                l_var.pop(6); l_x0.pop(6); l_bounds.pop(6); l_limbounds.pop(6); l_fixrange.pop(6)
        elif 'hybrid2' in alg_type[i]:
            l_var       = l_var_hyb2.copy()
            l_x0        = l_x0_hyb2.copy()
            l_bounds    = l_bounds_hyb2.copy()
            l_limbounds = l_limbounds_hyb2.copy()
            l_fixrange  = l_fixrange_hyb2.copy()
        elif 'hybrid' in alg_type[i]:
            l_var       = l_var_hyb.copy()
            l_x0        = l_x0_hyb.copy()
            l_bounds    = l_bounds_hyb.copy()
            l_limbounds = l_limbounds_hyb.copy()
            l_fixrange  = l_fixrange_hyb.copy()

        params = {'comb_type':comb_type,
                    'seed': 12345,
                    'start_seed': 0,
                    'startID': 0,
                    'num_sim': 20, #1000
                    'agent_num': 20, #20
                    'uniparam': uniparam,
                    'no_rew': no_rew,
                    'parallel': parallel,
                    'var_name': l_var,
                    'alg_type': alg_type[i],
                    'maxit': maxit[i]}
        
        if fixed_range:     
            params['kwargs'] = {"range":l_fixrange,"abs_bounds":l_bounds}
        else:               
            params['kwargs'] = {"bounds":l_limbounds}
            
        if 'hnor' in alg_type[i] or 'hnac' in alg_type[i] or 'hhybrid' in alg_type[i]:
            for ll in range(len(levels)):
                params['levels'] = [levels[ll]]
                name = f'multisim-{params["alg_type"]}_{comb_type}_{"uniparam" if uniparam else ("fixrange" if fixed_range else "varrange")}_seed-{params["seed"]}_l{levels[ll]}'
                with open(path / f'{name}.json', 'w') as fp:
                    json.dump(params, fp)
        else:
            name = f'multisim-{params["alg_type"]}_{comb_type}_{"uniparam" if uniparam else ("fixrange" if fixed_range else "varrange")}_seed-{params["seed"]}'
            with open(path / f'{name}.json', 'w') as fp:
                json.dump(params, fp)

# Make configs for MLE fit (parameter recovery) ################################################
if config_fit:
    for i in range(len(alg_type)):
        if 'nor' in alg_type[i]:
            l_var       = l_var_nor.copy()
            l_x0        = l_x0_nor.copy()
            l_bounds    = l_bounds_nor.copy()
        elif 'nac' in alg_type[i]:
            l_var       = l_var_nac.copy()
            l_x0        = l_x0_nac.copy()
            l_bounds    = l_bounds_nac.copy()
            if 'goi' in alg_type[i]:
                l_var.pop(6); l_x0.pop(6); l_bounds.pop(6)
        elif 'hybrid' in alg_type[i]:
            l_var       = l_var_hyb.copy()
            l_x0        = l_x0_hyb.copy()
            l_bounds    = l_bounds_hyb.copy()
        save_path = f'ParameterRecovery/FitData/{alg_type[i]}_{comb_type}_{opt_method}/'
        #sl.make_long_dir(save_path)
        params = {'num_fit': 20,
                    'data_path': f'ParameterRecovery/SimData{"_uniparam" if uniparam else ""}/{alg_type[i]}_{comb_type}/', # path to sim data (to be fitted)
                    'data_path_type': 'auto',
                    'comb_type': comb_type,
                    'var_name': l_var,
                    'kwargs': {"x0":l_x0,
                                "bounds":l_bounds,
                                "opt_method":opt_method,
                                "maxit": maxit},
                    'alg_type': alg_type[i],
                    'save_name': f'mle-recov_{alg_type[i]}_{comb_type}_{opt_method}',
                    'verbose': True,
                    'save_path': save_path,   # path where the fitting results are saved
                    'parallel': parallel,
                    'overwrite': overwrite
                    }
        if 'hnor' in alg_type[i] or 'hnac' in alg_type[i] or 'hhybrid' in alg_type[i]:
            params['levels'] = levels
        name = f'multifit-{params["alg_type"]}_{params["comb_type"]}_{params["kwargs"]["opt_method"]}'
        with open(path / f'{name}.json', 'w') as fp:
            json.dump(params, fp)

