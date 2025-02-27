import json
import numpy as np

import sys
import utils.saveload as sl

config_fit      = True

fixed_range     = True
range_perc      = 0.2
parallel        = True
maxit           = 1000
overwrite       = False
uniparam        = True
startID         = 0
num_fit         = 5
no_rew          = False

comb_type   = 'app' # 'app','sep'
alg_type    = ['hnor_notrace'] # 'nac','nor','hnac-gn','hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hhybrid2'
alg_type_sim = alg_type
alg_type_fit = alg_type
path        = './src/scripts/ParameterRecovery/model_recov_configs/'
opt_method  = 'Nelder-Mead' # 'SLSQP','L-BFGS-B'
levels_sim  = [1,2,3,4,5,6] # [1,5,6]
levels_fit  = [1,2,3,4,5,6]

l_var_nor       = ['lambda_N','beta_1','epsilon','k_leak']
l_x0_nor        = [0.5,5,0.0002,0.5] 
l_bounds_nor    = [[0.,0.999],
                    [0.1,30],
                    [0.001,1],
                    [0.001,0.999]]
l_limbounds_nor = [[0.,0.999],
                    [0.1,30],
                    [0.001,1],
                    [0.001,0.999]]
l_fixrange_nor  = [np.round(range_perc*(l_bounds_nor[i][1]-l_bounds_nor[i][0]),4) for i in range(len(l_bounds_nor))]

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

# Make configs for MLE fit (model recovery)
if config_fit:
    for i,j in zip(range(len(alg_type_sim)),range(len(alg_type_fit))):
        if 'nor' in alg_type_fit[i]:
            l_var       = l_var_nor
            l_x0        = l_x0_nor
            l_bounds    = l_bounds_nor
            l_limbounds = l_limbounds_nor.copy()
            l_fixrange  = l_fixrange_nor.copy()
        elif 'nac' in alg_type_fit[i]:
            l_var       = l_var_nac
            l_x0        = l_x0_nac
            l_bounds    = l_bounds_nac
            l_limbounds = l_limbounds_nac.copy()
            l_fixrange  = l_fixrange_nac.copy()
            if 'goi' in alg_type[i]:
                l_var.pop(6); l_x0.pop(6); l_bounds.pop(6); l_limbounds.pop(6); l_fixrange.pop(6)
        elif 'hybrid2' in alg_type_fit[i]:
            l_var       = l_var_hyb2.copy()
            l_x0        = l_x0_hyb2.copy()
            l_bounds    = l_bounds_hyb2.copy()
            l_limbounds = l_limbounds_hyb2.copy()
            l_fixrange  = l_fixrange_hyb2.copy()
        elif 'hybrid' in alg_type_fit[i]:
            l_var       = l_var_hyb.copy()
            l_x0        = l_x0_hyb.copy()
            l_bounds    = l_bounds_hyb.copy()
            l_limbounds = l_limbounds_hyb.copy()
            l_fixrange  = l_fixrange_hyb.copy()
        save_path = f'ParameterRecovery/FitData{"_uniparam" if uniparam else ""}/sim-{alg_type_sim[i]}-{comb_type}_fit-{alg_type_fit[j]}-{comb_type}_{opt_method}/'
        #sl.make_long_dir(save_path)
        params = {'num_fit': num_fit,
                  'startID': startID,
                    'data_path': f'ParameterRecovery/SimData{"_uniparam" if uniparam else ""}/{alg_type_sim[i]}_{comb_type}/', # path to sim data (to be fitted)
                    'data_path_type': 'auto',
                    'comb_type': comb_type,
                    'var_name': l_var,
                    'kwargs': {"x0":l_x0,
                                "bounds":l_bounds,
                                "opt_method":opt_method,
                                "maxit": maxit},
                    'alg_type_sim': alg_type_sim[i],
                    'alg_type_fit': alg_type_fit[j],
                    'save_name': f'mle-recov_sim-{alg_type_sim[i]}_fit-{alg_type_fit[j]}_{comb_type}_{opt_method}',
                    'verbose': True,
                    'save_path': save_path,   # path where the fitting results are saved
                    'parallel': parallel,
                    'overwrite': overwrite,
                    'uniparam': uniparam,
                    'no_rew': no_rew
                    }
        if 'hnor' in alg_type_sim[i] or 'hnac' in alg_type_sim[i] or 'hhybrid' in alg_type_sim[i]:
            for ll1 in range(len(levels_sim)):
                params['levels_sim'] = [levels_sim[ll1]]
                if 'hnor' in alg_type_sim[i] or 'hnac' in alg_type_sim[i] or 'hhybrid' in alg_type_sim[i]:
                    for ll2 in range(len(levels_fit)):
                        params['levels_fit'] = [levels_fit[ll2]]
                        name = f'multifit_sim-{alg_type_sim[i]}-l{levels_sim[ll1]}_fit-{alg_type_fit[j]}-l{levels_fit[ll2]}_{comb_type}_{opt_method}'
                        with open(path+name+'.json', 'w') as fp:
                            json.dump(params, fp)
    

    
        