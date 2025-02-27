import numpy as np
import pandas as pd

import os
import sys

import models.cnov.count_nov as cnov
import utils.saveload as sl
import fitting_neural.grid_search_complex_cells as gsc

# Save path for data
leaky = True
input_corr = True
save_path = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set1_cnov{"_leaky" if leaky else ""}{"_corr" if input_corr else ""}/'
sl.make_long_dir(save_path)

# Grid search over different values of k_alph
if leaky:
    grid_var1 = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999, 1] # k_alph
else:
    grid_var1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,20] # eps
grid_var2 = [0,2,10,50] # number of additional (non-observed) states

# Generate generic input (no variability)
params_input = {'n_fam':     [1,3,8,18,38],
                'n_im':      [3,6,9,12],
                'dN':        list(np.array([0,22,44,66,88,110,143])/0.3)} #[0,70,140,210,280,360,480]}
inputs, count_min, stim_unique = cnov.generate_input(params_input,input_corrected=input_corr)

# Load Homann data
homann_data = gsc.load_exp_homann(cluster=False)
# Set novelty response for L'=0 to zero
homann_data[1][1][0] = 0
# save_name='_corr-lp0'

# Simulate experiment for different values of k_alph
all_mse = []
c_grid = 0
for i in range(len(grid_var1)):
    if leaky:
        ka = grid_var1[i]
        eps = 1
    else:
        eps = grid_var1[i]
        ka = 0.1
    
    for j in range(len(grid_var2)):
        # Increase number of states
        stim_unique_j = np.concatenate([stim_unique,np.arange(stim_unique[-1]+1,stim_unique[-1]+1+grid_var2[j])])
    
        # Simulate tau_emerge
        stats_l, data_l = cnov.sim_tau_emerge(stim_unique,inputs[0],params_input['n_fam'],eps=eps,k_alph=ka,leaky=leaky,steady=True)

        # Simulate tau_recovery
        stats_lp, data_lp = cnov.sim_tau_recovery(stim_unique,inputs[1],params_input['dN'],eps=eps,k_alph=ka,leaky=leaky,steady=True)

        # Simulate tau_memory
        stats_m, stats_m_steady, data_m = cnov.sim_tau_memory(stim_unique,inputs[2],params_input['n_im'],eps=eps,k_alph=ka,leaky=leaky)

        # Save data
        save_path_i = os.path.join(save_path,f'grid_{c_grid}')
        # if leaky:
        #     save_path_i = save_path + 'S_' + str(grid_var2[j]) + '_k_alph_' + str(ka).replace('.','-') + '_eps_' +str(eps).replace('.','-') + '/'
        # else:
        #     save_path_i = save_path + 'S_' + str(grid_var2[j]) + 'eps_' + str(eps).replace('.','-') + '/'
        sl.make_long_dir(save_path_i)
        data_l.to_csv(os.path.join(save_path_i,'data_all_l.csv'))
        data_lp.to_csv(os.path.join(save_path_i,'data_all_lp.csv'))
        data_m.to_csv(os.path.join(save_path_i,'data_all_m.csv'))
        stats_l.to_csv(os.path.join(save_path_i,'stats_l.csv'))
        stats_lp.to_csv(os.path.join(save_path_i,'stats_lp.csv'))
        stats_m.to_csv(os.path.join(save_path_i,'stats_m.csv'))
        stats_m_steady.to_csv(os.path.join(save_path_i,'stats_m_steady.csv'))

        # Fit data to Homann model + compute MSE
        data_counts = []
        data_counts.append((stats_l['n_fam'].values,stats_l['nt_norm'].values))
        data_counts.append((stats_lp['dN'].values,stats_lp['tr_norm'].values))
        data_counts.append((stats_m['n_im'].values,stats_m['nt_norm'].values))
        data_counts.append((stats_m_steady['n_im'].values,stats_m_steady['steady'].values))

        pred_data, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gsc.fit_homann_exp(data_counts,homann_data,coef_steady=True,regr_meas='score',save_path='')
        pd.DataFrame({'n_fam': pred_data[0][0],'nt_norm': pred_data[0][1]}).to_csv(os.path.join(save_path_i,'pred_data_l.csv'))
        pd.DataFrame({'dN': pred_data[1][0],'tr_norm': pred_data[1][1]}).to_csv(os.path.join(save_path_i,'pred_data_lp.csv'))
        pd.DataFrame({'n_im': pred_data[2][0],'nt_norm': pred_data[2][1]}).to_csv(os.path.join(save_path_i,'pred_data_m.csv'))
        pd.DataFrame({'n_im': pred_data[2][0],'steady': pred_data[3][1]}).to_csv(os.path.join(save_path_i,'pred_data_m_steady.csv'))
        pd.DataFrame({'coef': coef,'shift': shift},index=[0]).to_csv(os.path.join(save_path_i,'coef_fit.csv'))
    
        mse_mean = np.nanmean([mse_tem, mse_trec, mse_tmem, mse_steady])
        mse_fit = pd.DataFrame({'mse_comb': mse_comb,
                                'mse_mean': mse_mean,
                                'mse_tem': mse_tem,
                                'mse_trec': mse_trec,
                                'mse_tmem': mse_tmem,
                                'mse_steady': mse_steady},index=[0])
        mse_fit.to_csv(os.path.join(save_path_i,'mse_fit.csv'))

        mse_fit['grid_id'] = c_grid
        mse_fit['k_alph'] = ka
        mse_fit['eps']    = eps
        mse_fit['num_states_input'] = count_min
        mse_fit['num_states_model'] = count_min + grid_var2[j]
        all_mse.append(mse_fit)
        with open(os.path.join(save_path,'done.txt'), 'a') as f:
            f.write(f'{c_grid}\n')
        print(f'Finished parameter: S = {count_min} + {grid_var2[j]}, k_alph = {ka}, eps = {eps}.')

        # Update grid counter
        c_grid += 1

# Save all MSE
all_mse = pd.concat(all_mse)
all_mse.to_csv(os.path.join(save_path,'grid.csv'))




    

    


    

