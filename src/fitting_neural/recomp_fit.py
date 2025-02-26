import numpy as np
import pandas as pd
from scipy.stats import sem
import os
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty')
# sys.path.append('/Volumes/lcncluster/becker/RL_reward_novelty')
sys.path.append('/lcncluster/becker/RL_reward_novelty')
import src.fitting_neural.grid_search_complex_cells as gscc #import fit_homann_exp, corr_homann_exp, load_exp_homann  
import src.utils.saveload as sl

def fit_homann_exp_jackknife(sim_data,homann_data,coef_steady=True,regr_meas='score',save_path='',save_name=''):
    ## Fit linear regression ##
    # Dimensions for fitting of multiplicative regression factor
    if coef_steady: x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],sim_data[3][1]]).reshape((-1,1)) 
    else:           x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],np.zeros(len(sim_data[3][1]))]).reshape((-1,1)) 

    # Dimension for fitting of shift (to steady state features)
    x2   = np.concatenate([np.zeros(len(sim_data[0][1])+len(sim_data[1][1])+len(sim_data[2][1])),np.ones(len(sim_data[3][1]))]).reshape((-1,1)) 
    x    = np.concatenate([x2,x1],axis=1)
    y    = np.concatenate([homann_data[0][1],homann_data[1][1],homann_data[2][1],homann_data[3][1]]).reshape((-1,1)) 

    num_jack = len(y)
    idx_map = dict(zip(np.arange(len(y)),np.concatenate([np.arange(len(sim_data[i][1])) for i in range(len(sim_data))])))
    mse_comb_jack = []; mse_tem_jack = []; mse_trec_jack = []; mse_tmem_jack = []; mse_steady_jack = []; mse_mean_jack = []
    for i in range(num_jack):

        # Drop one data point (jackknife data set)
        idx_i = np.arange(num_jack)
        idx_i = np.delete(idx_i,i)
        x_i = x[idx_i,:]
        y_i = y[idx_i,:]

        # Fit coefficients using least squares estimation
        fit_i    = np.dot(np.dot(np.linalg.inv(np.dot(x_i.transpose(),x_i)),x_i.transpose()),y_i).flatten()
    
        # Combined MSE (including all experiments)
        comp_meas  = eval(f'gscc.{regr_meas}_loss')
        ypred_i    = np.dot(x_i,fit_i.reshape((-1,1))).flatten()
        mse_comb_i = comp_meas(y_i,ypred_i)
        mse_comb_jack.append(mse_comb_i)

        # MSE for each experiment and mean MSE
        cc = 0; passed_i = False
        pred_data = []; keep_idx_all = []
        for j in range(len(sim_data)):
            keep_idx = np.arange(len(sim_data[j][1]))
            if not passed_i and i<cc+len(sim_data[j][1]): # drop index in pred data when applicable
                keep_idx = np.delete(keep_idx,idx_map[i])
                pred_data.append((sim_data[j][0][keep_idx],ypred_i[cc:cc+len(sim_data[j][1])-1]))
                passed_i = True
                cc += len(sim_data[j][1])-1
            else:
                pred_data.append((sim_data[j][0][keep_idx],ypred_i[cc:cc+len(sim_data[j][1])]))
                cc += len(sim_data[j][1])
            keep_idx_all.append(keep_idx)
            
        mse_tem_i    = comp_meas(homann_data[0][1][keep_idx_all[0]],pred_data[0][1]); mse_tem_jack.append(mse_tem_i)
        mse_trec_i   = comp_meas(homann_data[1][1][keep_idx_all[1]],pred_data[1][1]); mse_trec_jack.append(mse_trec_i)
        mse_tmem_i   = comp_meas(homann_data[2][1][keep_idx_all[2]],pred_data[2][1]); mse_tmem_jack.append(mse_tmem_i)
        mse_steady_i = comp_meas(homann_data[3][1][keep_idx_all[3]],pred_data[3][1]); mse_steady_jack.append(mse_steady_i)
        mse_mean_i   = np.mean([mse_tem_i, mse_trec_i, mse_tmem_i, mse_steady_i]); mse_mean_jack.append(mse_mean_i)
    
    # Compute the jackknife estimate of the MSE (as mean over all jackknife estimates)
    mse_tem = np.mean(mse_tem_jack)
    mse_trec = np.mean(mse_trec_jack)
    mse_tmem = np.mean(mse_tmem_jack)
    mse_steady = np.mean(mse_steady_jack)
    mse_comb = np.mean(mse_comb_jack)
    mse_mean = np.mean(mse_mean_jack)

    # Compute the jackknife standard error of the MSE (as the standard deviation of the jackknife estimates)
    se_tem = np.std(mse_tem_jack)/np.sqrt(num_jack)
    se_trec = np.std(mse_trec_jack)/np.sqrt(num_jack)
    se_tmem = np.std(mse_tmem_jack)/np.sqrt(num_jack)
    se_steady = np.std(mse_steady_jack)/np.sqrt(num_jack)
    se_comb = np.std(mse_comb_jack)/np.sqrt(num_jack)
    se_mean = np.std(mse_mean_jack)/np.sqrt(num_jack)

    # Save fitting results
    if len(save_path)>0:
        sl.make_long_dir(save_path)
        mse_df = pd.DataFrame({'mse_comb':mse_comb,
                               'mse_mean':mse_mean,
                               'mse_tem':mse_tem,
                               'mse_trec':mse_trec,
                               'mse_tmem':mse_tmem,
                               'mse_steady':mse_steady,
                               'se_comb':se_comb,
                               'se_mean':se_mean,
                               'se_tem':se_tem,
                               'se_trec':se_trec,
                               'se_tmem':se_tmem,
                               'se_steady':se_steady},
                               index=[0])
        mse_df.to_csv(os.path.join(save_path,f'jackknife_mse_fit{save_name}.csv'),index=False)
        data_names = ['l','lp','m','m_steady']
        data_var = ['n_fam','dN','n_im','n_im']
        data_val = ['nt_norm','tr_norm','nt_norm','steady']
        for i,n in zip(range(len(pred_data)),data_names):
            pd.DataFrame({data_var[i]:pred_data[i][0],data_val[i]:pred_data[i][1]}).to_csv(os.path.join(save_path,f'jackknife_pred_{n}{save_name}.csv'),index=False)

    return pred_data, mse_df

comp_fit    = True
regr_meas   = 'score'
comp_corr   = False
corr_type   = 'pearson'

# Load grid file
run_from_cluster    = False
run_cnov            = True

jackknife = False
bootstrap = False

if run_cnov:
    bootstrap = False
n_bootstrap = 50

if run_cnov:
    save_path_base = '/Users/sbecker/Projects/RL_reward_novelty/data/'
    name_proj      = '2024-08_grid_search_manual_cluster'
    name_set       = 'set1_cnov_leaky_corr' #'set1_cnov_corr'
    save_path      = save_path_base + f'{name_proj}/{name_set}/'
else:
    c_type  = 'complex' # 'complex'
    k_type  = 'triangle'
    set_num = 4 # 4
    save_path_base = '/lcncluster/becker/RL_reward_novelty/data/' if run_from_cluster else '/Volumes/lcncluster/becker/RL_reward_novelty/data/'
    name_proj      = '2024-08_grid_search_manual_corr'
    # name_set       = 'best_complex_cell_seqsep' 
    name_set       = f'set{set_num}_{c_type}_cells_seqsep' 
    study_name     = f'{c_type}_cells-corr_{k_type}_adj_w3_G-40' 
    save_path      = save_path_base + f'{name_proj}/{name_set}/{study_name}/'

# Load Homann data + manipulate as needed
homann_data = gscc.load_exp_homann(cluster=run_from_cluster)
homann_data[1][1][0] = 0    # set first value to 0
save_name='_corr-lp0'
# homann_data[1] = (homann_data[1][0][1:], homann_data[1][1][1:])
# save_name='_no-lp0'

# Parameters for sim data loading
data_names = ['l','lp','m','m_steady']
data_var = ['n_fam','dN','n_im','n_im']
data_val = ['nt_norm','tr_norm','nt_norm','steady']

# Load old grid file
grid_old = pd.read_csv(save_path+'grid.csv',index_col=False)
grid_df  = grid_old.copy()

if os.path.exists(os.path.join(save_path,'grid_new2.csv')): # MAKE SURE THAT THIS UPDATES CORRECTLY
    grid_new = pd.read_csv(os.path.join(save_path,'grid_new2.csv'),index_col=False)
    for cc in list(grid_new.columns):
        if cc not in list(grid_df.columns):
            grid_df[cc] = np.nan
    grid_df.update(grid_new)

# Loop over grid points and recompute fit / correlation
for i in range(len(grid_old)):

    # Load simulated data
    save_path_i = save_path + f'grid_{i}/'
    sim_data = []
    if bootstrap:
        boot_stats = []
    for j, nn in enumerate(data_names):
        if run_cnov:
            stats_nn = pd.read_csv(save_path_i+f'stats_{nn}.csv')
        else:
            data_nn = pd.read_csv(save_path_i+f'data_all_{nn}.csv')
            if bootstrap:
                all_sample_id = data_nn.sample_id.unique()
                all_exp_var = data_nn[data_var[j]].unique()
                boot_sample_id = [np.random.choice(all_sample_id, len(data_nn), replace=True) for bi in range(n_bootstrap)]  # create bootstrap sets
                boot_data_nn = [pd.concat([data_nn[data_nn.sample_id==bj] for bj in boot_sample_id[bi]]) for bi in range(n_bootstrap)]  # get bootstrap data sets
                boot_stats_nn = [boot_data_nn[bi][[data_var[j],data_val[j]]].groupby(data_var[j]).mean().reset_index() for bi in range(n_bootstrap)]  # get stats for each bootstrap data set
                boot_stats_nn_df = pd.DataFrame({'boot_id':   np.concatenate([np.ones(len(all_exp_var))*bi for bi in range(n_bootstrap)]),
                                                 data_var[j]: np.concatenate([boot_stats_nn[bi][data_var[j]].values for bi in range(n_bootstrap)]),
                                                 data_val[j]: np.concatenate([boot_stats_nn[bi][data_val[j]].values for bi in range(n_bootstrap)])
                                                 })
                boot_stats_nn_df.to_csv(save_path_i+f'boot_stats_{nn}.csv',index=False)
                boot_stats.append(boot_stats_nn)
            stats_nn = data_nn[[data_var[j],data_val[j]]].groupby(data_var[j]).mean().reset_index()
            stats_nn.to_csv(save_path_i+f'stats_{nn}.csv',index=False)
        sim_data.append([stats_nn[data_var[j]].values, stats_nn[data_val[j]].values])
    # sim_data[1] = (sim_data[1][0][1:], sim_data[1][1][1:])

    if comp_fit:
        # Fit simulated data to experimental data
        if jackknife:
            _, mse_df = fit_homann_exp_jackknife(sim_data,homann_data,coef_steady=True,regr_meas=regr_meas,save_path=save_path_i,save_name=save_name)
            for cc in list(mse_df.columns):
                grid_df.loc[grid_df.grid_id==i,f'jack_{cc}'] = mse_df[cc].values[0]
        else:
            _, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gscc.fit_homann_exp(sim_data,homann_data,coef_steady=True,regr_meas=regr_meas,save_path=save_path_i,save_name=save_name)
            grid_df.loc[grid_df.grid_id==i,'mse_comb'] = mse_comb
            grid_df.loc[grid_df.grid_id==i,'mse_tem'] = mse_tem
            grid_df.loc[grid_df.grid_id==i,'mse_trec'] = mse_trec
            grid_df.loc[grid_df.grid_id==i,'mse_tmem'] = mse_tmem
            grid_df.loc[grid_df.grid_id==i,'mse_steady'] = mse_steady
            grid_df.loc[grid_df.grid_id==i,'mse_mean'] = np.mean([mse_tem, mse_trec, mse_tmem, mse_steady])
            grid_df.loc[grid_df.grid_id==i,'coef'] = coef
            grid_df.loc[grid_df.grid_id==i,'shift'] = shift

        if bootstrap:
            boot_coef = []; boot_shift = []; boot_mse_comb = []; boot_mse_tem = []; boot_mse_trec = []; boot_mse_tmem = []; boot_mse_steady = []; boot_mse_mean = []
            for bi in range(n_bootstrap):
                sim_data_bi = []
                for j, nn in enumerate(data_names):
                    sim_data_bi.append([boot_stats[j][bi][data_var[j]].values, boot_stats[j][bi][data_val[j]].values])
                _, coef_bi, shift_bi, mse_comb_bi, [mse_tem_bi, mse_trec_bi, mse_tmem_bi, mse_steady_bi] = gscc.fit_homann_exp(sim_data_bi,homann_data,coef_steady=True,regr_meas=regr_meas)
                boot_coef.append(coef_bi); boot_shift.append(shift_bi); boot_mse_comb.append(mse_comb_bi); boot_mse_tem.append(mse_tem_bi); boot_mse_trec.append(mse_trec_bi); boot_mse_tmem.append(mse_tmem_bi); boot_mse_steady.append(mse_steady_bi); boot_mse_mean.append(np.mean([mse_tem_bi, mse_trec_bi, mse_tmem_bi, mse_steady_bi]))

            all_meas = ['mse_comb','mse_tem','mse_trec','mse_tmem','mse_steady','mse_mean','coef','shift']
            for meas in all_meas:
                grid_df.loc[grid_df.grid_id==i,f'boot-mean_{meas}'] = np.mean(eval(f'boot_{meas}'))
                grid_df.loc[grid_df.grid_id==i,f'boot-sem_{meas}'] =  sem(eval(f'boot_{meas}')) #np.std(eval(f'boot_{meas}'))/np.sqrt(n_bootstrap)

    if comp_corr:
        # Compute correlation between simulated and experimental data
        corr_comb, [corr_tem, corr_trec, corr_tmem, corr_steady] = gscc.corr_homann_exp(sim_data,homann_data,regr_meas='score',corr_type=corr_type,save_path=save_path_i)
        grid_df.loc[grid_df.grid_id==i,'corr_comb'] = corr_comb
        grid_df.loc[grid_df.grid_id==i,'corr_tem'] = corr_tem
        grid_df.loc[grid_df.grid_id==i,'corr_trec'] = corr_trec
        grid_df.loc[grid_df.grid_id==i,'corr_tmem'] = corr_tmem
        grid_df.loc[grid_df.grid_id==i,'corr_steady'] = corr_steady
        grid_df.loc[grid_df.grid_id==i,'corr_mean'] = np.mean([corr_tem, corr_trec, corr_tmem, corr_steady])

        if bootstrap:
            boot_corr_comb = []; boot_corr_tem = []; boot_corr_trec = []; boot_corr_tmem = []; boot_corr_steady = []; boot_corr_mean = []
            for bi in range(n_bootstrap):
                sim_data_bi = []
                for j, nn in enumerate(data_names):
                    sim_data_bi.append([boot_stats[j][bi][data_var[j]].values, boot_stats[j][bi][data_val[j]].values])
                corr_comb_bi, [corr_tem_bi, corr_trec_bi, corr_tmem_bi, corr_steady_bi] = gscc.corr_homann_exp(sim_data_bi,homann_data,regr_meas='score',corr_type=corr_type)
                boot_corr_comb.append(corr_comb_bi); boot_corr_tem.append(corr_tem_bi); boot_corr_trec.append(corr_trec_bi); boot_corr_tmem.append(corr_tmem_bi); boot_corr_steady.append(corr_steady_bi); boot_corr_mean.append(np.mean([corr_tem_bi, corr_trec_bi, corr_tmem_bi, corr_steady_bi]))

            all_meas = ['corr_comb','corr_tem','corr_trec','corr_tmem','corr_steady','corr_mean']
            for meas in all_meas:
                grid_df.loc[grid_df.grid_id==i,f'boot-mean_{meas}'] = np.mean(eval(f'boot_{meas}'))
                grid_df.loc[grid_df.grid_id==i,f'boot-sem_{meas}'] =  sem(eval(f'boot_{meas}')) #np.std(eval(f'boot_{meas}'))/np.sqrt(n_bootstrap)

    # Print progress
    print(f'Done with grid point {i}/{max(grid_df.grid_id.values)}.')

# Save results
grid_df.to_csv(save_path+'grid_new2.csv',index=False)
print('Done recomputing fit, saved results to file grid_new.csv.')