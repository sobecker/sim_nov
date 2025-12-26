import numpy as np
import pandas as pd
from scipy.stats import sem
import os
import json
from argparse import ArgumentParser
import fitting_neural.fit_homann as fit_homann
import fitting_neural.load_homann_data as load_homann

################################################################################################################################################
# This scripts computes the MSE fit / correlation of simulated to experimental data for all grid points, for four different types of models:
# 1. Leaky count-based novelty (model='cnov_leaky')
# 2. Fixed learning rate count-based novelty (model='cnov_fr')
# 3. Leaky similarity-based novelty (model='snov-complex-leaky')
# 4. Fixed learning rate similarity-based novelty (model='snov-complex-fr')
#
# Model and fitting parameters should be specified in the config file (see make_configs_fitting.py).
#
# To run the script from terminal, use:
# python -u ./src/fitting_neural/run_fit_withconfig.py --config ./src/fitting_neural/configs_fitting/<name_config>.json

################################################################################################################################################

def run_fit_new_withconfig(config):
    # Specify what to compute
    comp_type  = config['comp_type'] if 'comp_type' in config else 'fit'            # 'fit' or 'corr'
    model      = config['model']     # 'cnov_leaky', 'cnov_fr' or 'snov-complex-leaky', 'snov-complex-fr'
    robustness = config['robustness'] if 'robustness' in config else False         # whether to fit robustness simulations (TRUE) or regular simulations (FALSE)

    if comp_type=='fit':    
        weighting     = config['weighting'] if 'weighting' in config else 'equal-samples'         # 'equal-samples', 'equal-exp' or 'none' [not available yet for bootstrap!]
        drop_type     = config['drop_type'] if 'drop_type' in config else 'sample'                # 'sample' or 'exp' (drop one sample or one experiment - only has effect if 'jackknife' or 'loo' in sampling_type); recommended to choose drop consistent with weighting!
        sampling_type = config['sampling_type'] if 'sampling_type' in config else 'normal'        # 'normal','bootstrap','jackknife','loo','jackknife-loo', 'outerjack-cv' / 'outerboot-cv'
        n_bootstrap   = config['n_bootstrap'] if 'n_bootstrap' in config else 50
        refit         = config['refit'] if 'refit' in config else True  
        
        if 'cnov' in model and sampling_type=='bootstrap':
            sampling_type = 'normal'           # bootstrapping does not make sense for cnov since the model is deterministic
            print('Warning: Bootstrap not available for cnov models (deterministic). Sampling type set to normal.')

        add_name    = f'{sampling_type}'

        if 'jackknife' in sampling_type or 'loo' in sampling_type or 'outerjack' in sampling_type or 'outerboot' in sampling_type:
            add_name += f'-drop-{drop_type}'

        if weighting!='none':
            add_name += f'-{weighting}'

    elif comp_type=='corr':
        corr_type     = config['corr_type'] if 'corr_type' in config else 'pearson'         # 'pearson' or 'spearman'
        sampling_type = config['sampling_type'] if 'sampling_type' in config else 'normal'          # 'normal', bootstrap'
        n_bootstrap   = config['n_bootstrap'] if 'n_bootstrap' in config else 50

        add_name    = f'{corr_type}-{sampling_type}'

    # Whether to load grid file and Homann data from cluster 
    run_from_cluster    = config['run_from_cluster'] if 'run_from_cluster' in config else False

    # Where to save data
    if 'cnov' in model:
        save_path_base = '/Users/sbecker/Projects/sim_nov/data/'
        name_proj      = 'grid_search_results'
        name_set       = f'{model}' 
        save_path      = save_path_base + f'{name_proj}/{name_set}/'

    elif 'snov' in model and robustness:
        set_num        = config['set_num'] if 'set_num' in config else 1
        save_path_base = '/lcncluster/becker/sim_nov/data/' if run_from_cluster else '/Volumes/lcncluster/becker/sim_nov/data/'
        name_proj      = 'robustness2_width'
        name_set       = f'{model}_set{set_num}_sep' 
        save_path      = save_path_base + f'{name_proj}/{name_set}/combined_data/'

    elif 'snov' in model and not robustness:
        set_num        = config['set_num'] if 'set_num' in config else (9 if 'leaky' in model else 7) 
        save_path_base = '/lcncluster/becker/sim_nov/data/' if run_from_cluster else '/Volumes/lcncluster/becker/sim_nov/data/'
        name_proj      = 'grid_search_results'
        name_set       = f'{model}_set{set_num}_sep' 
        save_path      = save_path_base + f'{name_proj}/{name_set}/combined_data/'

    else:
        assert False, "Model not recognized. Use 'cnov' or 'snov'."

    ################################################################################################################################################
    if refit:
        # Load Homann data 
        homann_data = load_homann.load_exp_homann(cluster=run_from_cluster)
        homann_data[1][1][0] = 0    # set first value to 0

        # Parameters for sim data loading
        data_names = ['l','lp','m','m_steady']
        data_var   = ['n_fam','dN','n_im','n_im']
        data_val   = ['nt_norm','tr_norm','nt_norm','steady']

        # Load grid file
        grid_orig = pd.read_csv(save_path+'grid.csv',index_col=False)
        grid_df  = grid_orig.copy()

        # Remove error file from potential previous runs
        if os.path.exists(os.path.join(save_path,f'failed_{add_name}.txt')):
            os.remove(os.path.join(save_path,f'failed_{add_name}.txt'))

        # Loop over grid points and recompute fit / correlation
        for i, id_i in enumerate(grid_orig["grid_id"].values):

            try: 
                save_path_i = save_path + f'grid_{id_i}/'
                sim_data = []
                if sampling_type=='bootstrap':
                    boot_stats = []

                # Load simulated data
                for j, nn in enumerate(data_names):
                    if 'cnov' in model:
                        stats_nn = pd.read_csv(save_path_i+f'stats_{nn}.csv')

                    else:
                        data_nn = pd.read_csv(save_path_i+f'data_all_{nn}.csv')

                        if sampling_type=='bootstrap':
                            # Create bootstrap data sets
                            all_sample_id   = data_nn.sample_id.unique()
                            all_exp_var     = data_nn[data_var[j]].unique()
                            boot_sample_id  = [np.random.choice(all_sample_id, len(data_nn), replace=True) for bi in range(n_bootstrap)]  # create sets of simulation ids by bootstrapping
                            boot_data_nn    = [pd.concat([data_nn[data_nn.sample_id==bj] for bj in boot_sample_id[bi]]) for bi in range(n_bootstrap)]  # get simulation data for each bootstrap set of ids
                            boot_stats_nn   = [boot_data_nn[bi][[data_var[j],data_val[j]]].groupby(data_var[j]).mean().reset_index() for bi in range(n_bootstrap)]  # get stats for each bootstrap data set
                            boot_stats_nn_df = pd.DataFrame({'boot_id':   np.concatenate([np.ones(len(all_exp_var))*bi for bi in range(n_bootstrap)]),
                                                            data_var[j]: np.concatenate([boot_stats_nn[bi][data_var[j]].values for bi in range(n_bootstrap)]),
                                                            data_val[j]: np.concatenate([boot_stats_nn[bi][data_val[j]].values for bi in range(n_bootstrap)])
                                                            })
                            boot_stats_nn_df.to_csv(save_path_i+f'boot_stats_{nn}.csv',index=False)
                            boot_stats.append(boot_stats_nn)

                        stats_nn = data_nn[[data_var[j],data_val[j]]].groupby(data_var[j]).mean().reset_index()
                        stats_nn.to_csv(save_path_i+f'stats_{nn}.csv',index=False)

                    sim_data.append([stats_nn[data_var[j]].values, stats_nn[data_val[j]].values])

                # Fit simulated data to experimental data and compute (N)MSE
                if comp_type=='fit':

                    if sampling_type=='normal':
                        # Compute normal (N)MSE
                        coef, shift, fit_yres, fit_loss = fit_homann.fit_homann_exp(sim_data,homann_data,coef_steady=True,weighting=weighting,save_path=save_path_i)
                        grid_df.loc[grid_df.grid_id==i,'coef'] = coef
                        grid_df.loc[grid_df.grid_id==i,'shift'] = shift
                        for fl, fl_name in zip(fit_loss, ['train_rss','train_mse','train_nmse']):
                            grid_df.loc[grid_df.grid_id==i,fl_name] = fl

                    elif sampling_type=='bootstrap':
                        # Compute bootstrapped (N)MSE
                        boot_coef = []; boot_shift = []; boot_rss = []; boot_mse = []; boot_nmse = []
                        for bi in range(n_bootstrap):

                            sim_data_bi = []
                            for j, nn in enumerate(data_names):
                                sim_data_bi.append([boot_stats[j][bi][data_var[j]].values, boot_stats[j][bi][data_val[j]].values])

                            coef_bi, shift_bi, _, fit_loss_bi = fit_homann.fit_homann_exp(sim_data_bi,homann_data,coef_steady=True,weighting=weighting)

                            boot_coef.append(coef_bi)
                            boot_shift.append(shift_bi)
                            boot_rss.append(fit_loss_bi[0])
                            boot_mse.append(fit_loss_bi[1])
                            boot_nmse.append(fit_loss_bi[2])

                        all_meas = ['coef','shift','rss','mse','nmse']
                        for meas in all_meas:
                            grid_df.loc[grid_df.grid_id==i,f'boot-mean_{meas}'] = np.mean(eval(f'boot_{meas}'))
                            grid_df.loc[grid_df.grid_id==i,f'boot-sem_{meas}']  = sem(eval(f'boot_{meas}')) #np.std(eval(f'boot_{meas}'))/np.sqrt(n_bootstrap)
                    
                    elif 'jackknife' in sampling_type or 'loo' in sampling_type: 

                        # Compute jackknife (N)MSE
                        jack_mse_mean, jack_mse_std, jack_mse_se, df_sim_data = fit_homann.crossvalidation_homann_exp(sim_data,homann_data,coef_steady=True,sampling_type=sampling_type,weighting=weighting,drop_type=drop_type,combine_m=False,save_path=save_path_i)

                        if 'jackknife' in sampling_type:
                            for jm in['jack_mse_mean', 'jack_mse_std', 'jack_mse_se']:
                                grid_df.loc[grid_df.grid_id==i, jm] = eval(jm)
                    
                    elif 'outerjack' in sampling_type or 'outerboot' in sampling_type:
                        
                        # Compute resampled cross-validation error
                        jack_mse_mean, jack_mse_std, jack_mse_se, df_sim_data = fit_homann.jackknife_crossvalidation_homann_exp(sim_data,homann_data,coef_steady=True,sampling_type=sampling_type,weighting=weighting,drop_type=drop_type,combine_m=False,save_path=save_path_i)

                        if 'outerjack' in sampling_type:
                            for jm in ['jack_mse_mean', 'jack_mse_std', 'jack_mse_se']:
                                grid_df.loc[grid_df.grid_id==i, jm] = eval(jm)

                    else:
                        assert False, "Sampling type not recognized. Use 'normal', 'jackknife' or 'bootstrap'."

                # Compute correlation between simulated and experimental data
                elif comp_type=='corr':

                    if sampling_type=='normal':
                        # Compute normal correlation
                        corr_comb, corr_exp = fit_homann.corr_homann_exp(sim_data,homann_data,regr_meas='score',corr_type=corr_type,save_path=save_path_i)
                        grid_df.loc[grid_df.grid_id==i,'corr_comb'] = corr_comb
                        for ce, ce_name in zip(corr_exp, ['corr_tem', 'corr_trec', 'corr_tmem', 'corr_steady']):
                            grid_df.loc[grid_df.grid_id==i,ce_name] = ce
                        grid_df.loc[grid_df.grid_id==i,'corr_mean'] = np.mean(corr_exp)
                        grid_df.loc[grid_df.grid_id==i,'corr_mean2'] = np.average(corr_exp,weights=[1/3,1/3,1/6,1/6])

                    elif sampling_type=='bootstrap':
                        # Compute bootstrapped correlation
                        boot_corr_comb = []; boot_corr_tem = []; boot_corr_trec = []; boot_corr_tmem = []; boot_corr_steady = []; boot_corr_mean = []; boot_corr_mean2 = []
                        for bi in range(n_bootstrap):

                            sim_data_bi = []
                            for j, nn in enumerate(data_names):
                                sim_data_bi.append([boot_stats[j][bi][data_var[j]].values, boot_stats[j][bi][data_val[j]].values])

                            corr_comb_bi, [corr_tem_bi, corr_trec_bi, corr_tmem_bi, corr_steady_bi] = fit_homann.corr_homann_exp(sim_data_bi,homann_data,regr_meas='score',corr_type=corr_type)
                            boot_corr_comb.append(corr_comb_bi)
                            boot_corr_tem.append(corr_tem_bi)
                            boot_corr_trec.append(corr_trec_bi)
                            boot_corr_tmem.append(corr_tmem_bi)
                            boot_corr_steady.append(corr_steady_bi)
                            boot_corr_mean.append(np.mean([corr_tem_bi, corr_trec_bi, corr_tmem_bi, corr_steady_bi]))
                            boot_corr_mean2.append(np.average([corr_tem_bi, corr_trec_bi, corr_tmem_bi, corr_steady_bi],weights=[1/3,1/3,1/6,1/6]))

                        all_meas = ['corr_comb','corr_tem','corr_trec','corr_tmem','corr_steady','corr_mean','corr_mean2']
                        for meas in all_meas:
                            grid_df.loc[grid_df.grid_id==i,f'boot-mean_{meas}'] = np.mean(eval(f'boot_{meas}'))
                            grid_df.loc[grid_df.grid_id==i,f'boot-sem_{meas}'] =  sem(eval(f'boot_{meas}')) #np.std(eval(f'boot_{meas}'))/np.sqrt(n_bootstrap)
                        
                    else:
                        assert False, "Sampling type not recognized. Use 'normal' or 'bootstrap'."
                

                # Print progress
                print(f'Done with grid point {i}/{max(grid_df.grid_id.values)}.')
            
            except Exception as e:
                print(f'Error in grid point {i}/{max(grid_df.grid_id.values)}: {e}')
                # Save failed counter
                with open(os.path.join(save_path,f'failed_{add_name}.txt'), 'a') as f:
                    f.write(f'{id_i}: {e}\n')
                continue

        # Save results
        grid_df.to_csv(save_path+f'grid_{add_name}.csv',index=False)
        print(f'Done recomputing fit, saved results to file {save_path}grid_{add_name}.csv.')
    
    else:
        # Create df_sim_data for further computations
        exp_len = [5,7,4,4]
        df_sim_data = pd.DataFrame({'id':        np.arange(np.sum(exp_len)),
                                    'exp_id':    np.concatenate([[i]*exp_len[i] for i in range(len(exp_len))]),
                                    'id_in_exp': np.concatenate([np.arange(exp_len[i]) for i in range(len(exp_len))])})

    # If sampling type requires, compute test error based on loss landscape
    if 'loo' in sampling_type:
        mse_refitted, [train, train_std, train_se], [test, test_std, test_se] = fit_homann.get_testerror_cv(df_sim_data,sampling_type=sampling_type,weighting=weighting,drop_type=drop_type,load_path=save_path) 

        cv_errors = pd.DataFrame({'train_error': train,
                                'train_error_std': train_std,
                                'train_error_se': train_se,
                                'test_error': test,
                                'test_error_std': test_std,
                                'test_error_se': test_se
                                },index=[0])
        cv_errors.to_csv(save_path+f'cv_errors_{add_name}.csv',index=False)

        mse_refitted.to_csv(save_path+f'cv_all_errors_{add_name}.csv',index=False)

    elif 'outerjack-cv' in sampling_type or 'outerboot-cv' in sampling_type:
        mse_refitted, [train, train_std, train_se], [test, test_std, test_se], df_errors = fit_homann.get_testerror_jackknife_cv(df_sim_data,sampling_type=sampling_type,weighting=weighting,drop_type=drop_type,load_path=save_path) 

        cv_errors = pd.DataFrame({'train_error': train,
                                'train_error_std': train_std,
                                'train_error_se': train_se,
                                'test_error': test,
                                'test_error_std': test_std,
                                'test_error_se': test_se
                                },index=[0])
        cv_errors.to_csv(save_path+f'cv_errors_{add_name}.csv',index=False)

        df_errors.to_csv(save_path+f'cv_all_errors_df_{add_name}.csv',index=False)

        mse_refitted.to_csv(save_path+f'cv_all_all_errors_{add_name}.csv',index=False)


# Load config file
if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument(
            '-c',
            '--config',
            dest='config_file',
            type=str,
            default=None,
            help='config file',
        )
    args = parser.parse_args()
    print('Successfully loaded config file.\n')

    if args.config_file:
        config = json.load(open(args.config_file))
    else: 
        # config = json.load(open('./src/fitting_neural/configs_fitting/cnov_leaky_set_normal-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/cnov_leaky_set_jackknife-loo-drop-sample-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/cnov_leaky_set_outerjack-cv-drop-sample-equal-samples.json'))

        # config = json.load(open('./src/fitting_neural/configs_fitting/cnov_fr_set_normal-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/cnov_fr_set_jackknife-loo-drop-sample-equal-samples.json'))
        config = json.load(open('./src/fitting_neural/configs_fitting/cnov_fr_set_outerjack-cv-drop-sample-equal-samples.json'))

        # config = json.load(open('./src/fitting_neural/configs_fitting/snov-complex-leaky_set1_normal-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/snov-complex-leaky_set1_jackknife-loo-drop-sample-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/snov-complex-leaky_set1_outerjack-cv-drop-sample-equal-samples.json'))

        # config = json.load(open('./src/fitting_neural/configs_fitting/snov-complex-fr_set1_normal-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/snov-complex-fr_set1_jackknife-loo-drop-sample-equal-samples.json'))
        # config = json.load(open('./src/fitting_neural/configs_fitting/snov-complex-fr_set1_outerjack-cv-drop-sample-equal-samples.json'))
       
    print(config)

    run_fit_new_withconfig(config)

    