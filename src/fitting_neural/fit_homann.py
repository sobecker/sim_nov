import numpy as np
import pandas as pd
import os
import utils.saveload as sl

############################################################################################################
#               Helper functions                                                                           #
############################################################################################################
def weighted_av_se(values, weights):
    """
    Compute the weighted standard error of the mean.
    """
    average = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - average) ** 2, weights=weights))
    se = std / np.sqrt(np.sum(weights))
    return average, std, se

def get_lr_vectors(sim_data,homann_data,coef_steady=True):
    # Dimensions for fitting of multiplicative regression factor
    if coef_steady: x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],sim_data[3][1]]).reshape((-1,1)) 
    else:           x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],np.zeros(len(sim_data[3][1]))]).reshape((-1,1)) 

    # Dimension for fitting of shift (to steady state features)
    x2   = np.concatenate([np.zeros(len(sim_data[0][1])+len(sim_data[1][1])+len(sim_data[2][1])),np.ones(len(sim_data[3][1]))]).reshape((-1,1))

    # Concatenate data for fitting
    x     = np.concatenate([x2,x1],axis=1) 
    y     = np.concatenate([homann_data[0][1],homann_data[1][1],homann_data[2][1],homann_data[3][1]]).reshape((-1,1)) 

    # Compute sample weights for weighted linear regression
    df_sim_data = pd.DataFrame({'id':        np.arange(len(y)),
                                'exp_id':    np.concatenate([[i]*len(sim_data[i][1]) for i in range(len(sim_data))]),
                                'id_in_exp': np.concatenate([np.arange(len(sim_data[i][1])) for i in range(len(sim_data))]),
                                'data':      np.concatenate([sim_data[i][1] for i in range(len(sim_data))])
                                })
    return x, y, df_sim_data

############################################################################################################
#               Functions to compute linear regression weighting                                           #
############################################################################################################
def reweight_samples(df_sim_data, weighting='equal-samples', combine_exp_id=None):
    """
    Compute reweighting factor for sample vector.
    """

    if weighting=='equal-samples':
        len_samples = len(df_sim_data)
        w           = 1/len_samples * np.ones(len_samples).reshape((-1,1))  # note: equal weighting is equivalent to no weighting

    elif weighting=='equal-exp': 
        len_exps = dict(zip(df_sim_data['exp_id'].unique(),[len(df_sim_data[df_sim_data['exp_id']==i]) for i in df_sim_data['exp_id'].unique()]))

        # Combine experiments if specified
        if combine_exp_id is not None:
            min_comb_i = min(combine_exp_id) # get id for the combined experiment
            len_comb_i = np.sum([len_exps[i] if i in len_exps.keys() else 0 for i in combine_exp_id]) # get total length of the combined experiment
            len_exps_corr = dict(zip([i for i in len_exps.keys() if i not in combine_exp_id], [len_exps[i] for i in len_exps.keys() if i not in combine_exp_id]))
            if len_comb_i > 0:
                len_exps_corr[min_comb_i] = len_comb_i
        else:
            len_exps_corr = len_exps.copy()

        w = np.concatenate([1/len_exps_corr[i] * np.ones(len_exps_corr[i]) for i in len_exps_corr.keys()])

    elif weighting=='none':
        len_samples = len(df_sim_data)
        w           = np.ones(len_samples)
    
    return w.reshape((-1,1))

def reweight_exp(df_sim_data, weighting='equal-samples', combine_exp_id=None):
    """
    Compute reweighting factor for experiment vector.
    """

    if weighting=='equal-exp':
        len_samples = len(df_sim_data['exp_id'].unique())
        w = 1/len_samples * np.ones(len_samples).reshape((-1,1))  # note: equal weighting is equivalent to no weighting

    elif weighting=='equal-samples':
        len_exps    = dict(zip(df_sim_data['exp_id'].unique(),[len(df_sim_data[df_sim_data['exp_id']==i]) for i in df_sim_data['exp_id'].unique()]))
        len_exps_corr = len_exps.copy()  
        # Combine experiments if specified
        if combine_exp_id is not None:
            min_comb_i = min(combine_exp_id) # get id for the combined experiment
            len_comb_i = np.sum([len_exps[i] if i in len_exps.keys() else 0 for i in combine_exp_id]) # get total length of the combined experiment
            len_exps_corr = dict(zip([i for i in len_exps.keys() if i not in combine_exp_id], [len_exps[i] for i in len_exps.keys() if i not in combine_exp_id]))
            if len_comb_i > 0:
                len_exps_corr[min_comb_i] = len_comb_i
        else:
            len_exps_corr = len_exps.copy()
        # w           = np.array([len_exps_corr[i]/total_len for i in len_exps_corr.keys()])
        total_len = np.sum([len_exps_corr[i] for i in len_exps_corr.keys()])
        w = np.array([len_exps_corr[i]/total_len for i in len_exps_corr.keys()])

    elif weighting=='none':
        len_samples = len(df_sim_data['exp_id'].unique())
        w = np.ones(len_samples)

    return w.reshape((-1,1))

############################################################################################################
#               Weighted least-squares estimation                                                          #
############################################################################################################
def fit_wlse(x, y, w):
        # Check dimensions
        assert x.shape[0] == y.shape[0] == w.shape[0], "Dimensions of x, y, and w do not match."

        # Fit coefficients using weighted least squares estimation
        fit     = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), w*x)), x.transpose()), w*y).flatten()
    
        # Compute fitted data
        ypred   = np.dot(x,fit.reshape((-1,1)))

        # Compute residuals between fitted and experimental data
        yres = (y - ypred)**2

        # Compute train error of the weighted linear regression (residual sum of squares) and relevant train metrics (MSE, NMSE)
        train_rss = np.sum(w * yres)
        train_mse = np.mean(w * yres)
        train_nmse = np.mean(w * yres) / np.mean((y - np.mean(y)*np.ones(y.shape))**2)

        return fit, yres, ypred, train_rss, train_mse, train_nmse

############################################################################################################
#               Cross-validation                                                                           #
############################################################################################################
def get_losslandscape_cv(x,y,df_sim_data,sampling_type='jackknife',weighting='equal-samples',drop_type='sample',combine_exp_id=None):
        # Number of resampling sets (# data points or # experiments)
        if drop_type=='sample':
            num_resample    = len(y) 
        elif drop_type=='exp':
            num_resample    = len(df_sim_data['exp_id'].unique()) 

        # Initialize recording for resampling sets
        fit_all     = np.zeros((2,num_resample))  # store coefficients for each resampling set

        train_rss   = np.zeros(num_resample)  # store train residual sum of squares for each resampling set
        train_mse   = np.zeros(num_resample)  # store train mean squared error
        train_nmse  = np.zeros(num_resample)  # store train normalized mean squared
        
        if 'loo' in sampling_type:
            test_rss    = np.zeros(num_resample)  # store test residual sum of squares for each resampling set
            test_mse    = np.zeros(num_resample)  # store test mean squared error

            pred_full = np.nan * np.ones((len(y),num_resample))  # store predicted data (test data vector with dropout) for each resampling set
        
        else:
            pred_full = None

        # Loop over resampling sets ###############################################################################
        for i in range(num_resample):

            # Construct resampling sets (train and test data)
            if drop_type=='sample':
                mask_i = np.arange(len(df_sim_data)) != i   # drop single data point

            elif drop_type=='exp':
                id_exp_i = df_sim_data.loc[df_sim_data['exp_id']==i,'id'].values  # get sample ids of dropped experiment
                mask_i   = ~np.isin(np.arange(len(df_sim_data)),id_exp_i)   # drop one experiment (all samples of one experiment)

            xtrain_i = x[mask_i,:]
            ytrain_i = y[mask_i,:]

            xtest_i = x[~mask_i,:]
            ytest_i = y[~mask_i,:]

            # If dropping steady-state experiment, only fit coefficient
            if drop_type=='exp' and np.sum(x[mask_i,0])==0:
                xtrain_i = x[mask_i,1].reshape((-1,1))  
                xtest_i  = x[~mask_i,1].reshape((-1,1)) 

            # Compute sample weights for weighted linear regression
            wtrain_i = reweight_samples(df_sim_data[mask_i], weighting=weighting, combine_exp_id=combine_exp_id)

            # Fit coefficients using weighted least squares estimation
            fit_i, _, train_ypred_i, train_rss_i, train_mse_i, train_nmse_i = fit_wlse(xtrain_i, ytrain_i, wtrain_i)

            if drop_type=='exp' and np.sum(x[mask_i,0])==0:
                fit_all[0,i] = np.NaN
                fit_all[1,i] = fit_i[0]
            else:
                fit_all[:,i] = fit_i

            # Save train predictions + train metrics
            train_rss[i]         = train_rss_i
            train_mse[i]         = train_mse_i
            train_nmse[i]        = train_nmse_i
        
            if 'loo' in sampling_type:
                # Compute test predictions
                wtest_i = reweight_samples(df_sim_data[~mask_i], weighting=weighting, combine_exp_id=combine_exp_id)
                if drop_type=='exp' and np.sum(x[mask_i,0])==0:
                    test_ypred_i = np.dot(xtest_i,fit_i.reshape((-1,1))) 
                    test_ypred_i += (np.mean(ytest_i)-np.mean(test_ypred_i))  # set shift to mean of true steady-state data
                else:
                    test_ypred_i = np.dot(xtest_i,fit_i.reshape((-1,1)))
                # if steady-state exp was dropped during training, we didn't fit the shift - the only thing we can predict is the error to the incline of the steady-state exp. To make the prediction comparable between drop-out sets, we set the shift to the mean of the true steady-state data

                # Compute residuals between fitted and experimental data
                test_yres_i = (ytest_i - test_ypred_i)**2

                # Compute test RSS and MSE
                test_rss_i = np.sum(wtest_i * test_yres_i)
                test_mse_i = np.mean(wtest_i * test_yres_i)

                # Save train / test predictions + test metrics
                pred_full[mask_i,i]  = train_ypred_i.flatten()
                pred_full[~mask_i,i] = test_ypred_i.flatten()
                test_rss[i]          = test_rss_i
                test_mse[i]          = test_mse_i
    
        if 'jackknife' in sampling_type:
            # Compute the jackknife errors (with correct resampling weights) for current parameter set
            if drop_type=='sample':
                w_av = reweight_samples(df_sim_data, weighting=weighting, combine_exp_id=combine_exp_id)
            elif drop_type=='exp':
                w_av = reweight_exp(df_sim_data, weighting=weighting, combine_exp_id=combine_exp_id)
            
            jack_mse_mean, jack_mse_std, jack_mse_se = weighted_av_se(train_mse,weights=w_av.flatten())
        
        else:
            jack_mse_mean = np.NaN
            jack_mse_std  = np.NaN
            jack_mse_se   = np.NaN

        coef_all_df = pd.DataFrame({'resampling_id': np.arange(num_resample),
                                    'shift':          fit_all[0,:],
                                    'coef':         fit_all[1,:]})
        
        mse_df = pd.DataFrame({'resampling_id': np.arange(num_resample),
                               'train_rss':     train_rss,
                               'train_mse':     train_mse,
                               'train_nmse':    train_nmse})

        if 'loo' in sampling_type:
            test_df = pd.DataFrame({'resampling_id': np.arange(num_resample),
                                    'test_rss':      test_rss,
                                    'test_mse':      test_mse})
        else:
            test_df = None
        
        return coef_all_df, mse_df, test_df, pred_full, [jack_mse_mean, jack_mse_std, jack_mse_se]

############################################################################################################
#               Function to fit simulated data to experimental data                                        #
############################################################################################################
def fit_homann_exp(sim_data,homann_data,coef_steady=True,weighting='equal-samples',combine_m=False,nonleaky=False,save_path='',save_name=''):

    # Dimensions for fitting of multiplicative regression factor
    if coef_steady: x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],sim_data[3][1]]).reshape((-1,1)) 
    else:           x1 = np.concatenate([sim_data[0][1],sim_data[1][1],sim_data[2][1],np.zeros(len(sim_data[3][1]))]).reshape((-1,1)) 

    # Dimension for fitting of shift (to steady state features)
    x2   = np.concatenate([np.zeros(len(sim_data[0][1])+len(sim_data[1][1])+len(sim_data[2][1])),np.ones(len(sim_data[3][1]))]).reshape((-1,1))

    # Concatenate data for fitting
    x     = np.concatenate([x2,x1],axis=1) 
    y     = np.concatenate([homann_data[0][1],homann_data[1][1],homann_data[2][1],homann_data[3][1]]).reshape((-1,1)) 

    # Compute sample weights for weighted linear regression
    df_sim_data = pd.DataFrame({'exp_id':    np.concatenate([[i]*len(sim_data[i][1]) for i in range(len(sim_data))]),
                                'id_in_exp': np.concatenate([np.arange(len(sim_data[i][1])) for i in range(len(sim_data))]),
                                'data':      np.concatenate([sim_data[i][1] for i in range(len(sim_data))])
                                })
    
    combine_exp_id = [2,3] if combine_m else None # combine m and m_steady
    w = reweight_samples(df_sim_data, weighting=weighting,combine_exp_id=combine_exp_id)

    fit, yres, ypred, train_rss, train_mse, train_nmse = fit_wlse(x, y, w)
    coef = fit[1]  
    shift = fit[0]
    yres_df = None

    # Save results
    if len(save_path)>0:

        sl.make_long_dir(save_path)

        if weighting!='none':
            save_name = f'_{weighting}_{save_name}' if len(save_name)>0 else f'_{weighting}'
        
        if nonleaky:
            save_name = save_name + '_nonleaky'

        # Save coefficients and shift
        coef_df = pd.DataFrame({'coef':coef,
                                'shift':shift},
                                index=[0])
        coef_df.to_csv(os.path.join(save_path,f'coef_fit{save_name}.csv'),index=False)

        # Save error vector
        yres_df = pd.DataFrame({'exp_id':       df_sim_data['exp_id'],
                                'id_in_exp':    df_sim_data['id_in_exp'],
                                'yres':         yres.flatten(),
                                'ypred':        ypred.flatten(),
                                'w':            w.flatten()})
        yres_df.to_csv(os.path.join(save_path,f'yres_fit{save_name}.csv'),index=False)

        # Save train metrics
        mse_df = pd.DataFrame({'train_rss':     train_rss,
                               'train_mse':     train_mse,
                               'train_nmse':    train_nmse},index=[0])
        mse_df.to_csv(os.path.join(save_path,f'mse_fit{save_name}.csv'),index=False)

        # Save fitted data in compatible format
        data_names = ['l','lp','m','m_steady']
        data_var = ['n_fam','dN','n_im','n_im']
        data_val = ['nt_norm','tr_norm','nt_norm','steady']
        cc = 0
        ypred = ypred.flatten()
        pred_data = []
        for i, n in zip(range(len(sim_data)), data_names):
            pred_data.append((sim_data[i][0],ypred[cc:cc+len(sim_data[i][1])]))
            cc += len(sim_data[i][1])
            pd.DataFrame({data_var[i]:pred_data[i][0],data_val[i]:pred_data[i][1]}).to_csv(os.path.join(save_path,f'pred_{n}{save_name}.csv'),index=False)

    return coef, shift, yres_df, [train_rss, train_mse, train_nmse]

############################################################################################################
#               Function to run cross-validation                                                           #
############################################################################################################
def crossvalidation_homann_exp(sim_data,homann_data,coef_steady=True,sampling_type='jackknife',weighting='equal-samples',drop_type='sample',combine_m=False,nonleaky=False,save_path='',save_name=''):

    x, y, df_sim_data = get_lr_vectors(sim_data=sim_data,homann_data=homann_data,coef_steady=coef_steady)

    combine_exp_id = [2,3] if combine_m else None # combine m and m_steady

    coef_all_df, mse_df, test_df, pred_full, [jack_mse_mean, jack_mse_std, jack_mse_se] = get_losslandscape_cv(x,y,df_sim_data,sampling_type=sampling_type,weighting=weighting,drop_type=drop_type,combine_exp_id=combine_exp_id)

    # Save results
    if len(save_path)>0:
        sl.make_long_dir(save_path)
        save_name = f'_{sampling_type}_drop-{drop_type}' if len(save_name)==0 else f'_{sampling_type}_drop-{drop_type}_{save_name}'

        if weighting!='none':
            save_name = f'_{weighting}_{save_name}' if len(save_name)>0 else f'_{weighting}'

        if nonleaky:
            save_name = save_name + '_nonleaky'

        # Save coefficients and shift    
        coef_all_df.to_csv(os.path.join(save_path,f'coef_fit_all{save_name}.csv'),index=False)

        # Save train metrics
        mse_df.to_csv(os.path.join(save_path,f'train_metrics{save_name}.csv'),index=False)

        if 'loo' in sampling_type:
            # Save test metrics
            test_df.to_csv(os.path.join(save_path,f'test_metrics{save_name}.csv'),index=False)

            # Save predictions
            np.save(os.path.join(save_path,f'pred-test-all_{save_name}.npy'), pred_full, allow_pickle=True)

    return jack_mse_mean, jack_mse_std, jack_mse_se, df_sim_data

def jackknife_crossvalidation_homann_exp(sim_data,homann_data,coef_steady=True,sampling_type='outerjack-cv',weighting='equal-samples',drop_type='sample',combine_m=False,nonleaky=False,n_bootstrap=10,save_path='',save_name='',bootstrap_seed=98765):

    x, y, df_sim_data = get_lr_vectors(sim_data=sim_data,homann_data=homann_data,coef_steady=coef_steady)

    combine_exp_id = [2,3] if combine_m else None # combine m and m_steady  

    # Create outer resampling sets (either using bootstrap or jackknife, always drop single sample)
    if 'outerboot' in sampling_type:
        rng = np.random.default_rng(bootstrap_seed)  # set random seed for reproducibility
        masks_resample = np.zeros(len(y),n_bootstrap)

        # Bootstrap each experiment separately
        for i_exp in df_sim_data['exp_id'].unique():
            i_id = df_sim_data.loc[df_sim_data['exp_id']==i_exp,'id'].values
            boot_i_id = np.array([rng.choice(i_id, len(df_sim_data.loc[df_sim_data['exp_id']==i_exp]), replace=True) for bi in range(n_bootstrap)])
            masks_resample[i_id,:] = boot_i_id

    elif 'outerjack' in sampling_type:
        masks_resample = ~np.eye(len(y),dtype=bool)

    # For each outer resampling set, compute cross validation error landscape (for given inner weighting scheme and drop type)
    coef_all_df = []; mse_df = []; test_df = []
    pred_full = []   
    train_mse = []
    for i in range(masks_resample.shape[1]):

        mask_i = masks_resample[:,i]
        x_i    = x[mask_i,:]
        y_i    = y[mask_i,:]
        df_sim_data_i = df_sim_data[mask_i].copy().reset_index(drop=True)
        df_sim_data_i['id'] = np.arange(len(df_sim_data_i))  # reset id for current resampling set

        coef_all_df_i, mse_df_i, test_df_i, pred_full_i, [train_mse_i, _, _] = get_losslandscape_cv(x_i,y_i,df_sim_data_i,sampling_type='jackknife-loo',weighting=weighting,drop_type=drop_type,combine_exp_id=combine_exp_id)
        coef_all_df_i['outer_resampling_id']    = i
        mse_df_i['outer_resampling_id']         = i
        test_df_i['outer_resampling_id']        = i

        coef_all_df.append(coef_all_df_i); mse_df.append(mse_df_i); test_df.append(test_df_i)
        pred_full.append(pred_full_i)
        train_mse.append(train_mse_i)
        
    # Concatenate data for all resampling sets
    coef_all_df = pd.concat(coef_all_df)
    mse_df      = pd.concat(mse_df)
    test_df     = pd.concat(test_df)
    pred_full   = np.stack(pred_full,axis=2)
    train_mse   = np.array(train_mse)
    outer_mse_df = pd.DataFrame({'outer_resampling_id': np.arange(masks_resample.shape[1]),
                                 'train_mse':           train_mse})
    
    if 'outerboot' in sampling_type:
        # Compute the bootstrapped train error by averaging the train errors of all outer (jackknife) resampling sets
        w_av = 1/len(train_mse)*np.ones(len(train_mse))
        jack_mse_mean, jack_mse_std, jack_mse_se = weighted_av_se(train_mse,weights=w_av.flatten())

    elif 'outerjack' in sampling_type:
        # Compute the jackknifed train error by averaging the train errors of all outer (jackknife) resampling sets
        w_av = reweight_samples(df_sim_data, weighting=weighting, combine_exp_id=combine_exp_id) # I think this might be wrong - outer strategy is always drop-sample, equal-sample weighting
            
        jack_mse_mean, jack_mse_std, jack_mse_se = weighted_av_se(train_mse,weights=w_av.flatten())
    
    if len(save_path)>0:
        sl.make_long_dir(save_path)
        save_name = f'_{sampling_type}_drop-{drop_type}' if len(save_name)==0 else f'_{sampling_type}_drop-{drop_type}_{save_name}'

        if weighting!='none':
            save_name = f'_{weighting}_{save_name}' if len(save_name)>0 else f'_{weighting}'

        if nonleaky:
            save_name = save_name + '_nonleaky'

        # Save coefficients and shift    
        coef_all_df.to_csv(os.path.join(save_path,f'coef_fit_all{save_name}.csv'),index=False)

        # Save train metrics
        mse_df.to_csv(os.path.join(save_path,f'train_metrics{save_name}.csv'),index=False)

        # Save train metric averages
        outer_mse_df.to_csv(os.path.join(save_path,f'average_train_metrics{save_name}.csv'),index=False)
        
        # Save test metrics
        test_df.to_csv(os.path.join(save_path,f'test_metrics{save_name}.csv'),index=False)

        # Save predictions
        np.save(os.path.join(save_path,f'pred-test-all_{save_name}.npy'), pred_full, allow_pickle=True)

    return jack_mse_mean, jack_mse_std, jack_mse_se, df_sim_data

def get_testerror_cv(df_sim_data,sampling_type='jackknife-loo',weighting='equal-samples',drop_type='sample',combine_m=False,nonleaky=False,load_path=''):

    # Load grid (cross-validation)
    grid_name = f'grid_{sampling_type}-drop-{drop_type}-{weighting}.csv' 
    grid = pd.read_csv(os.path.join(load_path, grid_name))
    if nonleaky:
        grid = grid[grid['alph_leak']==0]
    gid_i = grid['grid_id'].values

    # Get individual train errors for each resampling set and each grid point
    mse_i = []
    for j, gid_ij in enumerate(gid_i):
        # Load MSEs
        if os.path.exists(os.path.join(load_path, f'grid_{int(gid_ij)}/train_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv')):
            mse_ij = pd.read_csv(os.path.join(load_path, f'grid_{int(gid_ij)}/train_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv'))
            if os.path.exists(os.path.join(load_path, f'grid_{int(gid_ij)}/test_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv')):
                mse_test_ij = pd.read_csv(os.path.join(load_path, f'grid_{int(gid_ij)}/test_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv'))
                mse_ij = mse_ij.merge(mse_test_ij, on='resampling_id', how='left')
        else:
            continue
        mse_ij['gid'] = gid_ij
        mse_i.append(mse_ij[['resampling_id', 'gid', f'train_mse', f'test_mse']])
    mse_i = pd.concat(mse_i, ignore_index=True).reset_index(drop=True)

    # Get best parameters for each resampling set (refitting of novelty model)
    mse_refitted_i = []
    for j in mse_i['resampling_id'].unique():
        mse_ij = mse_i[mse_i['resampling_id']==j]
        bestmse_ij = mse_ij[mse_ij[f'train_mse']==np.min(mse_ij[f'train_mse'])]
        mse_refitted_i.append(bestmse_ij)
    mse_refitted_i = pd.concat(mse_refitted_i, ignore_index=True).reset_index(drop=True)

    # Get reweighting of resampling sets
    combine_exp_id = [2,3] if combine_m else None # combine m and m_steady
    if drop_type=='sample':
        w_av = reweight_samples(df_sim_data, weighting=weighting, combine_exp_id=combine_exp_id)
    elif drop_type=='exp':
        w_av = reweight_exp(df_sim_data, weighting=weighting, combine_exp_id=combine_exp_id)
    
    # Compute train and test errors (cross-validation with refitting of novelty model)
    train_i, train_i_std, train_i_se = weighted_av_se(mse_refitted_i[f'train_mse'].values, w_av.flatten())
    test_i, test_i_std, test_i_se    = weighted_av_se(mse_refitted_i[f'test_mse'].values, w_av.flatten())
    
    return mse_refitted_i, [train_i, train_i_std, train_i_se], [test_i, test_i_std, test_i_se]


def get_testerror_jackknife_cv(df_sim_data,sampling_type='outerjack-cv',weighting='equal-samples',drop_type='sample',combine_m=False,nonleaky=False,load_path='',bootstrap_seed=98765,n_bootstrap=50):

    # Load grid (cross-validation)
    grid_name = f'grid_{sampling_type}-drop-{drop_type}-{weighting}.csv' 
    grid = pd.read_csv(os.path.join(load_path, grid_name))
    if nonleaky:
        grid = grid[grid['alph_leak']==0]
    gid = grid['grid_id'].values

    # Get individual train errors for each outer and inner resampling set and each grid point
    mse = []
    for j, gid_j in enumerate(gid):
        # Load MSEs
        if os.path.exists(os.path.join(load_path, f'grid_{int(gid_j)}/train_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv')):
            mse_j = pd.read_csv(os.path.join(load_path, f'grid_{int(gid_j)}/train_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv'))
            if os.path.exists(os.path.join(load_path, f'grid_{int(gid_j)}/test_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv')):
                mse_test_j = pd.read_csv(os.path.join(load_path, f'grid_{int(gid_j)}/test_metrics_{weighting}__{sampling_type}_drop-{drop_type}.csv'))
                mse_j = mse_j.merge(mse_test_j, on=['outer_resampling_id','resampling_id'], how='left')
        else:
            continue
        mse_j['gid'] = gid_j
        mse.append(mse_j[['outer_resampling_id','resampling_id', 'gid', f'train_mse', f'test_mse']])
    mse = pd.concat(mse, ignore_index=True).reset_index(drop=True)

    # Create outer resampling sets (either using bootstrap or jackknife) -- needed for correct reweighting of resampling sets (for each outer resampling set)
    # Note: we are recreating exactly the same resampling sets that were used to compute the loss landscape in the first place (to not be forced to save them)
    if 'outerboot' in sampling_type:
        rng = np.random.default_rng(bootstrap_seed)  # set random seed for reproducibility
        masks_resample = np.zeros(len(df_sim_data),n_bootstrap)

        # Bootstrap each experiment separately
        for i_exp in df_sim_data['exp_id'].unique():
            i_id = df_sim_data.loc[df_sim_data['exp_id']==i_exp,'id'].values
            boot_i_id = np.array([rng.choice(i_id, len(df_sim_data.loc[df_sim_data['exp_id']==i_exp]), replace=True) for bi in range(n_bootstrap)])
            masks_resample[i_id,:] = boot_i_id

    elif 'outerjack' in sampling_type:
        masks_resample = ~np.eye(len(df_sim_data),dtype=bool)

    # Get test error for each outer resampling set
    mse_refitted = []
    train        = []
    test         = []
    for i in mse['outer_resampling_id'].unique():
        
        # Get best parameters for each inner resampling set (refitting of novelty model)
        mse_i = mse.loc[mse['outer_resampling_id']==i]
        mse_refitted_i = []
        for j in mse_i['resampling_id'].unique():
            mse_ij = mse_i[mse_i['resampling_id']==j]
            bestmse_ij = mse_ij[mse_ij[f'train_mse']==np.min(mse_ij[f'train_mse'])]
            mse_refitted_i.append(bestmse_ij)
        mse_refitted_i = pd.concat(mse_refitted_i, ignore_index=True).reset_index(drop=True)
        mse_refitted.append(mse_refitted_i)

        # Get reweighting of inner resampling sets
        combine_exp_id = [2,3] if combine_m else None # combine m and m_steady
        if drop_type=='sample':
            w_av_i = reweight_samples(df_sim_data[masks_resample[i]], weighting=weighting, combine_exp_id=combine_exp_id)
        elif drop_type=='exp':
            w_av_i = reweight_exp(df_sim_data[masks_resample[i]], weighting=weighting, combine_exp_id=combine_exp_id)
        
        # Compute train and test errors (cross-validation with refitting of novelty model)
        train_i, _, _ = weighted_av_se(mse_refitted_i[f'train_mse'].values, w_av_i.flatten())
        test_i, _, _  = weighted_av_se(mse_refitted_i[f'test_mse'].values, w_av_i.flatten())
        train.append(train_i); test.append(test_i)

    if 'outerboot' in sampling_type:
        w_av = 1/len(train_mse)*np.ones(len(train_mse))
        jack_mse_mean, jack_mse_std, jack_mse_se = weighted_av_se(train_mse,weights=w_av.flatten())

    elif 'outerjack' in sampling_type:
        # Compute the combined errors over outer resampling sets (using same weighting of samples as in inner fitting procedure)
        w_av = reweight_samples(df_sim_data, weighting=weighting, combine_exp_id=combine_exp_id) 
        
    df_errors = pd.DataFrame({'outer_resampling_id': mse['outer_resampling_id'].unique(),
                              'train_mse':           train,
                              'test_mse':            test})
    train_mean, train_std, train_se = weighted_av_se(np.array(train), w_av.flatten())
    test_mean, test_std, test_se    = weighted_av_se(np.array(test), w_av.flatten())

    mse_refitted = pd.concat(mse_refitted, ignore_index=True).reset_index(drop=True)
    
    return mse_refitted, [train_mean, train_std, train_se], [test_mean, test_std, test_se], df_errors


if __name__ == '__main__':

    # Run tests

    print('Running tests for fit_homann_new.py...')

    ##############################################################################
    # Test reweighting of samples (weighting = 'equal-samples')                  #
    ##############################################################################
    df_sim_data = pd.DataFrame({'exp_id':    np.array([0,0,0,1,1,1,2,2,2]),
                                'id_in_exp': np.array([0,1,2,0,1,2,0,1,2]),
                                'data':      np.array([1,2,3,4,5,6,7,8,9])})
    w = reweight_samples(df_sim_data, weighting='equal-samples')
    assert np.all(w == 1/9 * np.ones((9,1))), "Reweighting for equal-samples failed."

    df_sim_data = pd.DataFrame({'exp_id':    np.array([0,0,1,1,1,2,2,2,2]),
                                'id_in_exp': np.array([0,1,0,1,2,0,1,2,3]),
                                'data':      np.array([1,2,3,4,5,6,7,8,9])})
    w = reweight_samples(df_sim_data, weighting='equal-samples')
    assert np.all(w == 1/9 * np.ones((9,1))), "Reweighting for equal-samples failed."

    # Add cases for masking

    ##############################################################################
    # Test reweighting of samples (weighting = 'equal-exp')                      #
    ##############################################################################
    df_sim_data = pd.DataFrame({'exp_id':    np.array([0,0,0,1,1,1,2,2,2]),
                                'id_in_exp': np.array([0,1,2,0,1,2,0,1,2]),
                                'data':      np.array([1,2,3,4,5,6,7,8,9])})
    w = reweight_samples(df_sim_data, weighting='equal-exp')
    assert np.all(w == 1/3 * np.ones((9,1))), "Reweighting for equal-exp failed."

    df_sim_data = pd.DataFrame({'exp_id':    np.array([0,0,1,1,1,2,2,2,2]),
                                'id_in_exp': np.array([0,1,0,1,2,0,1,2,3]),
                                'data':      np.array([1,2,3,4,5,6,7,8,9])})
    w = reweight_samples(df_sim_data, weighting='equal-exp')
    assert np.all(w == np.array([1/2, 1/2, 1/3, 1/3, 1/3, 1/4, 1/4, 1/4, 1/4]).reshape(w.shape)), "Reweighting for equal-exp failed."

    # Add cases for masking

    ##############################################################################
    # Test reweighting of exp (weighting = 'equal-samples')                      #
    ##############################################################################
    # df_sim_data = pd.DataFrame({'exp_id':    np.array([0,0,0,1,1,1,2,2,2]),
    #                             'id_in_exp': np.array([0,1,2,0,1,2,0,1,2]),
    #                             'data':      np.array([1,2,3,4,5,6,7,8,9])})
    # w = reweight_exp(df_sim_data, weighting='equal-samples', combine_m=True)
    # assert np.all(w == 1/3 * np.ones((9,1))), "Reweighting for equal-exp failed."

    ##############################################################################
    # Test reweighting of exp (weighting = 'equal-exp')                          #
    ##############################################################################
    # df_sim_data = pd.DataFrame({'exp_id':    np.array([0,0,0,1,1,1,2,2,2]),
    #                             'id_in_exp': np.array([0,1,2,0,1,2,0,1,2]),
    #                             'data':      np.array([1,2,3,4,5,6,7,8,9])})
    # w = reweight_exp(df_sim_data, weighting='equal-samples', combine_m=True)
    # assert np.all(w == 1/3 * np.ones((9,1))), "Reweighting for equal-exp failed."


    ##############################################################################
    # Test fitting                                                               #
    ##############################################################################
    shift = 0.76
    coef  = 1.234

    test_x = [np.sin(np.linspace(0,3,20)), np.cos(np.linspace(0,3,20)), np.linspace(0,3,20), np.ones(20)*shift]
    test_ytrue = [coef*test_x[i]+shift for i in range(len(test_x))]
    test_ytrue_noisy = [test_ytrue[i] + np.random.normal(0,0.1,len(test_ytrue[i])) for i in range(len(test_ytrue))]
    test_sdata = [(np.linspace(0,3,20), test_x[i]) for i in range(len(test_x))]
    test_hdata = [(np.linspace(0,3,20), test_ytrue_noisy[i]) for i in range(len(test_ytrue_noisy))]
    test_hdata_noisy = [(np.linspace(0,3,20), test_ytrue_noisy[i]) for i in range(len(test_ytrue_noisy))]

    coef, shift, yres_df, [train_rss, train_mse, train_nmse] = fit_homann_exp(test_sdata,test_hdata,coef_steady=True,weighting='equal-samples',save_path='',save_name='')


    print('All tests passed for fit_homann_new.py.')

