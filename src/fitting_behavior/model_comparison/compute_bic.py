import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils.saveload as sl

def compute_bic(path_models,candidates,opt_method,comb_type,path_save,name_save,maxit,n_app=None,n_sep=None):
    # Get number of data points from underlying mice data (see compute_sizemicedata.py)
    if n_app==None: n_app = 7972
    if n_sep==None: n_sep = {'B1': 1195, 'B2': 11, 'B3': 15, 'B4': 389, 'C1': 91, 'C3': 85, 'C6': 2183, 'C7': 359, 'C8': 39, 'C9': 469, 'B5': 401, 'B6': 79, 'B7': 1019, 'D3': 99, 'D4': 339, 'D5': 347, 'D6': 3, 'D7': 139, 'D8': 45, 'D9': 665}

    # Init results dataframe
    model_all   = []
    subID_all   = []
    bic_all     = []
    mle_all     = []
    if comb_type=='sep': 
        mean_ll_all = []
        sum_ll_all  = []
        sum_bic_all = []

    # Compute BICs for all models
    for i in range(len(candidates)):
        # Load LLs of each model (previously computed)
        # if candidates[i]=='hybrid-maxit':
        #     path = path_models+f'mle-maxit_hybrid-mice_{opt_method}/'
        #     file = f'mle-maxit_hybrid-mice_{opt_method}_{comb_type}.csv'
        path = path_models+f'mle{"-maxit" if maxit[i] else ""}_{candidates[i]}-mice_{opt_method}/'
        file = f'mle{"-maxit" if maxit[i] else ""}_{candidates[i]}-mice_{opt_method}_{comb_type}.csv'
        res  = sl.load_sim_data(path,file_data=file)

        # Compute combined LL 
        k  = len(res.var_name.unique())
        if comb_type=='app':
            LL = -res['mle_ll'].unique()[0] # minus since we recorded the -LL (minimizer function to optimize)
            bic = -2*LL+k*np.log(n_app)
            model_all.append(candidates[i])
            subID_all.append(-1)
            bic_all.append(bic)
            mle_all.append(LL)
            
        elif comb_type=='sep':  
            LL = res[['subID','mle_ll']].drop_duplicates()
            LL['mle_ll'] = - LL['mle_ll'] # minus since we recorded the -LL (minimizer function to optimize)
            n_sep_sorted = [n_sep[s] for s in LL['subID']]
            bic = -2*LL['mle_ll'].values+k*np.log(n_sep_sorted)
            model_all.extend([candidates[i]]*len(LL))
            subID_all.extend(LL['subID'].values)
            bic_all.extend(bic)
            mle_all.extend(LL['mle_ll'].values)
            mean_ll_all.extend([np.nanmean(LL['mle_ll'].values)]*len(LL['subID'].values))
            sum_ll_all.extend([np.sum(LL['mle_ll'].values)]*len(LL['subID'].values))
            sum_bic_all.extend([np.sum(bic)]*len(LL['subID'].values))
            #LL = -np.sum([np.nanmean(res.loc[res.var_name==v,'mle_ll'].values) for v in res.var_name.unique()]) 

    bic_df = pd.DataFrame({'model':model_all,'subID':subID_all,'LL':mle_all,'bic':bic_all})
    if comb_type=='sep':
        bic_df['mean_LL']  = mean_ll_all
        bic_df['sum_LL']   = sum_ll_all
        bic_df['sum_bic']  = sum_bic_all

    # Select model with minimum BIC value
    if comb_type=='app':
        best_bic = np.min(bic_all)
        best     = list(best_bic==bic_all)
    elif comb_type=='sep':
        best_bic = np.min(sum_bic_all)
        best     = best_bic==sum_bic_all
    bic_df['best'] = best
    print(f'The winning model is: {bic_df.loc[bic_df.best,"model"].unique()}.')

    bic_df.to_csv(os.path.join(path_save,f'bic_{name_save}_{comb_type}.csv'))
    bic_df.to_pickle(os.path.join(path_save,f'bic_{name_save}_{comb_type}.pickle'))

    return bic_df

def plot_bic_bar(bic_df,comb_type,measure_type,name,name_save,path_save,eps=100,xtl=[]):
    # comb_type='sep','app' / measure_type='bic','LL'
    plot_col  = measure_type if comb_type=='app' else ('sum_LL' if measure_type=='LL' else 'sum_bic') # mean_LL
    plot_data = bic_df[['model',plot_col]].drop_duplicates()
    plot_best_data  = bic_df.loc[bic_df.best,plot_col].unique()[0]

    # Plot model BIC/LL values
    f,ax = plt.subplots()
    ax.bar(np.arange(len(plot_data)),plot_data[plot_col].values)
    ax.set_xticks(np.arange(len(plot_data)))
    if len(xtl)==0:
        xtl = list(plot_data['model'].values)
    ax.set_xticklabels(xtl)
    ax.set_ylim([min(plot_data[plot_col].values)-eps,max(plot_data[plot_col].values)+eps])
    eps1 = 0.5
    ax.plot([0-eps1,len(plot_data)-eps1],[plot_best_data,plot_best_data],'k--')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(f'{measure_type.upper()} value')
    f.suptitle(f'{measure_type.upper()} for {name} ({comb_type})')

    save_name = f'{name_save}_{comb_type}_{measure_type}'
    plt.savefig(path_save+save_name+'.svg',bbox_inches='tight')
    plt.savefig(path_save+save_name+'.eps',bbox_inches='tight')

if __name__=="__main__":
    # Specify list of candidate models
    path_models = '/Volumes/lcncluster/becker/RL_reward_novelty/data/MLE_results/Fits/SingleRun/'
    candidates  = ['nac-oi-only',
                    'nac',
                    'nac-kpop',
                    'nac-kmix',
                    'nac-nooi',
                    'nor',
                    'hybrid2',
                    'hybrid']
    maxit   = [False, 
                False, 
                False,
                False,
                False,
                False,
                False,
                True]
    # candidates  = ['hybrid-maxit',
    #                 'hybrid']
    opt_method  = 'Nelder-Mead'
    comb_type   = 'app' # 'sep','app'
    measure_type= 'bic' # 'bic','LL'
    # name_save   = 'hybrid-vs-maxit'
    # name_save1  = 'hybrid-vs-maxit'
    # name        = 'hybrid with/without maxit'
    name_save   = 'basic-nov'
    name_save1  = 'basic-nov_which-alg'
    name        = 'basic nov'
    
    path_save   = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelSelection/BIC/'
    sl.make_long_dir(path_save)
    path_save1  = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/ModelSelection/BIC/'
    sl.make_long_dir(path_save1)

    bic_df = compute_bic(path_models,candidates,opt_method,comb_type,path_save,name_save,maxit)
    plot_bic_bar(bic_df,comb_type,measure_type,name,name_save1,path_save1)
    print('done')