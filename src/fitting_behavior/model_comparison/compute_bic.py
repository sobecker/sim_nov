import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.saveload as sl
import fitting_behavior.optimization.base_params_opt as bpo
from fitting_behavior.mle.mle_fit import preprocess_micedata

### Script to compute BIC for RL agents seeking count-based novelty ###

# Compute number of steps in mice data ########################################################################
def compute_size_micedata(UnrewNames=['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9'],RewNames=['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']):
    AllNames    = RewNames+UnrewNames

    params = bpo.base_params_nACtree.copy()
    P = params['P']

    d = []
    len_app = 0
    len_sep = []
    for i in range(len(AllNames)):
        dir  = sl.get_rootpath() / 'ext_data' / 'Rosenberg2021'
        file = dir / f'{AllNames[i]}_data' / f'{AllNames[i]}-stateseq_UntilG.pickle'
        df_i = preprocess_micedata(dir,file,P,subID=AllNames[i],epi=0)
        d.append(df_i)
        len_app += len(df_i)
        len_sep.append(len(df_i))

    dict_len = {a:l for a,l in zip(AllNames,len_sep)}

    return len_app, dict_len

# Compute BIC for all models ###################################################################################
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
        path = path_models / f'mle{"-maxit" if maxit[i] else ""}_{candidates[i]}-mice_{opt_method}'
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

    bic_df.to_csv(path_save / f'bic_{name_save}_{comb_type}.csv')
    bic_df.to_pickle(path_save / f'bic_{name_save}_{comb_type}.pickle')

    return bic_df

# Plot BIC for all models ###################################################################################
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
    plt.savefig(path_save / f'{save_name}.svg',bbox_inches='tight')
    plt.savefig(path_save / f'{save_name}.eps',bbox_inches='tight')

if __name__=="__main__":

    n_app, n_sep = compute_size_micedata()
    plot_bic    = False

    # Compute BIC for RL models seeking count-based novelty #####################################################
    path_models = sl.get_rootpath() / 'data' / 'mle_results' / 'fits' / 'singlerun'
    candidates  = ['nac',
                    'nor',
                    # 'hybrid2'
                    ]
    maxit   = [False] * len(candidates)
    opt_method  = 'Nelder-Mead'
    comb_type   = 'app' # 'sep','app'
    measure_type= 'bic' # 'bic','LL'
    name_save   = 'basic-nov'
    
    path_save   = sl.get_rootpath() / 'data' / 'model_selection' / 'bic'
    sl.make_long_dir(path_save)
    bic_df = compute_bic(path_models,candidates,opt_method,comb_type,path_save,name_save,maxit,n_app,n_sep)
    
    if plot_bic:
        name_save1  = 'basic-nov_which-alg'
        name        = 'basic nov'
        path_save1  = sl.get_rootpath() / 'output' / 'model_selection' / 'bic'
        sl.make_long_dir(path_save1)
        plot_bic_bar(bic_df,comb_type,measure_type,name,name_save1,path_save1)
    
    print('Computed BIC for count-based novelty-seeking RL models.')

    # Compute BIC for RL models seeking similarity-based novelty ################################################
    alg_types    = ['hhybrid2',
                    'hnor',
                    'hnac-gn'
                    ] # 'hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
    # alg_types    = ['hnor_center-triangle','hnor_notrace','hnor_notrace_center-box',
    #                 'hnac-gn_center-triangle','hnac-gn_notrace','hnac-gn_notrace_center-box',                    
    #                 'hhybrid2_center-triangle','hhybrid2_notrace','hhybrid2_notrace_center-box'
    #                 ] 
    data_type    = 'mice'           # 'mice','opt','naive'
    opt_method   = 'Nelder-Mead'    # 'Nelder-Mead','L-BFGS-B','SLSQP'
    comb_types   = ['app']          # 'sep','app'
    measure_type = 'LL'             # 'bic','LL'
    epss         = [[50,100]]*len(alg_types)
    maxit        = [False]*len(alg_types) 

    for i_alg in range(len(alg_types)):
        for i_comb in range(len(comb_types)):

            alg_type = alg_types[i_alg]
            comb_type = comb_types[i_comb]
            eps = epss[i_alg][i_comb]
            
            path_models = sl.get_rootpath() / 'data' / 'mle_results' / 'fits' / 'singlerun'
            levels      = [1,2,3,4,5,6]
            candidates  = [f'{alg_type}-l{i}' for i in levels]
            xtl         = [f'l{i}' for i in levels]

            path_save   = sl.get_rootpath() / 'data' / 'model_selection' / 'bic'
            sl.make_long_dir(path_save)
            name_save   = f'{alg_type}'

            bic_df = compute_bic(path_models,candidates,opt_method,comb_type,path_save,name_save,[maxit[i_alg]]*len(candidates),n_app,n_sep)

            if plot_bic:
                path_save1  = sl.get_rootpath() / 'output' / 'model_selection' / 'bic'
                sl.make_long_dir(path_save1)
                name        = f'{alg_type}'
                name_save1  = f'{alg_type}_which-alg'

                plot_bic_bar(bic_df,comb_type,measure_type,name,name_save1,path_save1,eps=eps,xtl=xtl)

        print('Computed BIC for similarity-based novelty-seeking RL models.')