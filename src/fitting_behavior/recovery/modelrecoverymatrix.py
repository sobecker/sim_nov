import numpy as np
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl

def get_n_recov(levels,sims,alg_type,uniparam=False):
    for ll in levels:
        # Get length of data for each level
        path = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData{"_uniparam" if uniparam else ""}/{alg_type}_app/{alg_type}_app_l{ll}/'
        path_s = glob.glob(os.path.join(path,'*_sim*'))
        sim_s = [int(p.split('sim')[-1]) for p in path_s]
        path_s = pd.DataFrame({'simID':sim_s,'path':path_s})
        path_s.set_index('simID')
        n_app = []
        n_sep = []
        for s in sims:
            if 'hybrid' in alg_type:
                data = sl.load_sim_data(os.path.join(path,path_s.loc[s,'path']),file_data='mf_data_basic.pickle')
            else:
                data = sl.load_sim_data(os.path.join(path,path_s.loc[s,'path'])) # get length of all individual simulations
            len_sep = data[['subID','it']].groupby(['subID']).count()
            len_sep_dict = dict(zip(list(len_sep.index),list(len_sep.it.values)))
            n_app.append(len(data))
            n_sep.append(len_sep_dict)
        
    return n_app,n_sep
  
def compute_bic_recov(path_models,models,sims,opt_method,comb_type,alg_type,path_save,name_save,n_app=None,n_sep=None,data_gen=''):
    # path_stats = '/'+os.path.join(*os.path.normpath(path_models[0]).split('/')[:-1])
    # opt_stats = sl.load_sim_data(path_stats,file_data='opt_stats.csv') # make sure to run stats_opt.py before so that opt_stats.csv is up-to-date
    # Compute BICs for all models
    list_bic_df = []
    for i in range(len(sims)):
        # Init results dataframe
        model_all   = []
        sim_all     = []
        subID_all   = []
        bic_all     = []
        mle_all     = []
        if comb_type=='sep': 
            mean_ll_all = []
            sum_ll_all  = []
            sum_bic_all = []
        path_models_i = [os.path.join(path_models[j],f'mle-recov_{alg_type}_sim{sims[i]}_{opt_method}/') for j in range(len(path_models))]
        file_i = f'mle-recov_{alg_type}_sim{sims[i]}_{opt_method}_{comb_type}.csv'
        f_exists = np.array([os.path.exists(os.path.join(path_models_i[j],file_i)) for j in range(len(path_models_i))]).all()
        f_conv = True
        # if f_exists:
        #     opt_stats_i = opt_stats.loc[opt_stats.simID==sims[i]]
        #     f_conv = np.array([opt_stats_i.loc[opt_stats_i.fit_comb==f'{data_gen}-{models[j]}','opt_success'] for j in range(len(path_models_i))]).all()
        if f_exists and f_conv:
            for j in range(len(path_models_i)):
                path = path_models_i[j] #os.path.join(path_models[j],f'mle-recov_{alg_type}_sim{sims[i]}_{opt_method}/')
                file = file_i  #f'mle-recov_{alg_type}_sim{sims[i]}_{opt_method}_{comb_type}.csv'
                res  = sl.load_sim_data(path,file_data=file)

                # Compute combined LL 
                k  = len(res.var_name.unique())
                if comb_type=='app':
                    LL = -res['mle_ll'].unique()[0] # minus since we recorded the -LL (minimizer function to optimize)
                    bic = -2*LL+k*np.log(n_app[i])
                    model_all.append(models[j])
                    sim_all.append(sims[i])
                    subID_all.append(-1)
                    bic_all.append(bic)
                    mle_all.append(LL)
            
                elif comb_type=='sep':  
                    LL = res[['subID','mle_ll']].drop_duplicates()
                    LL['mle_ll'] = - LL['mle_ll'] # minus since we recorded the -LL (minimizer function to optimize)
                    n_sep_sorted = [n_sep[i][s] for s in LL['subID']]
                    bic = -2*LL['mle_ll'].values+k*np.log(n_sep_sorted)
                    model_all.extend([models[j]]*len(LL))
                    sim_all.extend([sims[i]]*len(LL))
                    subID_all.extend(LL['subID'].values)
                    bic_all.extend(bic)
                    mle_all.extend(LL['mle_ll'].values)
                    mean_ll_all.extend([np.nanmean(LL['mle_ll'].values)]*len(LL['subID'].values))
                    sum_ll_all.extend([np.sum(LL['mle_ll'].values)]*len(LL['subID'].values))
                    sum_bic_all.extend([np.sum(bic)]*len(LL['subID'].values))
                    #LL = -np.sum([np.nanmean(res.loc[res.var_name==v,'mle_ll'].values) for v in res.var_name.unique()]) 

        bic_df = pd.DataFrame({'sampleID':sim_all,'model':model_all,'subID':subID_all,'LL':mle_all,'bic':bic_all})
        if comb_type=='sep':
            bic_df['mean_LL']  = mean_ll_all
            bic_df['sum_LL']   = sum_ll_all
            bic_df['sum_bic']  = sum_bic_all

        # Select model with minimum BIC value
        if len(bic_df)>0:
            if comb_type=='app':
                best_bic = np.min(bic_all)
                best     = list(best_bic==bic_all)
            elif comb_type=='sep':
                best_bic = np.min(sum_bic_all)
                best     = best_bic==sum_bic_all
            bic_df['best'] = best
            print(f'The winning model for data sample {i} is: {bic_df.loc[bic_df.best,"model"].unique()}.')
        else:
            bic_df['best'] = []
        list_bic_df.append(bic_df)

    all_bic_df = pd.concat(list_bic_df)
    
    #bic_df.to_csv(os.path.join(path_save,f'bic_{name_save}_{comb_type}.csv'))
    #bic_df.to_pickle(os.path.join(path_save,f'bic_{name_save}_{comb_type}.pickle'))

    return all_bic_df

if __name__=='__main__': 

    f_comp = False
    f_plot = True

    data_gen    = ['l1','l5','l6']
    candidates  = ['l1','l5','l6']
    path_model  = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/'
    
    sims = list(np.arange(6))
    opt_method  = 'Nelder-Mead'
    alg_type    = 'hnor'
    comb_type   = 'app' # 'sep','app'
    measure_type= 'bic' # 'bic','LL'
    name_save   = f'modelrecov-{alg_type}'
    name_save1  = f'modelrecov-{alg_type}'
    name        = f'Model recovery ({alg_type})'
            
    path_save   = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelRecovery/'
    path_save1  = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/ModelRecovery/'

    if f_comp:
        list_bic_df = []
        for ii in range(len(data_gen)):
            #if data_gen[ii]=='l5': candidates=['l1','l6']
            if data_gen[ii]=='l6': candidates=['l1','l5']
            path_candidates = []
            for jj in range(len(candidates)):
                if data_gen[ii]==candidates[jj]:
                    path_candidates.append(os.path.join(path_model,f'{alg_type}_{comb_type}_{opt_method}/{alg_type}_{comb_type}_{opt_method}_{data_gen[ii]}/'))
                else:
                    path_candidates.append(os.path.join(path_model,f'sim-{alg_type}-{comb_type}_fit-{alg_type}-{comb_type}_{opt_method}/sim-{alg_type}-{comb_type}-{data_gen[ii]}_{alg_type}-{comb_type}-{candidates[jj]}_{opt_method}/'))
            
            sl.make_long_dir(path_save)
            sl.make_long_dir(path_save1)

            n_app, n_sep = get_n_recov([1],sims,alg_type)
            print(f'Starting model comparison for data generated by {alg_type}-{data_gen[ii]}.\n')
            n = n_app if comb_type=='app' else n_sep
            bic_df_ii = compute_bic_recov(path_candidates,candidates,sims,opt_method,comb_type,alg_type,path_save,name_save,n)
            bic_df_ii = bic_df_ii.rename(columns={'model':'fit_model'})
            bic_df_ii['data_model'] = [data_gen[ii]]*len(bic_df_ii)
            # Compute percentages
            p_best = bic_df_ii.groupby(['fit_model']).agg(perc_best=('best',np.sum))/bic_df_ii[['fit_model','best']].groupby(['fit_model']).count().values
            bic_df_ii = bic_df_ii.join(p_best,on='fit_model')
            for fm in bic_df_ii.fit_model.unique():
                print(f'P({data_gen[ii]}|{fm})={bic_df_ii.loc[bic_df_ii.fit_model==fm,"perc_best"]}')
            list_bic_df.append(bic_df_ii)

        all_bic_df = pd.concat(list_bic_df)
        all_bic_df.to_csv(os.path.join(path_save,f'bic_{name_save}_{comb_type}.csv'))
        all_bic_df.to_pickle(os.path.join(path_save,f'bic_{name_save}_{comb_type}.pickle'))

    if f_plot:
        if not f_comp: 
            all_bic_df = sl.load_sim_data(path_save,file_data=f'bic_{name_save}_{comb_type}.pickle')
        p_win = all_bic_df[['data_model','fit_model','perc_best']].drop_duplicates().reset_index(drop=True)
        # Fill missing values with nan
        for i in data_gen:
            for j in candidates:
                if not ((p_win['data_model'] == i) & (p_win['fit_model'] == j)).any():
                    p_win = pd.concat([p_win,pd.DataFrame({'data_model':i,'fit_model':j,'perc_best':np.NaN},index=[0])])
        # Sort values, extract and reshape into matrix
        p_win = p_win.sort_values(['data_model','fit_model'])
        mat_win = p_win.perc_best.values.reshape((len(p_win.data_model.unique()),-1))
        # Plot matrix
        f,ax = plt.subplots(1,1)
        cmap = sn.diverging_palette(20, 220, n=200) #'bwr'
        sn.heatmap(mat_win,cmap=cmap,vmin=0,vmax=1,center=0,square=True,cbar=True,ax=ax,cbar_kws={'label':'P(model fit|model data)'},annot=True)
        ax.set_ylabel('Model (data)')
        ax.set_xlabel('Model (fit)')
        f.tight_layout()

        plt.savefig(path_save1+'/modelrecov_levels_confusion-matrix.eps')
        plt.savefig(path_save1+'/modelrecov_levels_confusion-matrix.svg')
