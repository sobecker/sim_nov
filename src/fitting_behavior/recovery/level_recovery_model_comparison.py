import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import utils.saveload as sl
from fitting_behavior.recovery.modelrecoverymatrix import get_n_recov, compute_bic_recov

def plot_modelrecov_levels(alg_type,measure_type,comb_type,opt_method,data_gen,candidates,sims,f_comp,save_plot,path_save_plot,name_save_plot,figshape,uniparam=True,title='',show_n=False):
    name_save_data  = f'modelrecov-{alg_type}'
    path_save_data  = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelRecovery/'
    path_model      = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData{"_uniparam" if uniparam else ""}/'
    f_plot          = True

    if f_comp:
        list_bic_df = []
        for ii in range(len(data_gen)):
            #if data_gen[ii]=='l5': candidates=['l1','l6']
            #if data_gen[ii]=='l6': candidates=['l1','l5']
            #if data_gen[ii]=='l1': candidates=['l5']
            path_candidates = []
            for jj in range(len(candidates)):
                if data_gen[ii]==candidates[jj]:
                    if alg_type=='hnor' and not uniparam:
                        path_candidates.append(path_model / f'{alg_type}_{comb_type}_{opt_method}/{alg_type}_{comb_type}_{opt_method}_{data_gen[ii]}/')
                    else:
                        path_candidates.append(path_model / f'sim-{alg_type}-{comb_type}_fit-{alg_type}-{comb_type}_{opt_method}/mle-recov_sim-{alg_type}_fit-{alg_type}_{comb_type}_{opt_method}_{candidates[jj]}/')
                else:
                    path_candidates.append(path_model / f'sim-{alg_type}-{comb_type}_fit-{alg_type}-{comb_type}_{opt_method}/sim-{alg_type}-{comb_type}-{data_gen[ii]}_{alg_type}-{comb_type}-{candidates[jj]}_{opt_method}/')
            
            sl.make_long_dir(path_save_data)
            sl.make_long_dir(path_save_plot)

            n_app, n_sep = get_n_recov([1],sims,alg_type,uniparam=uniparam)
            print(f'Starting model comparison for data generated by {alg_type}-{data_gen[ii]}.\n')
            n = n_app if comb_type=='app' else n_sep
            bic_df_ii = compute_bic_recov(path_candidates,candidates,sims,opt_method,comb_type,alg_type,path_save_data,name_save_data,n,data_gen=data_gen[ii])
            bic_df_ii = bic_df_ii.rename(columns={'model':'fit_model'})
            bic_df_ii['data_model'] = [data_gen[ii]]*len(bic_df_ii)
            # Compute percentages
            tot_fit = bic_df_ii[['fit_model','best']].groupby(['fit_model']).count().values
            p_best = bic_df_ii.groupby(['fit_model']).agg(perc_best=('best',np.sum))/tot_fit
            p_best['tot_fit'] = tot_fit
            p_best.loc[p_best.perc_best==np.inf,'perc_best'] = np.NaN
            bic_df_ii = bic_df_ii.join(p_best,on='fit_model')
            for fm in bic_df_ii.fit_model.unique():
                print(f'P({fm}|{data_gen[ii]})={bic_df_ii.loc[bic_df_ii.fit_model==fm,"perc_best"].values[0]} (sample size: {bic_df_ii.loc[bic_df_ii.fit_model==fm,"tot_fit"].values[0]}).')
            list_bic_df.append(bic_df_ii)

        all_bic_df = pd.concat(list_bic_df)
        all_bic_df.to_csv(path_save_data / f'bic_{name_save_data}_{comb_type}.csv')
        all_bic_df.to_pickle(path_save_data / f'bic_{name_save_data}_{comb_type}.pickle')

    if f_plot:
        if not f_comp: 
            all_bic_df = sl.load_sim_data(path_save_data,file_data=f'bic_{name_save_data}_{comb_type}.pickle')
        p_win = all_bic_df[['data_model','fit_model','perc_best','tot_fit']].drop_duplicates().reset_index(drop=True)
        # Fill missing values with nan
        for i in data_gen:
            for j in candidates:
                if not ((p_win['data_model'] == i) & (p_win['fit_model'] == j)).any():
                    p_win = pd.concat([p_win,pd.DataFrame({'data_model':i,'fit_model':j,'perc_best':np.NaN},index=[0])])
        # Sort values, extract and reshape into matrix
        p_win = p_win.sort_values(['data_model','fit_model'])
        mat_win = p_win.perc_best.values.reshape((len(p_win.data_model.unique()),-1))
        mat_tot = p_win.tot_fit.values.reshape((len(p_win.data_model.unique()),-1))
        #al = [[f'{np.round(mat_win[j][i],2)}\n(n={mat_tot[j][i]})' for i in range(mat_win.shape[0])] for j in range(mat_win.shape[1])]
        if show_n:
            al = [f'{np.round(i,2)}\n(n={j})' for i,j in zip(p_win.perc_best.values,p_win.tot_fit.values)]
        else:
            al = [f'{np.round(i,2)}' for i in p_win.perc_best.values]
        mat_al  = np.array(al).reshape((len(p_win.data_model.unique()),-1))
        # Plot matrix
        f,ax = plt.subplots(1,1,figsize=figshape)
        cmap = sn.diverging_palette(20, 220, n=200) #'bwr'
        sn.heatmap(mat_win,cmap=cmap,vmin=0,vmax=1,center=0,square=True,cbar=True,ax=ax,cbar_kws={'label':'P(fit|data)','shrink': 0.89},annot=mat_al,fmt='')
        # cmap = 'binary' 
        # sn.heatmap(mat_win,cmap=cmap,vmin=0,vmax=1,center=0.8,square=True,cbar=True,ax=ax,cbar_kws={'label':'P(fit|data)','shrink': 0.89},annot=True)
        ax.set_ylabel('Gran. level (data)')
        ax.set_yticklabels([i.replace('l','') for i in data_gen],rotation=0)
        ax.set_xlabel('Gran. level (fit)')
        ax.set_xticklabels([i.replace('l','') for i in candidates])
        if len(title)>0: ax.set_title(title)
        f.tight_layout()

        if save_plot:
            plt.savefig(path_save_plot / f'{name_save_plot}.eps', bbox_inches='tight')
            plt.savefig(os.path.join(path_save_plot / f'{name_save_plot}.svg'),bbox_inches='tight')


if __name__=='__main__': 

    # alg_type        = 'hnor'
    # sims            = list(np.arange(20))
    # title           = f'Recovery of granularity (MB)'
    # uniparam        = True
    # data_gen        = ['l4','l5','l6']
    # candidates      = ['l4','l5','l6']
    
    # alg_type        = 'hnor'
    # sims            = list(np.arange(20))
    # title           = f'Recovery of granularity (MB)'
    # uniparam        = False
    # data_gen        = ['l1','l5','l6']
    # candidates      = ['l1','l5','l6']

    # alg_type        = 'hhybrid'
    # sims            = list(np.arange(20))
    # title           = f'Recovery of granularity (Hybrid)'
    # uniparam        = False
    # data_gen        = ['l1','l5','l6']
    # candidates      = ['l1','l5','l6']
    
    alg_type        = 'hhybrid2'
    sims            = list(np.arange(30))
    title           = f'Recovery of granularity (Hybrid)'
    uniparam        = True
    data_gen        = ['l4','l5','l6']
    candidates      = ['l4','l5','l6']  

    measure_type    = 'bic' # 'bic','LL'
    comb_type       = 'app' 
    opt_method      = 'Nelder-Mead'
    
    f_comp          = True
    save_plot       = True
    name_save_plot  = f'modelrecov{"-uniparam" if uniparam else ""}-{alg_type}'
    path_save_plot  = os.path.join(sl.get_datapath().replace('data','output'),f'Figures_Paper/Fig_model_comparison/')
    figshape        = (4,3.5)
    plot_modelrecov_levels(alg_type,measure_type,comb_type,opt_method,data_gen,candidates,sims,f_comp,save_plot,path_save_plot,name_save_plot,figshape=figshape,uniparam=uniparam,title=title,show_n=True)