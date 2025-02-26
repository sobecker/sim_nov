import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl

if __name__=="__main__":
    algs_basic   = []#['nac-oi-only','nac','nac-kpop','nac-kmix','nac-nooi','nor','hybrid']
    algs_gran    = ['hnor','hnor_center-triangle','hnor_notrace','hnor_notrace_center-box',
                    'hnac-gn-goi','hnac-gn','hnac-gn_center-triangle','hnac-gn_notrace','hnac-gn_notrace_center-box',
                    'hhybrid','hhybrid2','hhybrid2_notrace','hhybrid2_center-triangle','hhybrid2_notrace','hhybrid2_notrace_center-box'] #'hnac-gn-gv','hnac-gn-gv-goi',
    data_type    = 'mice'           # 'mice','opt','naive'
    opt_method   = 'Nelder-Mead'    # 'Nelder-Mead','L-BFGS-B','SLSQP'
    comb_types   = ['sep','app']
    measure_type = 'LL'            # 'bic','LL'

    path_load   = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelSelection/BIC/'
    path_save   = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/ModelSelection/BIC/'

    for i_comb in range(len(comb_types)):
        comb_type   = comb_types[i_comb]
        plot_col    = measure_type if comb_type=='app' else ('sum_LL' if measure_type=='LL' else 'sum_bic')
        best_fun    = np.min if measure_type=='bic' else np.max
        worst_fun   = np.max if measure_type=='bic' else np.min

        bic_gran = []
        for i_alg in range(len(algs_gran)):
            alg_type    = algs_gran[i_alg]
            bic_df      = sl.load_sim_data(path_load,file_data=f'bic_{alg_type}_{comb_type}.csv')
            bic_df['model_type'] = [alg_type]*len(bic_df)
            bic_gran.append(bic_df)
        bic_df_gran = pd.concat(bic_gran)
        heat = bic_df_gran[['model_type','model',plot_col]].drop_duplicates()
        heat1 = heat.groupby(['model_type']).agg(best=(plot_col,best_fun),worst=(plot_col,worst_fun)).reset_index()
        
        if len(algs_basic)>0:
            bic_df_basic = sl.load_sim_data(path_load,file_data=f'bic_basic-nov_{comb_type}.csv')
            bic_df_basic = bic_df_basic[['model',plot_col]].drop_duplicates()
            bic_df_basic = bic_df_basic.set_index('model').loc[algs_basic].reset_index()
            bic_df_basic['best']       = bic_df_basic[plot_col]
            bic_df_basic['worst']      = bic_df_basic[plot_col]
            bic_df_basic['model_type'] = bic_df_basic['model']
            bic_df_all = pd.concat([bic_df_basic[['model_type','best','worst']],heat1],ignore_index=True)
        else:
            bic_df_all = heat1
    
        best_ofbest  = best_fun(bic_df_all['best'].values)
        best_ofworst = best_fun(bic_df_all['worst'].values)
        bic_df_all['best_ofbest'] = bic_df_all.best==best_ofbest
        bic_df_all['best_ofworst'] = bic_df_all.worst==best_ofworst

        print(f'The best model (best-case scenario) is: {bic_df_all.loc[bic_df_all.best_ofbest,"model_type"]}.')
        print(f'The best model (worst-case scenario) is: {bic_df_all.loc[bic_df_all.best_ofworst,"model_type"]}.')

        f,ax = plt.subplots(2,1,figsize=(len(algs_basic)+len(algs_gran),6))
        xtl = [bic_df_all['model_type'][i].replace('hybrid','hyb').replace('_center-triangle','\n(ct)').replace('_notrace_center-box','\n(cb)').replace('_notrace','\n(b)') for i in range(len(bic_df_all))]
        # xtl = [bic_df_all['model_type'][i].replace('nac-','nac\n') for i in range(len(bic_df_all))]
        ylabel = 'Loglikelihood' if measure_type=='LL' else 'BIC'
        ax[0].bar(np.arange(len(bic_df_all)),bic_df_all.best.values)
        eps1 = 0.5
        win = ax[0].plot([0-eps1,len(bic_df_all)-eps1],[best_ofbest,best_ofbest],'k--')
        ax[0].set_xticks(np.arange(len(bic_df_all)))
        ax[0].set_xticklabels(xtl)
        ax[0].set_ylabel(ylabel)
        ax[0].set_title(f'Best case ({comb_type})')
        eps=100
        ax[0].set_ylim([min(bic_df_all.best.values)-eps,max(bic_df_all.best.values)+eps])
        ax[0].legend([win[0]],['Winning model'])
    
        ax[1].bar(np.arange(len(bic_df_all)),bic_df_all.worst.values)
        eps1 = 0.5
        win = ax[1].plot([0-eps1,len(bic_df_all)-eps1],[best_ofworst,best_ofworst],'k--')
        ax[1].set_xticks(np.arange(len(bic_df_all)))
        ax[1].set_xticklabels(xtl)
        ax[1].set_ylabel(ylabel)
        ax[1].set_title(f'Worst case ({comb_type})')
        ax[1].set_ylim([min(bic_df_all.best.values)-eps,max(bic_df_all.best.values)+eps])
        ax[1].legend([win[0]],['Winning model'])
        
        f.tight_layout()

        save_name        = f'reduced_best-worst-{measure_type}_{comb_type}'
        plt.savefig(os.path.join(path_save,save_name+'.eps'))
        plt.savefig(os.path.join(path_save,save_name+'.svg'))
        