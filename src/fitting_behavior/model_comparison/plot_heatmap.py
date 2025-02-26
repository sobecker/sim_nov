import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl

if __name__=="__main__":
    # Specify list of candidate models
    # alg_types    = ['hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hhybrid']
    alg_types    = ['hnor','hnor_notrace','hnac-gn','hnac-gn_notrace','hnac-gn-goi','hhybrid','hhybrid2','hhybrid2_notrace'] 
    data_type    = 'mice' # 'mice','opt','naive'
    opt_method   = 'Nelder-Mead' # 'Nelder-Mead','L-BFGS-B','SLSQP'
    comb_types   = ['sep','app']
    measure_type = 'LL' # 'bic','LL'

    name        = 'heatmap_granular_reduced'
    # xtl         = ['hnor','hnac\n(gn)','hnac\n(gn-gv)','hnac\n(gn-goi)','hnac\n(gn-gv-goi)','hhybrid']
    xtl         = ['hnor','hnor\n(no trace)','hnac\n(gn)','hnac (gn,\n no trace)','hnac\n(gn-goi)','hhybrid','hhybrid2','hhybrid2\n(no trace)']
    path_load   = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelSelection/BIC/'
    path_save   = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/ModelSelection/BIC/'

    for i_comb in range(len(comb_types)):
        comb_type   = comb_types[i_comb]
        plot_col    = measure_type if comb_type=='app' else ('sum_LL' if measure_type=='LL' else 'sum_bic')
        title       = f'{measure_type.upper()} ({comb_type})'
        bic_df_list = []
        for i_alg in range(len(alg_types)):
            alg_type    = alg_types[i_alg]
            bic_df      = sl.load_sim_data(path_load,file_data=f'bic_{alg_type}_{comb_type}.csv')
            bic_df_list.append(bic_df)

        bic_df_all = pd.concat(bic_df_list)
        heat = bic_df_all[['model',plot_col]].drop_duplicates()
        heat = heat[~heat.model.str.contains('l0')]
        heat1 = heat[plot_col].values.reshape((len(alg_types),-1))
        heat2 = pd.DataFrame(heat1.transpose(),columns=alg_types)

        f,ax = plt.subplots()
        h = sns.heatmap(heat2,ax=ax,yticklabels=[f'l-{i+1}' for i in range(0,len(heat2))],xticklabels=xtl) 
        h.set_xticklabels(h.get_xticklabels(), rotation=45, fontsize=10)
        f.suptitle(title)

        fig_name = f'{name}_{plot_col}_{comb_type}'
        plt.savefig(os.path.join(path_save,fig_name+'.eps'))
        plt.savefig(os.path.join(path_save,fig_name+'.svg'))

    print('done')

