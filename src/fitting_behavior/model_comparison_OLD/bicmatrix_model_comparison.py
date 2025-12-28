import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils.saveload as sl

def plot_bic_matrix(comb_type,measure_type,alg_types,alg_labels,figshape,path_load,title,save_plot,path_save,save_name):
    plot_col    = measure_type if comb_type=='app' else ('sum_LL' if measure_type=='LL' else 'sum_bic')
    bic_df_list = []
    for i_alg in range(len(alg_types)):
        alg_type    = alg_types[i_alg]
        bic_df      = sl.load_sim_data(path_load,file_data=f'bic_{alg_type}_{comb_type}.csv')
        bic_df_list.append(bic_df)

    bic_df_all = pd.concat(bic_df_list)
    heat = bic_df_all[['model',plot_col]].drop_duplicates()
    heat1 = heat[plot_col].values.reshape((len(alg_types),-1))
    heat2 = pd.DataFrame(heat1.transpose(),columns=alg_types)

    f,ax = plt.subplots(figsize=figshape)
    ct = [int(np.floor(heat2.values.flatten().min()/100)*100),int(np.ceil(heat2.values.flatten().max()/100)*100)]
    ct1 = [int(i) for i in np.linspace(ct[0],ct[1],4)]    
    #cmap = vis.prep_cmap_log('magma',vmin,vmax)
    h = sns.heatmap(heat2.transpose(),ax=ax,
                    xticklabels=[f'{i+1}' for i in range(0,len(heat2))],
                    yticklabels=alg_labels,
                    # norm=LogNorm(),
                    vmin=ct[0],vmax=ct[1],
                    cbar_kws={'label': 'BIC','ticks':ct1}) 
    h.set_yticklabels(h.get_yticklabels(), rotation=0, fontsize=10)
    #ax.set_ylabel('Algorithm')
    ax.set_ylabel('')
    ax.set_xlabel('Gran. level')
    # ax.set_title(title)
    f.tight_layout()
    # xfmt = ScalarFormatter()
    # xfmt.set_powerlimits((-3,3))
    # ax.xaxis.set_major_formatter(xfmt)
    #ax.ticklabel_format(useMathText=True)
    #ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')

    if save_plot:
        fig_name = f'{save_name}_{measure_type}_{comb_type}'
        sl.make_long_dir(path_save)
        plt.savefig(path_save / fig_name+'.eps')
        plt.savefig(path_save / fig_name+'.svg')

if __name__=="__main__":
    # Specify list of candidate models
    # alg_types    = ['hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi','hhybrid']
    alg_types    = ['hnor','hnac-gn','hnac-gn-goi','hhybrid2'] 
    alg_labels   = ['MB','MF','MF\n(adapt.)','Hybrid']
    data_type    = 'mice' # 'mice','opt','naive'
    opt_method   = 'Nelder-Mead' # 'Nelder-Mead','L-BFGS-B','SLSQP'
    comb_types   = ['sep','app']
    measure_type = 'bic' # 'bic','LL'
    title        = 'Model selection across algorithms and levels'
    save_plot    = True
    save_name    = 'bic-matrix2'
    path_load    = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelSelection/BIC/'
    path_save    = sl.get_datapath().replace('data','output') / f'Figures_Paper/Fig_model_comparison/'
    figshape     = (6,6)


    for i_comb in range(len(comb_types)):
        comb_type   = comb_types[i_comb]
        plot_bic_matrix(comb_type,measure_type,alg_types,alg_labels,figshape,path_load,title,save_plot,path_save,save_name)


