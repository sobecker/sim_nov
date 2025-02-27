import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import sys
import pandas as pd
import utils.saveload as sl

def plot_corrs(path_true,path_fit,path_save):
    sl.make_long_dir(path_save)

    p_true  = sl.load_sim_data(path_true,file_data='params_overview.pickle')
    p_fit   = sl.load_sim_data(path_fit,file_data='fitparam_overview.pickle')

    id_shared = list(set(p_true.simID.unique()) & set(p_fit.simID.unique()))
    p_true    = p_true.loc[p_true.simID.isin(id_shared)]
    p_fit     = p_fit.loc[p_fit.simID.isin(id_shared)]
    p_fit_app = p_fit.loc[p_fit['mle_type']=='app']
    p_fit_sep = p_fit.loc[p_fit['mle_type']=='mean_sep']

    vars = list(p_true.columns)
    vars.pop()

    p_all = p_true[vars]
    p_all = p_all.rename(columns={p_all.columns[i]:f'true_{p_all.columns[i]}' for i in range(len(p_all.columns))})
    for i in range(len(vars)):
        p_all[f'app_{vars[i]}'] = p_fit_app[vars[i]].values
    for i in range(len(vars)):
        p_all[f'sep_{vars[i]}'] = p_fit_sep[vars[i]].values

    corr_all = p_all.corr()

    true_vars = [f'true_{vars[i]}' for i in range(len(vars))]
    app_vars = [f'app_{vars[i]}' for i in range(len(vars))]
    sep_vars = [f'sep_{vars[i]}' for i in range(len(vars))]

    corr_true_true = corr_all.iloc[:len(vars),:len(vars)]
    corr_app_app = corr_all.iloc[len(vars):2*len(vars),len(vars):2*len(vars)]
    corr_sep_sep = corr_all.iloc[2*len(vars):,2*len(vars):]
    corr_true_app = corr_all.iloc[:len(vars),len(vars):2*len(vars)]
    corr_true_sep = corr_all.iloc[:len(vars),2*len(vars):]

    f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4,figsize=(20,6),gridspec_kw={'width_ratios':[1,1,1,0.08]})
    ax1.get_shared_y_axes().join(ax2,ax3)
    cmap = sn.diverging_palette(20, 220, n=200) #'bwr'
    g1 = sn.heatmap(corr_true_true,cmap=cmap,vmin=-1,vmax=1,center=0,square=True,cbar=False,ax=ax1)
    g2 = sn.heatmap(corr_app_app,cmap=cmap,vmin=-1,vmax=1,center=0,square=True,cbar=False,ax=ax2)
    g3 = sn.heatmap(corr_sep_sep,cmap=cmap,vmin=-1,vmax=1,center=0,square=True,cbar=True,ax=ax3,cbar_ax=axcb)
    ax1.set_title('True vs. true parameters')
    ax2.set_title('MLE (app) vs. MLE (app)')
    ax3.set_title('Mean MLE (sep) vs. mean MLE (sep)')
    f.suptitle('Correlations within distribution')
    f.tight_layout()

    plt.savefig(path_save+'/corr_params_within-dist.eps')
    plt.savefig(path_save+'/corr_params_within-dist.svg')

    f,(ax1,ax2, axcb) = plt.subplots(1,3,figsize=(14,6),gridspec_kw={'width_ratios':[1,1,0.08]})
    ax1.get_shared_y_axes().join(ax1,ax2)
    cmap = sn.diverging_palette(20, 220, n=200) #'bwr'
    g1 = sn.heatmap(corr_true_app,cmap=cmap,vmin=-1,vmax=1,center=0,square=True,cbar=False,ax=ax1)
    g2 = sn.heatmap(corr_true_sep,cmap=cmap,vmin=-1,vmax=1,center=0,square=True,cbar=True,ax=ax2,cbar_ax=axcb)
    ax1.set_title('True vs. MLE (app) parameters')
    ax1.set_xlabel('MLE (app)')
    ax2.set_title('True vs. MLE (sep) parameters')
    ax2.set_xlabel('MLE (sep)')
    f.suptitle('Correlations across distributions')
    f.tight_layout()

    plt.savefig(path_save+'/corr_params_across-dist.eps')
    plt.savefig(path_save+'/corr_params_across-dist.svg')



if __name__=="__main__":
    # Cluster (nor)
    # path_true = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData//FullRandomnor/'
    # path_fit = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/FullRandom/nor_Nelder-Mead/'
    # path_save = '/Volumes/lcncluster/becker/RL_reward_novelty/output/ParameterRecovery/FullRandom/nor_Nelder-Mead/'

    # Cluster (nor, all params)
    path_true = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData/FullRandom/nor_allparams/'
    path_fit = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/FullRandom/nor_allparams_Nelder-Mead/'
    path_save = '/Volumes/lcncluster/becker/RL_reward_novelty/output/ParameterRecovery/FullRandom/nor_allparams_Nelder-Mead/'

    # Cluster (nac)
    # path_true = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData/FullRandom/nac/'
    # path_fit = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/FullRandom/nac_Nelder-Mead/'
    # path_save = '/Volumes/lcncluster/becker/RL_reward_novelty/output/ParameterRecovery/FullRandom/nac_Nelder-Mead/'

    # Local (nor)
    # path_true   = '/Users/sbecker/Projects/RL_reward_novelty/data/ParameterRecovery/SimData/FullRandom/nor/'
    # path_fit    = '/Users/sbecker/Projects/RL_reward_novelty/data/ParameterRecovery/FitData/FullRandom/nor_Nelder-Mead/'
    # path_save   = '/Users/sbecker/Projects/RL_reward_novelty/output/ParameterRecovery/FullRandom/nor_Nelder-Mead/'

    plot_corrs(path_true,path_fit,path_save)
