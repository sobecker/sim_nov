import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl

def plot_true_vs_mle(path_true,path_fit,path_save):
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

    for v in vars:
        fig,ax = plt.subplots(2,1,figsize=(6,6))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax[0].scatter(p_true[v],p_fit_app[v])
        a,b = np.polyfit(p_true[v],p_fit_app[v],1)
        x = np.linspace(min(p_true[v]),max(p_true[v]),100)
        ax[0].plot(x,a*x+b,'k--')
        str = '\n'.join((f'a={np.round(a,4)}',f'b={np.round(b,4)}'))
        ax[0].text(0.05, 0.95, str, transform=ax[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


        ax[1].scatter(p_true[v],p_fit_sep[v])
        a,b = np.polyfit(p_true[v],p_fit_sep[v],1)
        x = np.linspace(min(p_true[v]),max(p_true[v]),100)
        ax[1].plot(x,a*x+b,'k--')
        str = '\n'.join((f'a={np.round(a,4)}',f'b={np.round(b,4)}'))
        ax[1].text(0.05, 0.95, str, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


        ax[1].set_xlabel('True parameter value')
        ax[0].set_ylabel('MLE parameter estimate (app)')
        ax[1].set_ylabel('MLE parameter estimate (mean)')
        
        fig.suptitle(v)
        plt.savefig(path_save+f'{v}_true-vs-mle.svg')
        plt.savefig(path_save+f'{v}_true-vs-mle.eps')


if __name__=="__main__":
    # Cluster (nor)
    # path_true = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData/nor/'
    # path_fit = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/nor_Nelder-Mead/'
    # path_save = '/Volumes/lcncluster/becker/RL_reward_novelty/output/ParameterRecovery/nor_Nelder-Mead/'

    # Cluster (nor, all params)
    path_true = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData/nor_allparams/'
    path_fit = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/nor_allparams_Nelder-Mead/'
    path_save = '/Volumes/lcncluster/becker/RL_reward_novelty/output/ParameterRecovery/nor_allparams_Nelder-Mead/'

    # Cluster (nac)
    # path_true = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData/nac/'
    # path_fit = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/FitData/nac_Nelder-Mead/'
    # path_save = '/Volumes/lcncluster/becker/RL_reward_novelty/output/ParameterRecovery/nac_Nelder-Mead/'

    # Local (nor)
    # path_true   = '/Users/sbecker/Projects/RL_reward_novelty/data/ParameterRecovery/SimData/nor/'
    # path_fit    = '/Users/sbecker/Projects/RL_reward_novelty/data/ParameterRecovery/FitData/nor_Nelder-Mead/'
    # path_save   = '/Users/sbecker/Projects/RL_reward_novelty/output/ParameterRecovery/nor_Nelder-Mead/'

    plot_true_vs_mle(path_true,path_fit,path_save)
