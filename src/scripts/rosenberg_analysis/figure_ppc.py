import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import utils.saveload as sl
import utils.visualization as vis
from fitting_behavior.ppc.steps_to_goal_ppc import plot_steps_to_goal_all
from fitting_behavior.ppc.node_ratio_ppc import plot_node_ratio_all, plot_N32_all, plot_N32_all_singlestat, plot_integral_diff_all
from fitting_behavior.ppc.dur_per_partition_ppc import plot_dur_per_partition_all

if __name__=="__main__":
    # Load style file
    plt.style.use('/Users/sbecker/Projects/RL_reward_novelty/src/scripts/Figures_Paper/paper.mplstyle')

    # Folders for saving data and figures
    path_data_save = '/Users/sbecker/Projects/RL_reward_novelty/data/PPC'
    path_fig_save  = '/Users/sbecker/Projects/RL_reward_novelty/output/Figures_Paper/Fig_ppc_v2'
    # path_data_save  = os.path.join(sl.get_datapath(),'PPC')
    # path_fig_save   = os.path.join(sl.get_datapath().replace('data','output'),'Figures_Paper/Fig_ppc_v2')
    sl.make_long_dir(path_data_save)
    sl.make_long_dir(path_fig_save)

    # Folders for loading data
    path_exp    = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Rosenberg2021' # get it from cluster?
    path_sim    = '/Volumes/lcncluster/becker/RL_reward_novelty/data/ParameterRecovery/SimData_uniparam'

    # Which plots to make
    plot_hyb2 = True
    plot_nor  = False

    plot_other = False
    k_alg_name = 'hnac-gn'
    c_alg_name = 'nac'
    c_alg_leg = 'nac'
    k_alg_leg = 'hnac-gn (nt)'

    # Plot first row (hybrid model)
    if plot_hyb2:
        plot_medians    = False
        lw              = 1
        plot_type   = 'comp_hyb2'   # 'mice_only','comp_hyb2','comp_nor','comp_algs'
        no_rew      = False         # if true: include only unrewarded mice
        maxit       = 3000          # number of steps included (only relevant for node discovery plots)
        figsize     = (2.05,2.05)
        
        # paths_load  = [os.path.join(path_sim,'hybrid2_app_norew')] 
        paths_load  = [path_exp,os.path.join(path_sim,'hybrid2_app_norew')]
        # legendstr   = ['c-Hyb']
        legendstr   = ['Mice','c-Hyb']
        # savename    = ['hyb2']
        savename    = ['mice_expl' if no_rew else 'mice_full','hyb2']
        # cols        = ['b']
        cols        = ['k']
        # cols.extend([vis.prep_cmap(name='bwr',num=9)[-1]])
        cols.extend([vis.prep_cmap(name='Blues',num=len(np.arange(1,7)))[1]])
        [paths_load.append(os.path.join(path_sim,f'hhybrid2_app_norew/hhybrid2_app_l{ll}')) for ll in range(1,7)]
        [legendstr.append(f's{ll}-Hyb') for ll in range(1,7)] 
        [savename.append(f'hhyb2-l{ll}') for ll in range(1,7)]
        # cols.extend(vis.prep_cmap(name='Blues',num=len(np.arange(1,7))))
        cols.extend(vis.prep_cmap(name='Reds',num=len(np.arange(1,7))))
        # cols.extend(vis.prep_cmap(name='bwr',num=1+2*len(np.arange(1,7)))[:len(np.arange(1,7))])

        f_mean,ax_mean = plt.subplots(1,3,figsize=(8.1,2.2))
        ax1 = [ax_mean[0]]
        ax2 = [ax_mean[1]]
        ax3 = [ax_mean[2]]
        # ax4 = [ax_mean[3]]
        if plot_medians:
            f_med,ax_med = plt.subplots(1,3,figsize=(8.1,2.2))
            ax1.append(ax_med[0])
            ax2.append(ax_med[1])
            ax3.append(ax_med[2])
            # ax4.append(ax_med[3])

        # plot_steps_to_goal_all(plot_type, path_data_save, path_fig_save, no_rew, paths_load, legendstr, savename, cols, plot_dp=True, recomp=False, cut_outlier=False, axl=ax4, lw=lw, plot_medians=plot_medians)
        plot_node_ratio_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, window_size=5,recomp=False, axl=ax1, lw=lw)
        plot_N32_all_singlestat(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=True, recomp=False, axl=ax2, lw=lw)
        plot_integral_diff_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, recomp_ratios=False, recomp=True, plot_medians=plot_medians, axl=ax3, lw=lw)
        # plot_dur_per_partition_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, recomp=False, recomp_stats=True, dpp_level=5, plot_medians=plot_medians, axl=ax4, lw=lw)

        f_mean.tight_layout()
        f_mean.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_mean.svg'))
        f_mean.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_mean.eps'))

        if plot_medians:
            f_med.tight_layout()
            f_med.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_median.svg'))
            f_med.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_median.eps'))

    if plot_other:
        plot_medians    = False
        recomp          = False
        lw              = 1
        plot_type   = f'comp_{k_alg_name}'   # 'mice_only','comp_hyb2','comp_nor','comp_algs'
        no_rew      = False         # if true: include only unrewarded mice
        maxit       = 3000          # number of steps included (only relevant for node discovery plots)
        figsize     = (2.05,2.05)
        
        paths_load  = [path_exp,os.path.join(path_sim,f'{c_alg_name}_app_norew')]
        legendstr   = ['Mice',c_alg_leg]
        savename    = ['mice_full',c_alg_name]
        cols        = ['k','r']
        [paths_load.append(os.path.join(path_sim,f'{k_alg_name}_app_norew/{k_alg_name}_app_l{ll}')) for ll in range(1,7)]
        [legendstr.append(f'k{ll}-{k_alg_leg}') for ll in range(1,7)] 
        [savename.append(f'{k_alg_name}-l{ll}') for ll in range(1,7)]
        cols.extend(vis.prep_cmap(name='Blues',num=len(np.arange(1,7))))

        f_mean,ax_mean = plt.subplots(1,4,figsize=(8.2,2.2))
        ax1 = [ax_mean[0]]
        ax2 = [ax_mean[1]]
        ax3 = [ax_mean[2]]
        ax4 = [ax_mean[3]]
        if plot_medians:
            f_med,ax_med = plt.subplots(1,4,figsize=(8.2,2.2))
            ax1.append(ax_med[0])
            ax2.append(ax_med[1])
            ax3.append(ax_med[2])
            ax4.append(ax_med[3])

        plot_steps_to_goal_all(plot_type, path_data_save, path_fig_save, no_rew, paths_load, legendstr, savename, cols, plot_dp=True, recomp=recomp, cut_outlier=False, axl=ax4, lw=lw, plot_medians=plot_medians)
        print("done plotting steps to goal")
        plot_node_ratio_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, window_size=5,recomp=recomp, axl=ax1, lw=lw)
        print("done plotting node ratio")
        plot_N32_all_singlestat(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=True, recomp_ratios=False, recomp=recomp, axl=ax2, lw=lw)
        print("done plotting N32")
        plot_integral_diff_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, recomp_ratios=False, recomp=recomp, plot_medians=plot_medians, axl=ax3, lw=lw)
        print("done plotting integral diff")

        f_mean.tight_layout()
        f_mean.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_mean.svg'),bbox_inches='tight')
        f_mean.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_mean.eps'),bbox_inches='tight')

        if plot_medians:
            f_med.tight_layout()
            f_med.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_median.svg'),bbox_inches='tight')
            f_med.savefig(os.path.join(path_fig_save,f'ppc_{plot_type}_median.eps'),bbox_inches='tight')

    # Plot second row (MB model)
    if plot_nor:
        plot_type   = 'comp_nor'    # 'mice_only','comp_hyb2','comp_nor','comp_algs'
        no_rew      = False         # if true: include only unrewarded mice
        maxit       = 3000          # number of steps included (only relevant for node discovery plots)
        figsize     = (2.05,2.05)
        
        paths_load  = [path_exp,os.path.join(path_sim,'nor_app_norew')]
        legendstr   = ['Mice','c-MB']
        savename    = ['mice_full','nor']
        cols        = ['k','r']
        alg_range = np.arange(1,7)
        [paths_load.append(os.path.join(path_sim,f'hnor_app_norew/hnor_app_l{ll}')) for ll in alg_range]
        [legendstr.append(f'k{ll}-MB') for ll in alg_range]
        [savename.append(f'hnor-l{ll}') for ll in alg_range]
        cols.extend(vis.prep_cmap(name='Blues',num=len(alg_range)))

        plot_steps_to_goal_all(plot_type, path_data_save, path_fig_save, no_rew, paths_load, legendstr, savename, cols, plot_dp=True, recomp=False, cut_outlier=False, figsize=figsize)
        plot_node_ratio_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, window_size=5,recomp=False, figsize=figsize)
        plot_N32_all_singlestat(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=True, recomp=False, figsize=figsize)
        plot_integral_diff_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, recomp_ratios=False, recomp=True, figsize=figsize, plot_medians=True)
    
    print('done')
