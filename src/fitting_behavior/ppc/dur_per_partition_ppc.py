import matplotlib.pyplot as plt
from scipy.stats import sem
import os
import sys
import utils.saveload as sl
import fitting_behavior.ppc.dur_per_partition as dpp

def plot_dur_per_partition_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, recomp=False, recomp_stats=False, dpp_level=6, plot_medians=False, axl=None, lw=1):

    if recomp or recomp_stats:
        # savename[0] = f'mice_{"expl" if no_rew else "full"}'
        for i in [dpp_level]:
            num_sim = [list(range(1)) if 'mice' in sn else list(range(20)) for sn in savename]
            sl.make_long_dir(os.path.join(path_data_save,f'dur_per_partition_l{i}'))
            dpp.comp_dur_per_partition_parallel(path_data_save=os.path.join(path_data_save,f'dur_per_partition_l{i}'), 
                                                paths_load=paths_load, 
                                                name_save=savename, 
                                                num_sim_ll=num_sim,  #?
                                                levels=[i], 
                                                parallel=False, 
                                                recomp=recomp,
                                                recomp_stats=recomp_stats)
            if plot_medians:
                dpp.comp_dur_per_partition_parallel(path_data_save=os.path.join(path_data_save,f'dur_per_partition_l{i}'), 
                                                paths_load=paths_load, 
                                                name_save=savename, 
                                                num_sim_ll=num_sim,  #?
                                                levels=[i], 
                                                parallel=False, 
                                                stat_type='median',
                                                recomp=recomp,
                                                recomp_stats=recomp_stats)

    path_load = os.path.join(path_data_save,f'dur_per_partition_l{dpp_level}')
    # path_load = path_data_save

    lh = []; lstr = []
    ax = axl[0]
    x_all = []
    y_all = []
    for i, m in enumerate(savename): # iterate over models
        # names = [f'{models[i]}-l{ll}' for ll in levels[i][-1::-1]]
        # for j, n in enumerate(names):
        # Load data
        data = sl.load_sim_data(path_load,file_data=f'df_dur_l{dpp_level}_stats-mean_{m}.pkl')
        if 'mice' in m:
            path_exp_load = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Rosenberg2021'
            file_exp_load = 'df_stateseq_AllMiceFull.pickle'
            mice_data = sl.load_sim_data(path_exp_load,file_data=file_exp_load)
            mice_data = mice_data[['subID','subRew']].drop_duplicates()
            data = data.merge(mice_data,on='subID')
        data_mean_sub = data.groupby('subID').mean().reset_index()
        d_mean        = data_mean_sub['mean_dur'].mean()
        d_sem         = sem(data_mean_sub['mean_dur'])
        v_mean        = data_mean_sub['mean_count_visits'].mean()
        v_sem         = sem(data_mean_sub['mean_count_visits'])
        ax.scatter(data_mean_sub['mean_dur'],data_mean_sub['mean_count_visits'],marker='o',s=5,color=cols[i],alpha=0.1)
        a = ax.scatter(d_mean,v_mean,marker='+',s=15,color=cols[i],alpha=1)
        x_all.append(d_mean)
        y_all.append(v_mean)
        lh.append(a)
        lstr.append(legendstr[i])
        if 'mice' in m:
            ax.axhline(y=v_mean,color=cols[i],linestyle='--',alpha=1)
            ax.axvline(x=d_mean,color=cols[i],linestyle='--',alpha=1)
    
    ax.set_xlabel('Mean duration per partition')
    # ax.set_xlim([40,210])
    ax.set_ylabel('Mean number of visits per partition')
    # ax.set_ylim([380,1750])
    ax.legend(lh,lstr,fontsize=9,loc='upper right')

    if plot_medians:
        lh = []; lstr = []
        ax = axl[1]
        x_all = []
        y_all = []
        for i, m in enumerate(savename): # iterate over models
            # names = [f'{models[i]}-l{ll}' for ll in levels[i][-1::-1]]
            # for j, n in enumerate(names):
            # Load data
            data = sl.load_sim_data(path_load,file_data=f'df_dur_l{dpp_level}_stats-median_{m}.pkl')
            if 'mice' in m:
                path_exp_load = '/Users/sbecker/Projects/RL_reward_novelty/ext_data/Rosenberg2021'
                file_exp_load = 'df_stateseq_AllMiceFull.pickle'
                mice_data = sl.load_sim_data(path_exp_load,file_data=file_exp_load)
                mice_data = mice_data[['subID','subRew']].drop_duplicates()
                data = data.merge(mice_data,on='subID')
            data_mean_sub = data.groupby('subID').mean().reset_index()
            d_mean        = data_mean_sub['mean_dur'].mean()
            d_sem         = sem(data_mean_sub['mean_dur'])
            v_mean        = data_mean_sub['mean_count_visits'].mean()
            v_sem         = sem(data_mean_sub['mean_count_visits'])
            ax.scatter(data_mean_sub['mean_dur'],data_mean_sub['mean_count_visits'],marker='o',s=5,color=cols[i],alpha=0.1)
            a = ax.scatter(d_mean,v_mean,marker='+',s=15,color=cols[i],alpha=1)
            x_all.append(d_mean)
            y_all.append(v_mean)
            lh.append(a)
            lstr.append(legendstr[i])
            if 'mice' in m:
                ax.axhline(y=v_mean,color=cols[i],linestyle='--',alpha=1)
                ax.axvline(x=d_mean,color=cols[i],linestyle='--',alpha=1)
        
        ax.set_xlabel('Mean duration per partition')
        # ax.set_xlim([40,210])
        ax.set_ylabel('Mean number of visits per partition')
        # ax.set_ylim([380,1750])
        ax.legend(lh,lstr,fontsize=9,loc='upper right')
