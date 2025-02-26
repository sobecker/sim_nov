import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing as mp
import os
import sys
import glob
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')
import src.utils.saveload as sl
import src.utils.tree_env as tree
import src.utils.hnov as hn

##################################################################################################################
# Functions for computing stats                                                                                  #
##################################################################################################################
def comp_dur_per_partition(dir_data,file_data,dir_save,save_name,k_list=None,levels=None,save_data=True,rnode=None,comp_reduced=True):
    # Load data & params
    all_data = sl.load_sim_data(dir_data,file_data)
    if 'subRew' in all_data.columns:
        all_data = all_data.loc[all_data.subID!='D6'] # exclude animal that does not explore
    else: 
        all_data = all_data.rename(columns={'foundGoal':'goal_state','terminal':'goal_state','reward':'reward_received'})
        if np.isnan(all_data.action.values[0]): all_data = all_data[1:]
    
    # Compute duration per partition
    subID_list = []
    level_list = []
    kernel_list = []
    visit_list = []
    dur_list = []
    start_list = []
    ksizes_list = []
    g_list = []
    r_list = []
    for subID in np.unique(all_data.subID):     # iterate over subjects
        # print(f'subID: {subID}')
        data        = all_data[all_data.subID==subID]
        # Overwrite rewarded state if applicable
        if not rnode is None:
            data['goal_state'] = data['state']==rnode
            first_rew = np.where(data['state']==rnode)[0]
            first_rew = first_rew[0] if len(first_rew)>0 else len(data)
            data.loc[first_rew:,'reward_received'] = True
        # Compute time points of goal and reward encounters
        t_all_goal  = np.where(data.goal_state)[0]
        t_goal      = t_all_goal[0] if len(t_all_goal)>0 else len(data) # get time point at which goal was encountered first (or last index if not encountered at all)
        t_all_rew   = np.where(data.reward_received)[0]
        t_rew       = t_all_rew[0] if len(t_all_rew)>0 else len(data)   # get time point at which reward was first received (or last index if not received at all)
        # Reduce data to be processed if applicable
        if comp_reduced:
            t_stop  = np.max([t_goal,t_rew])
            data    = data.iloc[:t_stop]
        s_seq = data[['it','state']].to_numpy()
        for i in range(len(k_list)):            # iterate over levels in hierarchy
            k = k_list[i]
            for j in range(len(k[0])):                  # iterate over partition sets in a given level of the hierarchy
                # s = (k[:,j]==1).nonzero()[0]
                s       = (k[:,j]==np.max(k[:,j])).nonzero()[0]                                     # non-overlapping states in current partition
                s1      = np.array([s_seq[i,1] in s for i in range(len(s_seq))]).reshape(((-1,1)))  # time points at which current partition was visited
                s_seq   = np.append(s_seq,s1,axis=1)
                s2      = s_seq[:,-1].nonzero()[0]                                                  # time points spent in current kernel
                if len(s2)>0:
                    d     = s2[1:]-s2[:-1]                                      # time between subsequent time points spent in current kernel 
                    d     = np.append(d,np.array([len(s_seq)-s2[-1]]))          # append time difference to end of recording
                    d1    = (d>1).nonzero()[0]                                  # indices of switches between kernels (in d)
                    dur_ij    = np.zeros(len(d1),dtype=int)
                    di_old    = 0
                    for di in range(len(d1)):
                        dur_ij[di] = len((d[di_old:d1[di]]==1).nonzero()[0])+1  # duration of visits to kernel (# time points where time difference between consecutive partition visits == 1)
                        di_old=d1[di]
                    start_ij  = s2[d1]-dur_ij+1        # start time points of visits to kernels
                    goal_seen = (start_ij>=t_goal)     # bool: True if visit to kernel occured after first encounter with goal
                    rew_seen  = (start_ij>=t_rew)      # bool: True if visit to kernel occured after first reward delivery                  
                    subID_list.extend([subID]*len(dur_ij))
                    level_list.extend([i]*len(dur_ij))
                    kernel_list.extend([j]*len(dur_ij))
                    visit_list.extend(np.arange(len(dur_ij)))
                    dur_list.extend(dur_ij.copy())
                    start_list.extend(start_ij.copy())
                    ksizes_list.extend([len(s)]*len(dur_ij))
                    g_list.extend(goal_seen)
                    r_list.extend(rew_seen)
    df_dur = pd.DataFrame(dict(zip(['subID','level','kernel','visits','dur','start_time','kernel_size','goal_seen','rew_seen'],[subID_list,level_list,kernel_list,visit_list,dur_list,start_list,ksizes_list,g_list,r_list])))
    if 'subRew'in list(data.columns): df_dur = df_dur.merge(all_data[['subID','subRew']].drop_duplicates().reset_index(drop=True),how='left',on='subID')
    
    if save_data:
        df_dur.to_pickle(os.path.join(dir_save,f'df_dur_l{levels[0]}_{save_name}{"_reduced" if comp_reduced else ""}.pickle'))
        df_dur.to_csv(os.path.join(dir_save,f'df_dur_l{levels[0]}_{save_name}{"_reduced" if comp_reduced else ""}.csv'))
    
    return df_dur

##################################################################################################################
def comp_dur_per_partition_all(paths_load,legendstr,savename,dir_save,levels=[2]):
    # Make kernel structure for kernel-related stats
    rtree = tree.make_rtree_env(6,seed=None,rnode=117)
    x, P, R, T = tree.tree_df2list(rtree)
    w,h = hn.make_hierarchy_eps1(rtree,levels)
    k_list = h['kmat']

    # Compute duration per partition
    for i in range(len(paths_load)):
        pp = paths_load[i]
        s  = savename[i]

        if 'Rosenberg' in pp:   agg_col = ['subID'] 
        else:                   agg_col = ['simID','subID']
    
        if 'Rosenberg' in pp:
            if 'mice_full' in s: 
                dir_data    = pp
                file_data   = 'df_stateseq_AllMiceFull.pickle'
            elif 'mice_expl' in s: 
                dir_data    = pp
                file_data   = 'df_stateseq.pickle'
        else:
            g = glob.glob(os.path.join(pp,'*_sim0'))
            dir_data    = g[0]
            file_data   = f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle'
        df_dur = comp_dur_per_partition(dir_data,file_data,dir_save,s,k_list=k_list,levels=levels)
        # compute stats
        # df_dur_stats = None
        # df_dur_stats.to_pickle(os.path.join(dir_save,f'df_dur_stats_l{levels[0]}_{s}.pickle'))
        # df_dur_stats.to_csv(os.path.join(dir_save,f'df_dur_stats_l{levels[0]}_{s}.csv'))

##################################################################################################################
def compute_stats(df,agg_col=['subID'],stat='mean'): 
    # Preprocessing 
    df = df.loc[df.subID!='D6']
    df = df[agg_col + ['level','kernel','visits','dur','start_time','kernel_size','goal_seen','rew_seen']].drop_duplicates()
    df = df[df.kernel>0]                                # exclude home cage (since deterministic duration of visit=1)
    df['dur_norm_s'] = df['dur']/df['kernel_size']      # get duration normalized by size of partition
    num_visits = df[agg_col+['kernel','visits']].groupby(agg_col+['kernel']).count().rename(columns={'visits':'count_visits'})
    df = df.merge(num_visits,how='left',on=agg_col+['kernel'])
    df['dur_norm_v'] = df['dur']/df['count_visits']     # get duration normalized by number of visits to partition

    # Compute stats
    if stat=='mean':
        df_stats = df[agg_col+['dur','dur_norm_s','dur_norm_v']].groupby(agg_col).agg(mean_dur=('dur',np.mean),mean_dur_norm_s=('dur_norm_s',np.mean),mean_dur_norm_v=('dur_norm_v',np.mean))
        df_counts = df[agg_col+['count_visits']].drop_duplicates().groupby(agg_col).agg(mean_count_visits=('count_visits',np.mean))
        df_stats = df_stats.merge(df_counts,on=agg_col,how='left')
    elif stat=='median':
        df_stats = df[agg_col+['dur','dur_norm_s','dur_norm_v']].groupby(agg_col).agg(median_dur=('dur',np.median),median_dur_norm_s=('dur_norm_s',np.median),median_dur_norm_v=('dur_norm_v',np.median))
        df_counts = df[agg_col+['count_visits']].drop_duplicates().groupby(agg_col).agg(median_count_visits=('count_visits',np.median))
        df_stats = df_stats.merge(df_counts,on=agg_col,how='left')
    else:
        df_stats = None
    return df_stats

##################################################################################################################
def comp_dur_per_partition_i(i,path_data_save,pp,s,levels,k_list,num_sim,stat_type='mean',recomp=False,recomp_stats=False,rnode=None,comp_reduced=True):
    # if os.path.exists(os.path.join(path_data_save,f'df_dur_l{levels[0]}_{s}.pkl')) and os.path.exists(os.path.join(path_data_save,f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.pkl')) and os.path.exists(os.path.join(path_data_save,f'peri_center_l{levels[0]}_{s}.pkl')) and not recomp:
    if os.path.exists(os.path.join(path_data_save,f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.pkl')) and os.path.exists(os.path.join(path_data_save,f'peri_center_l{levels[0]}_{s}.pkl')) and not recomp:
        print(f'Already computed dur per partition for {s}.')
    else:
        print(f'Computing dur per partition for {s}...')
        if 'Rosenberg' in pp:
            if 'mice_full' in s: 
                dir_data    = pp
                file_data   = 'df_stateseq_AllMiceFull.pickle'
            elif 'mice_expl' in s: 
                dir_data    = pp
                file_data   = 'df_stateseq.pickle' # NO NO_REW FILTER?
            df_dur = comp_dur_per_partition(dir_data,file_data,path_data_save,s,k_list=k_list,levels=levels,save_data=True,rnode=rnode,comp_reduced=comp_reduced)
        else:
            df_ll = []
            for sim_id in num_sim:
                g = glob.glob(os.path.join(pp,f'*_sim{sim_id}'))
                dir_data  = g[0]
                file_data = f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle'
                df_dur = comp_dur_per_partition(dir_data,file_data,path_data_save,s,k_list=k_list,levels=levels,save_data=True,rnode=rnode,comp_reduced=comp_reduced)
                df_dur['simID'] = [sim_id]*len(df_dur)
                df_ll.append(df_dur)
            df_dur = pd.concat(df_ll)
        # df_dur.to_pickle(os.path.join(path_data_save,f'df_dur_l{levels[0]}_{s}.pickle'))
        # df_dur.to_csv(os.path.join(path_data_save,f'df_dur_l{levels[0]}_{s}.csv'))
        print(f'Finished computing dur per partition for {s}.\n')

    if recomp_stats:
        if not recomp:
            # Load stats files
            file_data = f'df_dur_l{levels[0]}_{s}{"_reduced" if comp_reduced else ""}.pickle'
            df_dur = sl.load_sim_data(dir_data=path_data_save,file_data=file_data)

        merge_cols = ['subID'] #if 'Rosenberg' in pp else ['simID','subID']

        # Compute stats of duration per partition
        print(f'Computing T/partition for {s}...')
        df_dur          = df_dur.loc[df_dur.goal_seen==False]
        df_steps        = sl.load_sim_data(os.path.join(path_data_save.removesuffix(f'dur_per_partition_l{levels[0]}'),'steps_to_goal'),file_data=f'steps_to_goal_{s}.pkl')
        # df_steps[(df_steps.subID.isin(df_dur.subID.unique()))]
        # df_steps[(df_steps.subID.isin(df_dur.subID.unique())) & (df_steps.simID.isin(df_dur.simID.unique()))]  # df_steps = df_steps.iloc[:len(df_steps.subID.unique())]
        df = df_dur.merge(df_steps,how='left',on=merge_cols)
        df['goal_seen'] = df.start_time>=df.it
        df              = df[df.goal_seen==False]
        df_stats = compute_stats(df,agg_col=merge_cols,stat=stat_type)
        df_stats.to_csv(os.path.join(path_data_save,f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.csv'))
        df_stats.to_pickle(os.path.join(path_data_save,f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.pkl'))
        print(f'Finished computing T/partition for {s}.\n')

        # Compute total time spent in periphery vs. home vs. center
        print(f'Computing C vs. P for {s}...')
        df['exceed']    = (np.heaviside(df['start_time']+df['dur']-df['it'],0)*(df['start_time']+df['dur']-df['it'])).astype(int)
        df['dur_corr']  = df['dur']-df['exceed']

        t_home = df.loc[df.kernel==0,merge_cols+['dur_corr']].groupby(merge_cols).sum().rename(columns={'dur_corr':'t_home'})
        t_peri = df.loc[df.kernel!=0,merge_cols+['dur_corr']].groupby(merge_cols).sum().rename(columns={'dur_corr':'t_peri'})
        df_steps = df_steps.merge(t_home,how='left',on=merge_cols)
        df_steps = df_steps.merge(t_peri,how='left',on=merge_cols)
        df_steps['t_center'] = df_steps['it']-df_steps['t_home']-df_steps['t_peri']
        df_steps['r_home']   = df_steps['t_home']/df_steps['it']
        df_steps['r_peri']   = df_steps['t_peri']/df_steps['it']
        df_steps['r_center']   = df_steps['t_center']/df_steps['it']
        df_steps['r_center_peri'] = df_steps['t_center']/df_steps['t_peri']
        df_steps.to_csv(os.path.join(path_data_save,f'peri_center_l{levels[0]}_{s}.csv'))
        df_steps.to_pickle(os.path.join(path_data_save,f'peri_center_l{levels[0]}_{s}.pkl'))
        print(f'Finished computing C vs. P for {s}.\n')
        print(f'Return{i}\n')
    return i

##################################################################################################################
def comp_dur_per_partition_parallel(path_data_save,paths_load,name_save,num_sim_ll,levels=[2],stat_type='mean',parallel=False,recomp=False,recomp_stats=False,comp_reduced=True):
    # Make kernel structure for kernel-related stats
    rnode = 117
    rtree = tree.make_rtree_env(6,seed=None,rnode=rnode)
    x, P, R, T = tree.tree_df2list(rtree)
    w,h = hn.make_hierarchy(rtree,levels,eps1=True)
    k_list = h['kmat']

    # Compute duration per partition
    if parallel:
        num_pool = mp.cpu_count()
        pool = mp.Pool(num_pool)
        res = [pool.apply_async(comp_dur_per_partition_i,args=(i,path_data_save,pp,s,levels,k_list,num_sim,stat_type,recomp,recomp_stats,rnode,comp_reduced)) for i,(pp,s,num_sim) in enumerate(zip(paths_load,name_save,num_sim_ll))]
        list_done = [r.get() for r in res]
        pool.close()
        pool.join()
    else:
        list_done = []
        for i in range(len(paths_load)):
            pp = paths_load[i]
            s  = name_save[i]
            num_sim = num_sim_ll[i]
            comp_dur_per_partition_i(i,path_data_save,pp,s,levels,k_list,num_sim,stat_type=stat_type,recomp=recomp,recomp_stats=recomp_stats,rnode=rnode,comp_reduced=comp_reduced)
            list_done.append(i)
    return list_done

##################################################################################################################
# Functions for plotting stats                                                                                   #
##################################################################################################################
def plot_meandur_l_all(plot_type, path_data_save, path_fig_save, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, recomp=False, recomp_stats=True, levels=[2], logscale=False, cut_outlier=False, plot_medians=True, stat_type='mean'):
    if recomp: comp_dur_per_partition_all(paths_load,legendstr,savename,path_data_save,levels=levels)

    # Get df_dur dataframes 
    dps1        = []
    means1      = []
    medians1    = []
    se_means1   = []
    se_medians1 = []
    dps2        = []
    means2      = []
    medians2    = []
    se_means2   = []
    se_medians2 = []
    dps3        = []
    means3      = []
    medians3    = []
    se_means3   = []
    se_medians3 = []
    dps4        = []
    means4      = []
    medians4    = []
    se_means4   = []
    se_medians4 = []
    # Iterate over algorithm types
    for i in range(len(paths_load)):
        s  = savename[i]

        # Compute / load mouse-wise stats 
        if recomp_stats:
            df_dur = sl.load_sim_data(path_data_save,file_data=f'df_dur_l{levels[0]}_{s}.pickle')
            df_dur = df_dur.loc[df_dur.goal_seen==False]
            df_steps        = sl.load_sim_data(path_data_save,file_data=f'steps_to_goal_{s}.pkl')
            df_steps        = df_steps.iloc[:len(df_steps.subID.unique())]
            df              = df_dur.merge(df_steps,how='left',on='subID')
            df['goal_seen'] = df.start_time>=df.it
            df              = df[df.goal_seen==False]
            df_stats = compute_stats(df,stat=stat_type)
            df_stats.to_csv(os.path.join(path_data_save,f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.csv'))
            df_stats.to_pickle(os.path.join(path_data_save,f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.pkl'))
        else:
            df_stats = sl.load_sim_data(path_data_save,file_data=f'df_dur_l{levels[0]}_stats-{stat_type}_{s}.pkl')

        # Compute agent-wise stats
        dps_size = np.min([20,len(df_stats)])
        dps      = df_stats.iloc[np.random.choice(len(df_stats),dps_size,replace=False)]
        dps1.append(dps['mean_dur'].values)
        dps2.append(dps['mean_dur_norm_s'].values)
        dps3.append(dps['mean_dur_norm_v'].values)
        dps4.append(dps['mean_count_visits'].values)

        means = np.mean(df_stats,axis=0).values
        means1.append(means[0])
        means2.append(means[1])
        means3.append(means[2])
        means4.append(means[3])

        se_means = stats.sem(df_stats,axis=0)
        se_means1.append(se_means[0])
        se_means2.append(se_means[1])
        se_means3.append(se_means[2])
        se_means4.append(se_means[3])

        if plot_medians:
            medians = np.median(df_stats,axis=0)
            medians1.append(medians[0])
            medians2.append(medians[1])
            medians3.append(medians[2])
            medians4.append(medians[3])

            se_medians = []
            for col in df_stats.columns:
                se_medians.append(stats.bootstrap((df_stats[col].values,),np.median,n_resamples=200).standard_error)
            se_medians1.append(se_medians[0])
            se_medians2.append(se_medians[1])
            se_medians3.append(se_medians[2])
            se_medians4.append(se_medians[3])

    # Plot means
    f1,ax1 = plt.subplots(1,3,figsize=(14,4))
    jitter = 0.2
    xt = np.arange(len(means1))
    ylim0 = []
    ylim1 = []

    ax1[0].bar(np.arange(len(means1)),means1,yerr=se_means1,ecolor='grey',capsize=4,color=cols)
    xlim = ax1[0].get_xlim()
    if cut_outlier: ylim0.append(ax1[0].get_ylim())
    ax1[0].plot(xlim,[means1[0]]*2,'k--')
    if plot_dp: [ax1[0].scatter(i*np.ones(len(dps1[i]))+np.random.uniform(0,jitter,len(dps1[i]))-jitter/2,dps1[i],c='grey',s=2) for i in range(len(dps1))]
    ax1[0].set_xlim(xlim)
    if not cut_outlier: ylim0.append(ax1[0].get_ylim())
    ax1[0].set_xticks(xt)
    ax1[0].set_xticklabels(legendstr)
    ax1[0].set_ylabel('Mean time per partition')
    # if logscale: ax1[0].set_yscale('log')

    ax1[1].bar(np.arange(len(means2)),means2,yerr=se_means2,ecolor='grey',capsize=4,color=cols)
    xlim = ax1[1].get_xlim()
    if cut_outlier: ylim1.append(ax1[1].get_ylim())
    ax1[1].plot(xlim,[means2[0]]*2,'k--')
    if plot_dp: [ax1[1].scatter(i*np.ones(len(dps2[i]))+np.random.uniform(0,jitter,len(dps2[i]))-jitter/2,dps2[i],c='grey',s=2) for i in range(len(dps2))]
    ax1[1].set_xlim(xlim)
    if not cut_outlier: ylim1.append(ax1[1].get_ylim())
    ax1[1].set_xticks(xt)
    ax1[1].set_xticklabels(legendstr)
    ax1[1].set_ylabel('Mean time per partition\n(norm. by kernel size)')
    # if logscale: ax1[1].set_yscale('log')

    # ax1[2].bar(np.arange(len(means3)),means3,yerr=se_means3,ecolor='grey',capsize=4,color=cols)
    # xlim = ax1[2].get_xlim()
    # if cut_outlier: ylim1.append(ax1[2].get_ylim())
    # ax1[2].plot(xlim,[means3[0]]*2,'k--')
    # if plot_dp: [ax1[2].scatter(i*np.ones(len(dps3[i]))+np.random.uniform(0,jitter,len(dps3[i]))-jitter/2,dps3[i],c='grey',s=2) for i in range(len(dps3))]
    # ax1[2].set_xlim(xlim)
    # if not cut_outlier: ylim1.append(ax1[2].get_ylim())
    # ax1[2].set_xticks(xt)
    # ax1[2].set_xticklabels(legendstr)
    # ax1[2].set_ylabel('Mean time per partition\n(norm. by number of visits)')
    # if logscale: ax1[2].set_yscale('log')

    h_legend = []
    for i in range(len(means1)):
        h = ax1[2].plot([means1[i]-se_means1[i],means1[i]+se_means1[i]],[means4[i],means4[i]],c=cols[i])
        ax1[2].plot([means1[i],means1[i]],[means4[i]-se_means4[i],means4[i]+se_means4[i]],c=cols[i])
        h_legend.append(h[0])
        if plot_dp: 
            ax1[2].scatter(dps1[i],dps4[i],c=cols[i],alpha=0.4,s=2)
    legendstr1 = [i.replace('\n','') for i in legendstr]
    ax1[2].legend(h_legend,legendstr1)
    ax1[2].set_xlabel('Mean time per partition')
    ax1[2].set_ylabel('Mean number of visits')
    if logscale: 
        ax1[2].set_xscale('log')
        ax1[2].set_yscale('log')

    # Plot medians
    if plot_medians:
        f2,ax2 = plt.subplots(1,3,figsize=(14,4))
        jitter = 0.2
        xt = np.arange(len(medians1))

        ax2[0].bar(np.arange(len(medians1)),medians1,yerr=se_medians1,ecolor='grey',capsize=4,color=cols)
        xlim = ax2[0].get_xlim()
        if cut_outlier: ylim0.append(ax2[0].get_ylim())
        ax2[0].plot(xlim,[medians1[0]]*2,'k--')
        if plot_dp: [ax2[0].scatter(i*np.ones(len(dps1[i]))+np.random.uniform(0,jitter,len(dps1[i]))-jitter/2,dps1[i],c='grey',s=2) for i in range(len(dps1))]
        ax2[0].set_xlim(xlim)
        if not cut_outlier: ylim0.append(ax2[0].get_ylim())
        ax2[0].set_xticks(xt)
        ax2[0].set_xticklabels(legendstr)
        ax2[0].set_ylabel('Median time per partition')
        # if logscale: ax2[0].set_yscale('log')

        ax2[1].bar(np.arange(len(medians2)),medians2,yerr=se_medians2,ecolor='grey',capsize=4,color=cols)
        xlim = ax2[1].get_xlim()
        if cut_outlier: ylim1.append(ax2[1].get_ylim())
        ax2[1].plot(xlim,[medians2[0]]*2,'k--')
        if plot_dp: [ax2[1].scatter(i*np.ones(len(dps2[i]))+np.random.uniform(0,jitter,len(dps2[i]))-jitter/2,dps2[i],c='grey',s=2) for i in range(len(dps2))]
        ax2[1].set_xlim(xlim)
        if not cut_outlier: ylim1.append(ax2[1].get_ylim())
        ax2[1].set_xticks(xt)
        ax2[1].set_xticklabels(legendstr)
        ax2[1].set_ylabel('Median time per partition\n(norm. by kernel size)')
        # if logscale: ax2[1].set_yscale('log')

        # ax2[2].bar(np.arange(len(medians3)),medians3,yerr=se_medians3,ecolor='grey',capsize=4,color=cols)
        # xlim = ax2[2].get_xlim()
        # if cut_outlier: ylim1.append(ax2[2].get_ylim())
        # ax2[2].plot(xlim,[medians3[0]]*2,'k--')
        # if plot_dp: [ax2[2].scatter(i*np.ones(len(dps3[i]))+np.random.uniform(0,jitter,len(dps3[i]))-jitter/2,dps3[i],c='grey',s=2) for i in range(len(dps3))]
        # ax2[2].set_xlim(xlim)
        # if not cut_outlier: ylim1.append(ax2[2].get_ylim())
        # ax2[2].set_xticks(xt)
        # ax2[2].set_xticklabels(legendstr)
        # ax2[2].set_ylabel('Median time per partition\n(norm. by number of visits)')
        # if logscale: ax2[2].set_yscale('log')

        h_legend = []
        for i in range(len(medians1)):
            h = ax2[2].plot([medians1[i]-se_medians1[i],medians1[i]+se_medians1[i]],[medians4[i],medians4[i]],c=cols[i])
            ax2[2].plot([medians1[i],medians1[i]],[medians4[i]-se_medians4[i],medians4[i]+se_medians4[i]],c=cols[i])
            h_legend.append(h[0])
            if plot_dp: 
                ax2[2].scatter(dps1[i],dps4[i],c=cols[i],alpha=0.4,s=2)
        ax2[2].legend(h_legend,legendstr1)
        ax2[2].set_xlabel('Median time per partition')
        ax2[2].set_ylabel('Median number of visits')
        if logscale: 
            ax2[2].set_xscale('log')
            ax2[2].set_yscale('log')

    # Set ylim for both mean/median figure
    # ylim_uni0 = [0,np.max(np.array([ylim0[i][-1] for i in range(len(ylim0))]))]
    # ylim_uni1 = [0,np.max(np.array([ylim1[i][-1] for i in range(len(ylim0))]))]
    ylim_uni0 = [0,250]
    ylim_uni1 = [0,10]
    # yt_uni0   = [0,1,10,100]
    # yt_uni1   = [0,1,100]

    # Format + save mean figure
    ax1[0].set_ylim(ylim_uni0) #; ax1[0].set_yticks(yt_uni0)
    ax1[1].set_ylim(ylim_uni1) #; ax1[1].set_yticks(yt_uni1)
    if logscale:
        ax1[0].set_yscale('log')
        ax1[1].set_yscale('log')
    # ax1[2].set_ylim(ylim_uni1); ax1[2].set_yticks(yt_uni1)
    f1.tight_layout()
    f1.savefig(os.path.join(path_fig_save,f'meandur-l_l{levels[0]}_stats-{stat_type}_{plot_type}.svg'),bbox_inches='tight')
    f1.savefig(os.path.join(path_fig_save,f'meandur-l_l{levels[0]}_stats-{stat_type}_{plot_type}.eps'),bbox_inches='tight')

    if plot_medians:
        ax2[0].set_ylim(ylim_uni0) #; ax2[0].set_yticks(yt_uni0)
        ax2[1].set_ylim(ylim_uni1) #; ax2[1].set_yticks(yt_uni1)
        if logscale:
            ax2[0].set_yscale('log')
            ax2[1].set_yscale('log')
        # ax2[2].set_ylim(ylim_uni1); ax2[2].set_yticks(yt_uni1)
        f2.tight_layout()
        f2.savefig(os.path.join(path_fig_save,f'mediandur-l_l{levels[0]}_stats-{stat_type}_{plot_type}.svg'),bbox_inches='tight')
        f2.savefig(os.path.join(path_fig_save,f'mediandur-l_l{levels[0]}_stats-{stat_type}_{plot_type}.eps'),bbox_inches='tight')