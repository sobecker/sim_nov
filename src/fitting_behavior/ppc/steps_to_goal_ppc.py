import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, bootstrap
import os
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
import glob
import src.utils.saveload as sl

def filter_expl(df):
    df_sub_ll = []
    for sub in df.subID.unique():
        if sub!='D6':
            df_sub = df.loc[df.subID==sub]
            sgoal = np.where(df_sub['state']==117)[0] 
            if len(sgoal)==0:
                sgoal = [len(df_sub)-1]
            df_sub_ll.append(df_sub.loc[df_sub.it<=sgoal[0]])
    df = pd.concat(df_sub_ll)
    return df

def get_steps_to_goal(df):
    df = df[['subID','epi','it']].groupby(['subID','epi']).count().reset_index()
    return df

def plot_steps_to_goal_all(plot_type, path_data_save, path_fig_save, no_rew, paths_load, legendstr, savename, cols, plot_dp=False,recomp=True,cut_outlier=False,logscale=True,plot_medians=True,figsize=(5,5),axl=[],lw=2):
    # Get steps to goal for all data sets
    dps         = []
    means       = []
    se_means    = []
    if plot_medians:
        medians     = []
        se_medians  = []
    for i in range(len(paths_load)):
        pp = paths_load[i]
        ll = legendstr[i]
        s  = savename[i]

        # means       = []
        # se_means    = []
        # if plot_medians:
        #     medians     = []
        #     se_medians  = []

        if recomp:
            # Load data
            if 'Rosenberg' in pp:
                if 'mice_full' in s:
                    df = sl.load_sim_data(pp,file_data='df_stateseq_AllMiceFull.pickle')
                elif 'mice_expl' in ll:
                    df = sl.load_sim_data(pp,file_data='df_stateseq.pickle')
                if no_rew: 
                    df = df.loc[df.subRew==0].reset_index(drop=True)
                df = filter_expl(df)
                df = get_steps_to_goal(df)
            else:
                df_ll = []
                for sim_id in range(5):
                    g = glob.glob(os.path.join(pp,f'*_sim{sim_id}'))
                    gg = g[0]
                    df = sl.load_sim_data(gg,file_data=f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle')
                    df = df.rename(columns={'foundGoal':'goal_state'})
                    df['it'] = df['it']-1
                    df = filter_expl(df)
                    df = get_steps_to_goal(df)
                    df['simID'] = [sim_id]*len(df)
                    df_ll.append(df)
                df = pd.concat(df_ll)
            df.to_csv(os.path.join(path_data_save,f'steps_to_goal_{s}.csv'))
            df.to_pickle(os.path.join(path_data_save,f'steps_to_goal_{s}.pkl'))
            print(f"done computing steps to goal for model {i}")
        else:
            df = sl.load_sim_data(path_data_save,file_data=f'steps_to_goal_{s}.pkl')

        # Append datapoints, means and sems for each dataset
        dps_size = np.min([20,len(df['it'].values)])
        dps.append(df['it'].values[np.random.choice(len(df['it'].values),dps_size,replace=False)])
        means.append(np.mean(df['it'].values))
        se_means.append(sem(df['it'].values))
        if plot_medians:
            medians.append(np.median(df['it'].values))
            se_median = bootstrap((df['it'].values,),np.median,n_resamples=200)
            se_medians.append(se_median.standard_error)

    # Plot mean steps to goal
    if len(axl)==0:
        f,ax = plt.subplots(1,1,figsize=figsize)
    else:
        f = None
        ax = axl[0]
    jitter = 0.2
    ax.bar(np.arange(len(means)),np.array(means),
           yerr=np.array(se_means),
           color=cols,
           ecolor='grey',
           capsize=4,
           error_kw={'elinewidth':lw,'capthick':lw})
    xlim = ax.get_xlim()
    ax.plot(xlim,[means[0]]*2,'k--')
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()
    if plot_dp:
        [ax.scatter(i*np.ones(len(dps[i]))+np.random.uniform(0,jitter,len(dps[i]))-jitter/2,dps[i],c='darkgrey',s=2*lw) for i in range(len(dps))]
    if cut_outlier:
        ax.set_ylim(ylim)
    if len(axl)==0:
        ax.set_xticks(np.arange(len(means)))
        ax.set_xticklabels(legendstr,rotation=90)
        ax.set_xlabel('')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('Algorithms')
    if len(axl)==0:
        ax.set_ylabel('Mean steps to goal')
    else:
        ax.set_title('Mean steps to goal')
    if logscale: ax.set_yscale('log')
    # vis.make_shifted_yaxis_logscale(ax,0)

    if len(axl)==0: 
        f.tight_layout()
        plt.savefig(os.path.join(path_fig_save,f'mean-steps_to_goal_{plot_type}.svg'),bbox_inches='tight')
        plt.savefig(os.path.join(path_fig_save,f'mean-steps_to_goal_{plot_type}.eps'),bbox_inches='tight')

    # Plot median steps to goal
    if plot_medians:
        if len(axl)==0:
            f,ax = plt.subplots(1,1,figsize=figsize)
        else:
            f = None
            ax = axl[1]
        jitter = 0.2
        ax.bar(np.arange(len(medians)),np.array(medians),
            yerr=np.array(se_medians),
            color=cols,
            ecolor='grey',
            capsize=4,
            error_kw={'elinewidth':lw,'capthick':lw})
        xlim = ax.get_xlim()
        ax.plot(xlim,[medians[0]]*2,'k--')
        ax.set_xlim(xlim)
        ylim = ax.get_ylim()
        if plot_dp:
            [ax.scatter(i*np.ones(len(dps[i]))+np.random.uniform(0,jitter,len(dps[i]))-jitter/2,dps[i],c='darkgrey',s=2*lw) for i in range(len(dps))]
        if cut_outlier:
            ax.set_ylim(ylim)
        ax.set_xticks(np.arange(len(medians)))
        if len(axl)==0:
            ax.set_xticklabels(legendstr,rotation=90)
            ax.set_xlabel('')
            ax.set_ylabel('Median steps to goal')
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('Algorithms')
            ax.set_title('Median steps to goal')
        if logscale: ax.set_yscale('log')
        # vis.make_shifted_yaxis(ax,0)

        if len(axl)==0:
            f.tight_layout()
            plt.savefig(os.path.join(path_fig_save,f'median-steps_to_goal_{plot_type}.svg'),bbox_inches='tight')
            plt.savefig(os.path.join(path_fig_save,f'median-steps_to_goal_{plot_type}.eps'),bbox_inches='tight')