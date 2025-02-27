import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, bootstrap
import utils.saveload as sl
import utils.visualization as vis

def get_bootstrap_matrix(data,stat_type,R=100):
    subs     = data.subID.unique()
    if 'D6' in list(subs): subs = np.delete(subs,np.where(subs=='D6')[0])
    stat_fun = np.nanmean if stat_type=='mean' else np.nanmedian
    bi       = np.random.choice(len(subs),size=(len(subs),R)) # choose random indices (with replacement)
    data1    = data.pivot(index='num_fam',columns='subID',values='num_nov') 
    ml = []
    for r in range(R):
        m = stat_fun(data1[subs[bi[:,r]]].values,axis=1) # compute mean/median curve for bootstrapping set
        ml.append(m)
    data_mat = np.stack(ml) # make matrix with bootstrapped median/mean curves [dim: R x len(curve)]
    return data_mat

def get_stat_and_error(df_list,stat_type,R=100):
    stat_fun = np.mean if stat_type=='mean' else np.median
    d0m      = df_list[0].groupby(['num_fam']).agg([stat_fun]) # get average/median curve for first (=reference) data set: mice
    d0m      = d0m['num_nov',stat_type].values

    d0mat     = get_bootstrap_matrix(df_list[0],stat_type=stat_type,R=R) # get bootstrapped average/median curve for first data set (random subsampling of mice IDs)
    diff_list = []
    se_list   = []
    m_list    = []
    for i in range(1,len(df_list)):
        dim   = df_list[i].groupby(['num_fam']).agg([stat_fun]) # get average/median curve for simulated model i
        diff  = np.sum(np.abs(dim['num_nov',stat_type].values-d0m))
        dimat   = get_bootstrap_matrix(df_list[i],stat_type=stat_type) # get bootstrapped average/median curve for model i
        diffmat = np.sum(np.abs(d0mat-dimat),axis=1) # compute difference between two bootstrapped curves
        diff_mean = np.nanmean(diffmat)
        diff_se = np.nanstd(diffmat)/np.sqrt(R)
        m_list.append(diff)
        se_list.append(diff_se)
        diff_list.append(diffmat)
    return m_list, se_list, diff_list

def comp_node_ratio(data_exp):
    id = np.unique(data_exp.subID)
    for i in id:
        st = data_exp.loc[data_exp.subID==i,'state'].values
        found = []
        c_nov = [0]
        c_fam = [0]
        for t in range(len(st)):
            c_fam.append(c_fam[-1])
            c_nov.append(c_nov[-1])
            if st[t]>63:
                c_fam[-1] += 1
                if not st[t] in found:
                    c_nov[-1] += 1
                    found.append(st[t])
        data_exp.loc[data_exp.subID==i,'num_nov'] = c_nov[1:]
        data_exp.loc[data_exp.subID==i,'num_fam'] = c_fam[1:]
    return data_exp

def plot_node_ratio(ax,data_exp,window_size=1,b=None,plot_dp=False,c='k',lw=1.5,plot_sem=False):
    if plot_dp:
        id = np.unique(data_exp.subID)
        [ax.plot(data_exp.loc[data_exp.subID==i,'num_fam'].values,data_exp.loc[data_exp.subID==i,'num_nov'].rolling(window_size).mean().values,c=c,alpha=0.4) for i in id] # plot individual animals

    d = data_exp[['num_fam','num_nov']].groupby(['num_fam']).agg([np.mean,np.std,sem]).reset_index()
    a = ax.plot(d['num_fam'],d['num_nov','mean'].rolling(window_size).mean().values,c=c,lw=lw) # plot average
    if plot_sem:
        ax.fill_between(d['num_fam'], d['num_nov','mean']-d['num_nov','sem'], d['num_nov','mean']+d['num_nov','sem'], alpha=0.2, edgecolor=c,facecolor=c)

    if b:
        nb = np.where(d['num_nov','mean']>=b)[0][0]
        ax.plot([nb]*2,[0,b],':',c=c,lw=lw)  # N_32
        # print(f'Exploration efficiency $E$ = 32/$N_{{2}}$ is {b/nb} (on average).')
    
    return a[0]

def plot_node_ratio_median(ax,data_exp,window_size=1,b=None,plot_dp=False,c='k',lw=1.5):
    if plot_dp:
        id = np.unique(data_exp.subID)
        [ax.plot(data_exp.loc[data_exp.subID==i,'num_fam'].values,data_exp.loc[data_exp.subID==i,'num_nov'].rolling(window_size).mean().values,c=c,alpha=0.4) for i in id] # plot individual animals

    d = data_exp[['num_fam','num_nov']].groupby(['num_fam']).agg([np.median]).reset_index()
    a = ax.plot(d['num_fam'],d['num_nov','median'].rolling(window_size).mean().values,c=c,lw=lw) # plot average
    # compute sem!
    ax.fill_between(d['num_fam'], d['num_nov','median']-d['num_nov','sem'], d['num_nov','mean']+d['num_nov','sem'], alpha=0.2, edgecolor=c,facecolor=c)

    if b:
        nb = np.where(d['num_nov','mean']>=b)[0][0]
        ax.plot([nb]*2,[0,b],':',c=c,lw=lw)  # N_32
        # print(f'Exploration efficiency $E$ = 32/$N_{{2}}$ is {b/nb} (on average).')
    return a[0]

def plot_node_ratio_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, window_size=1,plot_dp=False,recomp=True,figsize=(5,5),axl=[],lw=1.5,plot_b=False,plot_sem=False):
    if len(axl)==0:
        f,ax = plt.subplots(1,1,figsize=figsize)
    else:
        f = None
        ax = axl[0]
    legendh = []
    
    for i in range(len(paths_load)):
        pp = paths_load[i]
        ll = legendstr[i]
        s  = savename[i]
        cc = cols[i]

        if recomp:
            # Load data
            if 'Rosenberg' in pp:
                if 'mice_full' in s:
                    df = sl.load_sim_data(pp,file_data='df_stateseq_AllMiceFull.pickle')
                elif 'expl' in ll:
                    df = sl.load_sim_data(pp,file_data='df_stateseq.pickle')
                if no_rew: 
                    df = df.loc[df.subRew==0].reset_index(drop=True)
            else:
                g = glob.glob(os.path.join(pp,'*_sim0'))
                gg = g[0]
                df = sl.load_sim_data(gg,file_data=f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle')
            df = df.loc[df.it<maxit].reset_index(drop=True)

            # Compute and save node ratio
            df = comp_node_ratio(df)
            df.to_csv(os.path.join(path_data_save,f'node_ratio_{s}.csv'))
            df.to_pickle(os.path.join(path_data_save,f'node_ratio_{s}.pkl'))
            print(f"done computing node ratio for model {i}")
        else:
            df = sl.load_sim_data(path_data_save,file_data=f'node_ratio_{s}.pkl')

        # Plot node ratio
        b = 32 if plot_b else None
        plot_sem_i = True if 'Rosenberg' in pp else plot_sem
        a = plot_node_ratio(ax,df,b=b,c=cc,plot_dp=plot_dp,window_size=window_size,lw=lw,plot_sem=plot_sem_i)
        legendh.append(a)
        
    # Plot half-node boundary + optimal agent
    ax.plot([0,ax.get_xlim()[-1]],[b,b],'k:',lw=lw) # half the leaf nodes (32) visited
    # opt = np.arange(np.round(ax.get_ylim()[-1]))
    # a = ax.plot(opt,opt,'k')
    # legendh.append(a)
    # legendstr.append('optimal')
    ax.set_xlim([0,500])
    ax.set_ylim([0,65])

    # Format plot
    #ax.set_xscale('log')
    ax.set_xlabel('End node visits')
    if len(axl)==0:
        ax.set_ylabel('New end nodes found')
    else:
        ax.set_ylabel('New end nodes found')
    ncol,_ = divmod(len(legendstr),4)
    ax.spines[['top','right']].set_visible(False)
    if len(axl)==0:
        ax.legend(legendh,legendstr,ncol=max(1,ncol),handlelength=1)
        f.tight_layout()
        plt.savefig(os.path.join(path_fig_save,f'node_ratio_{plot_type}.svg'),bbox_inches='tight')
        plt.savefig(os.path.join(path_fig_save,f'node_ratio_{plot_type}.eps'),bbox_inches='tight')

        # plt.savefig(os.path.join(path_fig_save,f'node_ratio_{"_norew" if no_rew else ""}.svg'))
        # plt.savefig(os.path.join(path_fig_save,f'node_ratio_{"_norew" if no_rew else ""}.eps'))
    
def plot_N32_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, plot_dp=False, recomp_ratios=False, recomp=True, figsize=(5,5)):
    f,ax = plt.subplots(1,1,figsize=figsize)
    legendh = []

    cols = vis.prep_cmap_discrete('Paired')
    c1   = cols[1]
    c2   = cols[0]
    
    for i in range(len(paths_load)):
        pp = paths_load[i]
        ll = legendstr[i]
        s  = savename[i]

        if recomp_ratios:
            # Load data
            if 'Rosenberg' in pp:
                if 'unrew. mice' in ll:
                    df = sl.load_sim_data(pp,file_data='df_stateseq_AllMiceFull.pickle')
                elif 'expl' in ll:
                    df = sl.load_sim_data(pp,file_data='df_stateseq.pickle')
                if no_rew: 
                    df = df.loc[df.subRew==0].reset_index(drop=True)
            else:
                g = glob.glob(os.path.join(pp,'*_sim0'))
                gg = g[0]
                df = sl.load_sim_data(gg,file_data=f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle')
            df = df.loc[df.it<maxit].reset_index(drop=True)

            # Compute and save node ratio
            df = comp_node_ratio(df)
            df.to_csv(os.path.join(path_data_save,f'node_ratio_{s}.csv'))
            df.to_pickle(os.path.join(path_data_save,f'node_ratio_{s}.pkl'))
        else:
            df = sl.load_sim_data(path_data_save,file_data=f'node_ratio_{s}.pkl')

        if recomp:
            # Compute and save N32
            b  = 32
            nb = df.loc[df['num_nov']>=b,['subID','num_fam']].groupby('subID').first()
            nb = nb.rename(columns={'num_fam':'N32'})
            nb['N32_mean']       = np.mean(nb['N32'].values)
            nb['N32_median']     = np.median(nb['N32'].values)
            nb['N32_mean_err']   = sem(nb['N32'].values)
            nb['N32_median_err'] = bootstrap((nb['N32'].values,),np.median,n_resamples=200).standard_error
            nb.to_csv(os.path.join(path_data_save,f'N32_{s}.csv'))
            nb.to_pickle(os.path.join(path_data_save,f'N32_{s}.pkl'))
        else:
            nb = sl.load_sim_data(path_data_save,file_data=f'N32_{s}.pkl')
        
        # Plot N32
        eps=0.2
        a1 = ax.bar([i-eps],nb['N32_mean'].unique(),width=0.4,align='center',color=c1) 
        a2 = ax.bar([i+eps],nb['N32_median'].unique(),width=0.4,align='center',color=c2) 
        ax.errorbar(x=[i-eps],y=nb['N32_mean'].unique(),yerr=nb['N32_mean_err'].unique(),ecolor='k',elinewidth=1,linestyle='',capsize=2)
        ax.errorbar(x=[i+eps],y=nb['N32_median'].unique(),yerr=nb['N32_median_err'].unique(),ecolor='k',elinewidth=1,linestyle='',capsize=2)
        if plot_dp:
            ax.scatter([i-eps]*len(nb['N32']),nb['N32'],s=1,c='grey')
            ax.scatter([i+eps]*len(nb['N32']),nb['N32'],s=1,c='grey')
        if i==0: legendh.extend([a1,a2])
        if 'Rosenberg' in pp:
            mice_mean = nb['N32_mean'].unique()
            mice_median = nb['N32_median'].unique()
    
    # Plot optimal agent
    b=32
    xlim = ax.get_xlim()
    a3 = ax.plot(xlim,[b]*2,'k:',lw=1.5) 
    ax.set_xlim(xlim)
    legendh.append(a3[0])

    # Plot mice mean/median
    a4 = ax.plot(xlim,[mice_mean]*2,'--',c=c1,lw=1.5) 
    a5 = ax.plot(xlim,[mice_median]*2,'--',c=c2,lw=1.5) 
    ax.set_xlim(xlim)
    legendh.extend([a4[0],a5[0]])

    # Format plot
    ax.set_ylabel('Efficiency coefficient $N_{32}$')
    ax.set_xticks(np.arange(len(paths_load)))
    ax.set_xticklabels(legendstr,rotation=90)
    ax.legend(legendh,['Mean','Median','Optimal','Mice (mean)','Mice (median)'])
    vis.make_shifted_yaxis(ax,0)
    f.tight_layout()

    plt.savefig(os.path.join(path_fig_save,f'N32_{plot_type}.svg'),bbox_inches='tight')
    plt.savefig(os.path.join(path_fig_save,f'N32_{plot_type}.eps'),bbox_inches='tight')

def plot_N32_all_singlestat(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, plot_dp=False, recomp_ratios=False, recomp=True, figsize=(5,5), axl=[], lw=2, stat_type='median'):
    if len(axl)==0:
        f,ax = plt.subplots(1,1,figsize=figsize)
    else:
        f = None
        ax = axl[0]
    legendh = []

    for i in range(len(paths_load)):
        pp = paths_load[i]
        ll = legendstr[i]
        s  = savename[i]
        cc = cols[i]

        if recomp_ratios:
            # Load data
            if 'Rosenberg' in pp:
                if 'unrew. mice' in ll:
                    df = sl.load_sim_data(pp,file_data='df_stateseq_AllMiceFull.pickle')
                elif 'expl' in ll:
                    df = sl.load_sim_data(pp,file_data='df_stateseq.pickle')
                if no_rew: 
                    df = df.loc[df.subRew==0].reset_index(drop=True)
            else:
                g = glob.glob(os.path.join(pp,'*_sim0'))
                gg = g[0]
                df = sl.load_sim_data(gg,file_data=f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle')
            df = df.loc[df.it<maxit].reset_index(drop=True)

            # Compute and save node ratio
            df = comp_node_ratio(df)
            df.to_csv(os.path.join(path_data_save,f'node_ratio_{s}.csv'))
            df.to_pickle(os.path.join(path_data_save,f'node_ratio_{s}.pkl'))
        else:
            df = sl.load_sim_data(path_data_save,file_data=f'node_ratio_{s}.pkl')

        if recomp:
            # Compute and save N32
            b  = 32
            nb = df.loc[df['num_nov']>=b,['subID','num_fam']].groupby('subID').first()
            nb = nb.rename(columns={'num_fam':'N32'})
            nb['N32_mean']       = np.mean(nb['N32'].values)
            nb['N32_median']     = np.median(nb['N32'].values)
            nb['N32_mean_err']   = sem(nb['N32'].values)
            nb['N32_median_err'] = bootstrap((nb['N32'].values,),np.median,n_resamples=200).standard_error
            nb.to_csv(os.path.join(path_data_save,f'N32_{s}.csv'))
            nb.to_pickle(os.path.join(path_data_save,f'N32_{s}.pkl'))
        else:
            nb = sl.load_sim_data(path_data_save,file_data=f'N32_{s}.pkl')
        
        # Plot N32
        jitter=0.2
        if stat_type=='mean':
            a1 = ax.bar([i],nb['N32_mean'].unique(),color=cc,yerr=nb['N32_mean_err'].unique(),ecolor='grey',capsize=4,error_kw={'elinewidth':lw,'capthick':lw}) 
            if plot_dp:
                dps = nb['N32']
                [ax.scatter(i*np.ones(len(dps[i]))+np.random.uniform(0,jitter,len(dps[i]))-jitter/2,dps[i],c='darkgrey',s=2*lw) for i in range(len(dps))]
            # if i==0: legendh.extend([a1])
            if 'Rosenberg' in pp:
                mice_stat = nb['N32_mean'].unique()
        else:
            a1 = ax.bar([i],nb['N32_median'].unique(),color=cc,yerr=nb['N32_median_err'].unique(),ecolor='grey',capsize=4,error_kw={'elinewidth':lw,'capthick':lw}) 
            if plot_dp:
                dps = nb['N32'].values
                ax.scatter(i*np.ones(len(dps))+np.random.uniform(0,jitter,len(dps))-jitter/2,dps,c='darkgrey',s=2*lw)
            # if i==0: legendh.extend([a1])
            if 'Rosenberg' in pp:
                mice_stat = nb['N32_median'].unique()

    # Plot optimal agent
    b=32
    xlim = ax.get_xlim()
    # a3 = ax.plot(xlim,[b]*2,'k:',lw=lw) 
    # ax.set_xlim(xlim)
    # legendh.append(a3[0])

    # Plot mice mean/median
    a4 = ax.plot(xlim,[mice_stat]*2,'--',c=cols[0],lw=lw) 
    legendh.append(a4[0])
    ax.set_xlim(xlim)
    ax.set_ylim([40,320])

    # Format plot
    if len(axl)==0:
        ax.set_ylabel('Efficiency coefficient $N_{32}$')
        ax.set_xticks(np.arange(len(paths_load)))
        ax.set_xticklabels(legendstr,rotation=90)
        # ax.legend(legendh,['Optimal','Mice'])
    else:
        ax.set_ylabel('Efficiency coefficient $N_{32}$')
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('Algorithms')
    vis.make_shifted_yaxis(ax,0) #ax.get_ylim()[0])
    ax.spines[['top','right']].set_visible(False)
    if len(axl)==0:
        f.tight_layout()
        plt.savefig(os.path.join(path_fig_save,f'N32_{plot_type}.svg'),bbox_inches='tight')
        plt.savefig(os.path.join(path_fig_save,f'N32_{plot_type}.eps'),bbox_inches='tight')

def plot_integral_diff_all(plot_type, path_data_save, path_fig_save, maxit, no_rew, paths_load, legendstr, savename, cols, recomp_ratios=False, recomp=True, figsize=(5,5), axl=[], lw=2, plot_medians=True):
    if recomp:
        df_list = []
        for i in range(len(paths_load)):
            pp = paths_load[i]
            ll = legendstr[i]
            s  = savename[i]
            cc = cols[i]

            if recomp_ratios:
                # Load data
                if 'Rosenberg' in pp:
                    if 'unrew. mice' in ll:
                        df = sl.load_sim_data(pp,file_data='df_stateseq_AllMiceFull.pickle')
                    elif 'expl' in ll:
                        df = sl.load_sim_data(pp,file_data='df_stateseq.pickle')
                    if no_rew: 
                        df = df.loc[df.subRew==0].reset_index(drop=True)
                else:
                    g = glob.glob(os.path.join(pp,'*_sim0'))
                    gg = g[0]
                    df = sl.load_sim_data(gg,file_data=f'{"mf_" if "hybrid" in pp else ""}data_basic.pickle')
                df = df.loc[df.it<maxit].reset_index(drop=True)
                # Compute and save node ratio
                df = comp_node_ratio(df)
                df.to_csv(os.path.join(path_data_save,f'node_ratio_{s}.csv'))
                df.to_pickle(os.path.join(path_data_save,f'node_ratio_{s}.pkl'))
            else:
                df = sl.load_sim_data(path_data_save,file_data=f'node_ratio_{s}.pkl')
            # Drop duplicates + restrict range
            df = df[['subID','num_fam','num_nov']].drop_duplicates()
            df = df.loc[df.num_fam<=500]
            df_list.append(df)
        # Compute stats      
        means, se_means, _ = get_stat_and_error(df_list,'mean')
        medians, se_medians, _ = get_stat_and_error(df_list,'median')
        for i in range(len(means)):
            df_stats = pd.DataFrame(dict(zip(['mean','se_mean','median','se_median'],[means[i],se_means[i],medians[i],se_medians[i]])),index=[0])
            df_stats.to_csv(os.path.join(path_data_save,f'int-diff_{savename[i]}.csv'))
            df_stats.to_pickle(os.path.join(path_data_save,f'int-diff_{savename[i]}.pkl'))
    else:
        means=[]; se_means=[]; medians=[]; se_medians=[]
        for i in range(len(savename)):
            s  = savename[i]
            df_stats = sl.load_sim_data(path_data_save,file_data=f'int-diff_{s}.pkl')
            means.append(df_stats['mean'])
            se_means.append(df_stats['se_mean'])
            medians.append(df_stats['median'])
            se_medians.append(df_stats['se_median'])

    # Plot means
    if len(axl)==0:
        f,ax = plt.subplots(1,1,figsize=figsize)
    else:
        f = None
        ax = axl[0]
    ax.bar(np.arange(len(means)),means,color=cols[1:],yerr=se_means,ecolor='grey',capsize=4,error_kw={'elinewidth':lw,'capthick':lw})  
    if len(axl)==0:
        ax.set_xticks(np.arange(len(legendstr)-1))
        ax.set_xticklabels(legendstr[1:],rotation=90)
        ax.set_ylabel(f'Difference to mice')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Difference to mice')
    vis.make_shifted_yaxis(ax,0)
    ax.spines[['top','right']].set_visible(False)
    if len(axl)==0:
        f.tight_layout()
        plt.savefig(os.path.join(path_fig_save,f'mean-int-diff_{plot_type}.svg'),bbox_inches='tight')
        plt.savefig(os.path.join(path_fig_save,f'mean-int-diff_{plot_type}.eps'),bbox_inches='tight')

    # Plot medians
    if plot_medians:
        if len(axl)==0:
            f,ax = plt.subplots(1,1,figsize=figsize)
        else:
            f = None
            ax = axl[1]
        ax.bar(np.arange(len(medians)),medians,color=cols[1:],yerr=se_medians,ecolor='grey',capsize=4,error_kw={'elinewidth':2,'capthick':2})
        ax.set_xticks(np.arange(len(legendstr)-1)) 
        if len(axl)==0:
            ax.set_xticklabels(legendstr[1:],rotation=90)
            ax.set_ylabel(f'Difference to median mice efficiency')
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('Algorithms')
            ax.set_title('Difference to median mice efficiency')
        vis.make_shifted_yaxis(ax,0)
        if len(axl)==0:
            f.tight_layout()
            plt.savefig(os.path.join(path_fig_save,f'median-int-diff_{plot_type}.svg'),bbox_inches='tight')
            plt.savefig(os.path.join(path_fig_save,f'median-int-diff_{plot_type}.eps'),bbox_inches='tight')