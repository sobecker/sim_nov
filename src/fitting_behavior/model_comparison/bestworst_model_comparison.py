import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import utils.saveload as sl
from fitting_behavior.model_comparison.compute_bic import compute_bic

my_ceil = lambda num, prec: (10**(-prec))*np.ceil(num*10**(prec))
my_floor = lambda num, prec: (10**(-prec))*np.floor(num*10**(prec))

def make_shifted_yaxis(ax,shift):
    dt      = int(np.round(ax.get_yticks()[1]-ax.get_yticks()[0])) # get distance between two yticks (rounded to integers)
    rprec   = -(len(str(dt))-1)                                    # get rounding precision, e.g. -2 for yticks with distance ~100
    if shift<0:
        dyt     = my_floor(shift,rprec)-shift
    else:
        dyt     = my_ceil(shift,rprec)-shift                           # get distance between first ytick and first ytick for shifted yaxis
    yt      = ax.get_yticks()+dyt
    ytl     = [f'{int(ytj+shift)}' for ytj in yt]
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)

def compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col,best_fun,worst_fun):
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

    algs = algs_basic+algs_gran
    bic_df_all.sort_values(by="model_type", key=lambda column: column.map(lambda e: algs.index(e)), inplace=True)

    print(f'The best model (best-case scenario) is: {bic_df_all.loc[bic_df_all.best_ofbest,"model_type"]}.')
    print(f'The best model (worst-case scenario) is: {bic_df_all.loc[bic_df_all.best_ofworst,"model_type"]}.')

    return bic_df_all, best_ofbest, best_ofworst

def plot_best_worst(algs_basic,algs_gran,alg_labels,opt_method,comb_type,path_load,path_save,save_plot=False,save_name='',figshape=None,my_ylim=None,bic_only=False):
    # Compute BIC and LL
    plot_col1   = 'bic' if comb_type=='app' else 'sum_bic'
    best_fun1    = np.min; worst_fun1   = np.max
    bic_df_all, bic_bob, bic_bow = compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col1,best_fun1,worst_fun1)

    plot_col2   = 'LL' if comb_type=='app' else 'sum_LL'
    best_fun2    = np.max; worst_fun2   = np.min
    LL_df_all, LL_bob, LL_bow  = compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col2,best_fun2,worst_fun2)

    # Set figure color map
    cmap    = plt.cm.get_cmap('tab20')
    cnorm   = colors.Normalize(vmin=0, vmax=19)
    smap    = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    col_bic_gran    = smap.to_rgba(0)
    col_LL_gran     = smap.to_rgba(1)
    col_bic_basic   = smap.to_rgba(6)
    col_LL_basic    = smap.to_rgba(7)
    col_bic = [col_bic_basic]*len(algs_basic)+[col_bic_gran]*len(algs_gran)
    col_LL  = [col_LL_basic]*len(algs_basic)+[col_LL_gran]*len(algs_gran)

    # Make plots
    if not figshape:
        figshape = (0.8*(len(algs_basic)+len(algs_gran)),6)
    f,ax = plt.subplots(2,1,figsize=figshape)

    plot_col = ['best','worst']
    bic_bo   = [bic_bob,bic_bow]
    LL_bo    = [LL_bob,LL_bow]
    plot_title = ['Best','Worst']

    for i in range(len(ax)):
        # Compute ylims
        eps=100
        if not my_ylim:
            my_ylim = [np.round(min([min(bic_df_all[plot_col[i]].values),min(-2*LL_df_all[plot_col[i]].values)])-eps),np.round(max([max(bic_df_all[plot_col[i]].values),max(-2*LL_df_all[plot_col[i]].values)])+eps)]
        # Plot bars (BIC/-2*LL)
        eps_bar = 0.3
        if not bic_only:
            a1 = ax[i].bar(np.arange(len(bic_df_all))-eps_bar/2,bic_df_all[plot_col[i]].values-my_ylim[0],width=eps_bar,align='center',color=col_bic) 
            a2 = ax[i].bar(np.arange(len(LL_df_all))+eps_bar/2,-2*LL_df_all[plot_col[i]].values-my_ylim[0],width=eps_bar,align='center',color=col_LL) 
        else:
            a1 = ax[i].bar(np.arange(len(bic_df_all)),bic_df_all[plot_col[i]].values-my_ylim[0],width=2*eps_bar,color=col_bic) 
        # Plot best model (acc. to BIC/-2*LL)
        eps_line = 0.5
        win1 = ax[i].plot([0-eps_line,len(bic_df_all)-eps_line],[bic_bo[i]-my_ylim[0],bic_bo[i]-my_ylim[0]],'--',color='k')
        if not bic_only:
            win2 = ax[i].plot([0-eps_line,len(bic_df_all)-eps_line],[-2*LL_bo[i]-my_ylim[0],-2*LL_bo[i]-my_ylim[0]],'--',color='grey')
        # Format plot
        eps=100
        ax[i].set_title(f'{plot_title[i]} case')
        ax[i].set_xticks(np.arange(len(bic_df_all)))
        ax[i].set_xticklabels(alg_labels)
        #ax[i].set_xlim([-2*eps_bar,len(bic_df_all)-1+2*eps_bar])
        ax[i].set_xlim([-eps_line,len(bic_df_all)-eps_line])
        if not bic_only:
            ax[i].set_ylabel('BIC / -2*LL')
        else:
            ax[i].set_ylabel('BIC') 
        ax[i].set_ylim([0,my_ylim[1]-my_ylim[0]])
        make_shifted_yaxis(ax[i],my_ylim[0])
        ax[i].set_yticklabels([str(int(yt+my_ylim[0])) for yt in ax[i].get_yticks()])
        #ax[i].legend([a1[0],a2[0]],['BIC','-2*LL'])  
    f.tight_layout()

    if save_plot:
        save_name        = f'{save_name}_{comb_type}2{"_bic-only" if bic_only else ""}'
        sl.make_long_dir(path_save)
        plt.savefig(os.path.join(path_save,save_name+'.eps'))
        plt.savefig(os.path.join(path_save,save_name+'.svg'))

def plot_scenario(scenario,algs_basic,algs_gran,alg_labels,opt_method,comb_type,path_load,path_save,save_plot=False,save_name='',figshape=None,ax=None):
    # Compute BIC and LL
    plot_col1   = 'bic' if comb_type=='app' else 'sum_bic'
    best_fun1    = np.min; worst_fun1   = np.max
    bic_df_all, bic_bob, bic_bow = compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col1,best_fun1,worst_fun1)

    plot_col2   = 'LL' if comb_type=='app' else 'sum_LL'
    best_fun2    = np.max; worst_fun2   = np.min
    LL_df_all, LL_bob, LL_bow  = compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col2,best_fun2,worst_fun2)

    # Set figure color map
    cmap    = plt.cm.get_cmap('tab20c')
    cnorm   = colors.Normalize(vmin=0, vmax=19)
    smap    = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    col_bic = smap.to_rgba(0)
    col_LL  = smap.to_rgba(1)

    # Make plots
    if not ax:
        if not figshape:
            figshape = (len(algs_basic)+len(algs_gran),3)
        f,ax = plt.subplots(1,1,figsize=figshape)

    if scenario=='best':
        plot_col = ['best']
        bic_bo   = [bic_bob]
        LL_bo    = [LL_bob]
        plot_title = ['Best']
    else:
        plot_col = ['worst']
        bic_bo   = [bic_bow]
        LL_bo    = [LL_bow]
        plot_title = ['Worst']

    for i in range(len(ax)):
        # Compute ylims
        eps=100
        my_ylim = [min([min(bic_df_all[plot_col[i]].values),min(-2*LL_df_all[plot_col[i]].values)])-eps,max([max(bic_df_all[plot_col[i]].values),max(-2*LL_df_all[plot_col[i]].values)])+eps]
        # Plot bars (BIC/-2*LL)
        eps_bar = 0.3
        a1 = ax[i].bar(np.arange(len(bic_df_all))-eps_bar/2,bic_df_all[plot_col[i]].values-my_ylim[0],width=eps_bar,align='center',color=col_bic) 
        a2 = ax[i].bar(np.arange(len(LL_df_all))+eps_bar/2,-2*LL_df_all[plot_col[i]].values-my_ylim[0],width=eps_bar,align='center',color=col_LL) 
        # Plot best model (acc. to BIC/-2*LL)
        eps_line = 0.5
        win1 = ax[i].plot([0-eps_line,len(bic_df_all)-eps_line],[bic_bo[i]-my_ylim[0],bic_bo[i]-my_ylim[0]],'--',color=col_bic)
        win2 = ax[i].plot([0-eps_line,len(bic_df_all)-eps_line],[-2*LL_bo[i]-my_ylim[0],-2*LL_bo[i]-my_ylim[0]],'--',color=col_LL)
        # Format plot
        ax[i].set_title(f'{plot_title[i]} case')
        ax[i].set_xticks(np.arange(len(bic_df_all)))
        ax[i].set_xticklabels(alg_labels)
        ax[i].set_xlim([0-eps_line,len(bic_df_all)-eps_line])
        #ax[i].set_xlim([-2*eps_bar,len(bic_df_all)-1+2*eps_bar])
        ax[i].set_ylabel('BIC / -2*LL')
        ax[i].set_ylim([0,my_ylim[1]-my_ylim[0]])
        make_shifted_yaxis(ax[i],my_ylim[0])
        ax[i].set_yticklabels([str(int(yt+my_ylim[0])) for yt in ax[i].get_yticks()])
        ax[i].legend([a1[0],a2[0]],['BIC','-2*LL'])  
    f.tight_layout()

    if save_plot:
        save_name        = f'{save_name}_{plot_col[0]}_{comb_type}'
        sl.make_long_dir(path_save)
        plt.savefig(os.path.join(path_save,save_name+'.eps'))
        plt.savefig(os.path.join(path_save,save_name+'.svg'))

def plot_scenario_schwartz_approx(scenario,algs_basic,algs_gran,alg_labels,opt_method,comb_type,path_load,path_save,save_plot=False,save_name='',figshape=None,bar_pattern=None,ax=None,plot_legend=False,ylim=None,col_bic=None,col_LL=None):
    # Compute BIC and LL
    plot_col1   = 'bic' if comb_type=='app' else 'sum_bic'
    best_fun1    = np.min; worst_fun1   = np.max
    bic_df_all, bic_bob, bic_bow = compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col1,best_fun1,worst_fun1)

    plot_col2   = 'LL' if comb_type=='app' else 'sum_LL'
    best_fun2    = np.max; worst_fun2   = np.min
    LL_df_all, LL_bob, LL_bow  = compute_model_comp(algs_basic,algs_gran,opt_method,comb_type,path_load,plot_col2,best_fun2,worst_fun2)

    # Set figure color map
    if col_bic is None or col_LL is None:
        cmap    = plt.cm.get_cmap('tab20c')
        cnorm   = colors.Normalize(vmin=0, vmax=19)
        smap    = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        col_bic = smap.to_rgba(0)
        col_LL  = smap.to_rgba(1)

    # Make plots
    if not ax:
        if not figshape:
            figshape = (len(algs_basic)+len(algs_gran),3)
        f,ax = plt.subplots(1,1,figsize=figshape)

    if scenario=='best':
        plot_col = ['best']
        bic_bo   = [bic_bob]
        LL_bo    = [LL_bob]
        plot_title = ['Best']
    else:
        plot_col = ['worst']
        bic_bo   = [bic_bow]
        LL_bo    = [LL_bow]
        plot_title = ['Worst']

    if not isinstance(ax,list):
        ax = [ax]

    for i in range(len(ax)):
        eps_bar = 0.3
        y1 = LL_df_all[plot_col[i]].values
        y2 = -0.5*bic_df_all[plot_col[i]].values-y1
        # Compute ylims
        eps=100
        if ylim is None:
            my_ylim = [min(y1+y2)-eps,max(y1)+eps]
        else: 
            my_ylim = ylim 
        # Plot bars (LL, 1/2*k*log(n))
        if bar_pattern is None:
            bar_pattern=[""]*len(y1)
        a1 = ax[i].bar(np.arange(len(y1)),y1-my_ylim[1],width=2*eps_bar,align='center',color=col_LL,hatch=bar_pattern) 
        a2 = ax[i].bar(np.arange(len(y2)),y2,width=2*eps_bar,align='center',color=col_bic,bottom=y1-my_ylim[1],hatch=bar_pattern) 
        # Plot best model (acc. to BIC/-2*LL)
        eps_line = 0.5
        win1 = ax[i].plot([0-eps_line,len(bic_df_all)-eps_line],[LL_bo[i]-my_ylim[1]]*2,'--',color=col_LL)
        win2 = ax[i].plot([0-eps_line,len(bic_df_all)-eps_line],[-0.5*bic_bo[i]-my_ylim[1]]*2,'--',color=col_bic)
        # Format plot
        #ax[i].set_title(f'{plot_title[i]} case')
        ax[i].set_xticks(np.arange(len(bic_df_all)))
        ax[i].set_xticklabels(alg_labels,rotation=90,ha='center')
        #ax[i].set_xlim([-2*eps_bar,len(bic_df_all)-1+2*eps_bar])
        ax[i].set_xlim([0-eps_line,len(y1)-eps_line])
        ax[i].set_ylabel('Log evidence')
        # ax[i].set_ylim(my_ylim)
        ax[i].set_ylim([my_ylim[0]-my_ylim[1],0])
        make_shifted_yaxis(ax[i],my_ylim[1])
        ax[i].set_yticklabels([str(int(yt+my_ylim[1])) for yt in ax[i].get_yticks()])
        if plot_legend:
            if scenario=='best':
                #ax[i].legend([a1[0],a2[0]],['LL','Penalty'],fontsize=10,loc='upper right',bbox_to_anchor=(1,1.3),ncol=2,frameon=False,mode='expand')
                ax[i].legend([a1[0],a2[0]],['LL','Penalty'],fontsize=10,frameon=False,bbox_to_anchor=(0, 1.02, 1, 0.2),loc="lower left",mode="expand",borderaxespad=0,ncol=2)
            else:
                ax[i].legend([a1[0],a2[0]],['Log-Likelihood','Penalty'],fontsize=10)  
    f.tight_layout()

    if save_plot:
        save_name        = f'{save_name}_{plot_col[0]}_{comb_type}'
        sl.make_long_dir(path_save)
        plt.savefig(os.path.join(path_save,save_name+'.eps'),bbox_inches='tight')
        plt.savefig(os.path.join(path_save,save_name+'.svg'),bbox_inches='tight')   


def plot_bic_bar(path_load,alg_type,comb_type,path_save,save_name,title='',ax=None,figshape=None):
    # Load BIC and LL data            
    bic_df = sl.load_sim_data(path_load,file_data=f'bic_{alg_type}_{comb_type}.csv')

    # Set figure color map
    cmap    = plt.cm.get_cmap('tab20c')
    cnorm   = colors.Normalize(vmin=0, vmax=19)
    smap    = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    col_bic = smap.to_rgba(0)
    col_LL  = smap.to_rgba(1)

    # Make plots
    if not ax:
        if not figshape:
            figshape = (len(np.unique(bic_df.model))/2,3)
        f,ax = plt.subplots(1,1,figsize=figshape)
    
    # Get data to be plotted
    plot_col1  = 'bic' if comb_type=='app' else 'sum_bic'
    y1 = bic_df[['model',plot_col1]].drop_duplicates()
    by1 = bic_df.loc[bic_df.best,plot_col1].unique()[0]

    plot_col2  = 'LL' if comb_type=='app' else 'sum_LL'
    y2 = bic_df[['model',plot_col2]].drop_duplicates() 
    by2 = bic_df.loc[bic_df.best,plot_col2].unique()[0]

    # Compute ylims
    eps=100
    my_ylim = [min([min(y1[plot_col1].values),min(-2*y2[plot_col2].values)])-eps,max([max(y1[plot_col1].values),max(-2*y2[plot_col2].values)])+eps]
    # Plot BIC/-2*LL bars
    eps_bar = 0.3
    a1 = ax.bar(np.arange(len(y1))-eps_bar/2,y1[plot_col1].values-my_ylim[0],width=eps_bar,align='center',color=col_bic)
    a2 = ax.bar(np.arange(len(y1))+eps_bar/2,-2*y2[plot_col2].values-my_ylim[0],width=eps_bar,align='center',color=col_LL)
    # Plot best model lines
    eps_line = 0.5
    ax.plot([0-eps_line,len(y1)-eps_line],[by1-my_ylim[0]]*2,'--',color=col_bic)
    ax.plot([0-eps_line,len(y1)-eps_line],[-2*by2-my_ylim[0]]*2,'--',color=col_LL)
    # Format plot
    ax.set_title(title)
    ax.set_xticks(np.arange(len(y1)))
    xtl = list([j.split('-l')[-1] for j in y1['model'].values])
    ax.set_xticklabels(xtl)
    ax.set_xlim([0-eps_line,len(y1)-eps_line])
    # ax.set_xlim([-eps_line,len(y1)-eps_line])
    # ax.set_ylim(my_ylim)
    ax.set_ylim([0,my_ylim[1]-my_ylim[0]])
    make_shifted_yaxis(ax,my_ylim[0])
    ax.set_yticklabels([str(int(yt+my_ylim[0])) for yt in ax.get_yticks()])
    ax.set_xlabel('Granularity level')
    ax.set_ylabel('BIC / -2*LL')
    ax.legend([a1[0],a2[0]],['BIC','-2*LL'])  
    f.tight_layout()

    save_name = f'{save_name}_{alg_type}_{comb_type}'
    plt.savefig(path_save+save_name+'.svg',bbox_inches='tight')
    plt.savefig(path_save+save_name+'.eps',bbox_inches='tight')

def plot_schwartz_approx_per_level(path_load,alg_type,comb_type,path_save='',save_plot=True,save_name='',title='',ax=None,figshape=None,plot_legend=False,ylim=None,col_bic=None,col_LL=None):
    # Load BIC and LL data            
    bic_df = sl.load_sim_data(path_load,file_data=f'bic_{alg_type}_{comb_type}.csv')
    y_ll = bic_df['LL'].values
    y_pen = -0.5*bic_df['bic'].values-y_ll

    # Set figure color map
    if col_bic is None or col_LL is None:
        cmap    = plt.cm.get_cmap('tab20c')
        cnorm   = colors.Normalize(vmin=0, vmax=19)
        smap    = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        col_bic = smap.to_rgba(0)
        col_LL  = smap.to_rgba(1)

    # Make plots
    if not ax:
        if not figshape:
            figshape = (len(np.unique(bic_df.model))/2,3)
        f,ax = plt.subplots(1,1,figsize=figshape)
    else:
        f = None
    
    # Compute ylims
    eps=100
    eps_bar = 0.3
    if ylim is None:
        my_ylim = [min(y_ll+y_pen)-eps,max(y_ll)+eps]
    else:
        my_ylim = ylim #[-6900,-6500]
    # Plot bars (LL, 1/2*k*log(n))
    a1 = ax.bar(np.arange(len(y_ll)),y_ll-my_ylim[1],width=2*eps_bar,align='center',color=col_LL) 
    a2 = ax.bar(np.arange(len(y_pen)),y_pen,width=2*eps_bar,align='center',color=col_bic,bottom=y_ll-my_ylim[1]) 
    # Plot best model (acc. to y_ll, y_ll+y_pen)
    eps_line = 0.5
    win1 = ax.plot([0-eps_line,len(bic_df)-eps_line],[np.max(y_ll)-my_ylim[1]]*2,'--',color=col_LL[0] if isinstance(col_LL,list) else col_LL)
    win2 = ax.plot([0-eps_line,len(bic_df)-eps_line],[np.max(y_ll+y_pen)-my_ylim[1]]*2,'--',color=col_bic[0] if isinstance(col_bic,list) else col_bic)
    # Format plot
    ax.set_xticks(np.arange(len(bic_df)))
    ax.set_title(title)
    alg_labels = [i.replace(f'{alg_type}-l','') for i in list(bic_df.model.values)]
    ax.set_xticklabels(alg_labels)
    #ax.set_xlim([-2*eps_bar,len(bic_df_all)-1+2*eps_bar])
    ax.set_xlim([0-eps_line,len(y_ll)-eps_line])
    ax.set_xlabel('Granularity level')
    ax.set_ylabel('Log evidence')
    # ax.set_ylim(my_ylim)
    ax.set_ylim([my_ylim[0]-my_ylim[1],0])
    make_shifted_yaxis(ax,my_ylim[1])
    ax.set_yticklabels([str(int(yt+my_ylim[1])) for yt in ax.get_yticks()])
    if plot_legend: ax.legend([a1[0],a2[0]],['LL','Penalty'],fontsize=10)  
    if not (f is None): 
        f.tight_layout()
        save_name = f'{save_name}_{alg_type}_{comb_type}'
        plt.savefig(path_save+save_name+'.svg',bbox_inches='tight')
        plt.savefig(path_save+save_name+'.eps',bbox_inches='tight')

   
if __name__=="__main__":

    algs_basic      = ['nac','nor','hybrid2']
    algs_gran       = ['hnac-gn','hnac-gn-goi','hnor','hhybrid2'] #'hnac-gn-gv','hnac-gn-gv-goi',
    # alg_labels      = ['MF\n(count)','MB\n(count)','Hybrid\n(count)','MF\n(kernel)','adapt. MF\n(kernel)','MB\n(kernel)','Hybrid (kernel)']
    alg_labels      = ['MF\n(c)','MB\n(c)','Hybrid\n(c)','MF\n(k)','adapt. MF\n(k)','MB\n(k)','Hybrid\n(k)']
    opt_method      = 'Nelder-Mead'    # 'Nelder-Mead','L-BFGS-B','SLSQP'
    comb_types      = ['app']
    epss            = [[50,100]]*len(algs_gran)
    maxit           = [False,False,False,True]
    recomp_bic      = False 

    path_bicdata    = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/ModelSelection/BIC/'
    path_save       = os.path.join(sl.get_datapath().replace('data','output'),f'Figures_Paper/Fig_model_comparison/')

    if recomp_bic:
        for i_alg in range(len(algs_basic)):
            for i_comb in range(len(comb_types)):
                alg_type = algs_basic[i_alg]
                comb_type = comb_types[i_comb]
                eps = epss[i_alg][i_comb]

                path_models = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/MLE_results/Fits/SingleRun/'
                # mle{"-maxit" if maxit[i_alg] else ""}_{alg_type}-mice_{opt_method}/'
                candidates  = [f'{alg_type}']

                sl.make_long_dir(path_bicdata)
                name_save   = f'{alg_type}'
                bic_df = compute_bic(path_models,candidates,opt_method,comb_type,path_bicdata,name_save,[maxit[i_alg]]*len(candidates))

        for i_alg in range(len(algs_gran)):
            for i_comb in range(len(comb_types)):

                alg_type = algs_gran[i_alg]
                comb_type = comb_types[i_comb]
                eps = epss[i_alg][i_comb]
                
                path_models = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/MLE_results/Fits/SingleRun/mle{"-maxit" if maxit[i_alg] else ""}_{alg_type}-mice_{opt_method}/'
                levels      = [1,2,3,4,5,6]
                candidates  = [f'{alg_type}-l{i}' for i in levels]
                xtl         = [f'l{i}' for i in levels]

                sl.make_long_dir(path_bicdata)
                name_save   = f'{alg_type}'
                bic_df = compute_bic(path_models,candidates,opt_method,comb_type,path_bicdata,name_save,[maxit[i_alg]]*len(candidates))

    for i_comb in range(len(comb_types)):
        comb_type   = comb_types[i_comb]

        # Plot old Fig. 3A (BIC and -2*LL across model types)
        # figshape        = (0.8*(len(algs_gran)+len(algs_basic)),6)
        # plot_best_worst(algs_basic,algs_gran,alg_labels,opt_method,comb_type,path_bicdata,path_save,save_plot=True,save_name='best-worst',figshape=figshape,my_ylim=[13000,13600],bic_only=True)

        # Plot new Fig. 3A (LL and penalty across model types)
        # figshape        = (len(algs_gran),3)
        # plot_scenario_schwartz_approx('best',algs_basic,algs_gran,alg_labels,opt_method,comb_type,path_bicdata,path_save,save_plot=True,save_name='best-schwartz',figshape=figshape,ax=None)

        figshape    = (2.5,2.2)
        alg_details = 'hhybrid2'
        title       = 'Hybrid model'
        maxit       = True
        # plot_bic_bar(path_bicdata,alg_details,comb_type,path_save,save_name='bic-per-level',title=title,figshape=figshape)
        plot_schwartz_approx_per_level(path_bicdata,alg_details,comb_type,path_save,save_plot=True,save_name='schwartz-per-level',title=title,figshape=figshape,ax=None)
    
        # figshape    = (len(algs_gran),3)
        # alg_details = 'hnor'
        # title       = 'MB model'
        # maxit       = True
        # plot_bic_bar(path_bicdata,alg_details,comb_type,path_save,save_name='bic-per-level',title=title,figshape=figshape)
    
        print('done')