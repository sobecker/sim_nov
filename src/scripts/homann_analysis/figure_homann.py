import os
import sys

import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from scipy.stats import sem

import utils.saveload as sl
import fitting_neural.grid_search_complex_cells as gsc
import fitting_neural.simulate_data as sd
import utils.visualization as vis

################################################################################################################################################
# Helper functions 
################################################################################################################################################
def fit_cnov(homann_data,loadpath,best_id,data_names,data_var,data_val,save_name):
    sim_best = []; pd_best_all = []; stats_best_all = []
    for j, nn in enumerate(data_names):
        pd_best     = pd.read_csv(os.path.join(loadpath,f'grid_{best_id}/data_all_{data_names[j]}.csv'))
        # pd_best['novelty_notnorm'] = pd_best['novelty'].values
        # pd_best['novelty'] = pd_best['novelty_notnorm'].values - pd_best['steady'].values
        stats_best  = pd.read_csv(os.path.join(loadpath,f'grid_{best_id}/stats_{data_names[j]}.csv'))
        pd_best_all.append(pd_best)
        stats_best_all.append(stats_best)
        sim_best.append([stats_best[data_var[j]].values, stats_best[data_val[j]].values])
        if data_names[j]=='m':
            stats_best_steady = pd.read_csv(os.path.join(loadpath,f'grid_{best_id}/stats_m_steady.csv'))
            stats_best_all.append(stats_best_steady)
            sim_best.append([stats_best_steady[data_var[j]].values, stats_best_steady['steady'].values])

    pred_best, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gsc.fit_homann_exp(sim_best,homann_data,coef_steady=True,regr_meas='score',save_path=os.path.join(loadpath,f'grid_{best_id}'),save_name=save_name)
    return sim_best, pred_best, coef, shift, pd_best_all, stats_best_all

def fit_knov(homann_data,loadpath,grid_id,data_names,data_var,data_val,save_name):
    sim_best = []; pd_best_all = []; stats_best_all = []
    for j in range(len(data_names)):
        pd_best     = pd.read_csv(os.path.join(loadpath,f'grid_{grid_id}_detailedsim/sim_data_all_{data_names[j]}.csv'))
        if data_names[j]=='lp':
            stats_best, _ = sd.get_trans_response(pd_best,data_var[j],get_stats=False,steady=True)
            sim_best.append([stats_best[data_var[j]].values, stats_best[data_val[j]].values])
            stats_best_all.append(stats_best)
        elif data_names[j]=='l':
            stats_best, _ = sd.get_nov_response(pd_best,data_var[j],get_stats=False,steady=True)
            sim_best.append([stats_best[data_var[j]].values, stats_best[data_val[j]].values])
            stats_best_all.append(stats_best)
        elif data_names[j]=='m':
            stats_best, _ = sd.get_nov_response(pd_best,data_var[j],get_stats=False,steady=True)
            stats_steady_best = stats_best
            sim_best.append([stats_best[data_var[j]].values, stats_best[data_val[j]].values])
            sim_best.append([stats_best[data_var[j]].values, stats_best['steady'].values])
            stats_best_all.append(stats_best)
            stats_best_all.append(stats_steady_best)
        pd_best.rename(columns={'Unnamed: 0':'time_step','nt':'novelty','stim_type':'type'},inplace=True)
        pd_best_all.append(pd_best)
    
    pred_best, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gsc.fit_homann_exp(sim_best,homann_data,coef_steady=True,regr_meas='score',save_path=os.path.join(loadpath,f'grid_{best_id}'),save_name=save_name)
    return sim_best, pred_best, coef, shift, pd_best_all, stats_best_all

def plot_l(l,l_stats,f=None,ax=None,xlabel=False,legend=False,col_raw=None,col_nov='grey',homann_data=None,yl='Novelty'):
    if f is None or ax is None:
        f,ax = plt.subplots(1,len(l.n_fam.unique()),figsize=(4*len(l.n_fam.unique()),4),sharey=True)
    if col_raw is None:
        col_raw = ['k']*len(l.n_fam.unique())
    alpha_ll = np.linspace(0.5,1,len(l.n_fam.unique()))
    xnov = []; xsteady = []
    for i, n in enumerate(l.n_fam.unique()):
        lpi = l.loc[l.n_fam==n]
        lpi_stats = l_stats[l_stats.n_fam==n]
        xlim_i = [lpi['time_step'].min()-0.5,lpi['time_step'].max()+0.5]

        xnov_i = lpi.loc[lpi.type=='nov','time_step'].values[0]
        ax.axvline(x=xnov_i,c=col_nov,ls='-',lw=2,alpha=alpha_ll[i]) # novel stimulus presentation
        xnov.append(xnov_i)

        if legend:
            # ax.scatter(lpi['time_step'],lpi['novelty_fitted'],s=10,color=col_raw[i],label=f'$L = {n}$')
            ax.plot(lpi['time_step'],lpi['novelty_fitted'],c=col_raw[i],ls='-',lw=1,label=f'$L = {n}$') 
        else:
            # ax.scatter(lpi['time_step'],lpi['novelty_fitted'],s=10,color=col_raw[i])
            ax.plot(lpi['time_step'],lpi['novelty_fitted'],c=col_raw[i],ls='-',lw=1) # novelty traces, rescaled + shifted to fit data

        ax.axhline(y=lpi_stats['steady_fitted'].values[0],c=col_raw[i],ls='--',lw=1) # steady state
        if i==0: 
            xsteady = lpi_stats['steady_fitted'].values[0]
        
        ynov_i = lpi_stats['nt_norm_fitted'].values[0]+lpi_stats['steady_fitted'].values[0]
        ax.scatter(xnov_i,ynov_i,s=10,color=col_raw[i])
        # ax.axhline(y=lpi_stats['nt_norm_fitted'].values[0]+lpi_stats['steady_fitted'].values[0],c=col_raw[i],ls=':',lw=1) # peak novelty response
        ax.set_yticks([])
    if not homann_data is None:
        plot_homann = np.where(np.isin(homann_data[0][0],nfam_plot))[0]
        ax.scatter(xnov,homann_data[0][1][plot_homann]+xsteady,color='k')
    ax.tick_params(axis='x') #, labelsize=14)
    ax.set_xticks([0]+xnov)
    ax.set_xlim(xlim_i)
    if xlabel: ax[i].set_xlabel('Time steps') #,fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(yl) #,fontsize=14)
    # if legend: 
    #     ax.legend(ncol=1,loc='upper right',bbox_to_anchor=(0.87,0.95),fontsize=10,frameon=False,handlelength=0.8,handletextpad=0.3,columnspacing=0.8)

def run_gabor_model(loadpath,grid_id,grid,no_complex_cells=False,num_sim=1):
    # Run Homann experiments (complex cells)
    save_path_i = os.path.join(loadpath,f'grid_{grid_id}_detailedsim/')
    sl.make_long_dir(save_path_i)

    parallel_exp = False
    init_seed = 98765
    input_corrected = True
    input_sequence_mode = 'sep'

    params_input = {'num_gabor':     40,
                        'adj_w':     True,
                        'adj_f':     False,
                        'alph_adj':  3,
                        'n_fam':     [1,3,8,18,38],
                        'n_im':      [3,6,9,12],
                        'dN':        list(np.array([0,22,44,66,88,110,143])/0.3), #[0,70,140,210,280,360,480],
                        'idx':       True
                        }
        
    k_params = {'knum': 20,
                'type_complex': 8,
                'ratio_complex': 0.5,
                'num_complex': 4,
                'k_type': 'triangle',
                'cdens':            8,
                'k_alph':           0.5,
                'flr':              True,
                'ksig':             1,
                'kcenter':          1,
                'conv':             True,
                'gabor_seed':       12345,
                'gabor_sampling':   'equidist'
                }
    
    params_grid = grid.loc[grid.grid_id==grid_id].to_dict(orient='records')[0]
    params_grid = {k: v for k, v in params_grid.items() if not ('Unnamed' in k or 'mse' in k or 'grid_id' in k)}
    k_params.update(params_grid)
        
    kwargs = {'no_simple_cells': False,
                'no_complex_cells': no_complex_cells,
                'mode_complex': 'sum', # 'sum' or 'mean'
                'debug': False,
                'append_mode': False,
                'start_id': 0,
                'savepath_sim_data': save_path_i,
                'savename_sim_data': 'sim_data_all'
                }

    seed_input, inputs, params_input = gsc.generate_input(params_input,num_sim,init_seed,input_corrected=input_corrected,input_sequence_mode=input_sequence_mode)

    sim_data = gsc.run_homann_exp_n(k_params,params_input,inputs,parallel=parallel_exp,kwargs=kwargs)

    return sim_data

def get_grid(loadpath,plot_meas,grid_vars,w_meas_ll=None,bootstrap=False,jackknife=False,plain_mse=False):
    if w_meas_ll is None:
        w_meas_ll = np.array([1]*len(plot_meas.split('+')))
    # Load grid and done files
    if (bootstrap or jackknife) and plain_mse:
        grid = pd.read_csv(os.path.join(loadpath,'grid_new2.csv'))
    elif (bootstrap or jackknife) and not plain_mse:
        grid = pd.read_csv(os.path.join(loadpath,'grid_new.csv'))
    else:
        grid = pd.read_csv(os.path.join(loadpath,'grid.csv'))
    with open(os.path.join(loadpath,'done.txt')) as f:
        done = np.array([int(line.replace('\n','')) for line in f])

    mse_vars  = ['mse_comb','mse_tem','mse_trec','mse_tmem','mse_steady']
    corr_vars = ['corr_comb','corr_tem','corr_trec','corr_tmem','corr_steady']
    if bootstrap:
        mse_vars = mse_vars + [f'boot-mean_{mv}' for mv in mse_vars] + [f'boot-sem_{mv}' for mv in mse_vars+['mse_mean']]
        corr_vars = corr_vars + [f'boot-mean_{cv}' for cv in corr_vars] + [f'boot-sem_{cv}' for cv in corr_vars+['corr_mean']]
    elif jackknife:
        mse_vars = mse_vars + [f'jack_{mv}' for mv in mse_vars+['mse_mean']] + [f'jack_{mv}'.replace('mse','se') for mv in mse_vars+['mse_mean']]

    # Collect individual plot_meas values if necessary
    if not (bootstrap or jackknife) and ((not plot_meas in grid.columns) or (grid.loc[np.isin(grid.grid_id, done), plot_meas]==0).any()):
        if 'mse' in plot_meas:
            load_file = 'mse_fit_corr-lp0.csv'
            for ii in done:
                pd_ii = pd.read_csv(loadpath + f'grid_{ii}/{load_file}')
                for mv in mse_vars:
                    grid.loc[grid.grid_id==ii, mv] = pd_ii[mv].values[0]
        if 'corr' in plot_meas:
            load_file = 'corr.csv'
            for ii in done:
                pd_ii = pd.read_csv(loadpath + f'grid_{ii}/{load_file}')
                for cv in corr_vars:
                    grid.loc[grid.grid_id==ii, cv] = pd_ii[cv].values[0]

    # Get average plot_meas values for each combination of plot_vars
    select_grid = grid_vars
    if 'mse' in plot_meas:
        select_grid = select_grid + mse_vars + ['mse_mean']
        for mv in mse_vars:
            grid.loc[~grid.grid_id.isin(done), mv] = np.NaN
        if not 'mse_mean' in grid.columns:
            grid['mse_mean'] = np.nanmean(grid[['mse_tem','mse_trec','mse_tmem','mse_steady']],axis=1)
    if 'corr' in plot_meas:
        select_grid = select_grid + corr_vars + ['corr_mean']
        for cv in corr_vars:
            grid[cv] = -grid[cv] # inverse correlation (since we minimize later)
            grid.loc[~grid.grid_id.isin(done), cv] = np.NaN
        if not 'corr_mean' in grid.columns:
            grid['corr_mean'] = np.nanmean(grid[['corr_tem','corr_trec','corr_tmem','corr_steady']],axis=1) # maybe leave corr_steady out here

    # Compute combined measure if applicable
    if 'mse' in plot_meas and 'corr' in plot_meas:
        select_grid = select_grid + [plot_meas]
        plot_meas_ll = plot_meas.split('+')
        grid[plot_meas] = np.average(grid[plot_meas_ll],weights=w_meas_ll,axis=1) # grid[plot_meas_ll].sum(axis=1)

    return grid, done, select_grid

def get_opt_in_grid(grid,done,select_grid,grid_vars,plot_vars,plot_meas,type_best_worst):
    fun_best = np.nanmin #if 'mse' in plot_meas else np.nanmax if 'corr' in plot_meas else np.nanmean

    # Get best and worst samples
    if type_best_worst=='overall':
        imin = grid.loc[grid[plot_meas]==fun_best(grid[plot_meas].values.flatten()),'grid_id'].values[0] 
        val_min = dict(zip(grid_vars,grid.loc[grid.grid_id==imin,grid_vars].values.flatten()))
        imin = np.array([imin])
    elif type_best_worst=='plot_var':
        grid_plot = grid.groupby(plot_vars).agg(np.nanmean).reset_index()
        val_min = dict(zip(plot_vars,grid_plot.loc[grid_plot[plot_meas]==fun_best(grid_plot[plot_meas].values.flatten()),plot_vars].values[0]))
        imin = grid.loc[(grid[plot_vars[0]]==val_min[plot_vars[0]]) & (grid[plot_vars[1]]==val_min[plot_vars[1]]),'grid_id'].values
    imin = imin[np.where(np.isin(imin,done))]
    imin = np.array([int(imin[0])],dtype=int)
    mse_min = dict(zip(select_grid,grid.loc[grid.grid_id==imin[0],select_grid].values.flatten()))
    
    return imin, val_min, mse_min

def get_grid_vars(c_type,set_name=1):
    if c_type=='counts_leaky':
        grid_vars = ['k_alph','num_states_model']
    elif c_type=='counts':
        grid_vars = ['eps','num_states_model']
    elif c_type=='simple':
        grid_vars = ['k_alph','cdens','knum'] # variables for which grid search was run
    elif c_type=='complex' and set_name==1:
        grid_vars = ['type_complex','num_complex','knum'] # variables for which grid search was run
    elif c_type=='complex' and set_name!=1:
        grid_vars = ['k_alph','cdens','knum']
    return grid_vars

def get_paths(k_type,c_type,corr,set_name,str_types=''):
    if 'counts' in c_type:
        loadpath = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set{set_name}_cnov{"_leaky" if "leaky" in c_type else ""}{"_corr" if corr else ""}/'
        savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}{"_corr" if corr else ""}/'
        str_types += f'_{c_type}'
    else:
        loadpath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
        savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
        str_types += f'_{k_type}-{c_type[0]}'
    sl.make_long_dir(savepath)
    return loadpath, savepath, str_types

################################################################################################################################################
def plot_best(type,imin,val_min,plot_meas,plot_vars,homann_data,homann_lstd=None,homann_ustd=None,error_type='sem',type_best_worst='overall',loadpath='',savepath='',f=None,axl=None,color='r',label=None,figname=None,save_plot=True):
    # Plot specs
    data_files      = ['l','lp','m','m_steady']
    data_var        = ['n_fam','dN','n_im','n_im']
    data_val        = ['nt_norm','tr_norm','nt_norm','steady']
    titles_homann   = ['L-exp.','L\'-exp.','M-exp.','M-exp. (steady)']
    xl_homann       = ['L','L\'','M','M']
    yl_homann       = ['$\Delta$ N','$\Delta$ N','$\Delta$ N','$N_{\infty}$']
    err_fun = (lambda x: np.std(x,axis=0)) if error_type=='std' else (lambda x: sem(x,axis=0))

    if f is None or axl is None:
        f,axl = plt.subplots(1,4,figsize=(4*3,3))

    if homann_data is None or homann_lstd is None or homann_ustd is None:
        plot_homann = False
    else:
        plot_homann = True

    for i in range(len(data_files)):
        ax = axl[i]

        # Plot Homann data
        if plot_homann:
            ax.plot(homann_data[i][0],homann_data[i][1],'o',c='k')
            ax.errorbar(x=homann_data[i][0],y=homann_data[i][1],yerr=[homann_data[i][1]-homann_lstd[i][1],homann_ustd[i][1]-homann_data[i][1]],c='k')
            ax.plot(homann_data[i][0],homann_data[i][1],'-',c='k',label='Homann et al.')

        # Plot model data
        for j in range(len(imin)):
            # Compute fitted data
            data_j         = pd.read_csv(loadpath + f'grid_{imin[j]}/data_all_{data_files[i]}.csv')
            fit_j          = pd.read_csv(loadpath + f'grid_{imin[j]}/coef_fit.csv')
            data_j['pred'] = data_j[data_val[i]].apply(lambda x: fit_j['coef'].values[0]*x)
            if 'steady' in data_files[i]: 
                data_j['pred'] = data_j['pred'].apply(lambda x: x + fit_j['shift'].values[0])
            stats_j = data_j[[data_var[i],'pred']].groupby(data_var[i]).agg([np.nanmean,err_fun])

            # Plot model data
            if label is not None:
                lj = label
            else:
                lj = f'grid id {imin[j]}'
            ax.plot(homann_data[i][0],stats_j[stats_j.columns[0]],'o',c=color)
            ax.plot(homann_data[i][0],stats_j[stats_j.columns[0]],'-',c=color,label=lj)
            ax.fill_between(homann_data[i][0],stats_j[stats_j.columns[0]].values-stats_j[stats_j.columns[1]].values,stats_j[stats_j.columns[0]].values+stats_j[stats_j.columns[1]].values,color=color,alpha=0.2)
        # ax.set_title(titles_homann[i])  
        ax.set_xlabel(xl_homann[i])
        ax.set_ylabel(yl_homann[i])
        # if i==0: ax.legend()
    if save_plot:
        axl[0].legend()
        # f.suptitle(f'{type} sample: {val_min}')
        f.tight_layout()
        if figname is None:
            figname = f'{type}_{plot_meas}' + ('_overall' if type_best_worst=='overall' else f'_{plot_vars[0]}_{plot_vars[1]}') + '.png'
        f.savefig(savepath + figname,bbox_inches='tight')

################################################################################################################################################
def plot_heatmap(c_type, k_type, plot_meas, grid_vars, plot_vars, plot_heat=True, plot_best_worst=True, error_type='sem', type_best_worst='overall'):
    # loadpath = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set1_{c_type}_cells/{c_type}_cells_{k_type}_adj_w3_G-40/'
    # savepath = f'/Users/sbecker/Projects/RL_reward_novelty/output/2024-08_grid_search_manual_cluster/set1_{c_type}_cells/{c_type}_cells_{k_type}_adj_w3_G-40/'
    loadpath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual/set1_{c_type}_cells/{c_type}_cells_{k_type}_adj_w3_G-40/'
    savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual/set1_{c_type}_cells/{c_type}_cells_{k_type}_adj_w3_G-40/'
    sl.make_long_dir(savepath)

    # Load grid and done files
    grid = pd.read_csv(loadpath + 'grid.csv')  
    with open(loadpath + 'done.txt') as f:
        done = np.array([int(line.replace('\n','')) for line in f])
    
    # Collect individual plot_meas values if necessary
    if not plot_meas in grid.columns or (grid.loc[np.isin(grid.grid_id, done), plot_meas]==0).any():
        load_file = 'mse_fit.csv' if 'mse' in plot_meas else 'corr.csv' if 'corr' in plot_meas else 'coef_fit.csv'
        for i in done:
            if plot_meas=='mse_mean':
                pd_i = pd.read_csv(loadpath + f'grid_{i}/{load_file}')
                grid.loc[grid.grid_id==i, 'mse_tem'] = pd_i['mse_tem'].values[0]
                grid.loc[grid.grid_id==i, 'mse_trec'] = pd_i['mse_trec'].values[0]
                grid.loc[grid.grid_id==i, 'mse_tmem'] = pd_i['mse_tmem'].values[0]
                grid.loc[grid.grid_id==i, 'mse_steady'] = pd_i['mse_steady'].values[0]
                grid['mse_mean'] = np.nanmean(grid[['mse_tem','mse_trec','mse_tmem','mse_steady']],axis=1)
            else:
                grid.loc[grid.grid_id==i, plot_meas] = pd.read_csv(loadpath + f'grid_{i}/{load_file}')[plot_meas].values[0]

    # Get average plot_meas values for each combination of plot_vars
    grid = grid[grid_vars + ['grid_id', plot_meas]]
    grid.loc[~grid.grid_id.isin(done),plot_meas]= np.NaN
    grid_plot = grid.groupby(plot_vars).agg(np.nanmean).reset_index()
    grid_mat  = grid_plot.pivot(index=plot_vars[0], columns=plot_vars[1], values=plot_meas)

    if plot_heat:
        # Plot heatmap for mse / correlation values
        f,ax = plt.subplots(1,1,figsize=(5,5))
        cmap = mpl.cm.get_cmap('Reds')  # viridis is the default colormap for imshow
        cmap.set_bad(color='grey')
        vmin = np.floor(np.nanmin(grid_mat.values.flatten())*10)/10
        vmax = np.ceil(np.nanmax(grid_mat.values.flatten())*10)/10
        im = ax.imshow(grid_mat.values,cmap=cmap)
        # Add annotation for min and max value
        pos_min = np.where(grid_mat.values==np.nanmin(grid_mat.values))
        ax.text(pos_min[1][0], pos_min[0][0], np.round(grid_mat.values[pos_min][0],3), ha="center", va="center", color="k")
        pos_max = np.where(grid_mat.values==np.nanmax(grid_mat.values))
        ax.text(pos_max[1][0], pos_max[0][0], np.round(grid_mat.values[pos_max][0],3), ha="center", va="center", color="w")
        # Format plot
        ax.set_xlabel(plot_vars[1])
        ax.set_ylabel(plot_vars[0])
        ax.set_xticks(np.arange(len(list(grid_mat.columns))))
        ax.set_yticks(np.arange(len(list(grid_mat.index))))
        ax.set_xticklabels(list(grid_mat.columns))
        ax.set_yticklabels(list(grid_mat.index))
        # Add colorbar
        f.colorbar(im, label=plot_meas)
        f.tight_layout()
        # Save plot
        f.savefig(savepath + f'heatmap_{plot_meas}_{plot_vars[0]}_{plot_vars[1]}.png',bbox_inches='tight')
        print('done')
        
    if plot_best_worst:
        # Get best and worst samples
        if type_best_worst=='overall':
            imin = grid.loc[grid[plot_meas]==np.nanmin(grid[plot_meas].values.flatten()),'grid_id'].values[0] 
            imax = grid.loc[grid[plot_meas]==np.nanmax(grid[plot_meas].values.flatten()),'grid_id'].values[0]
            val_min = dict(zip(grid_vars,grid.loc[grid.grid_id==imin,grid_vars].values.flatten()))
            val_max = dict(zip(grid_vars,grid.loc[grid.grid_id==imax,grid_vars].values.flatten()))
            imin = np.array([imin])
            imax = np.array([imax])

        elif type_best_worst=='plot_var':
            val_min = dict(zip(plot_vars,grid_plot.loc[grid_plot[plot_meas]==np.nanmin(grid_plot[plot_meas].values.flatten()),plot_vars].values[0]))
            val_max = dict(zip(plot_vars,grid_plot.loc[grid_plot[plot_meas]==np.nanmax(grid_plot[plot_meas].values.flatten()),plot_vars].values[0]))
            imin = grid.loc[(grid[plot_vars[0]]==val_min[plot_vars[0]]) & (grid[plot_vars[1]]==val_min[plot_vars[1]]),'grid_id'].values
            imax = grid.loc[(grid[plot_vars[0]]==val_max[plot_vars[0]]) & (grid[plot_vars[1]]==val_max[plot_vars[1]]),'grid_id'].values

        imin = imin[np.where(np.isin(imin,done))]
        imax = imax[np.where(np.isin(imax,done))]

        # Load homann data
        homann_data = gsc.load_exp_homann(cluster=False)
        homann_lstd = gsc.load_exp_homann(cluster=False,type='lowerstd')
        homann_ustd = gsc.load_exp_homann(cluster=False,type='upperstd')

        # Plot best and worst samples
        plot_best('best',imin,val_min,plot_meas,plot_vars,homann_data,homann_lstd,homann_ustd,error_type,type_best_worst,loadpath,savepath)
        plot_best('worst',imax,val_max,plot_meas,plot_vars,homann_data,homann_lstd,homann_ustd,error_type,type_best_worst,loadpath,savepath)
    
        print('done')

################################################################################################################################################
def plot_single_var(plot_meas,plot_var,ll_c_type,ll_k_type,ll_color,xl,yl,xt,yt,plot_counts=False,color_counts=['grey']*2,best_params=None,color_other=[],corr=True,f=None,ax=None,legend=False,legend_color=False,legend_symbol=False,w_meas_ll=None, ll_set_names=None,ll_names=None,plot_sep_exp=False,add_name='',plain_mse=False,bootstrap=False,jackknife=False,broken_axes=None):
    # Init default values
    if w_meas_ll is None:
        w_meas_ll = np.array([1]*len(plot_meas.split('+')))
    if ll_set_names is None:
        ll_set_names = ['1']*len(ll_c_type)

    # Init figure
    if f is None or ax is None:
        fig,ax = plt.subplots(1,1,figsize=(2,2)) #figsize=(2.25,1.9)
        set_figtight = True
    else:
        set_figtight = False
    # if not broken_axes is None:
    #     ax2 = ax.twinx()
    str_types = ''
    if best_params is not None:
        bs_models = [bp[0] for bp in best_params]
        bs_params = [bp[1] for bp in best_params]

    # Load Homann data
    homann_data = gsc.load_exp_homann(cluster=False)
    homann_data[1][1][0] = 0 # set novelty response for L'=0 to zero

    # Plot specs (Homann)
    data_files      = ['l','lp','m','m_steady']
    data_var        = ['n_fam','dN','n_im','n_im']
    data_val        = ['nt_norm','tr_norm','nt_norm','steady']
    xl_homann       = ['L','L\'','M','M']
    yl_homann       = ['$\Delta$ N','$\Delta$ N','$\Delta$ N','$N_{\infty}$']
    err_fun = (lambda x: np.std(x,axis=0)) if error_type=='std' else (lambda x: sem(x,axis=0))

    if plot_counts:
        str_types += '_counts'

        # Simulation for count data
        h_types = ['tau_emerge','tau_recovery','tau_memory','steadystate']
        k_type  = 'box' 
        k_num   = 150
        t_eps   = 0.4

        path_save_data = '/Users/sbecker/Projects/RL_reward_novelty/data/GaborPredictions/Fig_neural_predictions/'
        data_counts = []
        data_errors = []
        for i, h_type in enumerate(h_types):
            # Load count data
            savename = f"{h_type}_{k_type}_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','')}"
            d_counts = pd.read_csv(os.path.join(path_save_data,f'{savename}.csv'),header=2,names=['n_fam','nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem'])
            if h_type=='tau_emerge':
                d_counts = d_counts.loc[np.isin(d_counts.n_fam,homann_data[0][0])]
            data_counts.append((d_counts['n_fam'].values,d_counts['nt_norm_mean'].values))
            data_errors.append((d_counts['n_fam'].values,d_counts[f'nt_norm_{error_type}'].values))

        # Fit count data
        pred_data, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gsc.fit_homann_exp(data_counts,homann_data,coef_steady=True,regr_meas='score',save_path='')
        mse_all = {'mse_comb':mse_comb,'mse_tem':mse_tem,'mse_trec':mse_trec,'mse_tmem':mse_tmem,'mse_steady':mse_steady,'mse_mean':np.nanmean([mse_tem,mse_trec,mse_tmem,mse_steady])}

        # Plot fitted count data
        if legend:
            ax.axhline(y=mse_all[plot_meas],ls=':',c=color_counts[0],label='Counts')
        else:
            ax.axhline(y=mse_all[plot_meas],ls=':',c=color_counts[0])

    # Load and plot data
    for i_model, (c_type, k_type, color, set_name) in enumerate(zip(ll_c_type,ll_k_type,ll_color,ll_set_names)):
        loadpath, savepath, str_types = get_paths(k_type,c_type,corr,set_name,str_types)
        
        # Variables of the grid search
        grid_vars = get_grid_vars(c_type,set_name)

        grid, done, select_grid = get_grid(loadpath,plot_meas,grid_vars,w_meas_ll,bootstrap=bootstrap,jackknife=jackknife,plain_mse=plain_mse)

        # Add percentage of simple/complex cells (if applicable)
        if c_type=='complex' and plot_var=='complex_perc':
            grid['simple_total']  = grid['type_complex']*grid['knum']                                   # total number of simple cells
            grid['complex_total'] = grid['num_complex']*grid['knum']                                    # total number of complex cells (on average)
            grid['simple_perc']   = grid['simple_total']/(grid['simple_total']+grid['complex_total'])   # percentage of simple cells
            grid['complex_perc']  = 1-grid['simple_perc']                                               # percentage of complex cells
            select_grid = select_grid + ['complex_perc']
        if 'count' in c_type:
            grid['k_alph'] = 1-grid['k_alph'] # inverse of alpha for plotting
            grid = grid[grid['k_alph']>=0.01]
            # grid = grid[grid['k_alph']<=0.09]
    
        # Plot all plot_meas across given plot_var
        if not plot_sep_exp:
            if 'mse' in plot_meas and 'corr' in plot_meas:
                plot_meas_all = ['mse_tem+corr_tem','mse_tmem+corr_tmem','mse_trec+corr_trec',plot_meas]
            elif 'mse' in plot_meas:
                plot_meas_all = ['mse_tem','mse_trec','mse_tmem',plot_meas]
            elif 'corr' in plot_meas:
                plot_meas_all = ['corr_tem','corr_trec','corr_tmem',plot_meas]
            plot_alpha = [1,1,1,1]
            plot_marker = ['none','left','right','full']
        else:
            plot_meas_all = [plot_meas]
            plot_alpha = [1]
            plot_marker = ['full']

        for i_pm, (pm, am, fm) in enumerate(zip(plot_meas_all,plot_alpha,plot_marker)):
            grid_plot = grid[[plot_var, pm]].groupby([plot_var]).agg(mean=(pm,np.nanmean),min=(pm,np.nanmin),max=(pm,np.nanmax)).reset_index()
            # Plot mses from individual experiments + plot_meas
            col_ls = '-' if pm==plot_meas else '--'
            if legend_symbol or legend:
                ls = "L exp." if 'tem' in pm else "L\' exp." if 'trec' in pm else "M exp." if 'tmem' in pm else "Exp. combined"
                # ax.plot(grid_plot[plot_var],grid_plot['mean'],'o',c=color[i_pm],fillstyle=fm,alpha=am,label=ls)
                ax.plot(grid_plot[plot_var],grid_plot['min'],'o',c=color[i_pm],fillstyle=fm,alpha=am,label=ls)
            else: 
                # ax.plot(grid_plot[plot_var],grid_plot['mean'],'o',c=color[i_pm],fillstyle=fm,alpha=am)
                ax.plot(grid_plot[plot_var],grid_plot['min'],'o',c=color[i_pm],fillstyle=fm,alpha=am)
            # Broken y-axis
            # if not broken_axes is None:
            #     ax2.plot(grid_plot[plot_var],grid_plot['min'],'o',c=color[i_pm],fillstyle=fm,alpha=am)

            if legend_color and fm=='full':
                ls = ll_names[i_model]
                ax.plot(grid_plot[plot_var],grid_plot['min'],'o',c=color[i_pm],fillstyle=fm,alpha=am,label=ls)
            
            ax.plot(grid_plot[plot_var],grid_plot['min'],ls=col_ls,c=color[i_pm],alpha=am) 
            # if not broken_axes is None:
            #     ax2.plot(grid_plot[plot_var],grid_plot['min'],ls=col_ls,c=color[i_pm],alpha=am) 
            #,label=f'{c_type} {k_type} (mean)')
            # ax.fill_between(grid_plot[plot_var],grid_plot['min'],grid_plot['max'],color=color[i_pm],alpha=0.2)

        # Plot best-sample point and best MSE of opposite model
        if best_params is not None:
            if 'counts' in c_type and f'{c_type}' in bs_models:
                idx = bs_models.index(f'{c_type}')
                idx_other = np.where(np.array(bs_models)!=f'{c_type}')[0]
                plot_best = True
            elif f'{k_type}-{c_type}' in bs_models:
                idx = bs_models.index(f'{k_type}-{c_type}')
                idx_other = np.where(np.array(bs_models)!=f'{k_type}-{c_type}')[0]
                plot_best = True
            else:
                plot_best = False
            if plot_best:
                # Best sample point
                if plot_var in bs_params[idx].keys():
                    best_var = bs_params[idx][plot_var] 
                    if plot_var=='k_alph' and 'count' in c_type:
                        best_var = 1-best_var
                    best_mse = bs_params[idx][plot_meas]
                    if legend:
                        ax.plot(best_var,best_mse,'x',c='k',linewidth=3,label='Best')
                    else:
                        ax.plot(best_var,best_mse,'x',c='k',linewidth=3)
                # Best MSE of opposite model
                # other_models = [bs_models[idx_other]] if len(bs_models)==2 else bs_models[idx_other]
                # for m_other in other_models:
                #     i_other = np.where(np.array(bs_models)==m_other)[0][0]
                #     mse_other = bs_params[i_other][plot_meas]
                #     if legend:
                #         lss = m_other.capitalize().split('-')[0]+(' kernels' if not 'counts' in m_other else '')
                #         ax.axhline(y=mse_other,ls='-',c=color_other[i_other][0],label=lss)
                #     else:
                #         ax.axhline(y=mse_other,ls='-',c=color_other[i_other][0])

    # Format plot
    ax.set_xlabel(xl)
    ax.set_xlim([min(xt),max(xt)])
    ax.set_xticks(xt)
    # if plot_var in ['knum','cdens']:
    #     ax.set_xscale('log')
    
    ax.set_ylabel(yl)
    ax.set_yticks(yt)
    if plain_mse:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylim([min(yt),max(yt)])
    # if not broken_axes is None: 
    #     # Set ylims
    #     ax.set_xlim(broken_axes[0])
    #     ax2.set_ylim(broken_axes[1])

    #     # Add diagonal lines to indicate the break
    #     d = .015  # how big to make the diagonal lines in axes coordinates
    #     kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    #     ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #     ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    #     kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    #     ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #     ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # else:
    #     ax.set_ylim([min(yt),max(yt)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    figname = f'{plot_meas}_{plot_var}' + str_types + add_name + '.svg'
    if set_figtight:
        fig.tight_layout()
        fig.savefig(os.path.join(savepath,figname))
    else:
        return savepath, figname

################################################################################################################################################
def plot_best_sim(ll_c_type, ll_k_type, ll_color, ll_names, plot_meas, plot_vars=None, error_type='sem', type_best_worst='overall',plot_homann=True, plot_counts=False, color_counts=['grey']*2,inset=True,corr=True,fixed_sim_path=None,fit_name='',w_meas_ll=None):
    # Set default values
    if fixed_sim_path is None:
        fixed_sim_path = [None]*len(ll_c_type)
    if fit_name=='':
        fit_name = ['']*len(ll_c_type)
    if w_meas_ll is None:
        plot_meas_ll = plot_meas.split('+')
        w_meas_ll    = np.array([1]*len(plot_meas_ll))

    # Init figure
    if inset:
        fig, axl = plt.subplots(1,3,figsize=(3*2.8,3.6))
        ax_inset = axl[2].inset_axes([0.3,0.1,0.4,0.4])
        axl = np.concatenate([axl,np.array([ax_inset])])
    else:
        fig, axl = plt.subplots(1,4,figsize=(4*2.7,3.6))
    str_types = ''
    best_params = []

    # Load Homann data
    homann_data = gsc.load_exp_homann(cluster=False)
    homann_lstd = gsc.load_exp_homann(cluster=False,type='lowerstd')
    homann_ustd = gsc.load_exp_homann(cluster=False,type='upperstd')
    # Set novelty response for L'=0 to zero
    homann_data[1][1][0] = 0 #homann_data[1][1][0]-delta_homann
    homann_lstd[1][1][0] = 0 #homann_lstd[1][1][0]-delta_homann
    homann_ustd[1][1][0] = 0 #homann_ustd[1][1][0]-delta_homann
    
    # Plot specs (Homann)
    data_files      = ['l','lp','m','m_steady']
    data_var        = ['n_fam','dN','n_im','n_im']
    data_val        = ['nt_norm','tr_norm','nt_norm','steady']
    xl_homann       = ['Sequence repetitions (L)','Recovery repetitions (L\')','Length of sequence (M)','Length of sequence (M)']
    yl_homann       = ['$\Delta$ N','$\Delta$ N','$\Delta$ N','$N_{\infty}$']
    err_fun = (lambda x: np.std(x,axis=0)) if error_type=='std' else (lambda x: sem(x,axis=0))

    # Plot Homann data
    if plot_homann:
        str_types += '_homann'
        for i in range(len(data_files)):
            ax = axl[i]
            if i==len(axl)-1 and inset:
                markersize = 3
                plot_line = False
            else:
                markersize = 6
                plot_line = True
            ax.plot(homann_data[i][0],homann_data[i][1],'o',c='k',markersize=markersize)
            ax.errorbar(x=homann_data[i][0],y=homann_data[i][1],yerr=[homann_data[i][1]-homann_lstd[i][1],homann_ustd[i][1]-homann_data[i][1]],c='k',capsize=2,fmt='none')
            if plot_line: 
                ax.plot(homann_data[i][0],homann_data[i][1],'-',c='k',label='Homann et al.')

    if plot_counts:
        str_types += '_counts'

        # Simulation for count data
        h_types = ['tau_emerge','tau_recovery','tau_memory','steadystate']
        k_type  = 'box' 
        k_num   = 150
        t_eps   = 0.4

        path_save_data = '/Users/sbecker/Projects/RL_reward_novelty/data/GaborPredictions/Fig_neural_predictions/'
        data_counts = []
        data_errors = []
        for i, h_type in enumerate(h_types):
            # Load count data
            savename = f"{h_type}_{k_type}_J-{k_num}_lr-{str(np.round(t_eps,4)).replace('.','')}"
            d_counts = pd.read_csv(os.path.join(path_save_data,f'{savename}.csv'),header=2,names=['n_fam','nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem'])
            if h_type=='tau_emerge':
                d_counts = d_counts.loc[np.isin(d_counts.n_fam,homann_data[0][0])]
            data_counts.append((d_counts['n_fam'].values,d_counts['nt_norm_mean'].values))
            data_errors.append((d_counts['n_fam'].values,d_counts[f'nt_norm_{error_type}'].values))

        # Fit count data
        pred_data, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gsc.fit_homann_exp(data_counts,homann_data,coef_steady=True,regr_meas='score',save_path='')

        best_params.append(('counts',{'k_type':'box','knum':150,'t_eps':0.4,'mse_comb':mse_comb,'mse_tem':mse_tem,'mse_trec':mse_trec,'mse_tmem':mse_tmem,'mse_steady':mse_steady,'mse_mean':np.nanmean([mse_tem,mse_trec,mse_tmem,mse_steady])}))

        # Plot fitted count data
        for i in range(len(pred_data)):
            ax = axl[i]
            if i==len(axl)-1 and inset:
                markersize = 3
                plot_line = False
            else:
                markersize = 6
                plot_line = True
            ax.plot(homann_data[i][0],pred_data[i][1],'o',c=color_counts[0],markersize=markersize)
            if plot_line: 
                ax.plot(homann_data[i][0],pred_data[i][1],'-',c=color_counts[0],label='Counts')
                ax.fill_between(homann_data[i][0],pred_data[i][1]-coef*data_errors[i][1],pred_data[i][1]+coef*data_errors[i][1],color=color_counts[0],alpha=0.2)
            else:
                ax.errorbar(x=homann_data[i][0],y=pred_data[i][1],yerr=coef*data_errors[i][1],c=color_counts[0],capsize=2,fmt='none')
            ax.set_xlabel(xl_homann[i])
            ax.set_ylabel(yl_homann[i])

    # Load and plot simulation data
    for j, (c_type, k_type, color, name, set_name) in enumerate(zip(ll_c_type, ll_k_type, ll_color, ll_names, ll_set_names)):
        if 'counts' in c_type:
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}{"_corr" if corr else ""}/'
            str_types += f'_{c_type}'
        else:
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
            str_types += f'_{k_type}-{c_type[0]}'
        sl.make_long_dir(savepath)

        # Variables of the grid search
        grid_vars = get_grid_vars(c_type,set_name)

        # Load and plot fixed simulation (if applicable)
        if fixed_sim_path[j] is not None:
            loadpath = fixed_sim_path[j]

            # Plot model data
            for i in range(len(data_files)):
                ax = axl[i]
                if i==len(axl)-1 and inset:
                    markersize = 3
                    label_font = 10
                    plot_line = False
                else:
                    markersize = 6
                    label_font = 12
                    plot_line = True

                # Compute fitted data
                if 'counts' in c_type:
                    data_j = pd.read_csv(os.path.join(loadpath,f'stats_{data_files[i]}.csv'))
                else:
                    data_j = pd.read_csv(os.path.join(loadpath,f'data_all_{data_files[i]}.csv'))
                fit_j          = pd.read_csv(os.path.join(loadpath,f'coef_fit{fit_name[j]}.csv'))
                data_j['pred'] = data_j[data_val[i]].apply(lambda x: fit_j['coef'].values[0]*x)
                if 'steady' in data_files[i]: 
                    data_j['pred'] = data_j['pred'].apply(lambda x: x + fit_j['shift'].values[0])
                stats_j = data_j[[data_var[i],'pred']].groupby(data_var[i]).agg([np.nanmean,err_fun])

                # Plot model data
                ax.plot(homann_data[i][0],stats_j[stats_j.columns[0]],'o',c=color[-1],markersize=markersize)
                if plot_line:
                    ax.plot(homann_data[i][0],stats_j[stats_j.columns[0]],'-',c=color[-1],label=ll_names[-1])
                    ax.fill_between(homann_data[i][0],stats_j[stats_j.columns[0]].values-stats_j[stats_j.columns[1]].values,stats_j[stats_j.columns[0]].values+stats_j[stats_j.columns[1]].values,color=color[-1],alpha=0.2)
                else:
                    ax.errorbar(x=homann_data[i][0],y=stats_j[stats_j.columns[0]],yerr=stats_j[stats_j.columns[1]].values,c=color[-1],capsize=2,fmt='none')
                ax.set_xlabel(xl_homann[i],loc='center',fontsize=label_font)
                ax.set_ylabel(yl_homann[i],loc='top',rotation='horizontal',fontsize=label_font)

            # Append best parameter values
            nongrid_path = loadpath.removesuffix('/').split('/')[-1]
            grid_path = '/' + os.path.join(*(loadpath.removesuffix('/').split('/')[:-1]))
            grid = pd.read_csv(os.path.join(grid_path,'grid.csv'))
            dict_best = {}
            if 'grid_' in nongrid_path:
                best_id = nongrid_path.split('grid_')[-1].split('/')[0]
                val_min = dict(zip(grid_vars,grid.loc[grid.grid_id==int(best_id),grid_vars].values[0]))
                dict_best = {**dict_best,**val_min}
            if 'mse' in plot_meas:
                mse = pd.read_csv(os.path.join(loadpath,'mse_fit.csv'))
                if 'mse_mean' not in mse.columns:
                    mse['mse_mean'] = np.nanmean(mse[['mse_tem','mse_trec','mse_tmem','mse_steady']],axis=1)
                mse_min = dict(zip(['mse_comb','mse_tem','mse_trec','mse_tmem','mse_steady','mse_mean'],mse[['mse_comb','mse_tem','mse_trec','mse_tmem','mse_steady','mse_mean']].values.flatten()))
                dict_best = {**dict_best,**mse_min}
            if 'corr' in plot_meas:
                corr_meas = pd.read_csv(os.path.join(loadpath,'corr.csv'))
                if 'corr_mean' not in corr_meas.columns:
                    corr_meas['corr_mean'] = np.nanmean(corr_meas[['corr_tem','corr_trec','corr_tmem','corr_steady']],axis=1)
                corr_min = dict(zip(['corr_comb','corr_tem','corr_trec','corr_tmem','corr_steady','corr_mean'],corr_meas[['corr_comb','corr_tem','corr_trec','corr_tmem','corr_steady','corr_mean']].values.flatten()))
                dict_best = {**dict_best,**corr_min}
            if 'mse' in plot_meas and 'corr' in plot_meas:
                plot_meas_ll = plot_meas.split('+')
                dict_best[plot_meas] = np.average([dict_best[pm] for pm in plot_meas_ll],weights=w_meas_ll)
            best_params.append((f'{k_type}-{c_type}',dict_best))

        # Load grid and plot best simulation
        else:
            if 'counts' in c_type:
                loadpath = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set1_cnov{"_leaky" if "leaky" in c_type else ""}{"_corr" if corr else ""}/'
            else:
                loadpath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual{"_corr" if corr else ""}/set1_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'

            grid, done, select_grid = get_grid(loadpath,plot_meas,grid_vars,w_meas_ll)
            imin, val_min, mse_min = get_opt_in_grid(grid,done,select_grid,grid_vars,plot_vars,plot_meas,type_best_worst)

            if 'counts' in c_type:
                best_params.append((f'{c_type}',{**val_min,**mse_min}))
            else:
                best_params.append((f'{k_type}-{c_type}',{**val_min,**mse_min}))

            # Plot model data
            for i in range(len(data_files)):
                ax = axl[i]
                if i==len(axl)-1 and inset:
                    markersize = 3
                    label_font = 10
                    plot_line = False
                else:
                    markersize = 6
                    label_font = 12
                    plot_line = True

                for jj in range(len(imin)):
                    # Compute fitted data
                    if 'counts' in c_type:
                        data_j = pd.read_csv(loadpath + f'grid_{imin[jj]}/stats_{data_files[i]}.csv')
                    else:
                        data_j = pd.read_csv(loadpath + f'grid_{imin[jj]}/data_all_{data_files[i]}.csv')
                    fit_j = pd.read_csv(loadpath + f'grid_{imin[jj]}/coef_fit.csv')
                    data_j['pred'] = data_j[data_val[i]].apply(lambda x: fit_j['coef'].values[0]*x)
                    if 'steady' in data_files[i]: 
                        data_j['pred'] = data_j['pred'].apply(lambda x: x + fit_j['shift'].values[0])
                    stats_j = data_j[[data_var[i],'pred']].groupby(data_var[i]).agg([np.nanmean,err_fun])

                    # Plot model data
                    ax.plot(homann_data[i][0],stats_j[stats_j.columns[0]],'o',c=color[-1],markersize=markersize)
                    if plot_line:
                        ax.plot(homann_data[i][0],stats_j[stats_j.columns[0]],'-',c=color[-1],label=name)
                        ax.fill_between(homann_data[i][0],stats_j[stats_j.columns[0]].values-stats_j[stats_j.columns[1]].values,stats_j[stats_j.columns[0]].values+stats_j[stats_j.columns[1]].values,color=color[-1],alpha=0.2)
                    else:
                        ax.errorbar(x=homann_data[i][0],y=stats_j[stats_j.columns[0]],yerr=stats_j[stats_j.columns[1]].values,c=color[-1],capsize=2,fmt='none')
                ax.set_xlabel(xl_homann[i],loc='center',fontsize=label_font)
                ax.set_ylabel(yl_homann[i],loc='top',rotation='horizontal',fontsize=label_font)
        
    # Format figure
    yeps = 0.0025
    if plot_meas=='mse_mean':
        ylim = [[0,0.05],[0,0.06],[-0.02,0.04],[0.06,0.085]] #[[0,0.05],[0,0.05],[-0.02,0.05],[0.06,0.085]]
    elif plot_meas=='mse_comb':
        ylim = [[0,0.05],[0,0.05],[-0.02,0.055],[0.06,0.085]]
    elif plot_meas=='mse_comb+corr_comb':
        ylim = [[0,0.05],[0,0.05],[-0.02,0.045],[0.06,0.085]]
    else:
        ylim = [[0,0.05],[0,0.05],[-0.02,0.055],[0.06,0.085]]
    xlim = [[0-0.9,40+0.9],[0-4,150+4],[3-0.5,12+0.5],[3-0.5,12+0.5]]
    xt = [[0,20,40],[0,50,100,150],[3,6,9,12],[3,6,9,12]]
    for i in range(len(axl)):
        axl[i].set_ylim(np.array(ylim[i])+yeps*np.array([-1,1]))
        axl[i].set_xlim(np.array(xlim[i]))
        axl[i].set_xticks(xt[i])
        axl[i].set_xticklabels(xt[i])
        axl[i].spines['right'].set_visible(False)
        axl[i].spines['top'].set_visible(False)
    axl[0].legend(fontsize=10,loc='lower right',handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=False)
    if inset:
        axl[-1].set_xlabel('')
        axl[-1].xaxis.set_tick_params(labelsize=10)
        axl[-1].yaxis.set_tick_params(labelsize=10)
    fig.tight_layout(pad=0.4)

    # Save figure
    figname = f'best-sim{"-inset" if inset else ""}_{plot_meas}' + str_types + '.svg'
    fig.savefig(savepath + figname,bbox_inches='tight')
        
    print('done')
    return best_params

################################################################################################################################################
def plot_average_sim(ll_c_type, ll_k_type, ll_color, ll_names, plot_homann=True, inset=True, corr=True, stat_type='mean', plot_samples=False, ll_set_names=None,add_name=''):
    # Set defaults
    if ll_set_names is None:
        ll_set_names = ['1']*len(ll_c_type)
                     
    # Init figure
    if inset:
        fig, axl = plt.subplots(1,3,figsize=(3*2.8,3.6))
        if stat_type=='mean':
            ax_inset = axl[2].inset_axes([0.3,0.1,0.4,0.4])
        elif stat_type=='median':
            ax_inset = axl[2].inset_axes([0.55,0.55,0.4,0.4])
        axl = np.concatenate([axl,np.array([ax_inset])])
    else:
        fig, axl = plt.subplots(1,4,figsize=(4*2.7,3.6))
    str_types = ''
    
    # Load Homann data
    homann_data = gsc.load_exp_homann(cluster=False)
    homann_lstd = gsc.load_exp_homann(cluster=False,type='lowerstd')
    homann_ustd = gsc.load_exp_homann(cluster=False,type='upperstd')
    # Set novelty response for L'=0 to zero
    homann_data[1][1][0] = 0 #homann_data[1][1][0]-delta_homann
    homann_lstd[1][1][0] = 0 #homann_lstd[1][1][0]-delta_homann
    homann_ustd[1][1][0] = 0 #homann_ustd[1][1][0]-delta_homann
    
    # Plot specs (Homann)
    data_files      = ['l','lp','m','m_steady']
    data_len        = [len(homann_data[dd][0]) for dd in range(len(homann_data))]
    data_var        = ['n_fam','dN','n_im','n_im']
    data_val        = ['nt_norm','tr_norm','nt_norm','steady']
    xl_homann       = ['Sequence repetitions (L)','Recovery repetitions (L\')','Length of sequence (M)','Length of sequence (M)']
    yl_homann       = ['$\Delta$ N','$\Delta$ N','$\Delta$ N','$N_{\infty}$']

    # Load simulations for all grid points
    average_params = []
    for j, (c_type, k_type, color, name, set_name) in enumerate(zip(ll_c_type, ll_k_type, ll_color, ll_names, ll_set_names)):
        if 'counts' in c_type:
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}{"_corr" if corr else ""}/'
            loadpath = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set{set_name}_cnov{"_leaky" if "leaky" in c_type else ""}{"_corr" if corr else ""}/'
            str_types += f'_{c_type}'
        else:
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
            loadpath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
            str_types += f'_{k_type}-{c_type[0]}'
        sl.make_long_dir(savepath)
            
        # Load done file
        with open(loadpath + 'done.txt') as f:
            done = np.array([int(line.replace('\n','')) for line in f])
        done = np.unique(done)

        dd_sim_all = []
        stats_sim_all = []
        for_fit_all = []
        for dd in range(len(data_files)):

            # Load all data files + combine into data frame
            dd_all = []
            for ii in done:
                d_ii = pd.read_csv(loadpath + f'grid_{ii}/stats_{data_files[dd]}.csv')
                d_ii = d_ii[:data_len[dd]]
                d_ii['grid_id'] = [ii]*len(d_ii)
                dd_all.append(d_ii)
            dd_all = pd.concat(dd_all)
            dd_sim_all.append(dd_all)
            
            # Compute mean + std across parameters
            dd_stats = dd_all[[data_var[dd], data_val[dd]]].groupby([data_var[dd]]).agg([np.mean,np.median,np.std,vis.quantile25err,vis.quantile75err]).reset_index()
            dd_for_fit = [dd_stats[data_var[dd]].values,dd_stats[(data_val[dd],stat_type)].values]
            stats_sim_all.append(dd_stats)
            for_fit_all.append(dd_for_fit)

        # Fit combined simulation data to experimental data (lin. regression)
        pred_data, coef, shift, mse_comb, [mse_tem,mse_trec,mse_tmem,mse_steady] = gsc.fit_homann_exp(for_fit_all,homann_data,coef_steady=True,regr_meas='score',save_path=os.path.join(loadpath,f'combined_{stat_type}/'),save_name=f'_combined_{stat_type}')
        if stat_type=='mean':
            pred_err = [[coef*stats_sim_all[dd][(data_val[dd],'std')].values for dd in range(len(stats_sim_all))]]*2
        elif stat_type=='median':
            pred_err = [[coef*stats_sim_all[dd][(data_val[dd],'quantile25err')].values for dd in range(len(stats_sim_all))],
                        [coef*stats_sim_all[dd][(data_val[dd],'quantile75err')].values for dd in range(len(stats_sim_all))]]
        
        # Compute regressed individual samples
        if plot_samples:
            for dd in range(len(dd_sim_all)):
                dd_sim_all[dd][f'pred_{data_val[dd]}'] = dd_sim_all[dd][data_val[dd]] * coef + int(dd==len(dd_sim_all)-1)*shift

        # Save dictionary with fitting results
        dict_average = dict(zip(['mse_comb','mse_tem','mse_trec','mse_tmem','mse_steady','coef','shift'],[mse_comb,mse_tem,mse_trec,mse_tmem,mse_steady,coef,shift]))
        if 'counts' in c_type:
            average_params.append((f'{c_type}',dict_average))
        else:
            average_params.append((f'{k_type}-{c_type[0]}',dict_average))

        # Plot average responses + standard deviation for all experiments
        for dd in range(len(data_files)):
            ax = axl[dd]
            if dd==len(axl)-1 and inset:
                markersize = 3
                label_font = 10
                plot_line = False
            else:
                markersize = 6
                label_font = 12
                plot_line = True

            # Plot samples 
            if plot_samples:
                dd_sim = dd_sim_all[dd]
                done_plot = done[done>=14]
                for ii in done_plot:
                    dd_ii = dd_sim[dd_sim.grid_id==ii]
                    ax.plot(homann_data[dd][0],dd_ii[f'pred_{data_val[dd]}'],'-',lw=0.5,c=color[0],alpha=0.4)

            # Plot model data
            ax.plot(homann_data[dd][0],pred_data[dd][1],'o',c=color[0],markersize=markersize)
            if plot_line:
                ax.plot(homann_data[dd][0],pred_data[dd][1],'-',c=color[0],label=name,lw=2)
                ax.fill_between(homann_data[dd][0],pred_data[dd][1]-pred_err[0][dd],pred_data[dd][1]+pred_err[1][dd],color=color[0],alpha=0.2)
            else:
                ax.errorbar(x=homann_data[dd][0],y=pred_data[dd][1],yerr=[pred_err[0][dd],pred_err[1][dd]],c=color[0],capsize=2,fmt='none')
            ax.set_xlabel(xl_homann[dd],loc='center',fontsize=label_font)
            ax.set_ylabel(yl_homann[dd],loc='top',rotation='horizontal',fontsize=label_font)
    
    # Plot Homann data
    if plot_homann:
        str_types += '_homann'
        for i in range(len(data_files)):
            ax = axl[i]
            if i==len(axl)-1 and inset:
                markersize = 3
                plot_line = False
            else:
                markersize = 6
                plot_line = True
            ax.plot(homann_data[i][0],homann_data[i][1],'o',c='k',markersize=markersize)
            ax.errorbar(x=homann_data[i][0],y=homann_data[i][1],yerr=[homann_data[i][1]-homann_lstd[i][1],homann_ustd[i][1]-homann_data[i][1]],c='k',capsize=2,fmt='none')
            if plot_line: 
                ax.plot(homann_data[i][0],homann_data[i][1],'-',c='k',label='Homann et al.')
        
    # Format figure
    yeps = 0.0025
    if stat_type=='mean':
        if plot_samples:
            ylim = [[-0.02,0.1],[-0.02,0.1],[-0.07,0.085],[0.02,0.1]]
        else:
            ylim = [[-0.02,0.1],[-0.02,0.1],[-0.07,0.085],[0.02,0.1]]
    elif stat_type=='median':
        if plot_samples:
            ylim = [[0,0.4],[0,0.4],[0,0.2],[-0.07,0.1]]
            # ylim = [[0,0.11],[0,0.11],[0,0.11],[0,0.1]]
        else:
            ylim = [[0,0.11],[0,0.11],[0,0.11],[0,0.1]]
    xlim = [[0-0.9,40+0.9],[0-4,150+4],[3-0.5,12+0.5],[3-0.5,12+0.5]]
    xt = [[0,20,40],[0,50,100,150],[3,6,9,12],[3,6,9,12]]
    for i in range(len(axl)):
        axl[i].set_ylim(np.array(ylim[i])+yeps*np.array([-1,1]))
        axl[i].set_xlim(np.array(xlim[i]))
        axl[i].set_xticks(xt[i])
        axl[i].set_xticklabels(xt[i])
        axl[i].spines['right'].set_visible(False)
        axl[i].spines['top'].set_visible(False)
    if stat_type=='mean':
        axl[0].legend(fontsize=10,loc='lower right',handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=True,framealpha=0.5)
    elif stat_type=='median':
        axl[0].legend(fontsize=10,loc=(0.12,0.6),handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=True,framealpha=0.5)
    if inset:
        axl[-1].set_xlabel('')
        axl[-1].xaxis.set_tick_params(labelsize=10)
        axl[-1].yaxis.set_tick_params(labelsize=10)
    fig.tight_layout(pad=0.4)

    # Save figure
    figname = f'{stat_type}{"-samples" if plot_samples else ""}-sim{"-inset" if inset else ""}' + str_types + add_name + '.svg'
    fig.savefig(savepath + figname,bbox_inches='tight')
        
    print('done')
    return average_params


################################################################################################################################################
def plot_average_sim_sepfit(ll_c_type, ll_k_type, ll_color, ll_names, plot_homann=True, inset=True, corr=True, stat_type='mean', plot_samples=False, ll_set_names=None, plot_meas=None, plot_vars=None, type_best_worst='overall', bootstrap=False, jackknife=False, ylim=None, legend_pos=None,add_name='',plain_mse=False):
    # Set defaults
    if ll_set_names is None:
        ll_set_names = ['1']*len(ll_c_type)
    plot_meas_ll = plot_meas.split('+')
    w_meas_ll    = np.array([1]*len(plot_meas_ll))

    # Init figure
    if inset:
        fig, axl = plt.subplots(1,3,figsize=(8.2,3.2))
        if stat_type=='mean':
            ax_inset = axl[2].inset_axes([0.55,0.6,0.4,0.4])
        elif stat_type=='median':
            ax_inset = axl[2].inset_axes([0.55,0.6,0.4,0.4])
        elif stat_type=='min':
            # ax_inset = axl[2].inset_axes([0.55,0.6,0.4,0.4])
            ax_inset = axl[2].inset_axes([0.55,0.7,0.4,0.3])
        axl = np.concatenate([axl,np.array([ax_inset])])
    else:
        fig, axl = plt.subplots(1,4,figsize=(4*2.7,3.6))
    str_types = ''
    
    # Load Homann data
    homann_data = gsc.load_exp_homann(cluster=False)
    homann_lstd = gsc.load_exp_homann(cluster=False,type='lowerstd')
    homann_ustd = gsc.load_exp_homann(cluster=False,type='upperstd')
    # Set novelty response for L'=0 to zero
    homann_data[1][1][0] = 0 #homann_data[1][1][0]-delta_homann
    homann_lstd[1][1][0] = 0 #homann_lstd[1][1][0]-delta_homann
    homann_ustd[1][1][0] = 0 #homann_ustd[1][1][0]-delta_homann
    
    # Plot specs (Homann)
    data_files      = ['l','lp','m','m_steady']
    data_len        = [len(homann_data[dd][0]) for dd in range(len(homann_data))]
    data_var        = ['n_fam','dN','n_im','n_im']
    data_val        = ['nt_norm','tr_norm','nt_norm','steady']
    xl_homann       = ['Sequence repetitions (L)','Recovery repetitions (L\')','Length of sequence (M)','Length of sequence (M)']
    yl_homann       = ['Novelty response $\Delta N$','$\Delta$ N','$\Delta$ N','Steady state $N_{\infty}$']

    # Load simulations for all grid points
    average_params = []
    for j, (c_type, k_type, color, name, set_name) in enumerate(zip(ll_c_type, ll_k_type, ll_color, ll_names, ll_set_names)):
        loadpath, savepath, str_types = get_paths(k_type,c_type,corr,set_name,str_types)
        # if 'counts' in c_type:
        #     savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}{"_corr" if corr else ""}/'
        #     loadpath = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set{set_name}_cnov{"_leaky" if "leaky" in c_type else ""}{"_corr" if corr else ""}/'
        #     str_types += f'_{c_type}'
        # else:
        #     savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
        #     loadpath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual{"_corr" if corr else ""}/set{set_name}_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
        #     str_types += f'_{k_type}-{c_type[0]}'
        # sl.make_long_dir(savepath)
            
        # Load done file
        with open(loadpath + 'done.txt') as f:
            done = np.array([int(line.replace('\n','')) for line in f])
        done = np.unique(done)

        if stat_type=='min':
            # Get best sample for each model
            grid_vars = get_grid_vars(c_type,set_name)
            grid, _, select_grid = get_grid(loadpath,plot_meas,grid_vars,w_meas_ll,bootstrap=bootstrap,jackknife=jackknife,plain_mse=plain_mse)
            imin, val_min, mse_min = get_opt_in_grid(grid,done,select_grid,grid_vars,plot_vars,plot_meas,type_best_worst)
            # Get filter for grid id
            keep_grid_id = grid.loc[(grid['k_alph']>=0.1) & ((grid['k_alph']<=0.9) | (grid['k_alph']==1)),'grid_id'].values
            # Get single grid id for each alpha
            grid_sorted = grid.sort_values(by=[plot_meas],ascending=True)
            alpha_grid = [grid_sorted.loc[grid_sorted['k_alph']==ka,'grid_id'].values[0] for ka in grid_sorted['k_alph'].unique()]
            if 'counts' in c_type:
                average_params.append((f'{c_type}',{**val_min,**mse_min}))
            else:
                average_params.append((f'{k_type}-{c_type}',{**val_min,**mse_min}))
        else:
            keep_grid_id = done

        dd_sim_all = []
        stats_sim_all = []
        fitted_all = []
        for dd in range(len(data_files)):

            # Load fitted data files + combine into data frame
            dd_all = []
            for ii in done:
                d_ii = pd.read_csv(loadpath + f'grid_{ii}/pred_{data_files[dd]}_corr-lp0.csv')
                d_ii = d_ii[:data_len[dd]]
                d_ii['grid_id'] = [ii]*len(d_ii)
                dd_all.append(d_ii)
            dd_all = pd.concat(dd_all)
            dd_sim_all.append(dd_all)
            
            if stat_type=='mean' or stat_type=='median':
                # Compute mean + std across parameters
                dd_stats = dd_all[[data_var[dd], data_val[dd]]].groupby([data_var[dd]]).agg([np.mean,np.median,np.std,vis.quantile25err,vis.quantile75err]).reset_index() 
                dd_fitted = [dd_stats[data_var[dd]].values,dd_stats[(data_val[dd],stat_type)].values]
            elif stat_type=='min':
                # Get best sample data
                dd_stats = dd_all.loc[dd_all['grid_id']==imin[0],[data_var[dd],data_val[dd]]]
                dd_fitted = [dd_stats[data_var[dd]].values,dd_stats[data_val[dd]].values]
                
            stats_sim_all.append(dd_stats)
            fitted_all.append(dd_fitted)

        # Plot average responses + standard deviation for all experiments
        for dd in range(len(data_files)):
            ax = axl[dd]
            if dd==len(axl)-1 and inset:
                markersize = 3
                label_font = 10
                plot_line = False
            else:
                markersize = 6
                label_font = 11
                plot_line = True

            # Plot samples 
            if plot_samples:
                dd_sim = dd_sim_all[dd]
                max_plot_sim = 36
                # done_plot = done[done>=14]
                done_plot = list(set(done).intersection(set(keep_grid_id)).intersection(set(alpha_grid)))
                if len(done_plot)>max_plot_sim:
                    done_plot = np.random.choice(done_plot,max_plot_sim,replace=False)
                    # done_plot = done_plot[:max_plot_sim]
                for ii in done_plot:
                    dd_ii = dd_sim[dd_sim.grid_id==ii] 
                    if dd==len(data_files)-1:
                        ax.plot(homann_data[dd][0],dd_ii[data_val[dd]].values,'+',markersize=markersize/2,c=color[-1],alpha=0.4)
                    else:
                        ax.plot(homann_data[dd][0],dd_ii[data_val[dd]].values,'-',lw=0.85,c=color[-1],alpha=0.4)

            # Plot model data
            xx = homann_data[dd][0]
            if stat_type=='min':
                yy_fit = pd.read_csv(os.path.join(loadpath,f'grid_{imin[0]}/coef_fit_corr-lp0.csv'))
                if 'counts' in c_type:
                    yy_data = pd.read_csv(os.path.join(loadpath,f'grid_{imin[0]}/pred_{data_files[dd]}_corr-lp0.csv'))
                    yy = yy_data[data_val[dd]]
                    yerrl = np.zeros(len(yy))
                    yerru = np.zeros(len(yy))
                else:
                    yy_data = pd.read_csv(os.path.join(loadpath,f'grid_{imin[0]}/data_all_{data_files[dd]}.csv'))
                    dv = data_val[dd]
                    yy_data['pred'] = yy_data[dv].values * yy_fit['coef'].values[0] 
                    if 'steady' in data_files[dd]:
                        yy_data['pred'] = yy_data['pred'].values + yy_fit['shift'].values[0]
                    yy_stats = yy_data[[data_var[dd],'pred']].groupby([data_var[dd]]).agg([np.mean, np.std, sem]).reset_index()
                    yy = yy_stats[('pred','mean')].values
                    yerrl = yy_stats[('pred','sem')].values
                    yerru = yy_stats[('pred','sem')].values
            else:
                yy = stats_sim_all[dd][(data_val[dd],stat_type)].values
                if stat_type=='mean':
                    yerrl = stats_sim_all[dd][(data_val[dd],'std')].values
                    yerru = stats_sim_all[dd][(data_val[dd],'std')].values
                elif stat_type=='median':
                    yerrl = stats_sim_all[dd][(data_val[dd],'quantile25err')].values
                    yerru = stats_sim_all[dd][(data_val[dd],'quantile75err')].values

            ax.plot(xx,yy,'o',c=color[-1],markersize=markersize)
            if plot_line:
                ax.plot(xx,yy,'-',c=color[-1],label=name,lw=2)
                # ax.fill_between(xx,yy-yerrl,yy+yerru,color=color[-1],alpha=0.2)
                ax.errorbar(x=xx,y=yy,yerr=[yerrl,yerru],c=color[-1],capsize=2,fmt='none')
            else:
                ax.errorbar(x=xx,y=yy,yerr=[yerrl,yerru],c=color[-1],capsize=2,fmt='none')
            ax.set_xlabel(xl_homann[dd],loc='center',fontsize=label_font)
            if dd==0:
                # ax.set_ylabel(yl_homann[dd],loc='top',rotation='horizontal',fontsize=label_font)
                ax.set_ylabel(yl_homann[dd],fontsize=label_font)
            elif dd==len(data_files)-1:
                ax.set_ylabel(yl_homann[dd],fontsize=label_font)
    
    # Plot Homann data
    if plot_homann:
        str_types += '_homann'
        for i in range(len(data_files)):
            ax = axl[i]
            if i==len(axl)-1 and inset:
                markersize = 3
                plot_line = False
            else:
                markersize = 6
                plot_line = True
            ax.plot(homann_data[i][0],homann_data[i][1],'o',c='k',markersize=markersize)
            ax.errorbar(x=homann_data[i][0],y=homann_data[i][1],yerr=[homann_data[i][1]-homann_lstd[i][1],homann_ustd[i][1]-homann_data[i][1]],c='k',capsize=2,fmt='none')
            if plot_line: 
                ax.plot(homann_data[i][0],homann_data[i][1],'-',c='k',label='Homann et al.')
        
    # Format figure
    yeps = 0.0025
    if stat_type=='mean' and ylim is None:
        ylim = [[0,0.05],[0,0.05],[0,0.07],[0.04,0.1]]
    elif stat_type=='median' and ylim is None:
        ylim = [[0,0.05],[0,0.05],[0,0.07],[0.04,0.1]]
    elif stat_type=='min' and ylim is None:
        # ylim = [[0,0.05],[0,0.05],[0,0.07],[0.04,0.1]]
        ylim = [[0,0.05],[0,0.05],[0,0.06],[0.04,0.1]]
    xlim = [[0-0.9,40+0.9],[0-4,150+4],[3-0.5,12+0.5],[3-0.5,12+0.5]]
    xt = [[0,20,40],[0,50,100,150],[3,6,9,12],[3,6,9,12]]
    for i in range(len(axl)):
        axl[i].set_ylim(np.array(ylim[i])+yeps*np.array([-1,1]))
        axl[i].set_xlim(np.array(xlim[i]))
        axl[i].set_xticks(xt[i])
        axl[i].set_xticklabels(xt[i])
        axl[i].spines['right'].set_visible(False)
        axl[i].spines['top'].set_visible(False)
    if stat_type=='mean' and legend_pos is None:
        axl[0].legend(fontsize=10,loc=(0.2,0.1),handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=True,framealpha=0.5)
    elif stat_type=='median' and legend_pos is None:
        axl[0].legend(fontsize=10,loc=(0.2,0.1),handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=True,framealpha=0.5)
    elif stat_type=='min' and legend_pos is None:
        axl[0].legend(fontsize=10,loc=(0.2,0.1),handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=True,framealpha=0.5)
    else:
        axl[0].legend(fontsize=10,loc=legend_pos,handlelength=1,borderaxespad=0,handletextpad=0.5,frameon=True,framealpha=0.5)
    if inset:
        axl[-1].set_xlabel('')
        axl[-1].xaxis.set_tick_params(labelsize=10)
        axl[-1].yaxis.set_tick_params(labelsize=10)
    fig.tight_layout(pad=0.4)

    # Save figure
    figname = f'sepfit-{stat_type}{"-samples" if plot_samples else ""}-sim{"-inset" if inset else ""}' + str_types + add_name + '.svg'
    fig.savefig(savepath + figname) #,bbox_inches='tight')
        
    print('done')
    return average_params


################################################################################################################################################
def plot_tradeoff_m_lp(c_type,k_type,ll_params,ll_colors,ll_names,xt,yt,yl,plot_meas=['mse_trec','mse_tmem','mse_steady'],pm_lines=['-','--'],pm_labels=['L\'-exp.','M-exp.'],best_params=None,corr=True):
    # Load MSEs for current model (M and L' experiment)
    loadpath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual{"_corr" if corr else ""}/set1_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
    savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual{"_corr" if corr else ""}/set1_{c_type}_cells{"_seqsep" if corr else ""}/{c_type}_cells{"-corr" if corr else ""}_{k_type}_adj_w3_G-40/'
    sl.make_long_dir(savepath)

    # Variables of the grid search
    if c_type=='simple':
        grid_vars = ['k_alph','cdens','knum'] # variables for which grid search was run
    elif c_type=='complex':
        grid_vars = ['knum','type_complex','num_complex'] # variables for which grid search was run

    if best_params is not None:
        bs_models = [bp[0] for bp in best_params]
        bs_params = [bp[1] for bp in best_params]

    # Load grid and done files
    grid = pd.read_csv(loadpath + 'grid.csv')  
    with open(loadpath + 'done.txt') as f:
        done = np.array([int(line.replace('\n','')) for line in f])
        
    # Collect individual plot_meas values if necessary
    if not np.array([pm in grid.columns for pm in plot_meas]).any() or (grid.loc[np.isin(grid.grid_id, done), plot_meas]==0).any().any():
        load_file = 'mse_fit.csv' if np.array(['mse' in pm for pm in plot_meas]).any() else 'corr.csv' if 'corr' in plot_meas else 'coef_fit.csv'
        for ii in done:
            pd_ii = pd.read_csv(loadpath + f'grid_{ii}/{load_file}')
            for jj in range(len(plot_meas)):
                grid.loc[grid.grid_id==ii, plot_meas[jj]] = pd_ii[plot_meas[jj]].values[0]
    if 'mse_tmem' in plot_meas and 'mse_steady' in plot_meas:
        grid['mse_tmem_steady'] = np.nanmean(grid[['mse_tmem','mse_steady']],axis=1) 
        plot_meas = [plot_meas[i] for i in range(len(plot_meas)) if (plot_meas[i]!='mse_tmem' and plot_meas[i]!='mse_steady')]
        plot_meas += ['mse_tmem_steady']

    # Fill missing values in grid with NaN
    grid = grid[grid_vars + ['grid_id'] + plot_meas]
    for pm in plot_meas:
        grid.loc[~grid.grid_id.isin(done), pm]   = np.NaN

    fig, ax1 = plt.subplots(1,1,figsize=(2.2,1.8))
    ax2 = ax1.twinx()
    axl = [ax1,ax2]
    # Plot MSE varying with different parameters
    str_types = ''

    for j, param in enumerate(ll_params):
        str_types += f'_{param}'

        # Plot min/max MSE for the different experiments (plot_meas)
        for i, (pm, pm_ls, pm_label) in enumerate(zip(plot_meas,pm_lines,pm_labels)):
            ax = axl[i]

            # Plot best-sample point
            if best_params is not None and f'{k_type}-{c_type}' in bs_models:
                idx = bs_models.index(f'{k_type}-{c_type}')
                if ll_params[0] in bs_params[idx].keys():
                    ax.axvline(x=bs_params[idx][ll_params[0]],linestyle=':',c='k')

            grid_plot = grid.groupby(param).agg(mean=(pm,np.nanmean),min=(pm,np.nanmin),max=(pm,np.nanmax)).reset_index()
            ax.plot(grid_plot[param],grid_plot['mean'],'o',c=ll_colors[i])
            ax.plot(grid_plot[param],grid_plot['mean'],ls=pm_ls,c=ll_colors[i],label=f'{pm_label}')
            ax.fill_between(grid_plot[param],grid_plot['min'],grid_plot['max'],color=ll_colors[i],alpha=0.2)
            
            if j==len(ll_params)-1:
                side = 'left' if i==0 else 'right'
                axl[-1].spines[side].set_color(ll_colors[i])
                ax.set_ylim(yl) 
                ax.set_yticks(yt,color=ll_colors[i])
                ax.set_ylabel(f'MSE ({pm_label})',color=ll_colors[i])
                ax.yaxis.label.set_color(ll_colors[i])
                ax.tick_params(axis='y', colors=ll_colors[i])

    for i in range(len(axl)):
        axl[i].spines['top'].set_visible(False)
        axl[i].set_xlabel(ll_names[0])
        axl[i].set_xticks(xt)
        # if ll_params[0] in ['knum','cdens']:
        #     axl[i].set_xscale('log')
    # ax.legend()
    fig.tight_layout()

    # Save figure
    exp_types = ''
    for pm in plot_meas:
        exp_types += pm.replace('mse','')
    figname = 'tradeoff' + exp_types + str_types + '.svg'
    fig.savefig(savepath + figname,bbox_inches='tight')
        
    print('done')

def plot_bar_best(ax,best_params,plot_meas,yl,ll_color,ll_names):
    pc = [ll_color[i][-1] for i in range(len(best_params))]
    pn = [ll_names[i].replace('-','\n').replace(' ','\n') for i in range(len(best_params))]
    pv = np.array([best_params[i][1][f'jack_{plot_meas}'] for i in range(len(best_params))])
    pv_err = np.array([best_params[i][1][f'jack_{plot_meas}'.replace('mse','se')] for i in range(len(best_params))])
    pp = ax.bar(np.arange(len(best_params)),pv,color=pc)
    ax.errorbar(np.arange(len(best_params)),pv,yerr=pv_err,fmt='none',ecolor='k',capsize=2)
    ax.bar_label(pp,label_type='center',fmt='%.2g')
    ax.set_xticks(np.arange(len(best_params)))
    ax.set_xticklabels(pn)    
    ax.set_ylabel(yl)

        
if __name__=="__main__":

    run_heatplot = False
    run_figure = True

    #####################################################################################################################
    #           Heat maps + best/worst sample                                                                           #
    #####################################################################################################################
    if run_heatplot:
        # Models to plot
        k_type = ['triangle']
        c_type = ['complex']
        plot_meas = ['mse_comb','mse_tem','mse_trec','mse_tmem','mse_steady','mse_mean']

        plot_heat = True
        plot_best_worst = True
        type_best_worst = 'overall' # 'overall' or 'plot_var'   
        error_type = 'sem' # 'sem' or 'std'

        for k in k_type:
            for c in c_type:
                for p in plot_meas:
                    # Measure to plot in heatmap
                    if c=='simple':
                        grid_vars = ['k_alph','cdens','knum'] # variables for which grid search was run
                        plot_vars = ['knum','cdens'] # variables for which to plot the heatmap
                    elif c=='complex':
                        grid_vars = ['type_complex','num_complex','knum'] # variables for which grid search was run
                        plot_vars = ['knum','type_complex'] # variables for which to plot the heatmap
                    plot_heatmap(c, k, p, grid_vars, plot_vars, plot_heat=plot_heat, plot_best_worst=plot_best_worst, error_type=error_type, type_best_worst=type_best_worst)    
        print('done')

    if run_figure:

        # Specify plot case (determines the models plotted)
        plot_case1  = False  # outdated?
        plot_case2  = False   # True for control figure in paper (snov-simple included)
        plot_case2a = True  # True for main figure in paper (cnov vs. snov-complex)
        plot_case3  = False  # True if plotting separate errors for cnov, snov-simple, snov-complex

        plot_simple = False   # Include snov-simple in plot_case2
        plot_leaky  = True   # True for main figure
        fixed_sim   = False  # outdated

        # Specify which measure to use
        plain_mse   = False   # if False, score is used
        if plain_mse: 
            add_name = '2'
        else:
            add_name = ''

        # Specify plots to be made 
        control_robustness = True # Plot robustness control (grid search for all params)
        plot_bestsim       = True # Plot best sim for each model

        # Set plot specs
        plt.rcParams.update({'font.size': 11}) 
        cmap_b = plt.cm.get_cmap('Blues')
        cmap_r = plt.cm.get_cmap('Reds')
        vmax   = 4
        cnorm  = colors.Normalize(vmin=0, vmax=vmax)
        smap_b = cm.ScalarMappable(norm=cnorm, cmap=cmap_b) # blue (cnov)
        smap_r = cm.ScalarMappable(norm=cnorm, cmap=cmap_r) # red (snov-complex)
        cb = [smap_b.to_rgba(i) for i in range(vmax)][::-1] + [smap_r.to_rgba(i) for i in range(vmax)]
        cmap2 = plt.cm.get_cmap('PiYG')
        cnorm2 = colors.Normalize(vmin=0, vmax=9)
        smap2 = cm.ScalarMappable(norm=cnorm2, cmap=cmap2) # purple (snov-simple)
        cb2 = [smap2.to_rgba(i) for i in range(10)] 

        # Specs for grid search plots: error type 
        plot_meas  = 'mse_mean' # 'mse_mean', 'mse_comb', 'mse_comb+corr_comb', 'corr_mean', 'corr_comb', ...
        yl         = 'Av. MSE' if plot_meas=='mse_mean' else 'Comb. MSE' if plot_meas=='mse_comb' else '- av. corr.' if plot_meas=='corr_mean' else '- comb. corr.' if plot_meas=='corr_comb' else 'Comb. MSE - corr.' if plot_meas=='mse_comb+corr_comb' else plot_meas  
        error_type = 'sem' # 'sem' or 'std'

        # Specs for grid search plots: y-ticks
        if plot_meas=='mse_mean' and (plot_case2 or plot_case2a):
            yt       = [[0,7,14]] if not plain_mse else [[0,1.5e-4,3e-4]]
            # if plain_mse:
                # ybreaks = [(0, 1e-4), (0.8e-3, 1e-3)]
        elif plot_meas=='mse_mean' and plot_case3:
            plot_meas_ll = ['mse_tem','mse_tmem','mse_trec',plot_meas]
            yt        = [[0,5,10],[0,10,20],[0,5,10],[0,10,20]]
        elif plot_meas=='mse_mean':
            yt       = [[0,7,14],[1,3,5]] if plot_simple else [[0,7,14],[0,2,4]] #[1,5,10]
        elif plot_meas=='mse_comb':
            yt       = [[0,7,14],[2,3,4]] if plot_simple else [[0,7,14],[2,3,4]] #[2,3,5]
        elif plot_meas=='mse_comb+corr_comb':
            yt       = [[0,2.5,5],[0,1,3]] if plot_simple else [[0,2.5,5],[0,1,3]] #[2,3,5]
        elif plot_meas=='corr_mean':
            yt       = [[-1,0,1],[-1,0,1]] if plot_simple else [[-1,0,1],[-1,0,1]] #[2,3,5]

        # Specify models to plot (best sim and grid search plots)
        if plot_case1 or plot_case2 or plot_case3:
            ll_set_names = [1,2,4]
            ll_k_type =  ['','triangle','triangle']
            ll_c_type = (['counts_leaky'] if plot_leaky else ['counts']) + ['simple','complex'] 
            ll_names  = (['Count-based'] if plot_leaky else ['Counts']) + ['Similarity-based\n(simple)','Similarity-based\n(full)'] 
            ll_color  = [[cb[3],cb[2],cb[1],cb[0]], [cb2[3],cb2[2],cb2[1],cb2[0]], [cb[-4],cb[-3],cb[-2],cb[-1]]] # blue, purple, red 
            color_counts = [cb[0],cb[1],cb[2],cb[3]] # blue colors, from intense to light
        else: # plot case 2a, control robustness
            ll_set_names = [1,4]
            ll_k_type =  ['','triangle']
            ll_c_type = (['counts_leaky'] if plot_leaky else ['counts']) + (['simple'] if plot_simple else ['complex'])
            ll_names  = (['Count-based'] if plot_leaky else ['Counts']) + ['Similarity-based'] 
            ll_color  = [[cb[3],cb[2],cb[1],cb[0]], [cb[-4],cb[-3],cb[-2],cb[-1]]] # red colors, from intense to light
            color_counts = [cb[0],cb[1],cb[2],cb[3]] # blue colors, from intense to light

        # Variables to plot in grid search plots
        if plot_case1:
            plot_var = [[('k_alph' if 'leaky' in ll_c_type[0] else 'eps'),'num_states_model']]  + [['knum','cdens','k_alph']]*2 #if plot_simple else [['knum','complex_perc','type_complex']]
            xl       = [[('Learning rate' if 'leaky' in ll_c_type[0] else 'Prior'),'# states']] + [['Conv. kernels','Conv. density','Learning rate']]*2 #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            xt       = [[[0,0.5,1] if 'leaky' in ll_c_type[0] else [0.01,1,10,20],[13,23,63]]] + [[[2,10,20,40],[4,8,16,32],[0.01,0.5,0.9]]]*2 #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]
        elif plot_case2:
            plot_var = [[('k_alph' if 'leaky' in ll_c_type[0] else 'eps')]]  + [['k_alph']]*2 #if plot_simple else [['knum','complex_perc','type_complex']]
            xl       = [[('Leakiness' if 'leaky' in ll_c_type[0] else 'Prior')]] + [['Leakiness']]*2 #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            xt       = [[[0,0.5,1] if 'leaky' in ll_c_type[0] else [0.01,1,10,20]]] + [[[0,0.5,1]]]*2 #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]
        elif plot_case2a:
            plot_var = [[('k_alph' if 'leaky' in ll_c_type[0] else 'eps')]]  + [['k_alph']] #if plot_simple else [['knum','complex_perc','type_complex']]
            xl       = [[('Leakiness' if 'leaky' in ll_c_type[0] else 'Prior')]] + [['Leakiness']] #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            xt       = [[[0,0.5,1] if 'leaky' in ll_c_type[0] else [0.01,1,10,20]]] + [[[0,0.5,1]]] #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]
        elif plot_case3:
            plot_var = ['k_alph'] #if plot_simple else [['knum','complex_perc','type_complex']]
            xl       = ['Learning rate'] #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            xt       = [[0.01,0.5,0.99]] #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]
        else:    
            plot_var = [[('k_alph' if 'leaky' in ll_c_type[0] else 'eps'),'num_states_model']]  + [['knum','cdens','k_alph']] #if plot_simple else [['knum','complex_perc','type_complex']]
            xl       = [[('Learning rate' if 'leaky' in ll_c_type[0] else 'Prior'),'# states']] + [['Conv. kernels','Conv. density','Learning rate']] #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            xt       = [[[0,0.5,1] if 'leaky' in ll_c_type[0] else [0.01,1,10,20],[13,23,63]]] + [[[2,10,20,40],[4,8,16,32],[0.01,0.5,0.9]]] #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]

        # Plot best-model simulation panels
        # if fixed_sim:
        #     fixed_sim_path = ['/Users/sbecker/Projects/RL_reward_novelty/data/2024-08_grid_search_manual_cluster/set1_cnov_leaky_corr/grid_4',
        #                       '/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual_corr/best_complex_cell_seqsep/complex_cells-corr_triangle_adj_w3_G-40/grid_0']
        #     fit_name = ['',
        #                 '_corr-lp0']
        #     best_params = plot_best_sim(ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,ll_names=ll_names,plot_meas=plot_meas,error_type='sem',type_best_worst='overall',color_counts=color_counts,fixed_sim_path=fixed_sim_path,fit_name=fit_name)
        # else:
        #     best_params = plot_best_sim(ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,ll_names=ll_names,plot_meas=plot_meas,error_type='sem',type_best_worst='overall',color_counts=color_counts)

        # Plot average simulation panels
        # average_params = plot_average_sim(ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,ll_names=ll_names, plot_homann=True, inset=True, corr=True, stat_type='mean', plot_samples=True, ll_set_names=ll_set_names)
        # average_params = plot_average_sim_sepfit(ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,ll_names=ll_names, plot_homann=True, inset=True, corr=True, stat_type='mean', plot_samples=True, ll_set_names=ll_set_names)

        # Specs for best-sim plots: y-limits and legend position
        if plot_case2 and plot_simple:
            ylim = [[0,0.05],[0,0.07],[0,0.06],[0.04,0.1]]
            legend_pos = (0.2,0)
        else:
            ylim = None
            legend_pos = None
        
        #####################################################################################################################
        #          Plot best simulation                                                                                     #
        #####################################################################################################################
        if plot_case2a or plot_case2:
            savepath_bestparams = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual_corr/data_for_figures/'
            savename_bestparams = f'best_params{"_with-simple" if plot_simple else ""}{add_name}.json'
            if plot_bestsim:
                # Plot best simulation panels
                best_params = plot_average_sim_sepfit(ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,ll_names=ll_names, plot_homann=True, inset=True, corr=True, stat_type='min', plot_samples=True, ll_set_names=ll_set_names, plot_meas=plot_meas, plot_vars=plot_var, bootstrap=False, jackknife=True, ylim=ylim, legend_pos=legend_pos, plain_mse=plain_mse)

                sl.make_long_dir(savepath_bestparams)
                with open(os.path.join(savepath_bestparams,savename_bestparams), 'w') as fout:
                    json.dump(best_params, fout)
            else:
                savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/data/2024-08_grid_search_manual_corr/data_for_figures/'
                with open(os.path.join(savepath_bestparams,savename_bestparams), 'r') as fin:
                    best_params = json.load(fin)

        # Plot error panels knov
        # for i in range(len(plot_var)):
        #     plot_single_var(plot_meas=plot_meas,plot_var=plot_var[i],ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,xl=xl[i],yl=yl,xt=xt[i],yt=yt,color_counts=color_counts,best_params=best_params)

        # # Plot tradeoff between M and L' experiments
        # ll_params = ['knum','cdens','k_alph'] if plot_simple else ['knum','num_complex','type_complex']
        # ll_param_names = ['Conv. kernels','Conv. density','Learning rate'] if plot_simple else ['Conv. kernels','Ratio complex','Type complex']
        # xt_params = [[2,10,20,40],[4,8,16,32],[0.01,0.5,0.9]] if plot_simple else [[2,10,20,40],[0.25,1,4],[2,4,6,8]]
        # # yt_params_all = [[0,4,8],[0,2,4]] if plot_simple else [[0,5,10],[0,2,4]]
        # # yl_params_all = [[0,8],[0,4]] if plot_simple else [[0,10],[0,4]]

        # yt_params_all = [[0,2,4]] 
        # yl_params_all = [[0,4]] 
        # for i in range(len(ll_k_type)):
        #     for j in range(len(ll_params)):
        #         yl_params = yl_params_all[i]  
        #         yt_params = yt_params_all[i]  
        #         plot_tradeoff_m_lp(c_type=ll_c_type[i],k_type=ll_k_type[i],ll_params=[ll_params[j]],ll_colors=ll_color[i],ll_names=[ll_param_names[j]],plot_meas=['mse_trec','mse_tmem','mse_steady'],pm_lines=['-','--'],pm_labels=['L\'-exp.','M-exp.'],xt=xt_params[j],yt=yt_params,yl=yl_params,best_params=best_params)

        if (plot_case2a or plot_case2) and control_robustness:
            if plot_case2a:
                ll_set_names_rob = [1,4]
            elif plot_case2 and plot_simple:
                ll_set_names_rob = [1,2,4]
            plot_var_rob = [[('k_alph' if 'leaky' in ll_c_type[0] else 'eps'),'num_states_model']]  + [['knum','cdens','k_alph']]*2 #if plot_simple else [['knum','complex_perc','type_complex']]
            xl_rob       = [[('Leakiness' if 'leaky' in ll_c_type[0] else 'Prior'),'# states']] + [['Conv. kernels','Conv. density','Leakiness']]*2 #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            xt_rob       = [[[0,0.5,1] if 'leaky' in ll_c_type[0] else [0.01,1,10,20],[13,23,63]]] + [[[2,10,20,40],[4,8,16,32],[0.01,0.5,0.9]]]*2 #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]

            # ll_set_names_rob = [1,4]
            # plot_var_rob = [[('k_alph' if 'leaky' in ll_c_type[0] else 'eps'),'num_states_model']]  + [['knum','complex_perc','type_complex']]*2 #if plot_simple else [['knum','complex_perc','type_complex']]
            # xl_rob       = [[('Learning rate' if 'leaky' in ll_c_type[0] else 'Prior'),'# states']] + [['Conv. kernels','% complex cells','Freq. components']]*2 #if plot_simple else [['Conv. kernels','% complex cells','Freq. components']]
            # xt_rob       = [[[0,0.5,1] if 'leaky' in ll_c_type[0] else [0.01,1,10,20],[13,23,63]]] + [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]*2 #if plot_simple else [[[2,10,20,40],[0.2,0.5,0.8],[2,4,6,8]]]

            # Plot error panels cnov into single figure
            for j in range(len(ll_set_names_rob)):
                fig, ax = plt.subplots(1, len(plot_var_rob[j]), figsize=(2*len(plot_var_rob[j]),2)) #figsize=(2.25,1.9)
                # fig = plt.figure(figsize=(2*len(plot_var_rob[j]),2))
                # gs = gridspec.GridSpec(1, len(plot_var_rob[j]), figure=fig)
                # if plain_mse:
                #     bax = []
                #     for i_gs in range(len(plot_var_rob[j])):
                #         bax_i = brokenaxes(ylims=ybreaks, subplot_spec=gs[i_gs])
                #         bax.append(bax_i)
                #     ax = bax
                for i in range(len(plot_var_rob[j])):
                    savepath, figname = plot_single_var(plot_meas=plot_meas, plot_var=plot_var_rob[j][i], ll_c_type=[ll_c_type[j]], ll_k_type=[ll_k_type[j]], ll_color=[ll_color[j]], xl=xl_rob[j][i], yl=yl ,xt=xt_rob[j][i], yt=yt[0], color_counts=color_counts, best_params=best_params, color_other=ll_color, f=fig, ax=ax[i], legend=i==0, ll_set_names=[ll_set_names_rob[j]], add_name=add_name, bootstrap=False, jackknife=True, plain_mse=plain_mse) #,broken_axes=ybreaks)
                    # if plain_mse and plot_var_rob[j][i]=='k_alph':
                    #     # make inset
                    #     ax_inset = ax[i].inset_axes([0.4,0.3,0.5,0.4])
                    #     yl_inset = [[0,1e-4,1e-3]]
                    #     # plot full data into inset
                    #     plot_single_var(plot_meas=plot_meas, plot_var=plot_var_rob[j][i], ll_c_type=[ll_c_type[j]], ll_k_type=[ll_k_type[j]], ll_color=[ll_color[j]], xl=xl_rob[j][i], yl=yl_inset ,xt=xt_rob[j][i], yt=yt[0], color_counts=color_counts, best_params=best_params, color_other=ll_color, f=fig, ax=ax_inset, legend=i==0, ll_set_names=[ll_set_names_rob[j]], add_name=add_name, bootstrap=False, jackknife=True, plain_mse=plain_mse)
                    #     ax_inset.set_ylabel('')

                [ax[i].set_ylabel('') for i in range(1,len(plot_var_rob[j]))]
                fig.legend(ncol=5+len(ll_c_type)-1,loc='upper center',bbox_to_anchor=(0.52,1.07),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
                fig.tight_layout()
                fig.savefig(os.path.join(savepath,figname), bbox_inches='tight')

            # Plot error panels knov (simple) into single figure
            # fig,ax = plt.subplots(1,len(plot_var_rob[1]),figsize=(2*len(plot_var_rob[1]),2)) #figsize=(2.25,1.9)
            # for i in range(len(plot_var_rob[1])):
            #     savepath, figname = plot_single_var(plot_meas=plot_meas,plot_var=plot_var_rob[1][i],ll_c_type=[ll_c_type[1]],ll_k_type=[ll_k_type[1]],ll_color=[ll_color[1]],xl=xl_rob[1][i],yl=yl,xt=xt_rob[1][i],yt=yt[0],color_counts=color_counts,best_params=None,color_other=ll_color,f=fig,ax=ax[i],legend=i==0,ll_set_names=[ll_set_names_rob[1]],add_name=add_name)
            # [ax[i].set_ylabel('') for i in range(1,len(plot_var_rob[1]))]
            # fig.legend(ncol=5+len(ll_c_type)-1,loc='upper center',bbox_to_anchor=(0.52,1.07),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
            # fig.tight_layout()
            # fig.savefig(os.path.join(savepath,figname), bbox_inches='tight') 

        # Plot error panels knov and cnov into single figure
        if plot_case1:
            for i in range(len(plot_var)):
                num_panels      = len(plot_var[i])
                all_plot_var    = plot_var[i]
                all_ll_c_type   = [ll_c_type[i]]*len(plot_var[i])
                all_ll_k_type   = [ll_k_type[i]]*len(plot_var[i]) 
                all_ll_color    = [ll_color[i]]*len(plot_var[i]) 
                all_ll_names    = [ll_set_names[i]]*len(plot_var[i]) 
                all_xl          = xl[i] 
                all_xt          = xt[i] 
                all_yt          = [yt[0]]*len(plot_var[i]) 
                add_legend      = [True] + [False]*(len(plot_var[i])-1)
                fig,ax = plt.subplots(1,num_panels,figsize=(2*num_panels,2)) #figsize=(2.25,1.9)
                for ii in range(num_panels):
                    sp, fn = plot_single_var(plot_meas=plot_meas,plot_var=all_plot_var[ii],ll_c_type=[all_ll_c_type[ii]],ll_k_type=[all_ll_k_type[ii]],ll_color=[all_ll_color[ii]],xl=all_xl[ii],yl=yl,xt=all_xt[ii],yt=all_yt[ii],color_counts=color_counts,best_params=None,color_other=ll_color,f=fig,ax=ax[ii],legend=add_legend[ii], ll_set_names=[all_ll_names[ii]])
                    if ii==0:
                        savepath = sp; figname = fn
                [ax[ii].set_ylabel('') for ii in range(1,num_panels)]
                fig.legend(ncol=8,loc='upper center',bbox_to_anchor=(0.52,1.07),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
                fig.tight_layout()
                fig.savefig(savepath + figname, bbox_inches='tight')

        elif plot_case2 and plot_simple:
            fig,axl = plt.subplots(1,1,figsize=(3.3,2)) #figsize=(2.25,1.9)

            # Plot best av. MSE
            yl_best = f'Best {yl.split(" ")[0].lower()} {" ".join(yl.split(" ")[1:])}'
            plot_bar_best(axl,best_params,plot_meas,yl_best,ll_color,ll_names)
            axl.spines[['right','top']].set_visible(False)

            save_str = 'plot2_'
            for i in range(len(ll_c_type)):
                if 'count' in ll_c_type[i]:
                    save_str += f'{ll_c_type[i]}-{ll_set_names[i]}_'
                else:
                    save_str += f'{ll_k_type[i]}-{ll_c_type[i]}-{ll_set_names[i]}_'
            savepath = '/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures'
            savename = f'{save_str}_mse-bars{add_name}.svg'
            fig.tight_layout()
            fig.savefig(os.path.join(savepath,savename))
                
        elif plot_case2:
            num_panels = np.sum([len(plot_var[i]) for i in range(len(plot_var))])
            fig,axl = plt.subplots(1,num_panels+1,figsize=(8.3,2),gridspec_kw={'width_ratios':[2,2,2,1]}) #figsize=(2.25,1.9)
            i_ax = 0
            for i in range(len(plot_var)):
                all_plot_var    = plot_var[i]
                all_ll_c_type   = [ll_c_type[i]]*len(plot_var[i])
                all_ll_k_type   = [ll_k_type[i]]*len(plot_var[i]) 
                all_ll_color    = [ll_color[i]]*len(plot_var[i]) 
                all_ll_names    = [ll_set_names[i]]*len(plot_var[i]) 
                all_xl          = xl[i] 
                all_xt          = xt[i] 
                all_yt          = [yt[0]]*len(plot_var[i]) 
                add_legend_col  = [True] + [False]*(len(plot_var[i])-1)
                for ii in range(len(plot_var[i])):
                    sp, fn = plot_single_var(plot_meas=plot_meas,plot_var=all_plot_var[ii],ll_c_type=[all_ll_c_type[ii]],ll_k_type=[all_ll_k_type[ii]],ll_color=[all_ll_color[ii]],xl=all_xl[ii],yl=yl,xt=all_xt[ii],yt=all_yt[ii],color_counts=color_counts,best_params=None,color_other=ll_color,f=fig,ax=axl[i_ax],legend_color=add_legend_col[ii],legend_symbol=i==0,ll_set_names=[all_ll_names[ii]],ll_names=[ll_names[i]])
                    i_ax += 1
            save_str = 'plot2_'
            if 'count' in ll_c_type[i]:
                save_str += f'{ll_c_type[i]}-{ll_set_names[i]}_'
            else:
                save_str += f'{ll_k_type[i]}-{ll_c_type[i]}-{ll_set_names[i]}_'
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures/'
            figname = f'{save_str}adj_w3_G-40.svg'
            sl.make_long_dir(savepath)
            [axl[i].set_ylabel('') for i in range(1,num_panels)]
            axl[-1].spines[['right','top','left','bottom']].set_visible(False)
            axl[-1].set_xticks([]); axl[-1].set_yticks([])
            fig.legend(ncol=1,loc='center right',bbox_to_anchor=(1.01,0.5),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
            fig.tight_layout()
            fig.savefig(os.path.join(savepath,figname), bbox_inches='tight')

        elif plot_case2a:
            num_panels = np.sum([len(plot_var[i]) for i in range(len(plot_var))])
            fig,axl = plt.subplots(1,num_panels+2,figsize=(8.3,2),gridspec_kw={'width_ratios':[2,2,2,1]}) #figsize=(2.25,1.9)

            # Plot best av. MSE
            yl_best = f'Best {yl.split(" ")[0].lower()} {" ".join(yl.split(" ")[1:])}'
            plot_bar_best(axl[0],best_params,plot_meas,yl_best,ll_color,ll_names)
            axl[0].spines[['right','top']].set_visible(False)

            # Plot grid search av. MSE
            i_ax = 1
            for i in range(len(plot_var)):
                all_plot_var    = plot_var[i]
                all_ll_c_type   = [ll_c_type[i]]*len(plot_var[i])
                all_ll_k_type   = [ll_k_type[i]]*len(plot_var[i]) 
                all_ll_color    = [ll_color[i]]*len(plot_var[i]) 
                all_ll_names    = [ll_set_names[i]]*len(plot_var[i]) 
                all_xl          = xl[i] 
                all_xt          = xt[i] 
                all_yt          = [yt[0]]*len(plot_var[i]) 
                add_legend_col  = [True] + [False]*(len(plot_var[i])-1)
                for ii in range(len(plot_var[i])):
                    sp, fn = plot_single_var(plot_meas=plot_meas,plot_var=all_plot_var[ii],ll_c_type=[all_ll_c_type[ii]],ll_k_type=[all_ll_k_type[ii]],ll_color=[all_ll_color[ii]],xl=all_xl[ii],yl=yl,xt=all_xt[ii],yt=all_yt[ii],color_counts=color_counts,best_params=None,color_other=ll_color,f=fig,ax=axl[i_ax],legend_color=add_legend_col[ii],legend_symbol=i==0,ll_set_names=[all_ll_names[ii]],ll_names=[ll_names[i]],add_name=add_name,bootstrap=False,jackknife=True,plain_mse=plain_mse)
                    i_ax += 1
            [axl[i].set_ylabel('') for i in range(2,num_panels)]
            axl[-1].spines[['right','top','left','bottom']].set_visible(False)
            axl[-1].set_xticks([]); axl[-1].set_yticks([])
            fig.legend(ncol=1,loc='center right',bbox_to_anchor=(1,0.5),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
            fig.tight_layout()

            # Save plot
            save_str = 'plot2a_'
            if 'count' in ll_c_type[i]:
                save_str += f'{ll_c_type[i]}-{ll_set_names[i]}_'
            else:
                save_str += f'{ll_k_type[i]}-{ll_c_type[i]}-{ll_set_names[i]}_'
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures/'
            figname = f'{save_str}adj_w3_G-40{add_name}.svg'
            sl.make_long_dir(savepath)
            fig.savefig(os.path.join(savepath,figname), bbox_inches='tight')
        
        elif plot_case3:
            num_panels = len(plot_meas_ll)
            fig,axl = plt.subplots(1,num_panels,figsize=(8.3,2)) #figsize=(2.25,1.9)
            for i in range(len(plot_meas_ll)):
                sp, fn = plot_single_var(plot_meas=plot_meas_ll[i],plot_var=plot_var[0],ll_c_type=ll_c_type,ll_k_type=ll_k_type,ll_color=ll_color,xl=xl[0],yl=yl,xt=xt[0],yt=yt[i],color_counts=color_counts,best_params=None,color_other=ll_color,f=fig,ax=axl[i],legend_color=i==0,ll_set_names=ll_set_names,ll_names=ll_names,plot_sep_exp=True)
            save_str = 'plot3_'
            for i in range(len(plot_var)):
                if 'count' in ll_c_type[i]:
                    save_str += f'{ll_c_type[i]}-{ll_set_names[i]}_'
                else:
                    save_str += f'{ll_k_type[i]}-{ll_c_type[i]}-{ll_set_names[i]}_'
            savepath = f'/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures/'
            figname = f'{save_str}adj_w3_G-40.svg'
            sl.make_long_dir(savepath)
            [axl[i].set_ylabel('') for i in range(1,num_panels)]
            fig.tight_layout()
            fig.savefig(os.path.join(savepath,figname), bbox_inches='tight')

        # else:
        #     num_panels      = len(plot_var[1])+len(plot_var[0])
        #     all_plot_var    = plot_var[1] + plot_var[0]
        #     all_ll_c_type   = [ll_c_type[1]]*len(plot_var[1]) + [ll_c_type[0]]*len(plot_var[0])
        #     all_ll_k_type   = [ll_k_type[1]]*len(plot_var[1]) + [ll_k_type[0]]*len(plot_var[0])
        #     all_ll_color    = [ll_color[1]]*len(plot_var[1]) + [ll_color[0]]*len(plot_var[0])
        #     ll_set_names    = [ll_set_names[1]]*len(plot_var[1]) + [ll_set_names[0]]*len(plot_var[0])
        #     all_xl          = xl[1] + xl[0]
        #     all_xt          = xt[1] + xt[0]
        #     all_yt          = [yt[0]]*len(plot_var[1]) + [yt[0]]*len(plot_var[0])
        #     add_legend      = [True] + [False]*(len(plot_var[1])-1) + [True] + [False]*(len(plot_var[0])-1)
        #     fig,ax = plt.subplots(1,num_panels,figsize=(2*num_panels,2)) #figsize=(2.25,1.9)
        #     for i in range(num_panels):
        #         sp, fn = plot_single_var(plot_meas=plot_meas,plot_var=all_plot_var[i],ll_c_type=[all_ll_c_type[i]],ll_k_type=[all_ll_k_type[i]],ll_color=[all_ll_color[i]],xl=all_xl[i],yl=yl,xt=all_xt[i],yt=all_yt[i],color_counts=color_counts,best_params=None,color_other=ll_color,f=fig,ax=ax[i],legend=add_legend[i], ll_set_names=[ll_set_names[i]])
        #         if i==0:
        #             savepath = sp; figname = fn
        #     [ax[i].set_ylabel('') for i in range(1,num_panels)]
        #     fig.legend(ncol=8,loc='upper center',bbox_to_anchor=(0.52,1.07),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
        #     fig.tight_layout()
        #     fig.savefig(savepath + figname, bbox_inches='tight')

        # if plot_case2a:
            # nfam_plot = [1,3,8]
            # recomp_sim = False

            # for j in range(len(best_params)):
            #     str_types = f'plot2a_{best_params[j][0]}'
            #     f,ax = plt.subplots(1,2,figsize=(8.3/2,2.3))
            #     # Get best and comparison alpha for each model
            #     best_alph = 0.1 #best_params[j][1]['k_alph']
            #     other_alph = 0.5
            #     loadpath, savepath, str_types = get_paths(ll_k_type[j],ll_c_type[j],True,ll_set_names[j],str_types)
            #     gridvars = get_grid_vars(ll_c_type[j],set_name=ll_set_names[j])
            #     grid, _, _ = get_grid(loadpath,plot_meas,gridvars)
            #     if 'count' in ll_c_type[j]:
            #         grid['k_alph'] = np.round(1-grid['k_alph'],6) # convert to learning rate
            #     best_id = grid.loc[grid.k_alph==best_alph,'grid_id'].values[0]
            #     other_id = grid.loc[grid.k_alph==other_alph,'grid_id'].values[0]

            #     # Simulate data for given parameter combination
            #     if not 'count' in ll_c_type[j] and recomp_sim:
            #         run_gabor_model(loadpath,best_id,grid,no_complex_cells=('complex' in ll_c_type[j]))
            #         run_gabor_model(loadpath,other_id,grid,no_complex_cells=('complex' in ll_c_type[j]))

            #     # Load and fit data
            #     homann_data = gsc.load_exp_homann(cluster=False)
            #     homann_data[1][1][0] = 0    # set first value to 0
            #     save_name='_corr-lp0'

            #     data_names = ['l','lp','m']
            #     data_var = ['n_fam','dN','n_im']
            #     data_val = ['nt_norm','tr_norm','nt_norm']
                
            #     if 'count' in ll_c_type[j]:
            #         sim_best, pred_best, coef_best, shift_best, pd_best_all, stats_best_all = fit_cnov(homann_data,loadpath,best_id,data_names,data_var,data_val,save_name)
            #         sim_other, pred_other, coef_other, shift_other, pd_other_all, stats_other_all = fit_cnov(homann_data,loadpath,other_id,data_names,data_var,data_val,save_name)
            #     else:
            #         sim_best, pred_best, coef_best, shift_best, pd_best_all, stats_best_all = fit_knov(homann_data,loadpath,best_id,data_names,data_var,data_val,save_name)
            #         sim_other, pred_other, coef_other, shift_other, pd_other_all, stats_other_all = fit_knov(homann_data,loadpath,other_id,data_names,data_var,data_val,save_name)

            #     # Plot fitted data
            #     pd_best_plot = pd_best_all[0].loc[pd_best_all[0].n_fam.isin(nfam_plot)]
            #     pd_other_plot = pd_other_all[0].loc[pd_other_all[0].n_fam.isin(nfam_plot)]
            #     pd_best_plot['novelty_fitted'] = pd_best_plot['novelty']*coef_best + shift_best
            #     pd_other_plot['novelty_fitted'] = pd_other_plot['novelty']*coef_other + shift_other

            #     stats_best_plot = stats_best_all[0].loc[stats_best_all[0].n_fam.isin(nfam_plot)]
            #     stats_other_plot = stats_other_all[0].loc[stats_other_all[0].n_fam.isin(nfam_plot)]
            #     stats_best_plot['nt_norm_fitted'] = stats_best_plot['nt_norm']*coef_best
            #     stats_best_plot['steady_fitted'] = stats_best_plot['steady']*coef_best + shift_best
            #     stats_other_plot['nt_norm_fitted'] = stats_other_plot['nt_norm']*coef_other
            #     stats_other_plot['steady_fitted'] = stats_other_plot['steady']*coef_other + shift_other

            #     col_raw = ll_color[j][1:]
            #     plot_l(pd_best_plot,stats_best_plot,f=f,ax=ax[0],col_raw=col_raw,legend=True,yl=ll_names[j])
            #     plot_l(pd_other_plot,stats_other_plot,f=f,ax=ax[1],col_raw=col_raw,yl=ll_names[j])
            
            #     ax[0].set_title(f'Slow update ($\\alpha$={best_alph})',fontsize=12)
            #     ax[1].set_title(f'Fast update ($\\alpha$={other_alph})',fontsize=12)
            #     ax[1].set_ylabel('')
            #     ax[0].set_xlabel('Time steps')
            #     ax[1].set_xlabel('Time steps')

            #     f.legend(loc='upper center',bbox_to_anchor=(0.35,1.1),ncol=len(nfam_plot),fontsize=10,frameon=False,handlelength=1.2,handletextpad=0.4,columnspacing=0.8)
            #     f.tight_layout()
            #     f.savefig(os.path.join('/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures',f'{str_types}_fast-slow.svg'),bbox_inches='tight')

            # print('done')

        # Plot simulations for best / control parameter sets
        if plot_case2a:
            recomp_sim = False

            for j in range(len(best_params)):
                if not 'count' in ll_c_type[j]:
                    str_types = f'plot2a_{best_params[j][0]}'
                
                    # Get best model parameters
                    loadpath, savepath, str_types = get_paths(ll_k_type[j],ll_c_type[j],True,ll_set_names[j],str_types)
                    gridvars = get_grid_vars(ll_c_type[j],set_name=ll_set_names[j])
                    grid, _, _ = get_grid(loadpath,plot_meas,gridvars,bootstrap=False,jackknife=True,plain_mse=plain_mse)

                    best_params_j = best_params[j][1]
                    best_id = grid.loc[(grid.k_alph==best_params_j['k_alph']) & (grid.cdens==best_params_j['cdens']) & (grid.knum==best_params_j['knum']),'grid_id'].values[0]
        
                    # Simulate data for given parameter combination
                    if recomp_sim:
                        run_gabor_model(loadpath,best_id,grid,no_complex_cells=('complex' in ll_c_type[j]),num_sim=10)

                    data_names = ['l','lp','m']
                    data_var = ['n_fam','dN','n_im']
                    data_val = ['nt_norm','tr_norm','nt_norm']
                    
                    pd_best_all = []
                    for i in range(len(data_names)):
                        pd_best = pd.read_csv(os.path.join(loadpath,f'grid_{best_id}_detailedsim/sim_data_all_{data_names[i]}.csv'))
                        pd_best.rename(columns={'Unnamed: 0':'time_step','nt':'novelty','stim_type':'type'},inplace=True)
                        pd_best_all.append(pd_best)
                        
                    # Plot L-experiment
                    nfam_plot = [pd_best_all[0].n_fam.unique()[2]]
                    pd_best_plot = pd_best_all[0].loc[pd_best_all[0].n_fam.isin(nfam_plot)]
                    f,ax = plt.subplots(1,figsize=(8.3/2,2))
                    all_seed = pd_best_plot.seed.unique()
                    for i_s, s in enumerate(all_seed):
                        pd_best_plot_s = pd_best_plot.loc[pd_best_plot.seed==s]
                        ax.plot(pd_best_plot_s.time_step,pd_best_plot_s.novelty,color='k',alpha=0.1)
                    pd_best_mean = pd_best_plot.groupby(['time_step']).agg(novelty=('novelty','mean')).reset_index()
                    ax.plot(pd_best_mean.time_step,pd_best_mean.novelty,color='k',linewidth=2)

                    ax.set_ylabel('Predicted responses')
                    ax.set_xlabel('Time steps')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    f.tight_layout()
                    f.savefig(os.path.join('/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures',f'{str_types}_traces-l.svg'))

                    # Plot L'-experiment
                    dN_plot = [pd_best_all[1].dN.unique()[1]]
                    pd_best_plot = pd_best_all[1].loc[pd_best_all[1].dN.isin(dN_plot)]
                    f,ax = plt.subplots(1,figsize=(8.3/2,2))
                    all_seed = pd_best_plot.seed.unique()
                    for i_s, s in enumerate(all_seed):
                        pd_best_plot_s = pd_best_plot.loc[pd_best_plot.seed==s]
                        ax.plot(pd_best_plot_s.time_step,pd_best_plot_s.novelty,color='k',alpha=0.1)
                    pd_best_mean = pd_best_plot.groupby(['time_step']).agg(novelty=('novelty','mean')).reset_index()
                    ax.plot(pd_best_mean.time_step,pd_best_mean.novelty,color='k',linewidth=2)
                
                    ax.set_ylabel('Predicted responses')
                    ax.set_xlabel('Time steps')
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    f.tight_layout()
                    f.savefig(os.path.join('/Volumes/lcncluster/becker/RL_reward_novelty/output/2024-08_grid_search_manual_corr/figures',f'{str_types}_traces-lp.svg'))

            print('done')

        print('done')







