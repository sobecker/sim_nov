import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys

import utils.saveload as sl
import utils.visualization as vis

if __name__=="__main__":
    plt.style.use('/Users/sbecker/Projects/RL_reward_novelty/src/scripts/Figures_Paper/paper.mplstyle')

    # Set paths
    # folder_name = '2024-11_exp-pred1_sum-of-parts_robustness-m' # simple cells (no permutations)
    folder_name = '2025-01_exp-pred1-simple_sum-of-parts_robustness-m' # simple cells (with permutations)
    # folder_name = '2025-01_exp-pred1-complex_sum-of-parts_robustness-m' # complex cells (no permutations)

    path_data = path_data = f'/Users/sbecker/Projects/RL_reward_novelty/data/{folder_name}/'
    sl.make_long_dir(path_data)

    path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/{folder_name}/'
    sl.make_long_dir(path_fig)

    ########################################################################################################################
    # Run M-experiment for different alpha values
    test_alphas_all = np.arange(0.1,1,0.1)
    test_alphas1    = np.arange(0.1,0.6,0.1)
    test_alphas2    = np.arange(0.5,1,0.1)

    color_list = vis.prep_cmap('coolwarm',len(test_alphas_all))
    color_list1 = color_list[:len(test_alphas1)]
    color_list2 = color_list[len(test_alphas1):]
    
    rotate      = np.array([0, 0.5, 0.509, 0.6, 1])*np.pi
    rotate_plot = np.array([0,0.2])*np.pi
    seed_parent = 12345

    name_fig1    = f'm_exp_seed-{seed_parent}_alph-up-to-05'.replace('.','-')
    name_fig2    = f'm_exp_seed-{seed_parent}_alph-above-05'.replace('.','-')

    # Load all data
    sim_stats_raw = []
    for i_alph, alph in enumerate(test_alphas_all):
        
        name_data   = f'm_exp_seed-{seed_parent}_alph-{np.round(alph,6)}'.replace('.','-')
        
        exp_type = 'm'

        field_vals   = 'n_fam' if exp_type=='l' else 'n_im' if exp_type=='m' else 'dN'
        field_stats  = 'tr_norm_mean' if exp_type=='lp'  else 'nt_norm_mean'
        field_stats2 = 'steady_mean' 
        field_stats3 = 'tr_mean' if exp_type=='lp' else 'nt_mean'

        col_extract = col_extract = [field_vals,'tr_mean','tr_std','tr_sem','tr_norm_mean','tr_norm_std','tr_norm_sem','steady_mean','steady_std','steady_sem','sample_id','parent_id','perm_id','rotate','ksim'] if exp_type=='lp' else [field_vals,'nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','steady_mean','steady_std','steady_sem','sample_id','parent_id','perm_id','rotate','ksim']

        sim_stats_raw_alph = pd.read_csv(os.path.join(path_data,f'{name_data}.csv'),header=1,names=col_extract,index_col=False)
        sim_stats_raw_alph['alph'] = alph
        sim_stats_raw.append(sim_stats_raw_alph)
    sim_stats_raw = pd.concat(sim_stats_raw)

    # Plot all data (part 1)
    fig,axl   = plt.subplots(1,len(rotate),figsize=(8.3,2.5))
    fig2,axl2 = plt.subplots(1,len(rotate),figsize=(8.3,2.5))
    fig3,axl3 = plt.subplots(1,len(rotate),figsize=(8.3,2.5))

    for i in range(len(rotate)):
        stats_i = sim_stats_raw[np.round(sim_stats_raw['rotate'],6)==np.round(rotate[i],6)]
        nov_i_mean = stats_i[['alph'] + [field_vals,field_stats]].groupby(['alph'] + [field_vals]).mean().reset_index()
        nov_i_std = stats_i[['alph'] + [field_vals,field_stats]].groupby(['alph'] + [field_vals]).std().reset_index()

        nov_i_mean2 = stats_i[['alph'] + [field_vals,field_stats2]].groupby(['alph'] + [field_vals]).mean().reset_index()
        nov_i_std2 = stats_i[['alph'] + [field_vals,field_stats2]].groupby(['alph'] + [field_vals]).std().reset_index()

        nov_i_mean3 = stats_i[['alph'] + [field_vals,field_stats3]].groupby(['alph'] + [field_vals]).mean().reset_index()
        nov_i_std3 = stats_i[['alph'] + [field_vals,field_stats3]].groupby(['alph'] + [field_vals]).std().reset_index()

        rotsim_i = np.round(rotate[i]/np.pi,2)
        label_i = f'{int(np.round(rotsim_i))}$\pi$' if rotsim_i%1==0 else f'{rotsim_i}$\pi$'

        ax = axl[i]; ax2 = axl2[i]; ax3 = axl3[i]
        for j in range(len(test_alphas1)):
            nov_i_mean_alph = nov_i_mean.loc[np.round(nov_i_mean['alph'],6)==np.round(test_alphas1[j],6)]
            nov_i_std_alph  = nov_i_std.loc[np.round(nov_i_std['alph'],6)==np.round(test_alphas1[j],6)]
            ax.plot(nov_i_mean_alph[field_vals],nov_i_mean_alph[field_stats],'-',c=color_list1[j])
            ax.plot(nov_i_mean_alph[field_vals],nov_i_mean_alph[field_stats],'o',c=color_list1[j],label=f'{np.round(test_alphas1[j],6)}')
            ax.fill_between(x=nov_i_std_alph[field_vals],y1=nov_i_mean_alph[field_stats]-nov_i_std_alph[field_stats],y2=nov_i_mean_alph[field_stats]+nov_i_std_alph[field_stats],color=color_list1[j],alpha=0.2)

            nov_i_mean2_alph = nov_i_mean2.loc[np.round(nov_i_mean2['alph'],6)==np.round(test_alphas1[j],6)]
            nov_i_std2_alph  = nov_i_std2.loc[np.round(nov_i_std2['alph'],6)==np.round(test_alphas1[j],6)]
            ax2.plot(nov_i_mean2_alph[field_vals],nov_i_mean2_alph[field_stats2],'-',c=color_list1[j])
            ax2.plot(nov_i_mean2_alph[field_vals],nov_i_mean2_alph[field_stats2],'o',c=color_list1[j],label=f'{np.round(test_alphas1[j],6)}')
            ax2.fill_between(x=nov_i_std2_alph[field_vals],y1=nov_i_mean2_alph[field_stats2]-nov_i_std2_alph[field_stats2],y2=nov_i_mean2_alph[field_stats2]+nov_i_std2_alph[field_stats2],color=color_list1[j],alpha=0.2)

            nov_i_mean3_alph = nov_i_mean3.loc[np.round(nov_i_mean3['alph'],6)==np.round(test_alphas1[j],6)]
            nov_i_std3_alph  = nov_i_std3.loc[np.round(nov_i_std3['alph'],6)==np.round(test_alphas1[j],6)]
            ax3.plot(nov_i_mean3_alph[field_vals],nov_i_mean3_alph[field_stats3],'-',c=color_list1[j])
            ax3.plot(nov_i_mean3_alph[field_vals],nov_i_mean3_alph[field_stats3],'o',c=color_list1[j],label=f'{np.round(test_alphas1[j],6)}')
            ax3.fill_between(x=nov_i_std3_alph[field_vals],y1=nov_i_mean3_alph[field_stats3]-nov_i_std3_alph[field_stats3],y2=nov_i_mean3_alph[field_stats3]+nov_i_std3_alph[field_stats3],color=color_list1[j],alpha=0.2)
        
        # ax.legend(loc='upper left',title='Rotation',frameon=False,handletextpad=0.1,bbox_to_anchor=(1,1))
        ax.set_title(f'Rotation: {np.round(rotate[i]/np.pi,3)}$\pi$')
        ax2.set_title(f'Rotation: {np.round(rotate[i]/np.pi,3)}$\pi$')
        ax3.set_title(f'Rotation: {np.round(rotate[i]/np.pi,3)}$\pi$')
        if i==0:
            ax.set_ylabel('Novelty response')
            ax2.set_ylabel('Steady state novelty')
            ax3.set_ylabel('Raw novelty')
        else:
            ax.set_ylabel('')
            ax2.set_ylabel('')
            ax3.set_ylabel('')
        ax.set_xlabel('M')
        ax.set_xticks([3,6,9,12])
        ax2.set_xlabel('M')
        ax2.set_xticks([3,6,9,12])
        ax3.set_xlabel('M')
        ax3.set_xticks([3,6,9,12])
        if i==len(rotate)-1:
            ax.legend(loc='upper left',title='Leakiness $\\alpha$',frameon=False,handletextpad=0.1,bbox_to_anchor=(1.1,1))
            ax2.legend(loc='upper left',title='Leakiness $\\alpha$',frameon=False,handletextpad=0.1,bbox_to_anchor=(1.1,1))
            ax3.legend(loc='upper left',title='Leakiness $\\alpha$',frameon=False,handletextpad=0.1,bbox_to_anchor=(1.1,1))

    fig.tight_layout()
    fig.savefig(os.path.join(path_fig,f'{name_fig1}.png'))
    fig.savefig(os.path.join(path_fig,f'{name_fig1}.svg'))
    fig2.tight_layout()
    fig2.savefig(os.path.join(path_fig,f'{name_fig1}_steady.png'))
    fig2.savefig(os.path.join(path_fig,f'{name_fig1}_steady.svg'))
    fig3.tight_layout()
    fig3.savefig(os.path.join(path_fig,f'{name_fig1}_raw.png'))
    fig3.savefig(os.path.join(path_fig,f'{name_fig1}_raw.svg'))

    # Plot all data (part 1)
    fig,axl   = plt.subplots(1,len(rotate),figsize=(8.3,2.5)) #,gridspec_kw={'width_ratios': [1,1,1,1,1,0.5]})
    fig2,axl2 = plt.subplots(1,len(rotate),figsize=(8.3,2.5)) #,gridspec_kw={'width_ratios': [1,1,1,1,1,0.5]})
    fig3,axl3 = plt.subplots(1,len(rotate),figsize=(8.3,2.5)) #,gridspec_kw={'width_ratios': [1,1,1,1,1,0.5]})
    for i in range(len(rotate)):
        stats_i = sim_stats_raw[np.round(sim_stats_raw['rotate'],6)==np.round(rotate[i],6)]
        nov_i_mean = stats_i[['alph'] + [field_vals,field_stats]].groupby(['alph'] + [field_vals]).mean().reset_index()
        nov_i_std = stats_i[['alph'] + [field_vals,field_stats]].groupby(['alph'] + [field_vals]).std().reset_index()
        
        nov_i_mean2 = stats_i[['alph'] + [field_vals,field_stats2]].groupby(['alph'] + [field_vals]).mean().reset_index()
        nov_i_std2 = stats_i[['alph'] + [field_vals,field_stats2]].groupby(['alph'] + [field_vals]).std().reset_index()

        nov_i_mean3 = stats_i[['alph'] + [field_vals,field_stats3]].groupby(['alph'] + [field_vals]).mean().reset_index()
        nov_i_std3 = stats_i[['alph'] + [field_vals,field_stats3]].groupby(['alph'] + [field_vals]).std().reset_index()


        rotsim_i = np.round(rotate[i]/np.pi,2)
        label_i = f'{int(np.round(rotsim_i))}$\pi$' if rotsim_i%1==0 else f'{rotsim_i}$\pi$'

        ax = axl[i]; ax2 = axl2[i]; ax3 = axl3[i]
        for j in range(len(test_alphas2)):
            nov_i_mean_alph = nov_i_mean.loc[np.round(nov_i_mean['alph'],6)==np.round(test_alphas2[j],6)]
            nov_i_std_alph  = nov_i_std.loc[np.round(nov_i_std['alph'],6)==np.round(test_alphas2[j],6)]
            ax.plot(nov_i_mean_alph[field_vals],nov_i_mean_alph[field_stats],'-',c=color_list2[j])
            ax.plot(nov_i_mean_alph[field_vals],nov_i_mean_alph[field_stats],'o',c=color_list2[j],label=f'{np.round(test_alphas2[j],6)}')
            ax.fill_between(x=nov_i_std_alph[field_vals],y1=nov_i_mean_alph[field_stats]-nov_i_std_alph[field_stats],y2=nov_i_mean_alph[field_stats]+nov_i_std_alph[field_stats],color=color_list2[j],alpha=0.2)

            nov_i_mean2_alph = nov_i_mean2.loc[np.round(nov_i_mean2['alph'],6)==np.round(test_alphas2[j],6)]
            nov_i_std2_alph  = nov_i_std2.loc[np.round(nov_i_std2['alph'],6)==np.round(test_alphas2[j],6)]
            ax2.plot(nov_i_mean2_alph[field_vals],nov_i_mean2_alph[field_stats2],'-',c=color_list2[j])
            ax2.plot(nov_i_mean2_alph[field_vals],nov_i_mean2_alph[field_stats2],'o',c=color_list2[j],label=f'{np.round(test_alphas2[j],6)}')
            ax2.fill_between(x=nov_i_std2_alph[field_vals],y1=nov_i_mean2_alph[field_stats2]-nov_i_std2_alph[field_stats2],y2=nov_i_mean2_alph[field_stats2]+nov_i_std2_alph[field_stats2],color=color_list2[j],alpha=0.2)

            nov_i_mean3_alph = nov_i_mean3.loc[np.round(nov_i_mean3['alph'],6)==np.round(test_alphas2[j],6)]
            nov_i_std3_alph  = nov_i_std3.loc[np.round(nov_i_std3['alph'],6)==np.round(test_alphas2[j],6)]
            ax3.plot(nov_i_mean3_alph[field_vals],nov_i_mean3_alph[field_stats3],'-',c=color_list2[j])
            ax3.plot(nov_i_mean3_alph[field_vals],nov_i_mean3_alph[field_stats3],'o',c=color_list2[j],label=f'{np.round(test_alphas2[j],6)}')
            ax3.fill_between(x=nov_i_std3_alph[field_vals],y1=nov_i_mean3_alph[field_stats3]-nov_i_std3_alph[field_stats3],y2=nov_i_mean3_alph[field_stats3]+nov_i_std3_alph[field_stats3],color=color_list2[j],alpha=0.2)
        
        ax.set_title(f'Rotation: {np.round(rotate[i]/np.pi,3)}$\pi$')
        ax2.set_title(f'Rotation: {np.round(rotate[i]/np.pi,3)}$\pi$')
        ax3.set_title(f'Rotation: {np.round(rotate[i]/np.pi,3)}$\pi$')
        if i==0:
            ax.set_ylabel('Novelty response')
            ax2.set_ylabel('Steady state novelty')
            ax3.set_ylabel('Raw novelty')
        else:
            ax.set_ylabel('')
            ax2.set_ylabel('')
            ax3.set_ylabel('')
        ax.set_xlabel('M')
        ax.set_xticks([3,6,9,12])
        ax2.set_xlabel('M')
        ax2.set_xticks([3,6,9,12])
        ax3.set_xlabel('M')
        ax3.set_xticks([3,6,9,12])
        if i==len(rotate)-1:
            ax.legend(loc='upper left',title='Leakiness $\\alpha$',frameon=False,handletextpad=0.1,bbox_to_anchor=(1.1,1))
            ax2.legend(loc='upper left',title='Leakiness $\\alpha$',frameon=False,handletextpad=0.1,bbox_to_anchor=(1.1,1))
            ax3.legend(loc='upper left',title='Leakiness $\\alpha$',frameon=False,handletextpad=0.1,bbox_to_anchor=(1.1,1))

    fig.tight_layout()
    fig.savefig(os.path.join(path_fig,f'{name_fig2}.png'))
    fig.savefig(os.path.join(path_fig,f'{name_fig2}.svg'))
    fig2.tight_layout()
    fig2.savefig(os.path.join(path_fig,f'{name_fig2}_steady.png'))
    fig2.savefig(os.path.join(path_fig,f'{name_fig2}_steady.svg'))
    fig3.tight_layout()
    fig3.savefig(os.path.join(path_fig,f'{name_fig2}_raw.png'))
    fig3.savefig(os.path.join(path_fig,f'{name_fig2}_raw.svg'))

    print('done')

