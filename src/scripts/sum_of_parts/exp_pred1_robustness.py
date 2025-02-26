import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/') 

import src.models.snov.gabor_stimuli as gs
import src.models.snov.run_gabor_knov2 as gknov2
import src.models.snov.run_gabor_knov_complex as gknovc

import src.utils.saveload as sl
import src.utils.visualization as vis
import src.scripts.sum_of_parts.test_parent_child_sim as tpcs
import src.scripts.sum_of_parts.plot_similarity_modulation_homann as smh
import src.scripts.sum_of_parts.exp_pred_helpers as eph

def run_perm_withalph(alph_k,rotate,unit_orient=0,plot_stim=False,permute_all=True,num_perm=20,seed_parent=12345,save_stats=True,save_path='',save_name='',exp_type='m',cell_type='simple'):
    # Set parameters according to experiment type
    if exp_type=='l': 
        num_child = 3
        plot_exp = [True,False,False]
        idx_exp = 0
        fun_plot_seq = eph.plot_sequence_l
    elif exp_type=='m': 
        num_child = 12
        plot_exp = [False,True,False]
        idx_exp = 1
        fun_plot_seq = eph.plot_sequence_m
    elif exp_type=='lp': 
        num_child = 3
        plot_exp = [False,False,True]
        idx_exp = 2
        fun_plot_seq = eph.plot_sequence_lp

    # Make kernel model
    if cell_type=='simple':
        k_params = gknov2.init_gabor_knov(gnum=4,k_type='triangle',ksig=1,kcenter=1,cdens=1,seed=12345,rng=None,mask=True,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='equidistant_fixed',fixed_freq=0.06,contrast='off',softmax_norm=False,eps_k=1,alph_k=alph_k,add_empty=False,debug=False)
        complex_kwargs = None
    elif cell_type=='complex':
        k_params = gknovc.init_gabor_knov(gnum_complex=4,dfreq=1.5,ctype=[4],cvar='frequency',k_type='triangle',ksig=1,kcenter=1,cdens=2,seed=12345,rng=None,mask=False,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='equidistant_fixed',fixed_freq=0.06,fixed_width=None,contrast='off',softmax_norm=False,eps_k=1,alph_k=alph_k,add_empty=False,debug=False)
        complex_kwargs = {'no_simple_cells': False,
              'no_complex_cells': False,
              'mode_complex': 'sum', # 'sum' or 'mean'
              'type_complex': [4],
              'num_complex': 1, #
              'debug': False,
              'append_mode': False,
              'start_id': 0
              }

    # Generate parent image
    parent, _ = gs.generate_teststim_parent(gs.dim_ranges_rad,init_orient=0,gabor_num=(6,2),fixed_freq=0.06,rng=np.random.default_rng(seed_parent))
    parent[0,:] = unit_orient

    # Generate shuffles of parent
    sim_data_raw = []; sim_stats_raw = []; sim_oddsim = []; sim_oddsim_vec = []; sim_rotate = []; sim_perm_id = []
    all_perm = []
    if permute_all:
        num_gabor = np.arange(parent.shape[1])
        for i_perm in range(num_perm):
            perm = np.random.permutation(num_gabor)
            all_perm.append(perm)
    else:
        all_perm.append(np.arange(parent.shape[1]))
        num_perm = 1

    for i_perm in range(len(all_perm)):
        perm = all_perm[i_perm]
        parent_perm = parent[:,perm]
        
        # Generate familiar sequence (constant children)
        if exp_type=='lp' or exp_type=='l':
            num_gpc =  np.floor(parent_perm.shape[1]/num_child).astype(int)
            children = [parent_perm[:,i*num_gpc:(i+1)*num_gpc] for i in range(num_child)]
            df_children = [pd.DataFrame(dict(zip(gs.dim_names,children[i]))) for i in range(len(children))] 

            im_children = []
            for j in range(len(children)):
                im_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],children[j],resolution=100,magn=1,add_eps=0)
                im_children.append(im_child)

        else:
            children = [parent_perm[:,i] for i in range(num_child)]
            df_children = [pd.DataFrame(dict(zip(gs.dim_names,children[i])),index=[0]) for i in range(len(children))]

            im_children = []
            for j in range(len(children)):
                im_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],children[j].reshape((-1,1)),resolution=100,magn=1,add_eps=0)
                im_children.append(im_child)  

        for i_rot in range(len(rotate)):
            if exp_type=='lp':
                # Generate similar replacement sequence
                df_children_i = df_children.copy()
                im_children_i = im_children.copy()

                for j in range(num_child):
                    last_child = children[j].copy()
                    last_child[0] += rotate[i_rot]
                    df_children_i.append(pd.DataFrame(dict(zip(gs.dim_names,last_child))))
                    im_last_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],last_child,resolution=100,magn=1)
                    im_children_i.append(im_last_child)

                all_children_i = [(df_children_i,im_children_i)]

            else:
                # Generate similar novel image
                df_children_i = df_children.copy()
                im_children_i = im_children.copy() 

                last_child = parent.copy() 
                last_child[0,:] += rotate[i_rot]
                df_children_i.append(pd.DataFrame(dict(zip(gs.dim_names,last_child))))
                im_last_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],last_child,resolution=100,magn=1)
                im_children_i.append(im_last_child)

                all_children_i = [(df_children_i,im_children_i)]

            # Compute + plot similarity matrix of sequence
            vec_children = np.stack(im_children_i)
            kmat_children = eph.compute_kernel_matrix((vec_children,None),k_params,idx=True,conv=True,parallel_k=False)
            kernels_children = [kmat_children[:,i] for i in range(kmat_children.shape[1])]

            pairwise_sim = eph.comp_pairwise_sim(kernels_children)
            pairwise_sim_hist = pairwise_sim[np.triu_indices(pairwise_sim.shape[0],k=1)].flatten()
            sim_oddsim_vec.append(pairwise_sim[:-1,-1])
            sim_oddsim.append(np.mean(pairwise_sim[:-1,-1]))

            if plot_stim:
                f,ax = plt.subplots(1,2,figsize=(12,6))
                eph.plot_pairwise_sim_matrix(pairwise_sim,f,ax[0])
                eph.plot_pairwise_sim_hist(pairwise_sim_hist,f,ax[1])
                ax[1].set_xlim([-1,1])
            
            # Transform into simulation inputs
            all_inputs_i = tpcs.generate_inputs_from_stimuli(all_children_i,sequence_mode='seq',plot_exp=plot_exp)

            # Simulate experiment
            all_data, all_stats, _ = smh.get_data_stats_allexp(exp_type=exp_type,gen_sim=True,gen_rand=False,p_rotate=np.array([0]),sim_kwargs={'seed_parent':[seed_parent]},k_params=k_params,inputs=[all_inputs_i[idx_exp]],recomp_data=True,recomp_stats=True,path_data=path_data,save_data=False,complex=(cell_type=='complex'),complex_kwargs=complex_kwargs)
            all_stats = pd.concat(all_stats)
            all_stats['parent_id'] = seed_parent
            all_stats['perm_id'] = i_perm
            all_stats['rotate'] = rotate[i_rot]
            all_stats['ksim'] = sim_oddsim[-1]
            all_stats['alph_k'] = alph_k
            sim_perm_id.append(i_perm)
            sim_rotate.append(rotate[i_rot])    
            sim_data_raw.append(all_data)
            sim_stats_raw.append(all_stats)
                
            if plot_stim:
                fun_plot_seq(rotate[i_rot],all_inputs_i)

    if save_stats:
        sim_stats = pd.concat(sim_stats_raw)
        sim_stats.to_csv(os.path.join(save_path,f'{save_name}.csv'),index=False)
        
    return sim_data_raw, sim_stats_raw, sim_rotate, sim_perm_id, sim_oddsim, sim_oddsim_vec

def plot_m(load_path,load_name,save_path,save_name):
    exp_type = 'm'

    field_vals   = 'n_fam' if exp_type=='l' else 'n_im' if exp_type=='m' else 'dN'
    field_stats  = 'tr_norm_mean' if exp_type=='lp'  else 'nt_norm_mean'
    field_stats2 = 'steady_mean' 

    col_extract = col_extract = [field_vals,'tr_mean','tr_std','tr_sem','tr_norm_mean','tr_norm_std','tr_norm_sem','steady_mean','steady_std','steady_sem','sample_id','parent_id','perm_id','rotate','ksim'] if exp_type=='lp' else [field_vals,'nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','steady_mean','steady_std','steady_sem','sample_id','parent_id','perm_id','rotate','ksim']

    sim_stats_raw = pd.read_csv(os.path.join(load_path,f'{load_name}.csv'),header=1,names=col_extract,index_col=False)
    rotate = sim_stats_raw['rotate'].unique()
    ksim = sim_stats_raw['ksim'].unique()

    # rotate      = np.array([0, 0.5, 0.509, 0.6, 1])*np.pi
    color_list  = vis.prep_cmap('Blues',len(rotate))

    rotate1 = np.array([0,1,2,3,4])
    rotate2 = np.array([1,2,3])

    ########################################################################################################################
    f,ax = plt.subplots(1,1,figsize=(2,2.5))
    for i in rotate1:
        stats_i = sim_stats_raw[np.round(sim_stats_raw['rotate'],6)==np.round(rotate[i],6)]
        nov_i_mean = stats_i[[field_vals,field_stats]].groupby([field_vals]).mean().reset_index()
        nov_i_std = stats_i[[field_vals,field_stats]].groupby([field_vals]).std().reset_index()

        rotsim_i = np.round(rotate[i]/np.pi,2)
        label_i = f'{int(np.round(rotsim_i))}$\pi$' if rotsim_i%1==0 else f'{rotsim_i}$\pi$'

        ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'-',c=color_list[i])
        ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'o',c=color_list[i],label=label_i)
        ax.fill_between(x=nov_i_std[field_vals],y1=nov_i_mean[field_stats]-nov_i_std[field_stats],y2=nov_i_mean[field_stats]+nov_i_std[field_stats],color=color_list[i],alpha=0.2)

    # ax.legend(loc='upper left',title='Rotation',frameon=False,handletextpad=0.1,bbox_to_anchor=(1,1))
    ax.set_ylabel('Novelty response')
    ax.set_xlabel('M')
    ax.set_xticks([3,6,9,12])

    f.tight_layout()
    f.savefig(os.path.join(save_path,f'{save_name}.svg'))
    f.savefig(os.path.join(save_path,f'{save_name}.png'))   

    ########################################################################################################################
    f,ax = plt.subplots(1,1,figsize=(1.5,1.6))
    for i in rotate2:
        stats_i = sim_stats_raw[np.round(sim_stats_raw['rotate'],6)==np.round(rotate[i],6)]
        nov_i_mean = stats_i[[field_vals,field_stats]].groupby([field_vals]).mean().reset_index()
        nov_i_std = stats_i[[field_vals,field_stats]].groupby([field_vals]).std().reset_index()

        rotsim_i = np.round(rotate[i]/np.pi,2)
        label_i = f'{int(np.round(rotsim_i))}$\pi$' if rotsim_i%1==0 else f'{rotsim_i}$\pi$'

        ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'-',c=color_list[i])
        ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'o',c=color_list[i],label=label_i)
        ax.fill_between(x=nov_i_std[field_vals],y1=nov_i_mean[field_stats]-nov_i_std[field_stats],y2=nov_i_mean[field_stats]+nov_i_std[field_stats],color=color_list[i],alpha=0.2)

    # ax.legend(loc='upper left',frameon=False,handletextpad=0.1,bbox_to_anchor=(-0.025,1.01),ncol=3,columnspacing=0.2)
    ax.set_ylabel('Novelty response')
    ax.set_xlabel('M')
    ax.set_xticks([3,6,9,12])
    # ax.set_ylim([3.2,8.5])

    f.tight_layout()
    f.savefig(os.path.join(save_path,f'{save_name}_zoom.svg'))
    f.savefig(os.path.join(save_path,f'{save_name}_zoom.png'))   


if __name__=="__main__":
    plt.style.use('/Users/sbecker/Projects/RL_reward_novelty/src/scripts/Figures_Paper/paper.mplstyle')

    # Set paths
    folder_name = '2025-01_exp-pred1-simple_sum-of-parts_robustness-m'

    path_data = path_data = f'/Users/sbecker/Projects/RL_reward_novelty/data/{folder_name}/'
    sl.make_long_dir(path_data)

    path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/{folder_name}/'
    sl.make_long_dir(path_fig)

    ########################################################################################################################
    # Run M-experiment for different alpha values
    test_alphas = np.arange(0.1,1,0.1)
    rotate      = np.array([0, 0.5, 0.509, 0.6, 1])*np.pi
    color_list  = vis.prep_cmap('Blues',len(rotate))
    rotate_plot = np.array([0,0.2])*np.pi
    seed_parent = 12345

    for i_alph, alph in enumerate(test_alphas):

        # name_data   = f'm_exp_seed-{seed_parent}_alph-{np.round(alph,6)}'.replace('.','-')
        name_data   = f'm_exp_seed-{seed_parent}_alph-{alph}'.replace('.','-')
        name_fig    = name_data 

        # plot_protocol(rotate_plot,unit_orient=0,seed_parent=seed_parent,save_path=path_fig,save_name=name_fig,exp_type='m')
        run_perm_withalph(alph,rotate,unit_orient=0,plot_stim=False,permute_all=True,seed_parent=seed_parent,save_stats=True,save_path=path_data,save_name=name_data,exp_type='m',cell_type='simple')
        print(f'Done simulating {alph} (value {i_alph+1}/{len(test_alphas)}).\n')

        # plot_m(load_path=path_data,load_name=name_data,save_path=path_fig,save_name=name_fig)

    print('done')

