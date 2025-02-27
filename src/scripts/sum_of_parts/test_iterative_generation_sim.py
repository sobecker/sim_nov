import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

import sys

import models.snov.gabor_stimuli as gs
import models.snov.run_gabor_knov2 as gknov2
import fitting_neural.create_homann_input as h_in
import utils.saveload as sl
import fitting_neural.simulate_data as sd
import utils.visualization as vis
import scripts.sum_of_parts.test_parent_child_sim as tpcs


############################################################################################################################
def generate_similar_input(sim_value,seed_parent,seed_child,init_orient,num_parent,num_child_per_parent,n_fam=[1,3,8,18,38],gabor_num=(4,4),loc_sigma=(0,0),features=[0],fixed_freq=0.04):
    # Children parameters
    child_colors = vis.prep_cmap('viridis',num_child_per_parent)

    # Experiment parameters
    sequence_mode = 'seq'
    len_fam = 3

    # Plot stimuli
    plot_stim = False

    # Generate input sequences
    all_children = []
    all_inputs = []
    all_overlap = []

    for i in range(num_parent):
        # Generate parent stimulus
        init_child, unique_vals = gs.generate_teststim_parent(gs.dim_ranges_rad,init_orient=init_orient,gabor_num=gabor_num,rng=np.random.default_rng(seed_parent[i]),loc_sigma=loc_sigma,return_features=features,fixed_freq=fixed_freq)
        
        # Generate child stimuli (four for L-experiment)
        if len(features)==1:
            unique_vals = unique_vals[0]
            children, overlap = gs.generate_teststim_iterative_1d(init_child,unique_vals,num_child_per_parent,field_transform=features,prob_overlap=[sim_value],rng=np.random.default_rng(seed_child[i]))
        else:
            children, overlap = gs.generate_teststim_iterative(init_child,unique_vals,num_child_per_parent,field_transform=features,prob_overlap=[sim_value],rng=np.random.default_rng(seed_child[i]))

        assert len(children)==num_child_per_parent, 'Number of children does not match number of children per parent.'
        all_overlap.append(overlap)

        df_children = [pd.DataFrame(dict(zip(gs.dim_names,children[i]))) for i in range(len(children))]  
        im_children = []
        if plot_stim:
            f_children = []
            ax_children = []
        for j in range(len(children)):
            im_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],children[j],resolution=100,magn=1)
            im_children.append(im_child)

            # Plot children
            if plot_stim:
                f,ax = plt.subplots(1,1,figsize=(8,16))
                ax.imshow(im_child,cmap='gray',vmin=-1,vmax=1,origin='lower')
                ax.axis('off')
                f_children.append(f)
                ax_children.append(ax)

                # Plot all Gabors of current child i into plot of child i
                xloc_i = (children[j][4,:] + 130)*200/260 
                yloc_i = (children[j][5,:] + 20)*100/90
                ax.scatter(xloc_i,yloc_i,s=30,color=child_colors[i],label=f'Unique to child {i}') 

                # Plot overlap between currently plotted image i and previous images j 
                for h in range(len(overlap[j])):
                    xloc_same = (children[h][4,overlap[j][h]] + 130)*200/260 
                    yloc_same = (children[h][5,overlap[j][h]] + 20)*100/90
                    ax.scatter(xloc_same,yloc_same,s=30,color=child_colors[h],label=f'Overlap with child {h}') # Plot overlap with child j into plot of current child i
                    ax_children[h].scatter(xloc_same,yloc_same,s=30,color=child_colors[j],label=f'Overlap with child {j}') # Plot overlap with current child i into plot of child j
                
        all_children.append((df_children,im_children))

        if plot_stim:
            for j in range(len(f_children)):
                ax_children[j].legend(bbox_to_anchor=(1.38,1),loc='upper right')
                f_children[j].tight_layout()

        gfam = [im_children[j] for j in range(len_fam)]
        gnov = [im_children[len_fam]]

        vec_list, type_list = tpcs.generate_l_seq(gfam,gnov,n_fam,sequence_mode=sequence_mode,len_fam=len_fam)
        
        all_inputs.append((vec_list,type_list))
    return all_children, all_inputs, all_overlap

############################################################################################################################
if __name__=='__main__':

    plot_similarities   = True
    plot_kernelact      = True
    plot_steadystate    = True
    recomp_data  = True
    recomp_stats = True
    plot_data    = True
    check_images        = False

    # Set parameters
    prec = 2
    p_rotate = np.round(np.arange(0.75,0.96,0.05),prec)
    n_fam = [18]
    len_fam = 3
    num_parent           = 20
    num_child_per_parent = len_fam+1
    features             = [0]
    gabor_orient         = 0 # pi/4 = 45 degrees
    gabor_diff           = 0 #np.linspace(0,np.pi/2,5), i.e. between 0 and 90 degrees
    init_orient          = gabor_orient - gabor_diff
    gabor_num            = (4,3) # (4,4)
    loc_sigma            = (0,0) # (0,0)
    fixed_freq           = 0.06
    ksig                 = 0.95
    cdens                = 2

    init_seed            = 12345
    seed_parent = h_in.get_random_seed(length=5,n=num_parent,init_seed=init_seed)
    seed_child  = h_in.get_random_seed(length=5,n=num_parent*num_child_per_parent,init_seed=init_seed+1)

    path_data = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-10_iterative-gen_gabor-12_denoised_ksig-095/'
    sl.make_long_dir(path_data)

    path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/2024-10_iterative-gen_gabor-12_denoised_ksig-095/'
    sl.make_long_dir(path_fig)

    # Generate children for different p_rotate
    sv_inputs = []
    sv_children = []
    sv_simf = []
    sv_simp = []
    sv_simk = []
    sv_kernels_children = []
    for pr in p_rotate:
        all_children, all_inputs, all_overlap = generate_similar_input(np.round(1-pr,prec),seed_parent,seed_child,init_orient,num_parent,num_child_per_parent,n_fam=[18],gabor_num=gabor_num,loc_sigma=loc_sigma,fixed_freq=fixed_freq)
        sv_inputs.append(all_inputs)
        sv_children.append(all_children)

        all_parent_simf, all_pairwise_simf, all_hist_pairwise_simf = tpcs.comp_feature_sim([],all_children)
        sv_simf.append((all_parent_simf, all_pairwise_simf, all_hist_pairwise_simf))
        all_parent_simp, all_pairwise_simp, all_hist_pairwise_simp = tpcs.comp_pixel_sim([],all_children)
        sv_simp.append((all_parent_simp, all_pairwise_simp, all_hist_pairwise_simp))
        all_parent_simk, all_pairwise_simk, all_hist_pairwise_simk, all_kmat_parent, all_kernels_children = tpcs.comp_kernel_sim([],all_children,ksig=ksig,cdens=cdens)
        sv_simk.append((all_parent_simk, all_pairwise_simk, all_hist_pairwise_simk))
        sv_kernels_children.append(all_kernels_children)

    print('Done generating similar inputs.\n')

    # Generate random images (Homann style)
    num_gabor = 40
    num_im = num_parent*num_child_per_parent
    im_seed = 98765
    df_random, im_random = tpcs.generate_random_input(num_gabor,num_im,im_seed)
    rand_children = [(df_random,im_random)]

    # Compute similarity of random images
    rand_simp = tpcs.comp_pixel_sim([],rand_children)
    parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk, rand_kmat_parent, rand_kernels_children = tpcs.comp_kernel_sim([],rand_children)
    rand_simk = (parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk)

    print('Done generating random images.\n')

    # Split random images into parent-child pairs
    rand_children_split = []
    for i in range(num_parent):
        df_i = [df_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1))]
        im_i = [im_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1))]
        rand_children_split.append((df_i,im_i))

    rand_simp_split = tpcs.comp_pixel_sim([],rand_children_split)
    parents_rand_simk_split, pairwise_rand_simk_split, hist_pairwise_rand_simk_split, rand_kmat_parent_split, rand_kernels_children_split = tpcs.comp_kernel_sim([],rand_children_split)
    rand_simk_split = (parents_rand_simk_split, pairwise_rand_simk_split, hist_pairwise_rand_simk_split)

    # Generate simulation input (L-sequence)
    inputs_rand = []
    for i in range(num_parent):
        gfam = [im_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1)-1)]
        gnov = [im_random[(i+1)*(len_fam+1)-1]]

        vec_list, type_list = tpcs.generate_l_seq(gfam,gnov,n_fam,len_fam=len_fam)
        inputs_rand.append((vec_list,type_list))

    print('Done generating random child sequences.\n')

    # Combine data
    sv_sim_all = [sv_simf,sv_simp,sv_simk]
    rand_sim_all = [None, rand_simp, rand_simk]
    rand_sim_all_split = [None, rand_simp_split, rand_simk_split]
    sv_sim_names = ['feature','pixel','kernel']

    # Plot average similarities
    if plot_similarities:
        f,ax = plt.subplots(1,1,figsize=(6,4))
        color_list = vis.prep_cmap('Blues',len(sv_sim_all))

        pairwise_means_all = []
        pairwise_std_all = []
        for j in range(len(sv_sim_all)):
            pairwise_means = []
            pairwise_stds = []
            for i in range(len(p_rotate)):
                all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim = sv_sim_all[j][i]
                pairwise_mean_i = np.mean(np.concatenate(all_hist_pairwise_sim))
                pairwise_std_i = np.std(np.concatenate(all_hist_pairwise_sim))
                pairwise_means.append(pairwise_mean_i)
                pairwise_stds.append(pairwise_std_i)
            # Plot each type of similarity for similar images
            ax.plot(p_rotate,pairwise_means,color=color_list[j],label=f'{sv_sim_names[j]}')
            ax.fill_between(p_rotate,np.array(pairwise_means)-np.array(pairwise_stds),np.array(pairwise_means)+np.array(pairwise_stds),color=color_list[j],alpha=0.2)
            pairwise_means_all.append(pairwise_means)
            pairwise_std_all.append(pairwise_stds)
            # Plot each type of similarity for random images
            if sv_sim_names[j]!="feature":
                rand_parent_sim, rand_pairwise_sim, rand_hist_pairwise_sim = rand_sim_all[j]
                pairwise_mean_rand = np.mean(np.concatenate(rand_hist_pairwise_sim))
                pairwise_std_rand = np.std(np.concatenate(rand_hist_pairwise_sim))
                ax.axhline(y=pairwise_mean_rand,ls='--',color=color_list[j],label=f'{sv_sim_names[j]} (random)')
                ax.fill_between(p_rotate,pairwise_mean_rand-pairwise_std_rand,pairwise_mean_rand+pairwise_std_rand,color=color_list[j],alpha=0.2)
        
        ax.legend()
        ax.set_title('Pairwise similarity')
        ax.set_xlabel('p_rotate')
        ax.set_ylabel('Mean similarity')
        ax.set_xlim([np.min(p_rotate),np.max(p_rotate)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        f.tight_layout()
        f.savefig(os.path.join(path_fig,f'mean-sim_vs_p-rotate.png'))

    # Plot kernel activations
    if plot_kernelact:
        for acc_type in ['mean','sum','max']:
            acc_fun = np.mean if acc_type=='mean' else np.sum if acc_type=='sum' else np.max
            f,ax = plt.subplots(1,2,figsize=(6,4),sharey=True,gridspec_kw={'width_ratios':[3,1]})

            all_kact = [sv_kernels_children,rand_kernels_children] # note: parents are identical for all p_rotate values
            all_kact_names = ['children','random']
            color_list = [vis.prep_cmap('Blues',len(p_rotate)),['grey']]
            xl = ['p_rotate','']

            for i in range(len(all_kact)): 
                kact = all_kact[i]
                for j in range(len(kact)): # number of p_rotate values, except for random
                    acc_kact_j = np.array([acc_fun(kact[j][h]) for h in range(len(kact[j]))]) # this has the statistic of kernel activations (scalar for each image in each experiment)
                    mean_kact_j = np.mean(acc_kact_j)
                    std_kact_j = np.std(acc_kact_j)
                    ax[i].scatter([j]*len(acc_kact_j.flatten()),acc_kact_j.flatten(),color=color_list[i][j],alpha=0.5)
                    ax[i].plot(j,mean_kact_j,'_',color=color_list[i][j],markersize=10,linewidth=4)
                    ax[i].errorbar(j,mean_kact_j,yerr=std_kact_j,color=color_list[i][j])
                ax[i].set_title(f'{all_kact_names[i]}')
                ax[i].set_xlabel(xl[i])
                if len(xl[i])>0: 
                    ax[i].set_xticks(np.arange(len(p_rotate)))
                    ax[i].set_xticklabels(np.round(p_rotate,2))
                else:
                    ax[i].set_xticks([])
                ax[i].set_ylabel(f'kernel activation ({acc_type})')
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
            f.tight_layout()
            f.savefig(os.path.join(path_fig,f'kact_{acc_type}.png'))
    
    # Plot steady state responses
    if plot_steadystate:
        if recomp_data:
            # Make kernel model
            k_params = gknov2.init_gabor_knov(gnum=4,k_type='triangle',ksig=1,kcenter=1,cdens=4,seed=12345,rng=None,mask=True,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='equidistant',contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,add_empty=False,debug=False)

            # Run experiment
            sv_data = []; sv_kl = []; sv_kwl = []
            for i in range(len(p_rotate)):
                data, kl, kwl = tpcs.run_homann_l(n_fam,sv_inputs[i],k_params,seed_parent)
                df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
                df_data.to_csv(os.path.join(path_data,f'homann_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                sv_data.append(df_data)
                sv_kl.append(kl)
                sv_kwl.append(kwl)
                    
            # Run experiment for random images
            data, kl, kwl = tpcs.run_homann_l(n_fam,inputs_rand,k_params,[im_seed]*len(inputs_rand))
            df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
            df_data.to_csv(os.path.join(path_data,'homann_rand.csv'))
            data_rand = [df_data]
            kl_rand = [kl]
            kwl_rand = [kwl]

            # Compute similarity statistics
            df_sim_rand = pd.DataFrame({'sample_id':df_data.sample_id.unique()})
            for j in range(1,len(rand_sim_all_split)):
                sim_j = np.array([np.mean(rand_sim_all_split[j][2][h]) for h in range(len(rand_sim_all_split[j][2]))]) # number of parents
                df_sim_rand[f'pairwise_sim_{sv_sim_names[j]}'] = sim_j
            df_sim_rand.to_csv(os.path.join(path_data,'homann_rand_sim.csv'))

            print('Done simulating experiments.\n')

        else:
            # Load simulation data
            sv_data = []
            for i in range(len(p_rotate)):
                df_data = pd.read_csv(os.path.join(path_data,f'homann_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                sv_data.append(df_data)
            
            # Load random data
            df_data = pd.read_csv(os.path.join(path_data,'homann_rand.csv'))
            data_rand = [df_data]
            print('Done loading experiments.\n')
        
        if recomp_stats:
            # Compute statistics
            sv_stats = []
            for i in range(len(p_rotate)):
                stats_i = tpcs.comp_stats(sv_data[i],steady=True)
                stats_i.to_csv(os.path.join(path_data,f'homann_stats-{p_rotate[i]}'.replace('.','') + '.csv'))
                sv_stats.append(stats_i)
                field_vals  = ('n_fam','')
                field_stats = ('nt_norm','mean')
                field_stats2 = ('steady','mean')
            
            # Compute random statistics
            stats_rand = tpcs.comp_stats(data_rand[0],steady=True)
            stats_rand.to_csv(os.path.join(path_data,'homann_stats-rand.csv'))
            stats_rand = [stats_rand]
            print('Done computing stats.\n')

        else:
            # Load statistics
            sv_stats = []
            for i in range(len(p_rotate)):
                col_extract = ['n_fam','nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','steady_mean','steady_std','steady_sem','sample_id']
                stats_i = pd.read_csv(os.path.join(path_data,f'homann_stats-{p_rotate[i]}'.replace('.','') + '.csv'),header=1,names=col_extract)
                sv_stats.append(stats_i)
                field_vals  = 'n_fam'
                field_stats = 'nt_norm_mean'
                field_stats2 = 'steady_mean'

            # Load random statistics
            stats_rand = pd.read_csv(os.path.join(path_data,'homann_stats-rand.csv'),header=1,names=col_extract)
            stats_rand = [stats_rand]
            print('Done loading stats.\n')

        if plot_data:
            # Plot novelty and steady state responses
            f1,axl1 = plt.subplots(1,len(sv_sim_all),figsize=(4*len(sv_sim_all),4),sharex=True,sharey=True)
            f2,axl2 = plt.subplots(1,len(sv_sim_all),figsize=(4*len(sv_sim_all),4),sharex=True,sharey=True)
            color_list = vis.prep_cmap('Blues',len(p_rotate))
            alpha_list = np.linspace(1,0.2,len(p_rotate))

            for i in range(len(p_rotate)):
                # Extract pairwise similarities for each similarity type
                sim_stats_all = []
                sim_mu_all    = []
                sim_std_all   = []
                sim_stats_all2 = []
                sim_mu_all2    = []
                sim_std_all2   = []
                for j in range(len(sv_sim_all)):
                    sim_i = np.array([np.mean(sv_sim_all[j][i][2][h]) for h in range(len(sv_sim_all[j][i][2]))]) # number of parents
                    sim_stats_all.append(sim_i)
                    sim_mu_all.append(np.mean(sim_i))
                    sim_std_all.append(np.std(sim_i))

                    rsas_j = sv_sim_all[j][i][1]
                    sim_i2 = []
                    for h in range(len(rsas_j)):
                        rsas_jh = rsas_j[h][:-1,:-1]
                        av_jh = [np.mean(rsas_jh[np.where(np.arange(rsas_jh.shape[0])!=li)[0],li]) for li in range(rsas_jh.shape[1])]
                        sim_i2.append(np.mean(av_jh))
                    sim_i2 = np.array(sim_i2) 
                    sim_stats_all2.append(sim_i2)
                    sim_mu_all2.append(np.mean(sim_i2))
                    sim_std_all2.append(np.std(sim_i2))

                # Extract simulation data
                stats_i = sv_stats[i]
                samples = stats_i['sample_id'].unique()
                distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index() 
                distribution_i2 = stats_i[[field_vals,field_stats2]].groupby([field_vals]).agg(mean_stats=(field_stats2,np.mean),std_stats=(field_stats2,np.std)).reset_index()

                for j in range(len(sv_sim_all)):
                    ax = axl1[j]
                    ax.scatter(sim_stats_all[j],stats_i[field_stats],s=10,c=color_list[i],label=f'P(rotate)={np.round(p_rotate[i],2)}',alpha=0.8)
                    ax.plot(sim_mu_all[j],distribution_i['mean_stats'],'x',c=color_list[i])
                    ax.errorbar(sim_mu_all[j],distribution_i['mean_stats'],xerr=sim_std_all[j],yerr=distribution_i['std_stats'],color=color_list[i])
                    ax.set_title('Novelty responses')
                    ax.set_xlabel(f'Av. pairwise similarity ({sv_sim_names[j]})')
                    ax.set_ylabel('Novelty responses')
        
                    ax = axl2[j]
                    # ax.scatter(sim_stats_all[j],stats_i[field_stats2],s=10,c=color_list[i],label=f'P(rotate)={np.round(p_rotate[i],2)}',alpha=0.8)
                    # ax.plot(sim_mu_all[j],distribution_i2['mean_stats'],'x',c=color_list[i])
                    # ax.errorbar(sim_mu_all[j],distribution_i2['mean_stats'],xerr=sim_std_all[j],yerr=distribution_i2['std_stats'],color=color_list[i])
                    ax.scatter(sim_stats_all2[j],stats_i[field_stats2],s=10,c=color_list[i],label=f'P(rotate)={np.round(p_rotate[i],2)}',alpha=0.8)
                    ax.plot(sim_mu_all2[j],distribution_i2['mean_stats'],'x',c=color_list[i])
                    ax.errorbar(sim_mu_all2[j],distribution_i2['mean_stats'],xerr=sim_std_all2[j],yerr=distribution_i2['std_stats'],color=color_list[i])
                    ax.set_title('Steady state responses')
                    ax.set_xlabel(f'Av. pairwise similarity ({sv_sim_names[j]})')
                    ax.set_ylabel('Steady state responses')
                
            # Extract pairwise similarities for each similarity type
            rand_stats_all = [None]
            rand_mu_all    = [None]
            rand_std_all   = [None]
            rand_stats_all2 = [None]
            rand_mu_all2    = [None]
            rand_std_all2   = [None]
            for j in range(1,len(rand_sim_all_split)):
                sim_i = np.array([np.mean(rand_sim_all_split[j][2][h]) for h in range(len(rand_sim_all_split[j][2]))]) # number of parents
                rand_stats_all.append(sim_i)
                rand_mu_all.append(np.mean(sim_i))
                rand_std_all.append(np.std(sim_i))

                rsas_j = rand_sim_all_split[j][1]
                sim_i2 = []
                for h in range(len(rsas_j)):
                    rsas_jh = rsas_j[h][:-1,:-1]
                    av_jh = [np.mean(rsas_jh[np.where(np.arange(rsas_jh.shape[0])!=li)[0],li]) for li in range(rsas_jh.shape[1])]
                    sim_i2.append(np.mean(av_jh))
                sim_i2 = np.array(sim_i2) 
                rand_stats_all2.append(sim_i2)
                rand_mu_all2.append(np.mean(sim_i2))
                rand_std_all2.append(np.std(sim_i2))

            # Extract simulation data
            stats_i = stats_rand[0]
            samples = stats_i['sample_id'].unique()
            distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index() 
            distribution_i2 = stats_i[[field_vals,field_stats2]].groupby([field_vals]).agg(mean_stats=(field_stats2,np.mean),std_stats=(field_stats2,np.std)).reset_index()

            for j in range(1,len(rand_sim_all_split)):
                ax = axl1[j]
                ax.scatter(rand_stats_all[j],stats_i[field_stats],s=10,c='k',label='random',alpha=0.8)
                ax.plot(rand_mu_all[j],distribution_i['mean_stats'],'x',c='k')
                ax.errorbar(rand_mu_all[j],distribution_i['mean_stats'],xerr=rand_std_all[j],yerr=distribution_i['std_stats'],color='k')
                ax.set_title('Novelty responses')
                ax.set_xlabel(f'Av. pairwise similarity ({sv_sim_names[j]})')
                ax.set_ylabel('Novelty responses')
                ax.set_ylim([1,6])
        
                ax = axl2[j]
                # ax.scatter(rand_stats_all[j],stats_i[field_stats2],s=10,c='k',label='random',alpha=0.8)
                # ax.plot(rand_mu_all[j],distribution_i2['mean_stats'],'x',c='k')
                # ax.errorbar(rand_mu_all[j],distribution_i2['mean_stats'],xerr=rand_std_all[j],yerr=distribution_i2['std_stats'],color='k')
                ax.scatter(rand_stats_all2[j],stats_i[field_stats2],s=10,c='k',label='random',alpha=0.8)
                ax.plot(rand_mu_all2[j],distribution_i2['mean_stats'],'x',c='k')
                ax.errorbar(rand_mu_all2[j],distribution_i2['mean_stats'],xerr=rand_std_all2[j],yerr=distribution_i2['std_stats'],color='k')
                ax.set_title('Steady state responses')
                ax.set_xlabel(f'Av. pairwise similarity ({sv_sim_names[j]})')
                ax.set_ylabel('Steady state responses')
                ax.set_ylim([1,6])

            axl1[-1].legend(bbox_to_anchor=(1.6,0.5),loc='center right')
            axl2[-1].legend(bbox_to_anchor=(1.6,0.5),loc='center right')
            f1.tight_layout()
            f1.savefig(os.path.join(path_fig,'nov_vs_p-rotate.png'),bbox_inches='tight')
            f2.tight_layout()
            f2.savefig(os.path.join(path_fig,'steady_vs_p-rotate.png'),bbox_inches='tight')

            # Plot novelty responses as Gaussians
            f,ax = plt.subplots(1,1,figsize=(6,4))

            for i in range(len(p_rotate)):
                # Extract simulation data
                stats_i = sv_stats[i]
                samples = stats_i['sample_id'].unique()
                distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index() 
                mu_i = distribution_i['mean_stats']
                std_i = distribution_i['std_stats']
                x_i = np.linspace(mu_i-3*std_i,mu_i+3*std_i,100)
            
                ax.plot(x_i,stats.norm.pdf(x_i,mu_i,std_i),color=color_list[i],label=f'P(rotate)={np.round(p_rotate[i],2)}')
                
            stats_i = stats_rand[0]
            samples = stats_i['sample_id'].unique()
            distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index() 
            mu_i = distribution_i['mean_stats']
            std_i = distribution_i['std_stats']
            x_i = np.linspace(mu_i-3*std_i,mu_i+3*std_i,100)
        
            ax.plot(x_i,stats.norm.pdf(x_i,mu_i,std_i),color='k',label='random')

            ax.set_xlabel('Novelty responses')  
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')

            f.tight_layout()
            f.savefig(os.path.join(path_fig,'novelty_gaussians.png'),bbox_inches='tight')

    if check_images:
        # Extract samples that have large / small steady state
        i_check = np.where(p_rotate==0.8)[0][0]
        stats_i = sv_stats[i_check]
        distribution_i2 = stats_i[[field_vals,field_stats2]].groupby([field_vals]).agg(mean_stats=(field_stats2,np.mean),std_stats=(field_stats2,np.std)).reset_index()
        set_large = stats_i.loc[stats_i[field_stats2]>distribution_i2['mean_stats'].values[0]]
        set_small = stats_i.loc[stats_i[field_stats2]<=distribution_i2['mean_stats'].values[0]]
        idx_large = set_large.sample_id.values
        idx_small = set_small.sample_id.values

        num_plot = 3
        data_i = sv_data[i_check]
        for j in range(min(num_plot,len(idx_large))):
            # Plot histogram of kernel activations
            # f,ax = plt.subplots(1,4,figsize=(4*3,3),sharex=True,sharey=True)
            # for i,ax_i in enumerate(ax):
            #     ax_i.hist(sv_kl[i_check][idx_large[j]][0][i,:],bins=20,color='k',alpha=0.5)
            #     ax_i.set_xlabel(f'Kernel activation')  
            #     ax_i.set_yscale('log')
            # f.suptitle(f'Sample {idx_large[j]}: large steady state')

            # Plot simulation traces
            f,ax = plt.subplots(1,1,figsize=(4,3))
            # data_ij = data_i.loc[data_i.sample_id==idx_large[j],'nt']
            data_ij = data_i.loc[data_i.sample_id==idx_large[j]]
            _, nov_resp_i = sd.get_nov_response(data_ij,'n_fam',steady=True)
            nov_resp_i = nov_resp_i.reset_index()

            ax.scatter(np.arange(len(data_ij)),data_ij['nt'],color='k')
            ax.plot(np.arange(len(data_ij)),data_ij['nt'],color='k')
            ax.axhline(y=distribution_i2['mean_stats'].values[0],ls='--',color='r')
            ax.axhline(y=set_large.loc[set_large.sample_id==idx_large[j],field_stats2].values[0],ls='--',color='k')
            f.suptitle(f'Sample {idx_large[j]}: large steady state')


            # Plot their similarity matrices
            f,ax = plt.subplots(2,1,figsize=(4,4*2))
            tpcs.plot_pairwise_sim_matrix(sv_simp[i_check][1][idx_large[j]],f,ax[0])
            tpcs.plot_pairwise_sim_matrix(sv_simk[i_check][1][idx_large[j]],f,ax[1])
            f.suptitle(f'Sample {idx_large[j]}: large steady state')

            # Plot their image sequences
            # f,ax = plt.subplots(1,4,figsize=(4*4,3))
            # for i,ax_i in enumerate(ax):
            #     ax_i.imshow(sv_children[i_check][idx_large[j]][-1][i],cmap='gray',vmin=-1,vmax=1)
            #     ax_i.axis('off')
            # f.tight_layout()
            # f.suptitle(f'Sample {idx_large[j]}: large steady state')
    
        num_plot = 3
        data_i = sv_data[i_check]
        for j in range(min(num_plot,len(idx_small))):
            # Plot histogram of kernel activations
            # f,ax = plt.subplots(1,4,figsize=(4*3,3),sharex=True,sharey=True)
            # for i,ax_i in enumerate(ax):
            #     ax_i.hist(sv_kl[i_check][idx_small[j]][0][i,:],bins=20,color='k',alpha=0.5)
            #     ax_i.set_xlabel(f'Kernel activation')  
            #     ax_i.set_yscale('log')
            # f.suptitle(f'Sample {idx_small[j]}: small steady state')

            # Plot simulation traces
            f,ax = plt.subplots(1,1,figsize=(4,3))
            # data_ij = data_i.loc[data_i.sample_id==idx_small[j],'nt']
            data_ij = data_i.loc[data_i.sample_id==idx_small[j]]
            _, nov_resp_i = sd.get_nov_response(data_ij,'n_fam',steady=True)
            nov_resp_i = nov_resp_i.reset_index()

            ax.scatter(np.arange(len(data_ij)),data_ij['nt'],color='k')
            ax.plot(np.arange(len(data_ij)),data_ij['nt'],color='k')
            ax.axhline(y=distribution_i2['mean_stats'].values[0],ls='--',color='r')
            ax.axhline(y=set_small.loc[set_small.sample_id==idx_small[j],field_stats2].values[0],ls='--',color='k')
            f.suptitle(f'Sample {idx_small[j]}: small steady state')
            
            # Plot their similarity matrices
            f,ax = plt.subplots(2,1,figsize=(4,4*2))
            tpcs.plot_pairwise_sim_matrix(sv_simp[i_check][1][idx_small[j]],f,ax[0])
            tpcs.plot_pairwise_sim_matrix(sv_simk[i_check][1][idx_small[j]],f,ax[1])
            f.suptitle(f'Sample {idx_small[j]}: small steady state')

            # Plot their image sequences
            # f,ax = plt.subplots(1,4,figsize=(4*4,3))
            # for i,ax_i in enumerate(ax):
            #     ax_i.imshow(sv_children[i_check][idx_small[j]][-1][i],cmap='gray',vmin=-1,vmax=1)
            #     ax_i.axis('off')
            # f.tight_layout()
            # f.suptitle(f'Sample {idx_small[j]}: small steady state')

    print('Done.\n')    

