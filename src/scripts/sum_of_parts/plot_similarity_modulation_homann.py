import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

import models.snov.gabor_stimuli as gs
import models.snov.run_gabor_knov2 as gknov2
import fitting_neural.create_homann_input as h_in
import utils.saveload as sl
import utils.visualization as vis

import scripts.sum_of_parts.test_parent_child_sim as tpcs

def generate_inputs_allexp(gen_sim=False,gen_rand=False,p_rotate=np.round(np.arange(0.1,1,0.1),2),sim_kwargs=None,rand_kwargs=None,k_params=None,max_len_fam=12,plot_exp=[False,False,False]):
    label     = []
    parents   = []
    children  = []
    inputs_l  = []
    inputs_m  = []
    inputs_lp = []
    simf      = []
    simp      = []
    simk      = []
    kmat_parent      = []
    kernels_children = []
    if gen_sim:
        for pr in p_rotate:
            # Generate similar images
            p, c = tpcs.generate_similar_stimuli(1-pr,sim_kwargs['seed_parent'],sim_kwargs['seed_child'],sim_kwargs['init_orient'],
                                                    sim_kwargs['num_parent'],sim_kwargs['num_child_per_parent'],transform_funs=sim_kwargs['fun_transform'],
                                                    transform_names=sim_kwargs['transform_names'],transform_cols=sim_kwargs['transform_cols'],gabor_num=sim_kwargs['gabor_num'],
                                                    loc_sigma=sim_kwargs['loc_sigma'],child_mode=sim_kwargs['child_mode'],sim_mode=sim_kwargs['sim_mode'])
            label.append(f'sim={np.round(1-pr,4)}')
            parents.append(p)
            children.append(c)

            # Compute similarities
            parent_simf, pair_simf, hist_pair_simf = tpcs.comp_feature_sim(p,c)
            simf.append((parent_simf, pair_simf, hist_pair_simf))
            
            parent_simp, pair_simp, hist_pair_simp = tpcs.comp_pixel_sim(p,c)
            simp.append((parent_simp, pair_simp, hist_pair_simp))

            parent_simk, pair_simk, hist_pair_simk, kmat_p, kernels_c = tpcs.comp_kernel_sim(p,c,k_params=k_params)
            simk.append((parent_simk, pair_simk, hist_pair_simk))
            kmat_parent.append(kmat_p)
            kernels_children.append(kernels_c)

            # Generate inputs for experiments
            i = tpcs.generate_inputs_from_stimuli(c,plot_exp=plot_exp)
            inputs_l.append(i[0])
            inputs_m.append(i[1])
            inputs_lp.append(i[2])

        print('Done generating similar inputs.\n')

    if gen_rand:
        # Generate random images (Homann style)
        num_im = rand_kwargs['num_parent']*rand_kwargs['num_child_per_parent']
        df_random, im_random = tpcs.generate_random_input(rand_kwargs['num_gabor'],num_im,rand_kwargs['im_seed'])
    
        rand_c = []
        for i in range(rand_kwargs['num_parent']):
            df_i = [df_random[j] for j in range(i*(max_len_fam+1),(i+1)*(max_len_fam+1))]
            im_i = [im_random[j] for j in range(i*(max_len_fam+1),(i+1)*(max_len_fam+1))]
            rand_c.append((df_i,im_i))
        label.append(f'rand')
        parents.append([])
        children.append(rand_c)

        # Compute similarity
        rand_simp = tpcs.comp_pixel_sim([],rand_c)

        parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk, _, rand_kernels_c = tpcs.comp_kernel_sim([],rand_c)
        rand_simk = (parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk)

        simf.append(([], [], []))
        simp.append(rand_simp)
        simk.append(rand_simk)
        kmat_parent.append([])
        kernels_children.append(rand_kernels_c)
        
        # Generate simulation input (L-sequence)
        rand_i = tpcs.generate_inputs_from_stimuli(rand_c,plot_exp=plot_exp)
        inputs_l.append(rand_i[0])
        inputs_m.append(rand_i[1])
        inputs_lp.append(rand_i[2])
    
        print('Done generating random inputs.\n')

    inputs = [inputs_l,inputs_m,inputs_lp]

    return label, parents, children, inputs, simf, simp, simk, kmat_parent, kernels_children

def generate_inputs(n_fam=[1,3,8,18,38],gen_sim=False,gen_rand=False,p_rotate=np.round(np.arange(0.1,1,0.1),2),sim_kwargs=None,rand_kwargs=None,k_params=None):
    label    = []
    parents  = []
    children = []
    inputs   = []
    simf     = []
    simp     = []
    simk     = []
    kmat_parent      = []
    kernels_children = []
    if gen_sim:
        for pr in p_rotate:
            # Generate similar images
            p, c, i = tpcs.generate_similar_input(1-pr,sim_kwargs['seed_parent'],sim_kwargs['seed_child'],sim_kwargs['init_orient'],
                                                    sim_kwargs['num_parent'],sim_kwargs['num_child_per_parent'],n_fam=n_fam,transform_funs=sim_kwargs['fun_transform'],
                                                    transform_names=sim_kwargs['transform_names'],transform_cols=sim_kwargs['transform_cols'],gabor_num=sim_kwargs['gabor_num'],
                                                    loc_sigma=sim_kwargs['loc_sigma'],child_mode=sim_kwargs['child_mode'])
            label.append(f'sim={np.round(1-pr,4)}')
            parents.append(p)
            children.append(c)
            inputs.append(i)

            # Compute similarities
            parent_simf, pair_simf, hist_pair_simf = tpcs.comp_feature_sim(p,c)
            simf.append((parent_simf, pair_simf, hist_pair_simf))
            
            parent_simp, pair_simp, hist_pair_simp = tpcs.comp_pixel_sim(p,c)
            simp.append((parent_simp, pair_simp, hist_pair_simp))

            parent_simk, pair_simk, hist_pair_simk, kmat_p, kernels_c = tpcs.comp_kernel_sim(p,c,k_params=k_params)
            simk.append((parent_simk, pair_simk, hist_pair_simk))
            kmat_parent.append(kmat_p)
            kernels_children.append(kernels_c)

        print('Done generating similar inputs.\n')

    if gen_rand:
        # Generate random images (Homann style)
        num_im = rand_kwargs['num_parent']*rand_kwargs['num_child_per_parent']
        df_random, im_random = tpcs.generate_random_input(rand_kwargs['num_gabor'],num_im,rand_kwargs['im_seed'])
    
        rand_c = []
        for i in range(rand_kwargs['num_parent']):
            df_i = [df_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1))]
            im_i = [im_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1))]
            rand_c.append((df_i,im_i))
        
        # Generate simulation input (L-sequence)
        rand_i = []
        for ii in range(rand_kwargs['num_parent']):
            gfam = [im_random[jj] for jj in range(ii*(len_fam+1),(ii+1)*(len_fam+1)-1)]
            gnov = [im_random[(ii+1)*(len_fam+1)-1]]

            vec_list, type_list = tpcs.generate_l_seq(gfam,gnov,n_fam,len_fam=len_fam)
            rand_i.append((vec_list,type_list))

        label.append(f'rand')
        parents.append([])
        children.append(rand_c)
        inputs.append(rand_i)

        # Compute similarity
        rand_simp = tpcs.comp_pixel_sim([],rand_c)

        parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk, _, rand_kernels_c = tpcs.comp_kernel_sim([],rand_c)
        rand_simk = (parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk)

        simf.append(([], [], []))
        simp.append(rand_simp)
        simk.append(rand_simk)
        kmat_parent.append([])
        kernels_children.append(rand_kernels_c)
    
        print('Done generating random inputs.\n')

    return label, parents, children, inputs, simf, simp, simk, kmat_parent, kernels_children

def get_data_stats(n_fam=[1,3,8,18,38],gen_sim=False,gen_rand=False,p_rotate=np.round(np.arange(0.1,1,0.1),2),sim_kwargs=None,rand_kwargs=None,k_params=None,inputs=None,recomp_data=False,recomp_stats=False,path_data=''):

    if recomp_data:
        all_data = []; all_kl = []; all_kwl = []

        # Run experiment for similar images
        if gen_sim:
            for i in range(len(p_rotate)):
                data, kl, kwl = tpcs.run_homann_l(n_fam,inputs[i],k_params,sim_kwargs['seed_parent'])
                df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
                df_data.to_csv(os.path.join(path_data,f'homann_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                all_data.append(df_data)
                all_kl.append(kl)
                all_kwl.append(kwl)
                
        # Run experiment for random images
        if gen_rand:
            data, kl, kwl = tpcs.run_homann_l(n_fam,inputs[-1],k_params,[rand_kwargs['im_seed']]*len(inputs[-1]))
            df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
            df_data.to_csv(os.path.join(path_data,f'homann_rand.csv'))
            all_data.append(df_data)
            all_kl.append(kl)
            all_kwl.append(kwl)

        print('Done running experiments.\n')

    else:
        # Load experiment data for similar images
        all_data = []
        if gen_sim:
            for i in range(len(p_rotate)):
                df_data = pd.read_csv(os.path.join(path_data,f'homann_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                all_data.append(df_data)
        
        # Load experiment data for random images
        if gen_rand:
            df_data = pd.read_csv(os.path.join(path_data,f'homann_rand.csv'))
            all_data.append(df_data)
    
        print('Done loading experiments.\n')
    
    if recomp_stats:

        field_vals   = (f'n_fam','')
        field_stats  = ('nt_norm','mean')
        field_stats2 = ('steady','mean')
        all_stats    = []

        # Compute statistics for similar images
        if gen_sim:
            for i in range(len(p_rotate)):
                stats_i = tpcs.comp_stats(all_data[i],steady=True)
                stats_i.to_csv(os.path.join(path_data,f'homann_stats-{p_rotate[i]}'.replace('.','') + '.csv'))
                all_stats.append(stats_i)
                
        # Compute statistics for random images
        if gen_rand:
            stats_rand = tpcs.comp_stats(all_data[-1],steady=True)
            stats_rand.to_csv(os.path.join(path_data,f'homan_stats-rand.csv'))
            all_stats.append(stats_rand)
        
        print('Done computing stats.\n')

    else:
        field_vals   = 'n_fam'
        field_stats  = 'nt_norm_mean'
        field_stats2 = 'steady_mean'
        all_stats    = []

        # Load statistics for similar images
        if gen_sim:
            for i in range(len(p_rotate)):
                col_extract = ['n_fam','nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','steady_mean','steady_std','steady_sem','sample_id']
                stats_i = pd.read_csv(os.path.join(path_data,f'homann_stats-{p_rotate[i]}'.replace('.','') + '.csv'),header=1,names=col_extract)
                all_stats.append(stats_i)
            
        # Load statistics for random images
        if gen_rand:
            stats_rand = pd.read_csv(os.path.join(path_data,f'homann_stats-rand.csv'),header=1,names=col_extract)
            all_stats.append(stats_rand)
        
        print('Done loading stats.\n')
    
    return all_data, all_stats, (field_vals,field_stats,field_stats2)

def get_data_stats_allexp(exp_type='l',gen_sim=False,gen_rand=False,p_rotate=np.round(np.arange(0.1,1,0.1),2),sim_kwargs=None,rand_kwargs=None,k_params=None,inputs=None,recomp_data=False,recomp_stats=False,path_data='',complex=False,complex_kwargs=None,save_data=True):

    exp_fun = eval(f'tpcs.run_homann_{exp_type}')
    exp_var = 'n_fam' if exp_type=='l' else 'n_im' if exp_type=='m' else 'dN'
    exp_val = [1,3,8,18,38] if exp_type=='l' else [3,6,9,12] if exp_type=='m' else list(np.array([0,22,44,66,88,110,143])/0.3)

    if recomp_data:
        all_data = []; all_kl = []; all_kwl = []

        # Run experiment for similar images
        if gen_sim:
            for i in range(len(p_rotate)):
                data, kl, kwl = exp_fun(exp_val,inputs[i],k_params,sim_kwargs['seed_parent'],complex=complex,complex_kwargs=complex_kwargs)
                df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
                if save_data: df_data.to_csv(os.path.join(path_data,f'homann-{exp_type}_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                all_data.append(df_data)
                all_kl.append(kl)
                all_kwl.append(kwl)
                
        # Run experiment for random images
        if gen_rand:
            data, kl, kwl = exp_fun(exp_val,inputs[-1],k_params,[rand_kwargs['im_seed']]*len(inputs[-1]),complex=complex,complex_kwargs=complex_kwargs)
            df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
            if save_data: df_data.to_csv(os.path.join(path_data,f'homann-{exp_type}_rand.csv'))
            all_data.append(df_data)
            all_kl.append(kl)
            all_kwl.append(kwl)

        print('Done running experiments.\n')

    else:
        # Load experiment data for similar images
        all_data = []
        if gen_sim:
            for i in range(len(p_rotate)):
                df_data = pd.read_csv(os.path.join(path_data,f'homann-{exp_type}_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                all_data.append(df_data)
        
        # Load experiment data for random images
        if gen_rand:
            df_data = pd.read_csv(os.path.join(path_data,f'homann-{exp_type}_rand.csv'))
            all_data.append(df_data)
    
        print('Done loading experiments.\n')
    
    if recomp_stats:

        field_vals  = (f'{exp_var}','')
        field_stats = ('tr_norm','mean') if exp_type=='lp' else ('nt_norm','mean')
        field_stats2 = ('steady','mean')
        all_stats = []

        # Compute statistics for similar images
        if gen_sim:
            for i in range(len(p_rotate)):
                stats_i = tpcs.comp_stats(all_data[i],steady=True,exp_var=exp_var)
                if save_data: stats_i.to_csv(os.path.join(path_data,f'homann-{exp_type}_stats-{p_rotate[i]}'.replace('.','') + '.csv'))
                all_stats.append(stats_i)
                
        # Compute statistics for random images
        if gen_rand:
            stats_rand = tpcs.comp_stats(all_data[-1],steady=True,exp_var=exp_var)
            if save_data: stats_rand.to_csv(os.path.join(path_data,f'homann-{exp_type}_stats-rand.csv'))
            all_stats.append(stats_rand)
        
        print('Done computing stats.\n')

    else:
        field_vals  = exp_var
        field_stats = 'tr_norm_mean' if exp_type=='lp' else 'nt_norm_mean'
        field_stats2 = 'steady_mean'
        all_stats = []

        # Load statistics for similar images
        if gen_sim:
            for i in range(len(p_rotate)):
                col_extract = [exp_var,'tr_mean','tr_std','tr_sem','tr_norm_mean','tr_norm_std','tr_norm_sem','steady_mean','steady_std','steady_sem','sample_id'] if exp_type=='lp' else [exp_var,'nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','steady_mean','steady_std','steady_sem','sample_id']
                stats_i = pd.read_csv(os.path.join(path_data,f'homann-{exp_type}_stats-{p_rotate[i]}'.replace('.','') + '.csv'),header=1,names=col_extract)
                all_stats.append(stats_i)
            
        # Load statistics for random images
        if gen_rand:
            stats_rand = pd.read_csv(os.path.join(path_data,f'homann-{exp_type}_stats-rand.csv'),header=1,names=col_extract)
            all_stats.append(stats_rand)
        
        print('Done loading stats.\n')
    
    return all_data, all_stats, (field_vals,field_stats,field_stats2)

if __name__=='__main__':
    # Set what should be plotted (random images, similarity levels)
    plot_similarities = True
    plot_l_exp  = True
    plot_m_exp  = False
    plot_lp_exp = False
    if plot_l_exp or plot_m_exp or plot_lp_exp:
        recomp_data = True
        recomp_stats = True
    generate_sim  = True
    generate_rand = True
    p_rotate = np.array([0.042, 0.083, 0.167, 0.25]) #np.array([0.042, 0.083, 0.125, 0.167, 0.21, 0.25])#np.array([0.2,0.4,0.6,0.8]) #np.round(np.arange(0.1,1,0.1),2)

    # Set colors
    color_list = []
    if generate_sim:
        color_list = color_list + vis.prep_cmap('Blues',len(p_rotate)-1)
    if generate_rand:
        color_list = color_list + ['k']

    # Set parameters for image generation
    num_parent = 50
    max_len_fam = 3
    num_child_per_parent = max_len_fam+1

    sim_kwargs = {'num_parent': num_parent,
                  'num_child_per_parent': num_child_per_parent,
                  'child_mode': 'fixed',
                  'sim_mode': 'exact',
                  'gabor_orient': 0, # pi/4 = 45 degrees
                  'gabor_diff': 0, #np.linspace(0,np.pi/2,5), i.e. between 0 and 90 degrees
                  'gabor_num': (8,3), # (4,4)
                  'loc_sigma': (0,0) # (0,0)
                  }
    init_seed = 12345
    sim_kwargs['init_orient'] = sim_kwargs['gabor_orient'] - sim_kwargs['gabor_diff']
    sim_kwargs['seed_parent'] = h_in.get_random_seed(length=5,n=sim_kwargs['num_parent'],init_seed=init_seed)        
    sim_kwargs['seed_child']  = h_in.get_random_seed(length=5,n=sim_kwargs['num_parent']*sim_kwargs['num_child_per_parent'],init_seed=init_seed+1)

    sim_kwargs['fun_transform']   = [gs.transform_identity,
                                        gs.transform_rotate_left] 
    # ,
    #                                     gs.transform_rotate_right,
    #                                     gs.transform_shift_right]
    sim_kwargs['transform_names'] = ['identity','rotation left'] # ,'rotation right','shift right']
    all_cols = vis.prep_cmap_discrete('tab20')
    sim_kwargs['transform_cols']  = [all_cols[0], all_cols[6]] #,all_cols[7], all_cols[9]]

    rand_kwargs = {'num_parent': num_parent,
                   'num_child_per_parent': num_child_per_parent,
                   'num_gabor': 24,
                   'im_seed': 98765
                  }

    # Set paths
    path_data = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-11_all-exp_parent-child-exact24/'
    sl.make_long_dir(path_data)

    path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/2024-11_all-exp_parent-child-exact24/'
    sl.make_long_dir(path_fig)

    # Make kernel model for kernel-based similarity + simulation
    ksig = 1
    cdens = 4
    k_params = gknov2.init_gabor_knov(gnum=4,k_type='triangle',ksig=ksig,kcenter=1,cdens=cdens,seed=init_seed,rng=None,mask=True,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='equidistant_fixed',contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,add_empty=False,debug=False)
    
    # Generate children for different p_rotate
    label, parents, children, inputs, simf, simp, simk, kmat_parent, kernels_children = generate_inputs_allexp(gen_sim=generate_sim,gen_rand=generate_rand,p_rotate=p_rotate,sim_kwargs=sim_kwargs,rand_kwargs=rand_kwargs,k_params=k_params,max_len_fam=max_len_fam,plot_exp=[plot_l_exp,plot_m_exp,plot_lp_exp])
    # label, parents, children, inputs, simf, simp, simk, kmat_parent, kernels_children = generate_inputs(n_fam=[1,3,8,18,38],gen_sim=generate_sim,gen_rand=generate_rand,p_rotate=p_rotate,sim_kwargs=sim_kwargs,rand_kwargs=rand_kwargs,k_params=k_params)

    # Plot distribution of pairwise (kernel) similarities
    if plot_similarities:
        f,ax = plt.subplots(1,1,figsize=(3,3))
        for i in range(len(simk)):
            simk_i = simk[i][2]
            av_pair_i = [np.mean(simk_i[j]) for j in range(len(simk_i))]
            av_i = np.mean(av_pair_i)
            std_i = np.std(av_pair_i)
            ax.hist(av_pair_i,color=color_list[i],alpha=0.2)
            ax.axvline(x=av_i,color=color_list[i],ls='-',label=label[i])
            ax.axvline(x=av_i-std_i,color=color_list[i],ls=':')
            ax.axvline(x=av_i+std_i,color=color_list[i],ls=':')
        ax.set_xlabel('Av. pairwise similarity')
        ax.set_ylabel('Count')
        f.legend()
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'sim.svg'),bbox_inches='tight')
        f.savefig(os.path.join(path_fig,'sim.eps'),bbox_inches='tight')

    if plot_l_exp:
        # all_data, all_stats, fields = get_data_stats(n_fam=[1,3,8,18,38],gen_sim=generate_sim,gen_rand=generate_rand,p_rotate=p_rotate,sim_kwargs=sim_kwargs,rand_kwargs=rand_kwargs,k_params=k_params,inputs=inputs,recomp_data=recomp_data,recomp_stats=recomp_stats,path_data=path_data)
        all_data, all_stats, fields = get_data_stats_allexp(exp_type='l',gen_sim=generate_sim,gen_rand=generate_rand,p_rotate=p_rotate,sim_kwargs=sim_kwargs,rand_kwargs=rand_kwargs,k_params=k_params,inputs=inputs[0],recomp_data=recomp_data,recomp_stats=recomp_stats,path_data=path_data)

        field_vals   = fields[0]
        field_stats  = fields[1]
        field_stats2 = fields[2]

        f,ax = plt.subplots(1,1,figsize=(3,3))
        for i in range(len(all_stats)):
            stats_i = all_stats[i]
            nov_i_mean = stats_i[[field_vals,field_stats]].groupby([field_vals]).mean().reset_index()
            nov_i_std = stats_i[[field_vals,field_stats]].groupby([field_vals]).std().reset_index()
            ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'-',c=color_list[i])
            ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'o',c=color_list[i])
            ax.fill_between(x=nov_i_std[field_vals],y1=nov_i_mean[field_stats]-nov_i_std[field_stats],y2=nov_i_mean[field_stats]+nov_i_std[field_stats],color=color_list[i],alpha=0.2)
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'l-exp.svg'),bbox_inches='tight')
        f.savefig(os.path.join(path_fig,'l-exp.eps'),bbox_inches='tight')
    
    if plot_m_exp:
        all_data, all_stats, fields = get_data_stats_allexp(exp_type='m',gen_sim=generate_sim,gen_rand=generate_rand,p_rotate=p_rotate,sim_kwargs=sim_kwargs,rand_kwargs=rand_kwargs,k_params=k_params,inputs=inputs[1],recomp_data=recomp_data,recomp_stats=recomp_stats,path_data=path_data)
        
        field_vals   = fields[0]
        field_stats  = fields[1]
        field_stats2 = fields[2]

        f,ax = plt.subplots(1,1,figsize=(3,3))
        for i in range(len(all_stats)):
            stats_i = all_stats[i]
            nov_i_mean = stats_i[[field_vals,field_stats]].groupby([field_vals]).mean().reset_index()
            nov_i_std = stats_i[[field_vals,field_stats]].groupby([field_vals]).std().reset_index()
            ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'-',c=color_list[i])
            ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'o',c=color_list[i])
            ax.fill_between(x=nov_i_std[field_vals],y1=nov_i_mean[field_stats]-nov_i_std[field_stats],y2=nov_i_mean[field_stats]+nov_i_std[field_stats],color=color_list[i],alpha=0.2)
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'m-exp.svg'),bbox_inches='tight')
        f.savefig(os.path.join(path_fig,'m-exp.eps'),bbox_inches='tight')

    if plot_lp_exp:
        all_data, all_stats, fields = get_data_stats_allexp(exp_type='lp',gen_sim=generate_sim,gen_rand=generate_rand,p_rotate=p_rotate,sim_kwargs=sim_kwargs,rand_kwargs=rand_kwargs,k_params=k_params,inputs=inputs[2],recomp_data=recomp_data,recomp_stats=recomp_stats,path_data=path_data)
        
        field_vals   = fields[0]
        field_stats  = fields[1]
        field_stats2 = fields[2]

        f,ax = plt.subplots(1,1,figsize=(3,3))
        for i in range(len(all_stats)):
            stats_i = all_stats[i]
            nov_i_mean = stats_i[[field_vals,field_stats]].groupby([field_vals]).mean().reset_index()
            nov_i_std = stats_i[[field_vals,field_stats]].groupby([field_vals]).std().reset_index()
            ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'-',c=color_list[i])
            ax.plot(nov_i_mean[field_vals],nov_i_mean[field_stats],'o',c=color_list[i])
            ax.fill_between(x=nov_i_std[field_vals],y1=nov_i_mean[field_stats]-nov_i_std[field_stats],y2=nov_i_mean[field_stats]+nov_i_std[field_stats],color=color_list[i],alpha=0.2)
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'lp-exp.svg'),bbox_inches='tight')
        f.savefig(os.path.join(path_fig,'lp-exp.eps'),bbox_inches='tight')




    
        

        

    


        
