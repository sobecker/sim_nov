import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import colors
import os

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/') 

import src.models.snov.gabor_stimuli as gs
import src.models.snov.run_gabor_knov2 as gknov2
import src.models.snov.run_gabor_knov_complex as gknovc
import src.models.snov.kernel_nov_vec as knov_vec
import src.fitting_neural.create_homann_input as h_in
import src.utils.saveload as sl
import src.fitting_neural.simulate_data as sd
import src.utils.visualization as vis

############################################################################################################################
def cosine_sim(a,b):
    return np.dot(a.flatten(),b.flatten()) / (np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()))

def orientation_sim_pergabor(a,b):
    return np.cos(a.flatten()-b.flatten())

def orientation_sim(a,b):
    return np.mean(np.heaviside(0.25*np.pi-np.abs(a.flatten()-b.flatten())%(2*np.pi),0))

def orientation_phase_sim(a,b): # num_samples x 2 (orientation, phase)
    return np.mean([np.mean(np.cos(a[0,:].flatten()-b[0,:].flatten())),np.mean(np.abs(a[1,:].flatten()-b[1,:].flatten()))])

############################################################################################################################
# Compute kernel activation matrix
def compute_kernel_matrix(stim,k_params,idx=True,conv=True,parallel_k=False):   
    ksig0 = k_params['ksig']   
    k = k_params['k']
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))
    # Initialize novelty
    if conv:
        cdens = k_params['cdens']
        num_conv = stim_unique[0,::cdens,::cdens].size
        knum = len(k) * num_conv
        kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=stim_unique,parallel=parallel_k)
    else:
        knum = len(k)
        kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov(k,ksig0,seq=stim_unique,parallel=parallel_k)
    return kmat_seq

# Compute parent-child similarity
def comp_parent_sim(parent,list_children,sim_meas='cosine_sim'):
    parent_sim = []
    for i in range(len(list_children)):
        parent_sim.append(eval(sim_meas)(parent.flatten(),list_children[i].flatten()))
    return parent_sim

# Compute pairwise similarity
def comp_pairwise_sim(list_children,sim_meas='cosine_sim'):
    pairwise_sim = np.ones((len(list_children),len(list_children)))
    for i in range(len(list_children)):
        for j in range(i+1,len(list_children)):
            sim_ij = eval(sim_meas)(list_children[i].flatten(),list_children[j].flatten())
            pairwise_sim[i][j] = sim_ij
            pairwise_sim[j][i] = sim_ij
    return pairwise_sim

def plot_parent_sim(parent_sim,f,ax):
    # Plot parent and pairwise similarity
    bars = ax.bar(np.arange(len(parent_sim)),parent_sim,color='grey')
    # a_mu = ax[0].axhline(np.mean(parent_sim),color='black',linestyle='--')
    ax.set_title('Parent-child similarity')
    ax.set_xlabel('Child stimulus')
    ax.set_ylabel('Similarity')
    ax.set_ylim([np.max([0,np.min(parent_sim)-0.1]),np.min([1,np.max(parent_sim)+0.1])])
    ax.bar_label(bars,fmt='%.2f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend([f'Mean={np.round(np.mean(parent_sim),2)}'],loc='upper right')

def plot_pairwise_sim_matrix(pairwise_sim,f,ax):
    asim = ax.imshow(pairwise_sim,cmap='bwr',norm=colors.CenteredNorm(vcenter=np.mean(pairwise_sim.flatten())))    
    for i in range(pairwise_sim.shape[0]):
        for j in range(pairwise_sim.shape[1]):
            ax.text(j, i, np.round(pairwise_sim[i, j],2), ha="center", va="center", color="k")
    ax.set_title('Pairwise similarity of children')
    ax.set_xticks(np.arange(pairwise_sim.shape[0]))
    ax.set_yticks(np.arange(pairwise_sim.shape[1]))
    f.colorbar(asim, ax=ax,shrink=0.7)

def plot_pairwise_sim_hist(all_sim,f,ax,title='Pairwise similarity distribution',xlabel='Similarity',color='black',legend=True):
    ax.hist(all_sim,bins=30,color=color,alpha=1)
    ylim = ax.get_ylim()
    mu = np.mean(all_sim)
    sigma = np.std(all_sim)
    a_mu = ax.axvline(mu,color=color,linestyle='--')
    a_sigma = ax.fill_between([mu-sigma,mu+sigma],[0]*2,[ylim[1]+0.1]*2,color=color,alpha=0.2)
    ax.set_ylim(ylim)    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend: ax.legend([a_mu,a_sigma],[f'Mean={np.round(mu,2)}',f'Std={np.round(sigma,2)}'],loc='upper right')

def plot_sim(parent_sim, pairwise_sim):

    f,ax = plt.subplots(1,3,figsize=(20,6))

    # Plot parent and pairwise similarity
    plot_parent_sim(parent_sim,f,ax[0])

    # Plot pairwise similarity
    plot_pairwise_sim_matrix(pairwise_sim,f,ax[1])

    # Plot pairwise similarity histogram
    all_sim = pairwise_sim[np.triu_indices(pairwise_sim.shape[0],k=1)].flatten()
    plot_pairwise_sim_hist(all_sim,f,ax[2])

    f.tight_layout()

############################################################################################################################
def generate_l_seq(gfam,gnov,n_fam=[1,3,8,18,38],sequence_mode='seq',len_fam=3):
    # Generate input sequence
    if sequence_mode=='app':
        mode_factor = len(n_fam)
        modes = np.arange(mode_factor,dtype=int)
    else:
        mode_factor = 1
        modes = np.zeros(len(n_fam),dtype=int)

    vec_list  = []
    type_list = []
    for i, m in zip(range(len(n_fam)), modes):

        # Pick new set of familiar and novel images
        if len_fam==1: gfam_m = [gfam[m]]
        else:          gfam_m = gfam[m*len_fam:(m+1)*len_fam]
        gnov_m = gnov[m]

        # Create sequence
        stim_unique = np.stack(gfam_m + [gnov_m])
        stim_idx    = (list(np.arange(len(gfam_m)))*(n_fam[i]+2))[:-1] + [len(gfam_m)] + list(np.arange(len(gfam_m)))*2 # n_fam+1 repetitions of fam. sequ.; fam. sequ. with novel image substituted; 1 repetition of fam. sequ.
        seq_vec     = (stim_unique,stim_idx)
        stim_type   = (['fam']*len_fam*(n_fam[i]+2))[:-1] + ['nov'] + ['fam']*len_fam*2
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, type_list

def generate_m_seq(gfam,gnov,len_fam=[3,6,9,12],sequence_mode='seq',n_fam=17):
    # Generate input sequence
    if sequence_mode=='app':
        mode_factor = len(len_fam)
        modes = np.arange(mode_factor,dtype=int)
    else:
        mode_factor = 1
        modes = np.zeros(len(len_fam),dtype=int)

    # Create sequences
    vec_list  = []
    type_list = []
    ic = 0
    for i, m in zip(range(len(len_fam)),modes):
        # Pick new set of familiar and novel images
        gfam_m = gfam[ic:ic+len_fam[i]]
        if len_fam[i]==1: gfam_m = [gfam_m]
        gnov_m = gnov[m]
        ic += m*len_fam[i]
        # Create sequence
        stim_unique = np.stack(gfam_m + [gnov_m])
        stim_idx    = (list(np.arange(len(gfam_m)))*(n_fam+1))[:-1] + [len(gfam_m)] + list(np.arange(len(gfam_m)))*2
        seq_vec     = (stim_unique,stim_idx)
        stim_type   = (['fam']*len_fam[i]*(n_fam+1))[:-1] + ['nov'] + ['fam']*len_fam[i]*2
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, type_list

def generate_lp_seq(gfam,gnov,dN=list(np.array([0,22,44,66,88,110,143])/0.3),sequence_mode='seq',n_fam=22,len_fam=3):
    # Generate input sequence
    if sequence_mode=='app':
        mode_factor = len(dN)
        modes = np.arange(mode_factor,dtype=int)
    else:
        mode_factor = 1
        modes = np.zeros(len(dN),dtype=int)

    vec_list = []
    type_list = []
    for i, m in zip(range(len(dN)),modes):
        # Chose familiar and novel images
        if len_fam==1: 
            gfam_m = [gfam[m]]
            gnov_m = [gnov[m]]
        else:          
            gfam_m = gfam[m*len_fam:(m+1)*len_fam]
            gnov_m = gnov[m*len_fam:(m+1)*len_fam]
        # Create sequence
        rep_nov = int(np.round((dN[i]/len_fam)))
        stim_unique = np.stack(gfam_m + gnov_m)
        stim_idx    = list(np.arange(len(gfam_m)))*n_fam + list(len(gfam_m)+np.arange(len(gnov_m)))*rep_nov + list(np.arange(len(gfam_m)))*5
        seq_vec     = (stim_unique,stim_idx)
        stim_type   = ['fam']*len_fam*n_fam + ['nov']*len_fam*rep_nov + ['fam_r']*len_fam*5
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, type_list


############################################################################################################################
def generate_similar_stimuli(sim_value,seed_parent,seed_child,init_orient,num_parent,num_child_per_parent,transform_funs=[gs.transform_identity,gs.transform_rotate_left],transform_names = ['identity','rotation'],transform_cols  = ['blue','red'],gabor_num=(4,4),loc_sigma=(0,0),child_mode='fixed',sim_mode='exact'):
    # Children parameters
    transform_probs = [sim_value] + [(1-sim_value)/(len(transform_funs)-1)]*(len(transform_funs)-1)

    # Plot stimuli
    plot_stim = False

    # Generate input sequences
    all_parents = []
    all_children = []

    for i in range(num_parent):
        # Generate parent stimulus
        parent, _ = gs.generate_teststim_parent(gs.dim_ranges_rad,init_orient=init_orient,gabor_num=gabor_num,rng=np.random.default_rng(seed_parent[i]),loc_sigma=loc_sigma)
        df_parent = pd.DataFrame(dict(zip(gs.dim_names,parent)))
        im_parent, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],parent,resolution=100,magn=1)
        all_parents.append((df_parent,im_parent))

        # Plot parent stimulus
        if plot_stim:
            f,ax = plt.subplots(1,1,figsize=(8,16))
            ax.imshow(im_parent,cmap='gray',vmin=-1,vmax=1,origin='lower')
            ax.axis('off')
            ax.set_title(f'Parent stimulus {i}')
        
        # Generate child stimuli (four for L-experiment)
        if sim_mode=='exact':
            children, set_transform = gs.generate_teststim_children_exact(parent,num_child=num_child_per_parent,fun_transform=transform_funs[1:],prob_transform=transform_probs[1:],rng=np.random.default_rng(seed_child[i]),mode=child_mode)
        else:
            children, set_transform = gs.generate_teststim_children(parent,num_child=num_child_per_parent,fun_transform=transform_funs,prob_transform=transform_probs,rng=np.random.default_rng(seed_child[i]),mode=child_mode)
        df_children = [pd.DataFrame(dict(zip(gs.dim_names,children[i]))) for i in range(len(children))]  
        im_children = []
        for j in range(len(children)):
            im_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],children[j],resolution=100,magn=1)
            im_children.append(im_child)

            # Plot children
            if plot_stim:
                f,ax = plt.subplots(1,1,figsize=(8,16))
                ax.imshow(im_child,cmap='gray',vmin=-1,vmax=1,origin='lower')
                ax.axis('off')
                # Plot type of transform for each Gabor
                for k in range(len(set_transform[j])):
                    xloc = (children[j][4,set_transform[j][k]] + 130)*200/260
                    yloc = (children[j][5,set_transform[j][k]] + 20)*100/90
                    ax.scatter(xloc,yloc,s=30,color=transform_cols[k],label=transform_names[k])
                ax.legend(bbox_to_anchor=(1.2,1),loc='upper right')
                ax.set_title(f'Child stimulus {j} for parent {i}')  
        
        all_children.append((df_children,im_children))
    
    return all_parents, all_children

def generate_inputs_from_stimuli(all_children,sequence_mode='seq',plot_exp=[False,False,False]):

    l_inputs = []; m_inputs = []; lp_inputs = []

    for i in range(len(all_children)):
        im_children = all_children[i][1]

        len_fam = 3
        max_len_fam = 12

        # L-exp inputs
        if plot_exp[0]:
            gfam = [im_children[j] for j in range(len_fam)]
            gnov = [im_children[len_fam]]
            vec_list, type_list = generate_l_seq(gfam,gnov,sequence_mode=sequence_mode)
            l_inputs.append((vec_list,type_list))

        # M-exp inputs
        if plot_exp[1]:
            gfam = [im_children[j] for j in range(max_len_fam)]
            gnov = [im_children[max_len_fam]]
            vec_list, type_list = generate_m_seq(gfam,gnov,sequence_mode=sequence_mode)
            m_inputs.append((vec_list,type_list))

        # L'-exp inputs
        if plot_exp[2]:
            gfam = [im_children[j] for j in range(len_fam)]
            gnov = [im_children[len_fam+j] for j in range(len_fam)]
            vec_list, type_list = generate_lp_seq(gfam,gnov,sequence_mode=sequence_mode)
            lp_inputs.append((vec_list,type_list))
        
    all_inputs = [l_inputs, m_inputs, lp_inputs]

    return all_inputs

def generate_similar_input(sim_value,seed_parent,seed_child,init_orient,num_parent,num_child_per_parent,n_fam=[1,3,8,18,38],transform_funs=[gs.transform_identity,gs.transform_rotate_left],transform_names = ['identity','rotation'],transform_cols  = ['blue','red'],gabor_num=(4,4),loc_sigma=(0,0),child_mode='fixed'):
    # Children parameters
    transform_probs = [sim_value] + [(1-sim_value)/(len(transform_funs)-1)]*(len(transform_funs)-1)

    # Experiment parameters
    sequence_mode = 'seq'
    len_fam = 3

    # Plot stimuli
    plot_stim = False

    # Generate input sequences
    all_parents = []
    all_children = []
    all_inputs = []

    for i in range(num_parent):
        # Generate parent stimulus
        parent, _ = gs.generate_teststim_parent(gs.dim_ranges_rad,init_orient=init_orient,gabor_num=gabor_num,rng=np.random.default_rng(seed_parent[i]),loc_sigma=loc_sigma)
        df_parent = pd.DataFrame(dict(zip(gs.dim_names,parent)))
        im_parent, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],parent,resolution=100,magn=1)
        all_parents.append((df_parent,im_parent))

        # Plot parent stimulus
        if plot_stim:
            f,ax = plt.subplots(1,1,figsize=(8,16))
            ax.imshow(im_parent,cmap='gray',vmin=-1,vmax=1,origin='lower')
            ax.axis('off')
            ax.set_title(f'Parent stimulus {i}')
        
        # Generate child stimuli (four for L-experiment)
        children, set_transform = gs.generate_teststim_children_exact(parent,num_child=num_child_per_parent,fun_transform=transform_funs[1:],prob_transform=transform_probs[1:],rng=np.random.default_rng(seed_child[i]),mode='fixed')
        df_children = [pd.DataFrame(dict(zip(gs.dim_names,children[i]))) for i in range(len(children))]  
        im_children = []
        for j in range(len(children)):
            im_child, _ = gs.comp_gabor(gs.dim_ranges[4],gs.dim_ranges[5],children[j],resolution=100,magn=1)
            im_children.append(im_child)

            # Plot children
            if plot_stim:
                f,ax = plt.subplots(1,1,figsize=(8,16))
                ax.imshow(im_child,cmap='gray',vmin=-1,vmax=1,origin='lower')
                ax.axis('off')
                # Plot type of transform for each Gabor
                for k in range(len(set_transform[j])):
                    xloc = (children[j][4,set_transform[j][k]] + 130)*200/260
                    yloc = (children[j][5,set_transform[j][k]] + 20)*100/90
                    ax.scatter(xloc,yloc,s=30,color=transform_cols[k],label=transform_names[k])
                ax.legend(bbox_to_anchor=(1.2,1),loc='upper right')
                ax.set_title(f'Child stimulus {j} for parent {i}')  
        
        all_children.append((df_children,im_children))

        gfam = [im_children[j] for j in range(len_fam)]
        gnov = [im_children[len_fam]]

        vec_list, type_list = generate_l_seq(gfam,gnov,n_fam,sequence_mode=sequence_mode,len_fam=len_fam)
        
        all_inputs.append((vec_list,type_list))
    return all_parents, all_children, all_inputs

############################################################################################################################
def comp_kernel_sim(all_parents,all_children,ksig=1,cdens=8,k_params=None):
    # Create standard kernel model
    if k_params is None:
        k_params = gknov2.init_gabor_knov(gnum=4,k_type='triangle',ksig=ksig,kcenter=1,cdens=cdens,seed=12345,rng=None,mask=True,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='equidistant',contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,add_empty=False,debug=False)

    all_parent_sim = []
    all_pairwise_sim = []
    all_hist_pairwise_sim = []
    all_kmat_parent = []
    all_kernels_children = []
    for i in range(len(all_children)):

        im_children = all_children[i][1]
        vec_children = np.stack(im_children)
        kmat_children = compute_kernel_matrix((vec_children,None),k_params,idx=True,conv=True,parallel_k=False)
        kernels_children = [kmat_children[:,i] for i in range(kmat_children.shape[1])]
        all_kernels_children.append(kernels_children)

        pairwise_sim = comp_pairwise_sim(kernels_children)
        hist_pairwise_sim = pairwise_sim[np.triu_indices(pairwise_sim.shape[0],k=1)].flatten()
        all_pairwise_sim.append(pairwise_sim)
        all_hist_pairwise_sim.append(hist_pairwise_sim)

        if len(all_parents)>0:  
            im_parent = all_parents[i][1]
            vec_parent = im_parent[None,:,:]
            kmat_parent = compute_kernel_matrix((vec_parent,None),k_params,idx=True,conv=True,parallel_k=False).flatten()
            all_kmat_parent.append(kmat_parent)
            parent_sim = comp_parent_sim(kmat_parent,kernels_children)
            all_parent_sim.append(parent_sim)

    return all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim, all_kmat_parent, all_kernels_children

############################################################################################################################
def comp_pixel_sim(all_parents,all_children): 

    all_parent_sim = []
    all_pairwise_sim = []
    all_hist_pairwise_sim = []

    for i in range(len(all_children)):

        im_children = all_children[i][1]
        pairwise_sim = comp_pairwise_sim(im_children)
        hist_pairwise_sim = pairwise_sim[np.triu_indices(pairwise_sim.shape[0],k=1)].flatten()
        all_pairwise_sim.append(pairwise_sim)
        all_hist_pairwise_sim.append(hist_pairwise_sim)

        if len(all_parents)>0:
            im_parent = all_parents[i][1]
            parent_sim = comp_parent_sim(im_parent,im_children)
            all_parent_sim.append(parent_sim)

    return all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim

############################################################################################################################
def comp_feature_sim(all_parents,all_children,sim_meas='orientation_sim'):
    sim_features = ['orientation']

    all_parent_sim = []
    all_pairwise_sim = []
    all_hist_pairwise_sim = []
    for i in range(len(all_children)):

        # Get children features 
        df_children = all_children[i][0]
        list_children_features = [df_children[j][sim_features].to_numpy() for j in range(len(df_children))]  

        # Compute similarities
        pairwise_sim = comp_pairwise_sim(list_children_features,sim_meas='orientation_sim')
        hist_pairwise_sim = pairwise_sim[np.triu_indices(pairwise_sim.shape[0],k=1)].flatten()
        all_pairwise_sim.append(pairwise_sim)
        all_hist_pairwise_sim.append(hist_pairwise_sim)

        # Get parent features
        if len(all_parents)>0:
            df_parent = all_parents[i][0]
            parent_features = df_parent[sim_features].to_numpy()  

            parent_sim = comp_parent_sim(parent_features,list_children_features,sim_meas='orientation_sim')
            all_parent_sim.append(parent_sim)

    return all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim

############################################################################################################################
def plot_similar_inputs(all_parents,all_children,plot=True):
    # Compute parent-child and pairwise similarity distribution (feature-similarity)
    all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim = comp_feature_sim(all_parents,all_children,sim_meas='orientation_sim')

    # Plot histogram of similarities
    if plot:
        f,ax = plt.subplots(1,2,figsize=(12,4))
        plot_pairwise_sim_hist(np.concatenate(all_parent_sim),f,ax[0],title='Parent-child similarity distribution',xlabel='Orientation similarity (cosine)')
        plot_pairwise_sim_hist(np.concatenate(all_hist_pairwise_sim),f,ax[1],title='Pairwise similarity distribution',xlabel='Orientation similarity (cosine)')

    return all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim

# all_inputs: List of all inputs
# level 0: number of experiments (with different random seed)
# level 1: inputs and input types
# level 2: number of sub-experiments (one for each value of L)
# level 3: stimulus unique and stimulus index for each sub-experiment

############################################################################################################################
def run_homann_l(n_fam,all_inputs,k_params,seed_parent,complex=False,complex_kwargs=None):
    data = []; kl = []; kwl = []
    for i in range(len(all_inputs)):
        print(f'Running realization {i} of L-experiment.\n')
        stim = all_inputs[i][0]
        itype = all_inputs[i][1]

        data_i = []; kl_i = []; kwl_i = []
        for j in range(len(stim)):
            print(f'Running experiment condition {j} of experiment realization {i}.\n')
            stim_j = stim[j]
            if complex: 
                data_ij, kl_gabor_ij, kwl_gabor_ij = gknovc.run_gabor_knov_withparams_flr(stim_j,k_params,idx=True,kwargs=complex_kwargs)
            else:
                data_ij, kl_gabor_ij, kwl_gabor_ij = gknov2.run_gabor_knov_withparams_flr(stim_j,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None,stop_nokernels=True)
            data_ij['n_fam'] = [n_fam[j]]*len(data_ij) 
            data_ij['seed']  = [seed_parent[i]]*len(data_ij)  
            data_ij['stim_type'] = itype[j] 
            data_ij['sample_id'] = [i]*len(data_ij)    
            data_i.append(data_ij)
            kl_i.append(kl_gabor_ij)
            kwl_i.append(kwl_gabor_ij)
        data.append(data_i)
        kl.append(kl_i)
        kwl.append(kwl_i)
    return data, kl, kwl

def run_homann_m(n_im,all_inputs,k_params,seed_parent,complex=False,complex_kwargs=None):
    data = []; kl = []; kwl = []
    for i in range(len(all_inputs)):
        print(f'Running realization {i} of M-experiment.\n')
        stim = all_inputs[i][0]
        itype = all_inputs[i][1]

        data_i = []; kl_i = []; kwl_i = []
        for j in range(len(stim)):
            print(f'Running experiment condition {j} of experiment realization {i}.\n')
            stim_j = stim[j]
            if complex: 
                data_ij, kl_gabor_ij, kwl_gabor_ij = gknovc.run_gabor_knov_withparams_flr(stim_j,k_params,idx=True,kwargs=complex_kwargs)
            else:
                data_ij, kl_gabor_ij, kwl_gabor_ij = gknov2.run_gabor_knov_withparams_flr(stim_j,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None,stop_nokernels=True)
            data_ij['n_im'] = [n_im[j]]*len(data_ij) 
            data_ij['seed']  = [seed_parent[i]]*len(data_ij)  
            data_ij['stim_type'] = itype[j] 
            data_ij['sample_id'] = [i]*len(data_ij)    
            data_i.append(data_ij)
            kl_i.append(kl_gabor_ij)
            kwl_i.append(kwl_gabor_ij)
        data.append(data_i)
        kl.append(kl_i)
        kwl.append(kwl_i)
    return data, kl, kwl

def run_homann_lp(dN,all_inputs,k_params,seed_parent,complex=False,complex_kwargs=None):
    data = []; kl = []; kwl = []
    for i in range(len(all_inputs)):
        print(f'Running realization {i} of L\'-experiment.\n')
        stim = all_inputs[i][0]
        itype = all_inputs[i][1]

        data_i = []; kl_i = []; kwl_i = []
        for j in range(len(stim)):
            print(f'Running experiment condition {j} of experiment realization {i}.\n')
            stim_j = stim[j]
            if complex: 
                data_ij, kl_gabor_ij, kwl_gabor_ij = gknovc.run_gabor_knov_withparams_flr(stim_j,k_params,idx=True,kwargs=complex_kwargs)
            else:
                data_ij, kl_gabor_ij, kwl_gabor_ij = gknov2.run_gabor_knov_withparams_flr(stim_j,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None,stop_nokernels=True)
            data_ij['dN'] = [dN[j]]*len(data_ij) 
            data_ij['seed']  = [seed_parent[i]]*len(data_ij)  
            data_ij['stim_type'] = itype[j] 
            data_ij['sample_id'] = [i]*len(data_ij)    
            data_i.append(data_ij)
            kl_i.append(kl_gabor_ij)
            kwl_i.append(kwl_gabor_ij)
        data.append(data_i)
        kl.append(kl_i)
        kwl.append(kwl_i)
    return data, kl, kwl

############################################################################################################################
def plot_homann_l(n_fam,data,stats,all_parent_sim,all_pairwise_sim,all_hist_pairwise_sim,sim_level):
    alpha_list = np.linspace(1,0.2,len(n_fam))
    samples = data.sample_id.unique()  

    for i,sample_i in enumerate(samples):
        f,ax = plt.subplots(1,5,figsize=(5*4,4))

        # Plot individual simulations
        for j in range(len(n_fam)):
            data_ij = data.loc[(data['sample_id']==sample_i) & (data['n_fam']==n_fam[j])]  
            ax[0].plot(np.arange(len(data_ij)),data_ij['nt'],color='k',alpha=alpha_list[j])
            ax[0].set_title(f'L={n_fam[j]}')
        ax[0].set_xlabel('Time steps')
        ax[0].set_ylabel('Response traces')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        # Plot novelty responses
        # data_i = data.loc[data['sample_id']==sample_i] 
        # _, nov_resp_i = sd.get_nov_response(data_i,'n_fam')
        # nov_resp_i = nov_resp_i.reset_index()
        # nov_resp_i['sample_id'] = sample_i
        # nov_resp_all.append(nov_resp_i)
        nov_resp_i = stats.loc[stats['sample_id']==sample_i]
        ax[1].plot(nov_resp_i['n_fam'],nov_resp_i[('nt_norm','mean')],'o',color='k')
        ax[1].plot(nov_resp_i['n_fam'],nov_resp_i[('nt_norm','mean')],'--',color='k')

        # Plot similarities
        plot_parent_sim(all_parent_sim[i],f,ax[-3])
        plot_pairwise_sim_matrix(all_pairwise_sim[i],f,ax[-2])
        plot_pairwise_sim_hist(all_hist_pairwise_sim[i],f,ax[-1])

        f.suptitle(f'Experiment run {i}')
        f.tight_layout()
        path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/2024-10_single-trace-sim/sim-{sim_level}'.replace('.','')
        sl.make_long_dir(path_fig) 
        f.savefig(os.path.join(path_fig,f'homann_sample-{i}.png'))

############################################################################################################################
def comp_stats(data,steady=False,exp_var='n_fam'):
    if exp_var=='n_fam' or exp_var=='n_im':
        stats_fun = sd.get_nov_response
    else:
        stats_fun = sd.get_trans_response
    nov_resp_all = []
    samples = data.sample_id.unique()   
    for i,sample_i in enumerate(samples):
        data_i = data.loc[data['sample_id']==sample_i] 
        _, nov_resp_i = stats_fun(data_i,exp_var,steady=steady)
        nov_resp_i = nov_resp_i.reset_index()
        nov_resp_i['sample_id'] = sample_i
        nov_resp_all.append(nov_resp_i)
    df_nov_resp = pd.concat(nov_resp_all)
    return df_nov_resp

############################################################################################################################
def generate_random_input(num_gabor,num_im,im_seed):
    random = gs.generate_stim(gs.dim_ranges_rad,num_gabor*num_im,np.random.default_rng(im_seed)) # num_dim x (num_gabor*num_im)
    # df_random_all = pd.DataFrame(random.transpose(),columns=gs.dim_names.copy())

    df_random = []
    im_random = []
    for i in range(num_im):
        random_i = random[:,i*num_gabor:(i+1)*num_gabor]
        df_i = pd.DataFrame(random_i.transpose(),columns=gs.dim_names.copy())
        im_i,_ = gs.comp_gabor(gs.dim_ranges_rad[4],gs.dim_ranges_rad[5],random_i,resolution=100,magn=1)
        im_random.append(im_i)
        df_random.append(df_i)

    return df_random, im_random 

############################################################################################################################
if __name__=='__main__':

    plot_similarities   = True
    plot_kernelact      = True
    plot_steadystate    = True
    recomp_data  = True
    recomp_stats = True
    plot_data    = True

    # Set parameters
    p_rotate = np.array([0.083, 0.167, 0.25]) #np.round(np.arange(0.1,0.25,0.05),3)
    n_fam = [18]
    len_fam = 3

    fun_transform=[gs.transform_identity,
               gs.transform_rotate_left] 
            #    gs.transform_rotate_right,
            #    gs.transform_shift_left,
            #    gs.transform_shift_right]
    
    # transform_names = ['identity','rotation left','rotation right','shift left','shift right']
    transform_names = ['identity','rotation left'] #,'rotation right','shift right']
    all_cols = vis.prep_cmap_discrete('tab20')
    # transform_cols = [all_cols[0], all_cols[6],all_cols[7], all_cols[8],all_cols[9]]
    transform_cols = [all_cols[0], all_cols[6]] #,all_cols[7], all_cols[9]]

    num_parent           = 20
    num_child_per_parent = len_fam+1
    child_mode           = 'fixed'
    gabor_orient         = 0 #0.714197 # pi/4 = 45 degrees
    gabor_diff           = 0 #np.linspace(0,np.pi/2,5), i.e. between 0 and 90 degrees
    init_orient          = gabor_orient - gabor_diff
    gabor_num            = (8,3) # (4,4)
    loc_sigma            = (0,0) # (0,0)

    init_seed            = 12345
    seed_parent = h_in.get_random_seed(length=5,n=num_parent,init_seed=init_seed)
    seed_child  = h_in.get_random_seed(length=5,n=num_parent*num_child_per_parent,init_seed=init_seed+1)

    path_data = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-11_parent-child-exact-24/'
    sl.make_long_dir(path_data)

    path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/2024-11_parent-child-exact-24/'
    sl.make_long_dir(path_fig)

    # Generate children for different p_rotate
    sv_inputs = []
    sv_parents = []
    sv_children = []
    sv_simf = []
    sv_simp = []
    sv_simk = []
    sv_kmat_parent = []
    sv_kernels_children = []
    for pr in p_rotate:
        all_parents, all_children, all_inputs = generate_similar_input(1-pr,seed_parent,seed_child,init_orient,num_parent,num_child_per_parent,n_fam=[18],transform_funs=fun_transform,transform_names=transform_names,transform_cols=transform_cols,gabor_num=gabor_num,loc_sigma=loc_sigma,child_mode=child_mode)
        sv_inputs.append(all_inputs)
        sv_parents.append(all_parents)
        sv_children.append(all_children)

        all_parent_simf, all_pairwise_simf, all_hist_pairwise_simf = comp_feature_sim(all_parents,all_children)
        sv_simf.append((all_parent_simf, all_pairwise_simf, all_hist_pairwise_simf))
        all_parent_simp, all_pairwise_simp, all_hist_pairwise_simp = comp_pixel_sim(all_parents,all_children)
        sv_simp.append((all_parent_simp, all_pairwise_simp, all_hist_pairwise_simp))
        all_parent_simk, all_pairwise_simk, all_hist_pairwise_simk, all_kmat_parent, all_kernels_children = comp_kernel_sim(all_parents,all_children)
        sv_simk.append((all_parent_simk, all_pairwise_simk, all_hist_pairwise_simk))
        sv_kmat_parent.append(all_kmat_parent)
        sv_kernels_children.append(all_kernels_children)

    print('Done generating similar inputs.\n')

    # Generate random images (Homann style)
    num_gabor = 40
    num_im = num_parent*num_child_per_parent
    im_seed = 98765
    df_random, im_random = generate_random_input(num_gabor,num_im,im_seed)
    rand_children = [(df_random,im_random)]

    # Compute similarity of random images
    rand_simp = comp_pixel_sim([],rand_children)
    parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk, rand_kmat_parent, rand_kernels_children = comp_kernel_sim([],rand_children)
    rand_simk = (parents_rand_simk, pairwise_rand_simk, hist_pairwise_rand_simk)

    print('Done generating random images.\n')

    # Split random images into parent-child pairs
    rand_children_split = []
    for i in range(num_parent):
        df_i = [df_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1))]
        im_i = [im_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1))]
        rand_children_split.append((df_i,im_i))

    rand_simp_split = comp_pixel_sim([],rand_children_split)
    parents_rand_simk_split, pairwise_rand_simk_split, hist_pairwise_rand_simk_split, rand_kmat_parent_split, rand_kernels_children_split = comp_kernel_sim([],rand_children_split)
    rand_simk_split = (parents_rand_simk_split, pairwise_rand_simk_split, hist_pairwise_rand_simk_split)

    # Generate simulation input (L-sequence)
    inputs_rand = []
    for i in range(num_parent):
        gfam = [im_random[j] for j in range(i*(len_fam+1),(i+1)*(len_fam+1)-1)]
        gnov = [im_random[(i+1)*(len_fam+1)-1]]

        vec_list, type_list = generate_l_seq(gfam,gnov,n_fam,len_fam=len_fam)
        inputs_rand.append((vec_list,type_list))

    print('Done generating random child sequences.\n')

    # Combine data
    sv_sim_all = [sv_simf,sv_simp,sv_simk]
    rand_sim_all = [None, rand_simp, rand_simk]
    rand_sim_all_split = [None, rand_simp_split, rand_simk_split]
    sv_sim_names = ['feature','pixel','kernel']

    # Plot average similarities
    if plot_similarities:
        f,ax = plt.subplots(1,2,figsize=(12,4))
        color_list = vis.prep_cmap('Blues',len(sv_sim_all))

        parent_means_all = []
        parent_std_all = []
        pairwise_means_all = []
        pairwise_std_all = []
        for j in range(len(sv_sim_all)):
            parent_means = []
            parent_stds = []
            pairwise_means = []
            pairwise_stds = []
            for i in range(len(p_rotate)):
                all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim = sv_sim_all[j][i]
                parent_mean_i = np.mean(np.concatenate(all_parent_sim))
                parent_std_i = np.std(np.concatenate(all_parent_sim))
                pairwise_mean_i = np.mean(np.concatenate(all_hist_pairwise_sim))
                pairwise_std_i = np.std(np.concatenate(all_hist_pairwise_sim))
                parent_means.append(parent_mean_i)
                parent_stds.append(parent_std_i)
                pairwise_means.append(pairwise_mean_i)
                pairwise_stds.append(pairwise_std_i)
            # Plot each type of similarity for similar images
            ax[0].plot(p_rotate,parent_means,color=color_list[j],label=sv_sim_names[j])
            ax[0].fill_between(p_rotate,np.array(parent_means)-np.array(parent_stds),np.array(parent_means)+np.array(parent_stds),color=color_list[j],alpha=0.2)
            ax[1].plot(p_rotate,pairwise_means,color=color_list[j])
            ax[1].fill_between(p_rotate,np.array(pairwise_means)-np.array(pairwise_stds),np.array(pairwise_means)+np.array(pairwise_stds),color=color_list[j],alpha=0.2)
            parent_means_all.append(parent_means)
            parent_std_all.append(parent_stds)
            pairwise_means_all.append(pairwise_means)
            pairwise_std_all.append(pairwise_stds)
            # Plot each type of similarity for random images
            if sv_sim_names[j]!="feature":
                rand_parent_sim, rand_pairwise_sim, rand_hist_pairwise_sim = rand_sim_all[j]
                pairwise_mean_rand = np.mean(np.concatenate(rand_hist_pairwise_sim))
                pairwise_std_rand = np.std(np.concatenate(rand_hist_pairwise_sim))
                ax[1].axhline(y=pairwise_mean_rand,ls='--',color=color_list[j],label=f'{sv_sim_names[j]} (random)')
                ax[1].fill_between(p_rotate,pairwise_mean_rand-pairwise_std_rand,pairwise_mean_rand+pairwise_std_rand,color=color_list[j],alpha=0.2)
        
        ax[0].set_title('Parent-child similarity')
        ax[0].set_xlabel('p_rotate')
        ax[0].set_ylabel('Mean similarity')
        ax[0].set_xlim([np.min(p_rotate),np.max(p_rotate)])
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].legend()
        
        ax[1].set_title('Pairwise similarity')
        ax[1].set_xlabel('p_rotate')
        ax[1].set_ylabel('Mean similarity')
        ax[1].set_xlim([np.min(p_rotate),np.max(p_rotate)])
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        f.tight_layout()
        f.savefig(os.path.join(path_fig,f'mean-sim_vs_p-rotate.png'))

    # Plot kernel activations
    if plot_kernelact:
        for acc_type in ['mean','sum','max']:
            acc_fun = np.mean if acc_type=='mean' else np.sum if acc_type=='sum' else np.max
            f,ax = plt.subplots(1,3,figsize=(8,4),sharey=True,gridspec_kw={'width_ratios':[1,3,1]})

            all_kact = [[sv_kmat_parent[0]],sv_kernels_children,rand_kernels_children] # note: parents are identical for all p_rotate values
            all_kact_names = ['parent','children','random']
            color_list = [['black'],vis.prep_cmap('Blues',len(p_rotate)),['grey']]
            xl = ['','p_rotate','']

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
                data, kl, kwl = run_homann_l(n_fam,sv_inputs[i],k_params,seed_parent)
                df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
                df_data.to_csv(os.path.join(path_data,f'homann_sim-{p_rotate[i]}'.replace('.','') + '.csv'))
                sv_data.append(df_data)
                sv_kl.append(kl)
                sv_kwl.append(kwl)
                    
            # Run experiment for random images
            data, kl, kwl = run_homann_l(n_fam,inputs_rand,k_params,[im_seed]*len(inputs_rand))
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
                stats_i = comp_stats(sv_data[i],steady=True)
                stats_i.to_csv(os.path.join(path_data,f'homann_stats-{p_rotate[i]}'.replace('.','') + '.csv'))
                sv_stats.append(stats_i)
                field_vals  = ('n_fam','')
                field_stats = ('nt_norm','mean')
                field_stats2 = ('steady','mean')
            
            # Compute random statistics
            stats_rand = comp_stats(data_rand[0],steady=True)
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
                ax.set_ylim([-0.5,3.5])
        
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
                
                ax.set_ylim([0,4])

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

    check_images=False
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
            plot_pairwise_sim_matrix(sv_simp[i_check][1][idx_large[j]],f,ax[0])
            plot_pairwise_sim_matrix(sv_simk[i_check][1][idx_large[j]],f,ax[1])
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
            plot_pairwise_sim_matrix(sv_simp[i_check][1][idx_small[j]],f,ax[0])
            plot_pairwise_sim_matrix(sv_simk[i_check][1][idx_small[j]],f,ax[1])
            f.suptitle(f'Sample {idx_small[j]}: small steady state')

            # Plot their image sequences
            # f,ax = plt.subplots(1,4,figsize=(4*4,3))
            # for i,ax_i in enumerate(ax):
            #     ax_i.imshow(sv_children[i_check][idx_small[j]][-1][i],cmap='gray',vmin=-1,vmax=1)
            #     ax_i.axis('off')
            # f.tight_layout()
            # f.suptitle(f'Sample {idx_small[j]}: small steady state')

    print('Done.\n')    

