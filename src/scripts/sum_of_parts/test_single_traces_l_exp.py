import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import os

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/') 

import src.models.snov.gabor_stimuli as gs
import src.models.snov.run_gabor_knov2 as gknov2
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

def plot_pairwise_sim_hist(all_sim,f,ax,title='Pairwise similarity distribution',xlabel='Similarity'):
    ax.hist(all_sim,bins=30,color='grey',alpha=1)
    ylim = ax.get_ylim()
    mu = np.mean(all_sim)
    sigma = np.std(all_sim)
    a_mu = ax.axvline(mu,color='black',linestyle='--')
    a_sigma = ax.fill_between([mu-sigma,mu+sigma],[0]*2,[ylim[1]+0.1]*2,color='black',alpha=0.2)
    ax.set_ylim(ylim)    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend([a_mu,a_sigma],[f'Mean={np.round(mu,2)}',f'Std={np.round(sigma,2)}'],loc='upper right')

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
def generate_similar_input(sim_value,seed_parent,seed_child,num_parent,num_child_per_parent):
    # Children parameters
    transform_funs  = [gs.transform_identity,gs.transform_rotate]
    transform_probs = [sim_value,1-sim_value]
    transform_names = ['identity','rotation']
    transform_cols  = ['blue','red']

    # Experiment parameters
    sequence_mode = 'seq'
    n_fam = [1,3,8,18,38]
    len_fam = 3

    # Plot stimuli
    plot_stim = False

    # Generate input sequences
    all_parents = []
    all_children = []
    all_inputs = []

    for i in range(num_parent):
        # Generate parent stimulus
        parent, _ = gs.generate_teststim_parent(gs.dim_ranges_rad,gabor_num=(4,4),rng=np.random.default_rng(seed_parent[i]))
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
        children, set_transform = gs.generate_teststim_children(parent,num_child=num_child_per_parent,fun_transform=transform_funs,prob_transform=transform_probs,rng=np.random.default_rng(seed_child[i]))
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
        
        all_inputs.append((vec_list,type_list))
    return all_parents, all_children, all_inputs

############################################################################################################################
def plot_similar_inputs(all_parents,all_children):
    # Plot parent-child and pairwise similarity distribution (feature-similarity)
    sim_features = ['orientation']

    all_parent_sim = []
    all_pairwise_sim = []
    all_hist_pairwise_sim = []
    for i in range(len(all_parents)):
        # Get parent features
        df_parent = all_parents[i][0]
        parent_features = df_parent[sim_features].to_numpy()  

        # Get children features 
        df_children = all_children[i][0]
        list_children_features = [df_children[j][sim_features].to_numpy() for j in range(len(df_children))]  

        # Compute similarities
        parent_sim = comp_parent_sim(parent_features,list_children_features,sim_meas='orientation_sim')
        pairwise_sim = comp_pairwise_sim(list_children_features,sim_meas='orientation_sim')

        hist_pairwise_sim = pairwise_sim[np.triu_indices(pairwise_sim.shape[0],k=1)].flatten()

        all_parent_sim.append(parent_sim)
        all_pairwise_sim.append(pairwise_sim)
        all_hist_pairwise_sim.append(hist_pairwise_sim)

    # print(np.concatenate(all_parent_sim))
    # print(np.concatenate(all_hist_pairwise_sim))

    # Plot histogram of similarities
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
def run_homann_l(n_fam,all_inputs,k_params,seed_parent):
    data = []; kl = []; kwl = []
    for i in range(len(all_inputs)):
        print(f'Running realization {i} of L-experiment.\n')
        stim = all_inputs[i][0]
        itype = all_inputs[i][1]

        data_i = []; kl_i = []; kwl_i = []
        for j in range(len(stim)):
            print(f'Running experiment condition {j} of experiment realization {i}.\n')
            stim_j = stim[j]
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

def comp_stats(data,steady=False):
    nov_resp_all = []
    samples = data.sample_id.unique()   
    for i,sample_i in enumerate(samples):
        data_i = data.loc[data['sample_id']==sample_i] 
        _, nov_resp_i = sd.get_nov_response(data_i,'n_fam',steady=steady)
        nov_resp_i = nov_resp_i.reset_index()
        nov_resp_i['sample_id'] = sample_i
        nov_resp_all.append(nov_resp_i)
    df_nov_resp = pd.concat(nov_resp_all)
    return df_nov_resp


############################################################################################################################
if __name__=='__main__':

    regenerate_inputs = True
    recomp_data = True
    recomp_stats = True
    plot_samples = False

    plot_steady = True

    # Set parameters
    num_parent           = 10
    num_child_per_parent = 4
    init_seed            = 12345
    seed_parent = h_in.get_random_seed(length=5,n=num_parent,init_seed=init_seed)
    seed_child  = h_in.get_random_seed(length=5,n=num_parent*num_child_per_parent,init_seed=init_seed+1)

    n_fam = [1,3,8,18,38]
    input_type = 'similar'
    if input_type=='random':
        sim_values = None
    elif input_type=='similar':
        sim_values = np.array([0.25,0.5,0.75])
    elif input_type=='similar_fixed_overlap':
        sim_values = np.array([0.1,0.2,0.3])

    path_data = f'/Users/sbecker/Projects/RL_reward_novelty/data/2024-10_single-trace-sim/'
    sl.make_long_dir(path_data)

    path_fig = f'/Users/sbecker/Projects/RL_reward_novelty/output/2024-10_single-trace-sim/'

    # Generate inputs
    if regenerate_inputs:
        sv_inputs = []
        sv_parents = []
        sv_children = []
        sv_sim = []
        for sv in sim_values:
            if input_type=='random':
                pass
            elif input_type=='similar':
                all_parents, all_children, all_inputs = generate_similar_input(sv,seed_parent,seed_child,num_parent,num_child_per_parent)
                all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim = plot_similar_inputs(all_parents,all_children)
            elif input_type=='similar_fixed_overlap':
                pass    
            sv_inputs.append(all_inputs)
            sv_parents.append(all_parents)
            sv_children.append(all_children)
            sv_sim.append((all_parent_sim, all_pairwise_sim, all_hist_pairwise_sim))
        print('Done generating inputs.\n')

    # Simulate or load experiments
    if recomp_data:
        # Make kernel model
        k_params = gknov2.init_gabor_knov(gnum=4,k_type='triangle',ksig=1,kcenter=1,cdens=4,seed=12345,rng=None,mask=True,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='equidistant',contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,add_empty=False,debug=True)

        # Run experiment
        sv_data = []; sv_kl = []; sv_kwl = []
        for i in range(len(sim_values)):
            data, kl, kwl = run_homann_l(n_fam,sv_inputs[i],k_params,seed_parent)
            df_data = pd.concat([pd.concat(data[j]) for j in range(len(data))])
            df_data.to_csv(os.path.join(path_data,f'homann_sim-{sim_values[i]}'.replace('.','') + '.csv'))
            sv_data.append(df_data)
            sv_kl.append(kl)
            sv_kwl.append(kwl)
        print('Done simulating experiments.\n')

    else:
        # Load simulation data
        sv_data = []
        for i in range(len(sim_values)):
            df_data = pd.read_csv(os.path.join(path_data,f'homann_sim-{sim_values[i]}'.replace('.','') + '.csv'))
            sv_data.append(df_data)
        print('Done loading experiments.\n')

    # Plot individual experiments and compute stats
    if recomp_stats:
        sv_stats = []
        for i in range(len(sim_values)):
            stats_i = comp_stats(sv_data[i],steady=plot_steady)
            stats_i.to_csv(os.path.join(path_data,f'homann_stats-{sim_values[i]}'.replace('.','') + '.csv'))
            sv_stats.append(stats_i)
            field_vals  = ('n_fam','')
            field_stats = ('nt_norm','mean')
            if plot_steady:
                field_stats2 = ('steady','mean')
        print('Done plotting experiments and computing stats.\n')

    else:
        sv_stats = []
        for i in range(len(sim_values)):
            if plot_steady:
                col_extract = ['n_fam','nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','steady_mean','steady_std','steady_sem','sample_id']
            else:
                col_extract = ['n_fam','nt_mean','nt_std','nt_sem','nt_norm_mean','nt_norm_std','nt_norm_sem','sample_id']
            stats_i = pd.read_csv(os.path.join(path_data,f'homann_stats-{sim_values[i]}'.replace('.','') + '.csv'),header=1,names=col_extract)
            sv_stats.append(stats_i)
            field_vals  = 'n_fam'
            field_stats = 'nt_norm_mean'
            if plot_steady:
                field_stats2 = 'steady_mean'
        print('Done loading stats.\n')

    if plot_samples:
        for i in range(len(sim_values)):
            df_stats = plot_homann_l(n_fam=n_fam,data=sv_data[i],stats=sv_stats[i],all_parent_sim=sv_sim[i][0],all_pairwise_sim=sv_sim[i][1],all_hist_pairwise_sim=sv_sim[i][2],sim_level=sim_values[i])
        print('Done plotting samples.\n')

    # Plot stats (1)
    color_list = vis.prep_cmap('Blues',len(sim_values))
    alpha_list = np.linspace(1,0.2,len(sim_values))

    f,ax = plt.subplots(1,1,figsize=(4,4))
    for i in range(len(sim_values)):
        stats_i = sv_stats[i]
        samples = stats_i['sample_id'].unique()
        distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index()
        ax.plot(distribution_i[field_vals],distribution_i['mean_stats'],'o',c=color_list[i],label=f'sim={sim_values[i]}')
        ax.plot(distribution_i[field_vals],distribution_i['mean_stats'],'--',c=color_list[i])
        ax.fill_between(distribution_i[field_vals],distribution_i['mean_stats']-distribution_i['std_stats'],distribution_i['mean_stats']+distribution_i['std_stats'],color=color_list[i],alpha=0.2)
    ax.set_xlabel('L')
    ax.set_yticks([])
    ax.set_ylabel('Novelty responses')
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(path_fig,'nov_vs_l_all-sim.png'))

    if plot_steady:
        f,ax = plt.subplots(1,1,figsize=(4,4))
        for i in range(len(sim_values)):
            stats_i = sv_stats[i]
            samples = stats_i['sample_id'].unique()
            distribution_i2 = stats_i[[field_vals,field_stats2]].groupby([field_vals]).agg(mean_stats=(field_stats2,np.mean),std_stats=(field_stats2,np.std)).reset_index()
            ax.plot(distribution_i2[field_vals],distribution_i2['mean_stats'],'o',c=color_list[i],label=f'sim={sim_values[i]}')
            ax.plot(distribution_i2[field_vals],distribution_i2['mean_stats'],'--',c=color_list[i])
            ax.fill_between(distribution_i2[field_vals],distribution_i2['mean_stats']-distribution_i2['std_stats'],distribution_i2['mean_stats']+distribution_i2['std_stats'],color=color_list[i],alpha=0.2)
        ax.set_xlabel('L')
        ax.set_yticks([])
        ax.set_ylabel('Steady state responses')
        ax.legend()
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'steady_vs_l_all-sim.png'))

    # Plot stats (2)
    color_list = vis.prep_cmap('Blues',len(sim_values))
    alpha_list = np.linspace(1,0.2,len(sim_values))

    f,ax = plt.subplots(1,len(n_fam),figsize=(4*len(n_fam),4),sharex=True,sharey=True)
    for i in range(len(sim_values)):
        sim_i = np.array([np.mean(sv_sim[i][2][j]) for j in range(len(sv_sim[i][2]))])
        sim_mu, sim_std = np.mean(sim_i), np.std(sim_i)
        stats_i = sv_stats[i]
        samples = stats_i['sample_id'].unique()
        distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index()
        for j in range(len(ax)):
            stats_ij = stats_i.loc[stats_i[field_vals]==n_fam[j]]
            mu_ij = distribution_i.loc[distribution_i[field_vals]==n_fam[j],'mean_stats']
            std_ij = distribution_i.loc[distribution_i[field_vals]==n_fam[j],'std_stats']
            ax[j].plot(sim_i,stats_ij[field_stats],'o',c=color_list[i],label=f'sim={sim_values[i]}',alpha=0.4)
            ax[j].plot(sim_mu,mu_ij,'x',c=color_list[i])
            ax[j].errorbar(sim_mu,mu_ij,xerr=sim_std,yerr=std_ij,color=color_list[i])
            if i==len(sim_values)-1:
                ax[j].set_title(f'L={n_fam[j]}')
                ax[j].set_xlabel('Av. pairwise similarity')
                ax[j].set_yticks([])
                ax[j].set_ylabel('Novelty responses')
                ax[j].legend()
    f.tight_layout()
    f.savefig(os.path.join(path_fig,'nov_vs_sim_all-l.png'))

    if plot_steady:
        f,ax = plt.subplots(1,len(n_fam),figsize=(4*len(n_fam),4),sharex=True,sharey=True)
        for i in range(len(sim_values)):
            sim_i = np.array([np.mean(sv_sim[i][2][j]) for j in range(len(sv_sim[i][2]))])
            sim_mu, sim_std = np.mean(sim_i), np.std(sim_i)
            stats_i = sv_stats[i]
            samples = stats_i['sample_id'].unique()
            distribution_i2 = stats_i[[field_vals,field_stats2]].groupby([field_vals]).agg(mean_stats=(field_stats2,np.mean),std_stats=(field_stats2,np.std)).reset_index()
            for j in range(len(ax)):
                stats_ij = stats_i.loc[stats_i[field_vals]==n_fam[j]]
                mu_ij = distribution_i2.loc[distribution_i2[field_vals]==n_fam[j],'mean_stats']
                std_ij = distribution_i2.loc[distribution_i2[field_vals]==n_fam[j],'std_stats']
                ax[j].plot(sim_i,stats_ij[field_stats2],'o',c=color_list[i],label=f'sim={sim_values[i]}',alpha=0.4)
                ax[j].plot(sim_mu,mu_ij,'x',c=color_list[i])
                ax[j].errorbar(sim_mu,mu_ij,xerr=sim_std,yerr=std_ij,color=color_list[i])
                if i==len(sim_values)-1:
                    ax[j].set_title(f'L={n_fam[j]}')
                    ax[j].set_xlabel('Av. pairwise similarity')
                    ax[j].set_yticks([])
                    ax[j].set_ylabel('Steady state responses')
                    ax[j].legend()
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'steady_vs_sim_all-l.png'))
    
    if plot_steady:
        n_fam_plot = 18
        cmap = plt.get_cmap('Blues')
        f,ax = plt.subplots(1,1,figsize=(4,4))
        # f,axl = plt.subplots(1,len(sim_values),figsize=(4*len(sim_values),4),sharex=True,sharey=True)
        for i in range(len(sim_values)):
            # ax = axl[i]

            sim_i = np.array([np.mean(sv_sim[i][2][j]) for j in range(len(sv_sim[i][2]))])
            sim_mu, sim_std = np.mean(sim_i), np.std(sim_i)
            stats_i = sv_stats[i]

            samples = stats_i['sample_id'].unique()
            distribution_i = stats_i[[field_vals,field_stats]].groupby([field_vals]).agg(mean_stats=(field_stats,np.mean),std_stats=(field_stats,np.std)).reset_index()
            distribution_i2 = stats_i[[field_vals,field_stats2]].groupby([field_vals]).agg(mean_stats=(field_stats2,np.mean),std_stats=(field_stats2,np.std)).reset_index()

            stats_ij = stats_i.loc[stats_i[field_vals]==n_fam_plot]
            mu_ij = distribution_i.loc[distribution_i2[field_vals]==n_fam_plot,'mean_stats']
            std_ij = distribution_i.loc[distribution_i2[field_vals]==n_fam_plot,'std_stats']

            ax.scatter(sim_i,stats_ij[field_stats2],s=20,c=stats_ij[field_stats].values,alpha=0.8,cmap='Blues')
            ax.axhline(y=mu_ij.values[0],color='k',ls='--')
            ax.axvline(x=sim_mu,color='k',ls='--')

            ax.set_title(f'L={n_fam_plot}')
            ax.set_xlabel('Av. pairwise similarity')
            ax.set_yticks([])
            ax.set_ylabel('Steady state responses')
            ax.legend()
        f.tight_layout()
        f.savefig(os.path.join(path_fig,'nov_vs_steady_vs_sim_all-sim.png'))
    

    print('Done plotting experiments.\n')
    plt.close('all')
