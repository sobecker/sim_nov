import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/') 

import src.models.snov.kernel_nov_vec as knov_vec

def get_random_seed(length,n,init_seed=None):
    if not init_seed: 
        np.random.seed()
    else:
        np.random.seed(init_seed)
    min = 10**(length-1)
    max = 9*min + (min-1)
    return np.random.randint(min, max, n)

# Compute cosine similarity
def cosine_sim_nonorm(a,b):
    return np.dot(a.flatten(),b.flatten()) 

def cosine_sim1(a,b):
    return np.dot(a.flatten(),b.flatten()) / (np.linalg.norm(a.flatten(),ord=1) * np.linalg.norm(b.flatten(),ord=1))

def cosine_sim(a,b):
    return np.dot(a.flatten(),b.flatten()) / (np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()))

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
def comp_pairwise_sim(list_children,sim_meas='cosine_sim',include_prior_sim=False):
    if include_prior_sim:
        prior = 1/len(list_children[0].flatten())*np.ones(list_children[0].shape)
        list_children = [prior] + list_children
    pairwise_sim = np.ones((len(list_children),len(list_children)))
    for i in range(len(list_children)):
        for j in range(i,len(list_children)):
            sim_ij = eval(sim_meas)(list_children[i].flatten(),list_children[j].flatten())
            pairwise_sim[i][j] = sim_ij
            pairwise_sim[j][i] = sim_ij
    # if include_prior_sim:
    #     prior = 1/len(list_children[0].flatten())*np.ones(len(list_children[0].flatten()))
    #     sim_prior = 
    #     for i in range(len(list_children)):
    #         sim_prior_i = eval(sim_meas)(list_children[i].flatten(),prior)
    #         sim_prior.append(sim_prior_i)
    #     pairwise_sim = np.concatenate((np.array(sim_prior).reshape(-1,1),pairwise_sim),axis=1)
    #     pairwise_sim = np.concatenate((np.array(sim_prior).reshape(1,-1),pairwise_sim),axis=0)
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

def plot_pairwise_sim_matrix(pairwise_sim,f,ax,title='Pairwise similarity of children'):
    # minsim = np.min(pairwise_sim.flatten())
    # maxsim = np.max(pairwise_sim.flatten())
    asim = ax.imshow(pairwise_sim,cmap='bwr',norm=colors.CenteredNorm(vcenter=np.mean(pairwise_sim.flatten()))) 
    for i in range(pairwise_sim.shape[0]):
        for j in range(pairwise_sim.shape[1]):
            ax.text(j, i, np.round(pairwise_sim[i, j],2), ha="center", va="center", color="k")
    ax.set_title(title)
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

############################################################################################################
# Plot sequence for M-experiment
def plot_sequence_m(rotate_i,all_inputs_i):
    print('Sequence for rotation:',rotate_i)
    mark_novel = [3,6,9,12]
    f_list = []; axl_list = []
    for j in range(len(mark_novel)):
        m_unique = all_inputs_i[1][0][0][j][0]
        f,axl = plt.subplots(1,m_unique.shape[0],figsize=(2*m_unique.shape[0],2))
        f_list.append(f); axl_list.append(axl)
        for ii in range(len(axl)):
            ax = axl[ii]
            ax.imshow(m_unique[ii,:,:],cmap='gray',vmin=-1,vmax=1,origin='lower')
            ax.set_ylabel(''); ax.set_xlabel(''); ax.set_yticks([]); ax.set_xticks([])
            if ii==mark_novel[j]:
                [ax.spines[spine].set_color('r') for spine in ax.spines]
    return [f_list, axl_list]

def plot_sequence_l(rotate_i,all_inputs_i):
    print('Sequence for rotation:',rotate_i)
    mark_novel = [3]
    l_unique = all_inputs_i[0][0][0][0][0]
    f,axl = plt.subplots(1,l_unique.shape[0],figsize=(3*l_unique.shape[0],2))
    for i in range(len(axl)):
        ax = axl[i]
        ax.imshow(l_unique[i,:,:],cmap='gray',vmin=-1,vmax=1,origin='lower')
        ax.set_ylabel(''); ax.set_xlabel(''); ax.set_yticks([]); ax.set_xticks([])
        if i in mark_novel:
            [ax.spines[spine].set_color('r') for spine in ax.spines]
    return [f, axl]
        
def plot_sequence_lp(rotate_i,all_inputs_i):
    print('Sequence for rotation:',rotate_i)
    lp_unique = all_inputs_i[2][0][0][0][0]
    f,axl = plt.subplots(1,lp_unique.shape[0],figsize=(3*lp_unique.shape[0],2))
    mark_novel = [3,4,5]
    for i in range(len(axl)):
        ax = axl[i]
        ax.imshow(lp_unique[i,:,:],cmap='gray',vmin=-1,vmax=1,origin='lower')
        ax.set_ylabel(''); ax.set_xlabel(''); ax.set_yticks([]); ax.set_xticks([])
        if i in mark_novel:
            [ax.spines[spine].set_color('r') for spine in ax.spines]
    return [f, axl]
    