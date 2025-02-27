import numpy as np
import pandas as pd
import sys

import models.snov.gabor_stimuli as gs

def get_random_seed(length,n,init_seed=None):
    if not init_seed: 
        np.random.seed()
    else:
        np.random.seed(init_seed)
    min = 10**(length-1)
    max = 9*min + (min-1)
    return np.random.randint(min, max, n)

def create_tau_emerge_input_gabor(n_fam=[1,3,8,18,38],len_fam=3,num_gabor=1,seed=0,plot=False,alph_adj=3,adj_w=True,adj_f=False,idx=True,sampling='basic',patches=None,sequence_mode='sep'): # sequence_mode: 'sep' or 'app'
    if seed==0: seed = get_random_seed(5,1)
    rng = np.random.default_rng(seed=seed)

    if sequence_mode=='app':
        mode_factor = len(n_fam)
        modes = np.arange(mode_factor,dtype=int)
    else:
        mode_factor = 1
        modes = np.zeros(len(n_fam),dtype=int)
    
    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,mode_factor*num_gabor*len_fam,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    nov = gs.generate_stim(dim_ranges,mode_factor*num_gabor*1,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    params = [df_fam, df_nov]
    gfam = []
    for i in range(mode_factor*len_fam):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
    gnov = []
    for i in range(mode_factor):
        gnov_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gnov.append(gnov_i)

    # Create sequences
    vec_list  = []
    type_list = []
    for i, m in zip(range(len(n_fam)), modes):
        # Pick new set of familiar and novel images
        if len_fam==1: gfam_m = [gfam[m]]
        else:          gfam_m = gfam[m*len_fam:(m+1)*len_fam]
        gnov_m = gnov[m]
        # Create sequence
        if idx:
            stim_unique = np.stack(gfam_m + [gnov_m])
            stim_idx    = (list(np.arange(len(gfam_m)))*(n_fam[i]+2))[:-1] + [len(gfam_m)] + list(np.arange(len(gfam_m)))*2 # n_fam+1 repetitions of fam. sequ.; fam. sequ. with novel image substituted; 1 repetition of fam. sequ.
            seq_vec     = (stim_unique,stim_idx)
        else:
            seq = (gfam_m*(n_fam[i]+2))[:-1] + [gnov_m] + gfam_m*2
            seq_vec = np.squeeze(np.stack(seq))
        stim_type = (['fam']*len_fam*(n_fam[i]+2))[:-1] + ['nov'] + ['fam']*len_fam*2
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, seed, type_list, params

def create_tau_memory_input_gabor(n_fam=17,len_fam=[3,6,9,12],num_gabor=1,seed=0,plot=False,alph_adj=3,adj_w=True,adj_f=False,idx=True,sampling='basic',patches=None,sequence_mode='sep'): # sequence_mode: 'sep' or 'app'
    if seed==0: seed = get_random_seed(5,1)
    rng = np.random.default_rng(seed=seed)

    if sequence_mode=='app':
        mode_factor = len(len_fam)
        modes = np.arange(mode_factor,dtype=int)
    else:
        mode_factor = 1
        modes = np.zeros(len(len_fam),dtype=int)
    
    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,mode_factor*num_gabor*np.max(len_fam),rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    nov = gs.generate_stim(dim_ranges,mode_factor*num_gabor*1,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    params = [df_fam, df_nov]
    gfam = []
    for i in range(mode_factor*np.max(len_fam)):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
    gnov = []
    for i in range(mode_factor):
        gnov_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gnov.append(gnov_i)

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
        if idx:
            stim_unique = np.stack(gfam_m + [gnov_m])
            stim_idx    = (list(np.arange(len(gfam_m)))*(n_fam+1))[:-1] + [len(gfam_m)] + list(np.arange(len(gfam_m)))*2
            seq_vec     = (stim_unique,stim_idx)
        else:
            seq = (gfam_m*(n_fam+1))[:-1] + [gnov_m] + gfam_m*2
            seq_vec = np.squeeze(np.stack(seq))
        stim_type = (['fam']*len_fam[i]*(n_fam+1))[:-1] + ['nov'] + ['fam']*len_fam[i]*2
        vec_list.append(seq_vec)
        type_list.append(stim_type)

    return vec_list, seed, type_list, params

# dN=[0,70,140,210,280,360,480] 
# dN = list(3*np.array([0,24,49,73,98,122,159])) # number of recovery images presented, computed based number of recovery repetitions L' closest to Homann graphs
# dN = list(np.array([0,22,44,66,88,110,143])/0.3) # number of recovery images presented, computed based read out from Homann graphs
def create_tau_recovery_input_gabor(n_fam=22,len_fam=3,dN=list(np.array([0,22,44,66,88,110,143])/0.3),num_gabor=1,seed=0,plot=False,alph_adj=3,adj_w=True,adj_f=False,idx=True,sampling='basic',patches=None,sequence_mode='sep'): # sequence_mode: 'sep' or 'app'
    if seed==0: seed = get_random_seed(5,1)
    rng = np.random.default_rng(seed=seed)

    if sequence_mode=='app':
        mode_factor = len(dN)
        modes = np.arange(mode_factor,dtype=int)
    else:
        mode_factor = 1
        modes = np.zeros(len(dN),dtype=int)

    # Generate novel and familiar images
    dim_ranges = gs.dim_ranges_rad.copy()
    fam = gs.generate_stim(dim_ranges,mode_factor*num_gabor*len_fam,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    nov = gs.generate_stim(dim_ranges,mode_factor*num_gabor*len_fam,rng,alph_adj=alph_adj,adj_w=adj_w,adj_f=adj_f,sampling=sampling,patches=patches)
    df_fam = pd.DataFrame(fam.transpose(),columns=gs.dim_names.copy())
    df_nov = pd.DataFrame(nov.transpose(),columns=gs.dim_names.copy())
    params = [df_fam, df_nov]
    gfam = []
    gnov = []
    for i in range(mode_factor*len_fam):
        gfam_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],fam[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gfam.append(gfam_i)
        gnov_i,_ = gs.comp_gabor(dim_ranges[4],dim_ranges[5],nov[:,i*num_gabor:(i+1)*num_gabor].reshape((-1,num_gabor)),resolution=100,magn=1)
        gnov.append(gnov_i)

    # Create sequence
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
        if idx:
            stim_unique = np.stack(gfam_m + gnov_m)
            stim_idx    = list(np.arange(len(gfam_m)))*n_fam + list(len(gfam_m)+np.arange(len(gnov_m)))*rep_nov + list(np.arange(len(gfam_m)))*5
            seq_vec     = (stim_unique,stim_idx)
        else:
            seq = gfam_m*n_fam + gnov_m*rep_nov + gfam_m*5
            seq_vec = np.squeeze(np.stack(seq))
        stim_type = ['fam']*len_fam*n_fam + ['nov']*len_fam*rep_nov + ['fam_r']*len_fam*5
        vec_list.append(seq_vec)
        type_list.append(stim_type)
    
    return vec_list, seed, type_list, params
    
