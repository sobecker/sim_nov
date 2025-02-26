import numpy as np
rng = np.random.default_rng(seed=12345)
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.special import softmax
import multiprocessing as mp
import src.models.snov.kernel_nov_vec as knov_vec  
import src.models.snov.gabor_stimuli as gs

### This file contains functions to simulate Gabor novelty using a new approach (non-negative kernel functions, more similar to 1D model) ###

####################################################################################################################################################################
# Compute similarity of stimulus with single (conv./non-conv.) Gabor kernel
####################################################################################################################################################################
# Compute similarity of stimulus with given Gabor kernel
def comp_overlap(stim,k,contrast='off'):
    if contrast=='off':
        m = np.dot(stim.flatten(),k.flatten()) / (np.linalg.norm(stim.flatten()) * np.linalg.norm(k.flatten())) # cosine similarity (contrast-invariant)
    elif contrast=='on':
        m = np.tanh(np.dot(stim.flatten(),k.flatten()) / np.linalg.norm(k.flatten())) # bounded but contrast-variant similarity
    return m

# Compute similarity of stimulus (masked) with given Gabor kernel
def comp_overlap_mask(stim,mask,k,contrast='off',add_eps=1e-10):
    if contrast=='off':
        mm = np.multiply(mask.flatten(),stim.flatten()) + add_eps #*np.ones(stim.flatten().shape)
        m = np.dot(mm,k.flatten()) / (np.linalg.norm(mm) * np.linalg.norm(k.flatten())) # cosine similarity (contrast-invariant)
    elif contrast=='on':
        m = np.tanh(np.dot(np.multiply(mask.flatten(),stim.flatten()),k.flatten()) / np.linalg.norm(k.flatten())) # bounded but contrast-variant similarity
    return m

# Compute similarity of stimulus with given Gabor kernel for all points in the convolution
def comp_conv(stim,k,dens=2,contrast='off'):
    if contrast=='off':
        norm = (np.linalg.norm(stim.flatten()) * np.linalg.norm(k.flatten()))
    else:
        norm = np.linalg.norm(k.flatten())
    if norm == 0:
        # conv = ndi.convolve(stim, k, mode='constant')[::dens,::dens]
        conv = np.nan*np.ones(stim[::dens,::dens].shape)
    else:
        conv = ndi.convolve(stim, k, mode='constant')[::dens,::dens] / norm 
    if contrast=='off':
        m = conv          # cosine similarity (contrast-invariant)
    elif contrast=='on':
        m = np.tanh(conv) # bounded but contrast-variant similarity
    return m

def comp_conv_parallel(id,stim,k,dens=2,contrast='off'):
    m = comp_conv(stim,k,dens=dens,contrast=contrast)
    return (id, m)

####################################################################################################################################################################
# Compute similarity of stimulus vector with single (conv./non-conv.) Gabor kernel
####################################################################################################################################################################
# Compute similarity of stimulus vector with given Gabor kernel
def comp_overlap_vec(stim_vec,k,contrast='off',axis=0):
    ml = []
    for i in range(stim_vec.shape[axis]):
        m = comp_overlap(stim_vec[i],k,contrast=contrast)
        ml.append(m)
    m = np.stack(ml)
    return m

# Compute similarity of stimulus vector (masked) with given Gabor kernel
def comp_overlap_mask_vec(stim_vec,mask_vec,k,contrast='off',axis=0):
    ml = []
    for i in range(stim_vec.shape[axis]):
        m = comp_overlap_mask(stim_vec[i],mask_vec,k,contrast=contrast)
        ml.append(m)
    m = np.stack(ml)
    return m

# Compute similarity of stimulus vector with given Gabor kernel for all points in the convolution
def comp_conv_vec(stim_vec,k,dens=2,contrast='off',axis=0):
    ml = []
    for i in range(stim_vec.shape[axis]):
        m = comp_conv(stim_vec[i],k,dens,contrast=contrast)
        ml.append(m.flatten())
    m = np.stack(ml)
    return m

# Compute similarity of stimulus vector with given Gabor kernel for all points in the convolution
def comp_conv_vec_parallel(stim_vec,k,dens=2,contrast='off',axis=0):
    num_pool = mp.cpu_count()
    pool = mp.Pool(num_pool)
    jobs = [pool.apply_async(comp_conv_parallel,args=(ii,stim_vec[ii],k,dens,contrast)) for ii in range(stim_vec.shape[axis])]
    # jobs = [pool.apply_async(comp_conv_parallel,args=(i,stim_vec[i],k,dens,contrast=contrast)) for i in range(stim_vec.shape[axis])]
    data = [r.get() for r in jobs]
    pool.close()
    pool.join() 
    data.sort(key=lambda tup: tup[0])
    # Stack data
    ml = []
    for i in range(len(data)):
        ml.append(data[i][1].flatten())
    m = np.stack(ml)
    return m

####################################################################################################################################################################
# Function wrapper to apply similarity with Gabor and activation for single stimulus
####################################################################################################################################################################
def k_gabor_fun(gs,gref,fun,sig,contrast='off'): 
    sim = comp_overlap(gs,gref,contrast=contrast)   # map stimulus gs into similarity space of reference filter gref
    act = fun(sim,1,sig)                            # compute kernel function fun in similarity space
    return act
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

def k_gabor_mask_fun(gs,mref,gref,fun,sig,contrast='off'): 
    sim = comp_overlap_mask(gs,mref,gref,contrast=contrast)   # map stimulus gs into similarity space of reference filter gref
    act = fun(sim,1,sig)                            # compute kernel function fun in similarity space
    return act
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

def k_gabor_fun_conv(gs,gref,fun,sig,dens=2,contrast='off'): 
    sim = comp_conv(gs,gref,dens,contrast=contrast)   # map stimulus gs into similarity space of reference filter gref
    act = fun(sim,1,sig)                            # compute kernel function fun in similarity space
    return act
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

def k_gabor_fun_convint(gs,gref,fun,sig,dens=2,contrast='off'): 
    sim = comp_conv(gs,gref,dens,contrast=contrast)   # map stimulus gs into similarity space of reference filter gref, shape: (n_conv_points,)
    act = fun(sim,1,sig)                            # compute kernel function fun in similarity space
    act_int = np.mean(act)
    return act_int
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

####################################################################################################################################################################
# Function wrapper to apply similarity with Gabor and activation for stimulus vector
####################################################################################################################################################################
def k_gabor_fun_vec(gs_vec,gref,fun,sig,center=1,contrast='off',softmax_norm=False): 
    sim = comp_overlap_vec(gs_vec,gref,contrast=contrast)  # map all stimuli from gs_vec into similarity space of reference filter gref
    act = fun(sim,center,sig)                           # compute kernel function fun for each stimulus in similarity space
    if softmax_norm:
        act = softmax(act)                              # softmax normalization
    return act
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

def k_gabor_fun_mask_vec(gs_vec,mref,gref,fun,sig,center=1,contrast='off',softmax_norm=False): 
    sim = comp_overlap_mask_vec(gs_vec,mref,gref,contrast=contrast)  # map all stimuli from gs_vec into similarity space of reference filter gref
    act = fun(sim,center,sig)                           # compute kernel function fun for each stimulus in similarity space
    if softmax_norm:
        act = softmax(act)                              # softmax normalization
    return act
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

def k_gabor_fun_conv_vec(gs_vec,gref,fun,sig,center=1,dens=2,contrast='off',softmax_norm=False,parallel=False): 
    if parallel:
        sim = comp_conv_vec_parallel(gs_vec,gref,dens,contrast=contrast)
    else:
        sim = comp_conv_vec(gs_vec,gref,dens,contrast=contrast)  # map all stimuli from gs_vec into similarity space of reference filter gref
    act = fun(sim,center,sig)                           # compute kernel function fun for each stimulus in similarity space
    if softmax_norm:
        act = softmax(act)                              # softmax normalization
    return act
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

def k_gabor_fun_convint_vec(gs_vec,gref,fun,sig,center=1,dens=2,contrast='off',softmax_norm=False,parallel=False): 
    if parallel:
        sim = comp_conv_vec_parallel(gs_vec,gref,dens,contrast=contrast) # returns vector of sim. in R^(n_stimuli x n_conv_points)
    else:
        sim = comp_conv_vec(gs_vec,gref,dens,contrast=contrast)          # returns vector of sim. in R^(n_stimuli x n_conv_points)
    act = fun(sim,center,sig)                           # compute kernel function fun for each stimulus in similarity space
    act_int = np.mean(act,axis=1)
    if softmax_norm:
        act = softmax(act_int)                              # softmax normalization
    return act_int
## Note: doesn't work yet for contrast='on' (then the similarity space needs to be normalized before applying the kernel function) ##

####################################################################################################################################################################
# def apply_conv(x,sig,kg,k_type,kcenter,cdens,parallel): 
#     return k_gabor_fun_conv_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,dens=cdens,parallel=parallel)

# def apply_mask(x,sig,kg,km,k_type,kcenter): 
#     return k_gabor_fun_mask_vec(x,km,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter) 

# def apply_fun(x,sig,kg,k_type,kcenter): 
#     return k_gabor_fun_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter)

def init_gabor_knov_from_images(kgabor,k_type='box',ksig=0.1,kcenter=1,cdens=2,conv=False,parallel=False,contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,debug=False,xw=12,yw=12):

    # Plot reference Gabors and masks (debug mode)
    if debug:
        for i in range(len(kgabor)):
            f,ax = plt.subplots(1,1,figsize=(1,1))
            ax.imshow(kgabor[i],cmap='binary',vmin=-1,vmax=1,origin='lower') 
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Gabor filter {i}',c='w')

    # Define kernels (one centered around each Gabor reference filter)
    if conv:
        k = [lambda x,sig,kg=kg: k_gabor_fun_conv_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,dens=cdens,parallel=parallel,contrast=contrast,softmax_norm=softmax_norm) for kg in kgabor]
    else:
        k = [lambda x,sig,kg=kg: k_gabor_fun_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg in kgabor]

    # Format parameters
    k_params = {'k_type': k_type,
                'gnum':kgabor.shape[0],
                'k':k,
                # 'kmu':[kcenter]*len(k),
                'ksig':[ksig]*len(k),
                'eps_k':eps_k,
                'alph_k':alph_k,
                't0_update':0,
                'ref_gabors':None,
                'ref_gabors_im':kgabor
                # 'ref_gabors_fullim':o_kgabor
    }

    if conv:
        k_params['size_conv'] = 2 * xw * 2 * yw
        k_params['cdens']     = cdens

    return k_params
    
    

# Initialize Gabor novelty and return dictionary with parameters
def init_gabor_knov(gnum=16,k_type='triangle',ksig=1,kcenter=1,cdens=2,seed=12345,rng=None,mask=True,conv=False,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='basic',fixed_freq=None,fixed_width=None,contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,add_empty=False,debug=False):
    
    # Define random generator and stimulus dimensions
    if not rng: rng = np.random.default_rng(seed)
    dim_ranges_gabor = gs.dim_ranges_rad.copy()
    dim_names_gabor = gs.dim_names.copy()

    # Make gnum Gabor dimensions randomly
    kcl    = gs.generate_stim(dim_ranges_gabor,gnum,rng,adj_w=adj_w,adj_f=adj_f,alph_adj=alph_adj,sampling=sampling,fixed_freq=fixed_freq,fixed_width=fixed_width)
    kcl_df = pd.DataFrame(dict(zip(dim_names_gabor,kcl)))
    kgabor = []
    if conv:
        # pwx = stim_df['width'] / 260 * 200
        # pwy = stim_df['width'] / 90 * 100
        # gc = lambda x: int(np.ceil(x))+5
        # yw = gc(pwy)
        # xw = gc(pwx)

        # kratio = 0.05 #0.06
        # kint   = lambda x: int(np.round(x))
        # xw = kint(kratio*200); yw = xw      # get target width of filter (in pixels)
        yw = xw = 12
        # o_kgabor = []
    if mask:
        kmasks = []
    for i in range(len(kcl_df)):
        if conv:
            # Generate conv. kernels by cutting them from the original image (need padding if the kernel is at the boundary - not yet implemented)
            # gi,_ = gs.comp_gabor(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1) # add resolution, magn as input parameters (later)
            # px = (kcl_df.iloc[i]['x-position'] + 130) / 260 * 200  # get x-position of Gabor center
            # py = (kcl_df.iloc[i]['y-position'] + 20) / 90 * 100    # get y-position of Gabor center
            # o_kgabor.append(gi.copy())
            # gi = gi[kint(py)-yw:kint(py)+yw,kint(px)-xw:kint(px)+xw]
            # Generate conv. kernels directly at the center of the small patch
            kcl_corr = kcl_df.iloc[i].values
            kcl_corr[4] = 0; kcl_corr[5] = 0
            gi,_ = gs.comp_gabor([-xw,xw],[-yw,yw],kcl_corr.reshape((-1,1)),resolution=2*xw,magn=1,ratio_y_x=1) # add resolution, magn as input parameters (later)
        else:
            gi,_ = gs.comp_gabor(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1) # add resolution, magn as input parameters (later)
        kgabor.append(gi)
        if mask and not conv:
            mi,_ = gs.comp_gauss(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=100,magn=1) # add resolution, magn as input parameters (later)
            kmasks.append(mi)
    if add_empty:
        if conv:
            gi = gs.get_empty([-xw,xw],[-yw,yw],resolution=2*xw,ratio_y_x=1)
        else:
            gi = gs.get_empty(dim_ranges_gabor[4],dim_ranges_gabor[5],resolution=100)
        kgabor.append(gi)

    # Plot reference Gabors and masks (debug mode)
    if debug:
        for i in range(len(kgabor)):
            f,ax = plt.subplots(1,1,figsize=(1,1))
            ax.imshow(kgabor[i],cmap='binary',vmin=-1,vmax=1,origin='lower') 
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Gabor filter {i}',c='w')
        if mask:
            for i in range(len(kmasks)):
                f,ax = plt.subplots(1,1,figsize=(1,1))
                ax.imshow(kmasks[i],cmap='binary',vmin=-1,vmax=1,origin='lower') 
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Mask {i}',c='w')

    # Define kernels (one centered around each Gabor reference filter)
    if conv:
        k = [lambda x,sig,kg=kg: k_gabor_fun_conv_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,dens=cdens,parallel=parallel,contrast=contrast,softmax_norm=softmax_norm) for kg in kgabor]
    elif mask:
        k = [lambda x,sig,kg=kg: k_gabor_fun_mask_vec(x,km,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg,km in zip(kgabor,kmasks)]
    else:
        k = [lambda x,sig,kg=kg: k_gabor_fun_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg in kgabor]

    # Format parameters
    k_params = {'k_type': k_type,
                'gnum':gnum,
                'k':k,
                # 'kmu':[kcenter]*len(k),
                'ksig':[ksig]*len(k),
                'eps_k':eps_k,
                'alph_k':alph_k,
                't0_update':0,
                'ref_gabors':kcl_df,
                'ref_gabors_im':kgabor
                # 'ref_gabors_fullim':o_kgabor
    }

    if conv:
        k_params['size_conv'] = 2 * xw * 2 * yw
        k_params['cdens']     = cdens
    return k_params

####################################################################################################################################################################
# # Simulate kernel-based novelty with given Gabor kernels
# def sim_knov_gabor(stim,k,ksig0,eps_k,t0_update=0,cdens=2,conv=False,plot=False,figsize=None,rec_time=100,save_plot=False,save_plot_dir='',save_plot_name=''):

#     # Initialize novelty
#     if conv:
#         num_conv = stim[0,::cdens,::cdens].size
#         knum = len(k) * num_conv
#         kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=stim)
#     else:
#         knum = len(k)
#         kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov(k,ksig0,seq=stim)

#     # Check if all stimuli are represented
#     represented = (np.sum(kmat_seq,axis=0)!=0).all()
#     if not represented:
#         print('Simulation will note be executed because at least one input stimulus is not represented by the current kernel model. All recordings are set to NaN. \n')
#         klist = []
#         plist = np.NaN * np.ones(kmat_seq.shape[1])
#         nlist = np.NaN * np.ones(kmat_seq.shape[1])
#         kwlist = []
#     else:
#         # Initialize recording
#         klist = []
#         plist = []
#         nlist = []
#         kwlist = []

#         # Run task
#         for t in range(stim.shape[0]):
#             # Compute novelty values
#             kk,pk,nk = knov_vec.comp_nov(kwl,kmat_seq[:,t].reshape((-1,1)))
#             klist.append(kk)
#             plist.append(pk[0])
#             nlist.append(nk[0])
#             kwlist.append(kwl)

#             # Update novelty parameters in response to stimulus
#             rk_new    = knov_vec.update_rk_approx(kwl,kmat_seq,t)                             # Update responsibilities
#             kwl       = knov_vec.update_nov_approx(kwl,t+t0_update,rk_new,knum,eps=eps_k)     # Update kernel weights

#         kwlist.append(kwl)

#     return klist,plist,nlist,kwlist

# Simulate kernel-based novelty with given Gabor kernels (fixed learning rate)
# def sim_knov_gabor_flr(stim,k,ksig0,alph_k,t0_update=0,cdens=2,conv=False,plot=False,figsize=None,rec_time=100,save_plot=False,save_plot_dir='',save_plot_name=''):
#     # Initialize novelty
#     if conv:
#         num_conv = stim[0,::cdens,::cdens].size
#         knum = len(k) * num_conv
#         kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=stim)
#     else:
#         knum = len(k)
#         kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov(k,ksig0,seq=stim)

#     # Check if all stimuli are represented
#     represented = (np.sum(kmat_seq,axis=0)!=0).all()
#     if not represented:
#         print('Simulation will note be executed because at least one input stimulus is not represented by the current kernel model. All recordings are set to NaN. \n')
#         klist = []
#         plist = np.NaN * np.ones(kmat_seq.shape[1])
#         nlist = np.NaN * np.ones(kmat_seq.shape[1])
#         kwlist = []
#     else:
#         # Initialize recording
#         klist = []
#         plist = []
#         nlist = []
#         kwlist = []

#         # Run task
#         for t in range(stim.shape[0]):
#             # Compute novelty values
#             kk,pk,nk = knov_vec.comp_nov(kwl,kmat_seq[:,t].reshape((-1,1)))
#             klist.append(kk)
#             plist.append(pk[0])
#             nlist.append(nk[0])
#             kwlist.append(kwl)

#             # Update novelty parameters in response to stimulus
#             rk_new    = knov_vec.update_rk_approx(kwl,kmat_seq,t)                  # Update responsibilities
#             kwl       = knov_vec.update_nov_approx_flr(kwl,rk_new,alph=alph_k)     # Update kernel weights
        
#         kwlist.append(kwl)

#     return klist,plist,nlist,kwlist

####################################################################################################################################################################
# Simulations based on indexing input stimuli
####################################################################################################################################################################
# Adaptive lr
def sim_knov_gabor(stim,k,ksig0,eps_k,t0_update=0,cdens=2,conv=False,plot=False,figsize=None,rec_time=100,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,flip_idx=None,kmat_seq_flipped=None,stop_nokernels=True):
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))
    # Initialize novelty
    if conv:
        num_conv = stim_unique[0,::cdens,::cdens].size
        knum = len(k) * num_conv
        kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=stim_unique,parallel=parallel_k)
    else:
        knum = len(k)
        kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov(k,ksig0,seq=stim_unique,parallel=parallel_k)
    # Flip bits of the kmat_seq matrix
    if flip:
        if kmat_seq_flipped is not None:
            print(f'Flip sum: {np.sum(np.abs(kmat_seq-kmat_seq_flipped))}')
            kmat_seq = kmat_seq_flipped
        else:
            kmat_seq_old        = kmat_seq.copy()
            for i in range(len(flip_idx)):
                flip_nz = flip_idx[i][0]
                flip_z  = flip_idx[i][1]
                kmat_seq[flip_nz]   = kmat_seq_old[flip_z]
                kmat_seq[flip_z]    = kmat_seq_old[flip_nz]
            print(f'Flip sum: {np.sum(np.abs(kmat_seq-kmat_seq_old))}')

    # Check if all stimuli are represented
    represented = (np.sum(kmat_seq,axis=0)!=0).all()
    if stop_nokernels and not represented:
        print('Simulation will note be executed because at least one input stimulus is not represented by the current kernel model. All recordings are set to NaN. \n')
        klist = []
        plist = np.NaN * np.ones(len(stim_idx))
        nlist = np.NaN * np.ones(len(stim_idx))
        kwlist = []
    else:
        # Initialize recording
        klist = []
        plist = []
        nlist = []
        kwlist = []

        # Run task
        for t in range(len(stim_idx)):
            # Compute novelty values
            kk,pk,nk = knov_vec.comp_nov(kwl,kmat_seq[:,stim_idx[t]].reshape((-1,1)))
            klist.append(kk)
            plist.append(pk[0])
            nlist.append(nk[0])
            kwlist.append(kwl)

            # Update novelty parameters in response to stimulus
            rk_new    = knov_vec.update_rk_approx(kwl,kmat_seq,stim_idx[t])                   # Update responsibilities
            kwl       = knov_vec.update_nov_approx(kwl,t+t0_update,rk_new,knum,eps=eps_k)     # Update kernel weights

        kwlist.append(kwl)

    return klist,plist,nlist,kwlist

# Fixed lr
def sim_knov_gabor_flr(stim,k,ksig0,alph_k,t0_update=0,cdens=2,conv=False,plot=False,figsize=None,rec_time=100,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,flip_idx=None,kmat_seq_flipped=None,stop_nokernels=True):
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))
    # Initialize novelty
    if conv:
        num_conv = stim_unique[0,::cdens,::cdens].size
        knum     = len(k) * num_conv
        kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=stim_unique,parallel=parallel_k)
    else:
        knum = len(k)
        kwl, kmat, kmat_seq, _, _, _ = knov_vec.init_nov(k,ksig0,seq=stim_unique,parallel=parallel_k)
    # Flip bits of the kmat_seq matrix
    if flip:
        # # Switch num_flips non-zero and a zero kernels
        # kmat_seq_nz = np.where(kmat_seq!=0)
        # kmat_seq_z  = np.where(kmat_seq==0)
        # flip_nz = np.choice(kmat_seq_nz[0],num_flips)
        # flip_z  = np.choice(kmat_seq_z[0],num_flips)
        if kmat_seq_flipped is not None:
            print(f'Flip sum: {np.sum(np.abs(kmat_seq-kmat_seq_flipped))}')
            kmat_seq = kmat_seq_flipped
        else:
            kmat_seq_old        = kmat_seq.copy()
            for i in range(len(flip_idx)):
                flip_nz = flip_idx[i][0]
                flip_z  = flip_idx[i][1]
                kmat_seq[flip_nz]   = kmat_seq_old[flip_z]
                kmat_seq[flip_z]    = kmat_seq_old[flip_nz]
            print(f'Flip sum: {np.sum(np.abs(kmat_seq-kmat_seq_old))}')

    # Check if all stimuli are represented
    represented = (np.sum(kmat_seq,axis=0)!=0).all()
    if stop_nokernels and not represented:
        print('Simulation will note be executed because at least one input stimulus is not represented by the current kernel model. All recordings are set to NaN. \n')
        klist = []
        plist = np.NaN * np.ones(len(stim_idx))
        nlist = np.NaN * np.ones(len(stim_idx))
        kwlist = []
    else:
        # Initialize recording
        klist = []
        plist = []
        nlist = []
        kwlist = []

        # Run task
        for t in range(len(stim_idx)):
            # Compute novelty values
            kk,pk,nk = knov_vec.comp_nov(kwl,kmat_seq[:,stim_idx[t]].reshape((-1,1)))
            klist.append(kk)
            plist.append(pk[0])
            nlist.append(nk[0])
            kwlist.append(kwl)

            # Update novelty parameters in response to stimulus
            rk_new    = knov_vec.update_rk_approx(kwl,kmat_seq,stim_idx[t])        # Update responsibilities
            kwl       = knov_vec.update_nov_approx_flr(kwl,rk_new,alph=alph_k)     # Update kernel weights
        
        kwlist.append(kwl)

    return klist,plist,nlist,kwlist

####################################################################################################################################################################
# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment
def run_gabor_knov_withparams(stim,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None,stop_nokernels=True):
    # Run novelty experiment
    klist_gabor,plist_gabor,nlist_gabor,kwlist_gabor = sim_knov_gabor(stim,
                                                                        k=k_params['k'],
                                                                        ksig0=k_params['ksig'],
                                                                        eps_k=k_params['eps_k'],
                                                                        t0_update=k_params['t0_update'],
                                                                        cdens=k_params['cdens'] if 'cdens' in k_params.keys() else None,
                                                                        conv=True if 'cdens' in k_params.keys() else False,
                                                                        plot=plot_kernels,
                                                                        rec_time=20,
                                                                        save_plot=save_plot,
                                                                        save_plot_dir=save_plot_dir,
                                                                        save_plot_name=save_plot_name,
                                                                        idx=idx,
                                                                        parallel_k=parallel_k,
                                                                        flip=flip,
                                                                        flip_idx=k_params['flip_idx'] if 'flip_idx' in k_params.keys() else None,
                                                                        kmat_seq_flipped=kmat_seq_flipped,
                                                                        stop_nokernels=stop_nokernels)
    # Format data
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    kl_gabor = np.squeeze(np.stack(klist_gabor)) if len(klist_gabor)>0 else []
    kwl_gabor = np.squeeze(np.stack(kwlist_gabor)) if len(kwlist_gabor) else []
    df_all = pd.DataFrame({'nt':nl_gabor.flatten(),'pt':pl_gabor.flatten()})
    return df_all, kl_gabor, kwl_gabor

# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment (fixed learning rate)
def run_gabor_knov_withparams_flr(stim,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None,stop_nokernels=True):
    # Run novelty experiment
    klist_gabor,plist_gabor,nlist_gabor,kwlist_gabor = sim_knov_gabor_flr(stim,
                                                                            k=k_params['k'],
                                                                            ksig0=k_params['ksig'],
                                                                            alph_k=k_params['alph_k'],
                                                                            cdens=k_params['cdens'] if 'cdens' in k_params.keys() else None,
                                                                            conv=True if 'cdens' in k_params.keys() else False,
                                                                            plot=plot_kernels,
                                                                            rec_time=20,
                                                                            save_plot=save_plot,
                                                                            save_plot_dir=save_plot_dir,
                                                                            save_plot_name=save_plot_name,
                                                                            idx=idx,
                                                                            parallel_k=parallel_k,
                                                                            flip=flip,
                                                                            flip_idx=k_params['flip_idx'] if 'flip_idx' in k_params.keys() else None,
                                                                            kmat_seq_flipped=kmat_seq_flipped,
                                                                            stop_nokernels=stop_nokernels)
    # Format data
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    kl_gabor = np.squeeze(np.stack(klist_gabor)) if len(klist_gabor)>0 else []
    kwl_gabor = np.squeeze(np.stack(kwlist_gabor)) if len(kwlist_gabor)>0 else []
    df_all = pd.DataFrame({'nt':nl_gabor.flatten(),'pt':pl_gabor.flatten()})
    return df_all, kl_gabor, kwl_gabor

