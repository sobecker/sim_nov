import numpy as np
rng = np.random.default_rng(seed=12345)
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import models.snov.run_gabor_knov2 as gknov2
import models.snov.kernel_nov_vec as knov_vec  
import models.snov.gabor_stimuli as gs

### This file contains functions to simulate Gabor novelty using convolutional or non-convolutional kernels (simple + complex cells) ###

####################################################################################################################################################################
# Helper functions
####################################################################################################################################################################
def kw(i,k,x): 
    kk = k(x)
    return (i,kk)

def kwsig(i,k,x,ksig): 
    kk = k(x,ksig)
    return (i,kk)

def kfun_complex(k_arr, i, mode='mean'): # k_arr has shape: n_stimuli x n_conv_points x n_kernels
    aggfun = np.sum if mode=='sum' else np.mean
    k_arr = k_arr * (k_arr>0)
    return i, aggfun(k_arr,axis=-1) # output shape: n_stimuli x n_conv_points

def kfun_complex_comb(k_arr, i, num_complex=1, type_complex=[4], mode='mean'): # k_arr has shape: n_stimuli x n_conv_points x n_kernels
    k_arr = k_arr * (k_arr>0) 
    aggfun = np.sum if mode=='sum' else np.mean

    # Compute all combinations of complex cells
    num_simple = k_arr.shape[-1]
    if isinstance(type_complex,int):
        type_complex = np.arange(2,type_complex+1)
    elif len(type_complex)==0:
        type_complex = np.arange(2,num_simple+1)
    else:
        type_complex = np.sort(np.array(type_complex))
    c_all = []
    for i in type_complex:
        c_i = [aggfun(k_arr[:,:,idx_i],axis=-1) for idx_i in itertools.combinations(range(k_arr.shape[-1]),i)] 
        c_all.extend(c_i)

    # Draw a given number of them at random
    num_complex     = min(num_complex,len(c_all)) # make sure we don't draw more complex cells than available
    rand_complex    = 1-num_complex%1             # extract probabilistic part of num_complex
    num_complex     = int(num_complex)            # extract deterministic part of num_complex

    if len(c_all)>=num_complex+1 and rand_complex!=1: # draw additional cell if enough cells available and num_complex has probabilistic part
        num_complex += 1

    if k_arr.shape[-1] in type_complex:
        c = [c_all[-1]] # largest complex cell is always selected
        if num_complex>1:
            c.extend([c_all[i] for i in np.random.choice(range(len(c_all)-1),num_complex-1,replace=False)])
    else:
        c = [c_all[i] for i in np.random.choice(range(len(c_all)),num_complex,replace=False)]
    
    # Drop last drawn complex cell to account for the probabilistic part of num_complex
    if rand_complex!=1:
        rr = np.random.rand()
        if rr<rand_complex:
            c = c[:-1]
    
    # Concatenate complex cells
    if len(c)>0: 
        c = np.concatenate(c,axis=-1)

    return i, c

####################################################################################################################################################################
# Compute kmat and kmat_seq for given stimulus sequence
####################################################################################################################################################################
def compute_kmat(k,kfun_idx,kfun_complex_idx,ksig=[],x=np.array([]),kwargs={}): 
    if len(x)>0:
        # Set optional args
        no_simple_cells = kwargs['no_simple_cells'] if 'no_simple_cells' in kwargs.keys() else False
        no_complex_cells = kwargs['no_complex_cells'] if 'no_complex_cells' in kwargs.keys() else False
        debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
        mode_complex = kwargs['mode_complex'] if 'mode_complex' in kwargs.keys() else 'mean'

        kmat_all = []
        # Compute simple cell kernels
        kmat_s = np.concatenate([k[i](x,ksig[i])[:,None] for i in range(len(k))],axis=-1) # kmat_s has shape: n_stimuli x n_kernels
        if not no_simple_cells:
            kmat_all.append(kmat_s)

        # Compute complex cell kernels
        if not no_complex_cells:
            for j in range(len(kfun_complex_idx)):
                # kmat_c = np.concatenate([kfun_complex(kmat_s[:,np.where(kfun_idx[j]==i)[0]],i,mode=mode_complex)[1] for i in kfun_complex_idx[j]],axis=-1) # output of kfun_complex: i, kk (dim: n_stimuli x n_conv), kmat_c has dim: n_stimuli x n_conv x n_kernels_complex
                # kmat_c = np.concatenate([kfun_complex_comb(kmat_s[:,np.where(kfun_idx[j]==i)[0]],i,
                #                                            num_complex=kwargs['num_complex'] if 'num_complex' in kwargs.keys() else 1,
                #                                            type_complex=kwargs['type_complex'] if 'type_complex' in kwargs.keys() else [4],
                #                                            mode=mode_complex)[1] for i in kfun_complex_idx[j]],axis=-1)
                kmat_c = []
                for i in kfun_complex_idx[j]:
                    _, kmat_ci = kfun_complex_comb(kmat_s[:,np.where(kfun_idx[j]==i)[0]],i,
                                                           num_complex=kwargs['num_complex'] if 'num_complex' in kwargs.keys() else 1,
                                                           type_complex=kwargs['type_complex'] if 'type_complex' in kwargs.keys() else [4],
                                                           mode=mode_complex)
                    if len(kmat_ci)>0:
                        kmat_c.append(kmat_ci)
                if len(kmat_c)==0:
                    # Choose one random complex cell
                    i = np.random.choice(kfun_complex_idx[j])
                    _, kmat_ci = kfun_complex_comb(kmat_s[:,np.where(kfun_idx[j]==i)[0]],i,
                                                           num_complex=1,
                                                           type_complex=kwargs['type_complex'] if 'type_complex' in kwargs.keys() else [4],
                                                           mode=mode_complex)
                    kmat_c.append(kmat_ci)
                kmat_c = np.concatenate(kmat_c,axis=-1)
                kmat_all.append(kmat_c)
                    
        # Append matrices
        kmat = np.concatenate([kmat.reshape((kmat.shape[0],-1)) for kmat in kmat_all],axis=-1).transpose()

        # Compute number of simple vs. complex cells
        len_simple = kmat_s.shape[-1] if not no_simple_cells else 0
        len_complex = kmat.shape[0]-len_simple
    else:        
        kmat = None
    return kmat, (len_simple, len_complex)

def compute_kmat_conv(k,kfun_idx,kfun_complex_idx,ksig=[],x=np.array([]),kwargs={}): 
    if len(x)>0: 
        # Set optional args
        no_simple_cells = kwargs['no_simple_cells'] if 'no_simple_cells' in kwargs.keys() else False
        no_complex_cells = kwargs['no_complex_cells'] if 'no_complex_cells' in kwargs.keys() else False
        debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
        mode_complex = kwargs['mode_complex'] if 'mode_complex' in kwargs.keys() else 'mean'

        kmat_all = []
        # Compute simple cell kernels
        kmat_s = np.concatenate([k[i](x,ksig[i])[:,:,None] for i in range(len(k))],axis=-1) # output of k: kk (dim: n_stimuli x n_conv), kmat has dim: n_stimuli x n_conv x n_kernels_simple
        if not no_simple_cells:
            kmat_all.append(kmat_s)

        # Compute complex cell kernels
        if not no_complex_cells:
            for j in range(len(kfun_complex_idx)):
                # kmat_c = np.concatenate([kfun_complex(kmat_s[:,:,np.where(kfun_idx[j]==i)[0]],i,mode=mode_complex)[1] for i in kfun_complex_idx[j]],axis=-1) # output of kfun_complex: i, kk (dim: n_stimuli x n_conv), kmat_c has dim: n_stimuli x n_conv x n_kernels_complex
                # kmat_c = np.concatenate([kfun_complex_comb(kmat_s[:,:,np.where(kfun_idx[j]==i)[0]],i,
                #                                        num_complex=kwargs['num_complex'] if 'num_complex' in kwargs.keys() else 1,
                #                                        type_complex=kwargs['type_complex'] if 'type_complex' in kwargs.keys() else [4],
                #                                        mode=mode_complex)[1] for i in kfun_complex_idx[j]],axis=-1)
                kmat_c = []
                for i in kfun_complex_idx[j]:
                    _, kmat_ci = kfun_complex_comb(kmat_s[:,:,np.where(kfun_idx[j]==i)[0]],i,
                                                           num_complex=kwargs['num_complex'] if 'num_complex' in kwargs.keys() else 1,
                                                           type_complex=kwargs['type_complex'] if 'type_complex' in kwargs.keys() else [4],
                                                           mode=mode_complex)
                    if len(kmat_ci)>0:
                        kmat_c.append(kmat_ci)
                if len(kmat_c)==0:
                    # Choose one random complex cell
                    i = np.random.choice(kfun_complex_idx[j])
                    _, kmat_ci = kfun_complex_comb(kmat_s[:,:,np.where(kfun_idx[j]==i)[0]],i,
                                                           num_complex=1,
                                                           type_complex=kwargs['type_complex'] if 'type_complex' in kwargs.keys() else [4],
                                                           mode=mode_complex)
                    kmat_c.append(kmat_ci)
                kmat_c = np.concatenate(kmat_c,axis=-1)
                kmat_all.append(kmat_c)

        # Append matrices
        kmat = np.concatenate([kmat.reshape((kmat.shape[0],-1)) for kmat in kmat_all],axis=-1).transpose() # n_kernels(all) x n_stimuli

        # Compute number of simple vs. complex cells
        len_simple = kmat_s.shape[-1] if not no_simple_cells else 0
        len_conv   = kmat_s.shape[-2]
        len_complex = int((kmat.shape[0]-len_simple*len_conv)/len_conv)

        if debug:
            # Plot complete kmat
            f,ax = plt.subplots(1,1,figsize=(10,4))
            vmin = np.min(kmat.flatten())
            vmax = np.max(kmat.flatten())
            a_all = ax.imshow(kmat.transpose(),cmap='binary',vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
            ax.axvline(len_simple*len_conv,color='k',lw=2)
            for i in range(kmat_s.shape[0]-1):
                ax.axhline(i+1/2,color='k',lw=1)
            ax.set_xticks([])
            ax.set_yticks(np.arange(kmat_s.shape[0]))
            ax.set_xlabel('Kernels')
            ax.set_ylabel('Stimuli')
            f.colorbar(a_all,ax=ax,label='Kernel activation'); 
            f.suptitle('Kernel activation matrix')
            f.tight_layout()

            # Plot average, cumulative and maximum activation of simple and complex cells per stimulus
            f,ax = plt.subplots(1,3,figsize=(3*3,4))
            avg_all = np.concatenate([np.mean(kmat[:len_simple,:],axis=0).reshape((-1,1)),
                                      np.mean(kmat[len_simple:,:],axis=0).reshape((-1,1))],axis=1)
            sum_all = np.concatenate([np.sum(kmat[:len_simple,:],axis=0).reshape((-1,1)),
                                      np.mean(kmat[len_simple:,:],axis=0).reshape((-1,1))],axis=1)
            max_all = np.concatenate([np.max(kmat[:len_simple,:],axis=0).reshape((-1,1)),
                                      np.max(kmat[len_simple:,:],axis=0).reshape((-1,1))],axis=1)
            stats_all = [avg_all, sum_all, max_all]
            label_all = ['Av.','Cum.','Max.']
            for i in range(len(stats_all)):
                vmin, vmax = np.min(stats_all[i].flatten()), np.max(stats_all[i].flatten())
                a_i = ax[i].imshow(stats_all[i],vmin=vmin,vmax=vmax,cmap='binary',aspect='equal',origin='lower')
                ax[i].axvline(0.5,color='k',lw=1)
                for j in range(kmat_s.shape[0]-1):
                    ax[i].axhline(j+1/2,color='k',lw=1)
                ax[i].set_xticks([])
                ax[i].set_yticks(np.arange(stats_all[i].shape[0]))
                ax[i].set_xlabel('Simple | Complex')
                ax[i].set_ylabel('Stimuli')
                f.colorbar(a_i,ax=ax[i],label=f'{label_all[i]} activation')
            f.suptitle('Kernel activation per stimulus')
            f.tight_layout()
    else:        
        kmat = None
    return kmat, (len_simple, len_complex, len_conv)

####################################################################################################################################################################
# Initialize kmat and kmat_seq for given stimulus sequence
####################################################################################################################################################################
def init_nov(k,kfun_idx,kfun_complex_idx,ksig0,x=np.array([]),seq=np.array([]),full_update=False,kwargs={}):
     # Compute kernel matrices
    if len(x)>0: 
        kmat, klen = compute_kmat(k,kfun_idx,kfun_complex_idx,ksig0,x,kwargs)
        knum = kmat.shape[-1]
    else:        
        kmat = None
        knum = 0
    if len(seq)>0: 
        kmat_seq, klen = compute_kmat(k,kfun_idx,kfun_complex_idx,ksig0,seq,kwargs)
        knum = kmat_seq.shape[-1] 
    else:          
        kmat_seq = None
        knum = 0

    # Init weights
    # print(f"Number of kernels: {knum} (simple: {klen[0]}, complex: {klen[1]}).")
    kw = (1/knum)*np.ones(knum).reshape(-1,1)   

    # Compute responsibility matrices
    rkmat = None                
    if full_update and not knum==0:
        rkmat = (1/knum)*np.ones((knum,len(seq)))  
    return kw, kmat, kmat_seq, rkmat, klen

def init_nov_conv(k,kfun_idx,kfun_complex_idx,ksig0,num_conv,x=np.array([]),seq=np.array([]),full_update=False,kwargs={}):  
    # Compute kernel matrices
    if len(x)>0: 
        kmat, klen = compute_kmat_conv(k,kfun_idx,kfun_complex_idx,ksig0,x,kwargs)
        knum = kmat.shape[0] 
    else:        
        kmat = None
        knum = 0
    if len(seq)>0: 
        kmat_seq, klen = compute_kmat_conv(k,kfun_idx,kfun_complex_idx,ksig0,seq,kwargs)
        knum = kmat_seq.shape[0] 
    else:          
        kmat_seq = None
        knum = 0

    # Init weights
    # print(f"Number of kernels: {knum} (simple: {klen[0]}, complex: {klen[1]}, conv: {klen[2]}).")
    kw = (1/knum)*np.ones(knum).reshape(-1,1)  

    # Compute responsibility matrices
    rkmat = None                    
    if full_update and not knum==0:
        rkmat = (1/knum)*np.ones((knum,len(seq)))  
    return kw, kmat, kmat_seq, rkmat, klen

####################################################################################################################################################################
# Initialize Gabor novelty and return dictionary with parameters
####################################################################################################################################################################
def init_gabor_knov(gnum_complex=4,dfreq=1.5,ctype=[4],cvar='frequency',k_type='triangle',ksig=1,kcenter=1,cdens=2,seed=12345,rng=None,mask=False,conv=True,parallel=False,adj_w=True,adj_f=False,alph_adj=3,sampling='basic',fixed_freq=None,fixed_width=None,contrast='off',softmax_norm=False,eps_k=1,alph_k=0.1,add_empty=False,debug=False):
    
    # Define random generator and stimulus dimensions
    if not rng: rng = np.random.default_rng(seed)
    dim_ranges_gabor = gs.dim_ranges_rad.copy()
    dim_names_gabor = gs.dim_names.copy()

    # Generate Gabor reference filters (complex, 4 cells per complex cell)
    kcl    = gs.generate_stim_complex(dim_ranges_gabor,gnum_complex,rng,adj_w=adj_w,adj_f=adj_f,alph_adj=alph_adj,dfreq=dfreq,sampling=sampling,ctype=ctype,cvar=cvar,fixed_freq=fixed_freq,fixed_width=fixed_width)
    kcl_df = pd.DataFrame(dict(zip(dim_names_gabor + ['cell_id'],kcl)))
    kcl_df.sort_values(by=['cell_id'],inplace=True)
    if debug:
        print(kcl_df)
    
    # Generate convolutional filters and masks (if applicable)
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
            kcl_corr = kcl_df.iloc[i].values[:len(dim_ranges_gabor)]
            kcl_corr[4] = 0; kcl_corr[5] = 0
            gi,_ = gs.comp_gabor([-xw,xw],[-yw,yw],kcl_corr.reshape((-1,1)),resolution=2*xw,magn=1,ratio_y_x=1) # add resolution, magn as input parameters (later)
        else:
            gi,_ = gs.comp_gabor(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values[:len(dim_ranges_gabor)].reshape((-1,1)),resolution=100,magn=1) # add resolution, magn as input parameters (later)
        kgabor.append(gi)
        if mask and not conv:
            mi,_ = gs.comp_gauss(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values[:len(dim_ranges_gabor)].reshape((-1,1)),resolution=100,magn=1) # add resolution, magn as input parameters (later)
            kmasks.append(mi)
    if add_empty:
        if conv:
            gi = gs.get_empty([-xw,xw],[-yw,yw],resolution=2*xw,ratio_y_x=1)
        else:
            gi = gs.get_empty(dim_ranges_gabor[4],dim_ranges_gabor[5],resolution=100)
            if mask:
                mi = gs.get_empty(dim_ranges_gabor[4],dim_ranges_gabor[5],resolution=100,add_eps=1) # Note: we want the mask for the empty filter to be 1 (i.e. no masking)
                kmasks.append(mi.copy())
        kgabor.append(gi)

    # Plot reference Gabors and masks (debug mode)
    if debug:
        for i in np.sort(kcl_df.cell_id.unique()):
            idx = np.where(kcl_df.cell_id==i)[0]
            f,axl = plt.subplots(1,len(idx),figsize=(len(idx),1))
            if mask: 
                fm, axlm = plt.subplots(1,len(idx),figsize=(len(idx),1))
            for j in range(len(idx)):
                ax = axl[j]
                ax.imshow(kgabor[idx[j]],cmap='binary',vmin=-1,vmax=1,origin='lower') 
                ax.set_xticks([])
                ax.set_yticks([])
                if mask:
                    axm = axlm[j]
                    axm.imshow(kmasks[idx[j]],cmap='binary',vmin=-1,vmax=1,origin='lower') 
                    axm.set_xticks([])
                    axm.set_yticks([])
            # f.suptitle(f'Complex cell {int(i)}\n',c='w')
            if mask:
                fm.suptitle(f'Masks for complex cell {int(i)}',c='w')

    # Define kernels (one centered around each Gabor reference filter) 
    if conv:
        k = [lambda x,sig,kg=kg: gknov2.k_gabor_fun_conv_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,dens=cdens,parallel=parallel,contrast=contrast,softmax_norm=softmax_norm) for kg in kgabor]
    elif mask:
        k = [lambda x,sig,kg=kg: gknov2.k_gabor_fun_mask_vec(x,km,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg,km in zip(kgabor,kmasks)]
    else:
        k = [lambda x,sig,kg=kg: gknov2.k_gabor_fun_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg in kgabor]
    cell_id = kcl_df.cell_id.values

    # Format parameters
    k_params = {'k_type': k_type,
                'gnum':gnum_complex,
                'k':k,
                'kfun_idx': [cell_id],
                'kfun_complex_idx': [np.unique(cell_id)],
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
# Simulations based on indexing input stimuli
####################################################################################################################################################################
# Adaptive lr
def sim_knov_gabor(stim,k,kfun_idx,kfun_complex_idx,ksig0,eps_k,t0_update=0,cdens=2,conv=False,idx=True,flip=False,flip_idx=None,kmat_seq_flipped=None,stim_type=None,kwargs={}):
    # Format inputs
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))

    # Initialize novelty
    if conv:
        num_conv = stim_unique[0,::cdens,::cdens].size
        kwl, _, kmat_seq, _, klen = init_nov_conv(k,kfun_idx,kfun_complex_idx,ksig0,num_conv,seq=stim_unique,kwargs=kwargs)
    else:
        kwl, _, kmat_seq, _, klen = init_nov(k,kfun_idx,kfun_complex_idx,ksig0,seq=stim_unique,kwargs=kwargs)
    knum = kmat_seq.shape[0]

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

    debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
    if debug and not (stim_type is None): 
        # Identify time step of the novelty event
        if 'fam_r' in stim_type:
            nov_t = np.where(np.array(stim_type)=='fam_r')[0][0]
        else:
            nov_t = np.where(np.array(stim_type)=='nov')[0][0]
    else:
        nov_t = -1

    # Check if all stimuli are represented
    represented = (np.sum(kmat_seq,axis=0)!=0).all()
    if not represented:
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

            if t==nov_t:
                # Plot weight matrix
                f,ax = plt.subplots(1,1,figsize=(10,1))
                aw = ax.imshow(kwl.transpose(),cmap='binary',origin='lower',aspect='auto')
                len_simple = klen[0] * klen[2] if len(klen)==3 else klen[0]
                ax.axvline(len_simple,color='k',lw=2)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel('Simple | Complex')
                f.colorbar(aw,ax=ax,label=f'Kernel weight at t={t} (novelty event)')
                f.tight_layout()

                # Plot stats of weight matrix
                f,ax = plt.subplots(1,3,figsize=(3*3,1))
                stats_all = [np.array([fun_i(kwl[:len_simple]), fun_i(kwl[len_simple:])]).reshape((1,-1)) for fun_i in [np.mean, np.sum, np.max]]
                label_all = ['Av.','Cum.','Max.']
                for i in range(len(stats_all)):
                    vmin, vmax = np.min(stats_all[i]), np.max(stats_all[i])
                    a_i = ax[i].imshow(stats_all[i],vmin=vmin,vmax=vmax,cmap='binary',aspect='equal',origin='lower')
                    ax[i].axvline(0.5,color='k',lw=1)
                    ax[i].set_xticks([]); ax[i].set_yticks([])
                    ax[i].set_xlabel('Simple | Complex')
                    f.colorbar(a_i,ax=ax[i],label=f'{label_all[i]} weight')
                f.tight_layout()

            # Update novelty parameters in response to stimulus
            rk_new    = knov_vec.update_rk_approx(kwl,kmat_seq,stim_idx[t])                   # Update responsibilities
            kwl       = knov_vec.update_nov_approx(kwl,t+t0_update,rk_new,knum,eps=eps_k)     # Update kernel weights

        kwlist.append(kwl)

    return klist,plist,nlist,kwlist

# Fixed lr
def sim_knov_gabor_flr(stim,k,kfun_idx,kfun_complex_idx,ksig0,alph_k,t0_update=0,cdens=2,conv=False,idx=True,flip=False,flip_idx=None,kmat_seq_flipped=None,stim_type=None,kwargs={}):
    # Format inputs
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))

    # Initialize novelty
    if conv:
        num_conv = stim_unique[0,::cdens,::cdens].size
        kwl, _, kmat_seq, _, klen = init_nov_conv(k,kfun_idx,kfun_complex_idx,ksig0,num_conv,seq=stim_unique,kwargs=kwargs)
    else:
        kwl, _, kmat_seq, _, klen = init_nov(k,kfun_idx,kfun_complex_idx,ksig0,seq=stim_unique,kwargs=kwargs)

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
    
    debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
    if debug and not (stim_type is None): 
        # Identify time step of the novelty event
        if 'fam_r' in stim_type:
            nov_t = np.where(np.array(stim_type)=='fam_r')[0][0]
        else:
            nov_t = np.where(np.array(stim_type)=='nov')[0][0]
    else:
        nov_t = -1

    # Check if all stimuli are represented
    represented = (np.sum(kmat_seq,axis=0)!=0).all()
    if not represented:
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

            if t==nov_t:
                # Plot weight matrix
                f,ax = plt.subplots(1,1,figsize=(10,1))
                aw = ax.imshow(kwl.transpose(),cmap='binary',origin='lower',aspect='auto')
                len_simple = klen[0] * klen[2] if len(klen)==3 else klen[0]
                ax.axvline(len_simple,color='k',lw=2)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel('Simple | Complex')
                f.colorbar(aw,ax=ax,label=f'Kernel weight\n at t={t}')
                f.tight_layout()

                # Plot stats of weight matrix
                f,ax = plt.subplots(1,3,figsize=(3*3,1))
                stats_all = [np.array([fun_i(kwl[:len_simple]), fun_i(kwl[len_simple:])]).reshape((1,-1)) for fun_i in [np.mean, np.sum, np.max]]
                label_all = ['Av.','Cum.','Max.']
                for i in range(len(stats_all)):
                    vmin, vmax = np.min(stats_all[i]), np.max(stats_all[i])
                    a_i = ax[i].imshow(stats_all[i],vmin=vmin,vmax=vmax,cmap='binary',aspect='equal',origin='lower')
                    ax[i].axvline(0.5,color='k',lw=1)
                    ax[i].set_xticks([]); ax[i].set_yticks([])
                    ax[i].set_xlabel('Simple | Complex')
                    f.colorbar(a_i,ax=ax[i],label=f'{label_all[i]} weight')
                f.tight_layout()

            # Update novelty parameters in response to stimulus
            rk_new    = knov_vec.update_rk_approx(kwl,kmat_seq,stim_idx[t])        # Update responsibilities
            kwl       = knov_vec.update_nov_approx_flr(kwl,rk_new,alph=alph_k)     # Update kernel weights
        
        kwlist.append(kwl)

    return klist,plist,nlist,kwlist

####################################################################################################################################################################
# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment
def run_gabor_knov_withparams(stim,k_params,idx=True,flip=False,kmat_seq_flipped=None,stim_type=None,kwargs={}):
    # Run novelty experiment
    klist_gabor,plist_gabor,nlist_gabor,kwlist_gabor = sim_knov_gabor(stim,
                                                                        k=k_params['k'],
                                                                        kfun_idx=k_params['kfun_idx'],
                                                                        kfun_complex_idx=k_params['kfun_complex_idx'],
                                                                        ksig0=k_params['ksig'],
                                                                        eps_k=k_params['eps_k'],
                                                                        t0_update=k_params['t0_update'],
                                                                        cdens=k_params['cdens'] if 'cdens' in k_params.keys() else None,
                                                                        conv=True if 'cdens' in k_params.keys() else False,
                                                                        idx=idx,
                                                                        flip=flip,
                                                                        flip_idx=k_params['flip_idx'] if 'flip_idx' in k_params.keys() else None,
                                                                        kmat_seq_flipped=kmat_seq_flipped,
                                                                        stim_type=stim_type,
                                                                        kwargs=kwargs)
    # Format data
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    kl_gabor = np.squeeze(np.stack(klist_gabor)) if len(klist_gabor)>0 else []
    kwl_gabor = np.squeeze(np.stack(kwlist_gabor)) if len(kwlist_gabor) else []
    df_all = pd.DataFrame({'nt':nl_gabor.flatten(),'pt':pl_gabor.flatten()})
    plt.close('all')
    return df_all, kl_gabor, kwl_gabor

# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment (fixed learning rate)
def run_gabor_knov_withparams_flr(stim,k_params,idx=True,flip=False,kmat_seq_flipped=None,stim_type=None,kwargs={}):
    # Run novelty experiment
    klist_gabor,plist_gabor,nlist_gabor,kwlist_gabor = sim_knov_gabor_flr(stim,
                                                                            k=k_params['k'],
                                                                            kfun_idx=k_params['kfun_idx'],
                                                                            kfun_complex_idx=k_params['kfun_complex_idx'],
                                                                            ksig0=k_params['ksig'],
                                                                            alph_k=k_params['alph_k'],
                                                                            cdens=k_params['cdens'] if 'cdens' in k_params.keys() else None,
                                                                            conv=True if 'cdens' in k_params.keys() else False,
                                                                            idx=idx,
                                                                            flip=flip,
                                                                            flip_idx=k_params['flip_idx'] if 'flip_idx' in k_params.keys() else None,
                                                                            kmat_seq_flipped=kmat_seq_flipped,
                                                                            stim_type=stim_type,
                                                                            kwargs=kwargs)
    # Format data
    pl_gabor = np.array(plist_gabor)
    nl_gabor = np.array(nlist_gabor)
    kl_gabor = np.squeeze(np.stack(klist_gabor)) if len(klist_gabor)>0 else []
    kwl_gabor = np.squeeze(np.stack(kwlist_gabor)) if len(kwlist_gabor)>0 else []
    df_all = pd.DataFrame({'nt':nl_gabor.flatten(),'pt':pl_gabor.flatten()})
    plt.close('all')
    return df_all, kl_gabor, kwl_gabor

