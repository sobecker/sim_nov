import numpy as np
rng = np.random.default_rng(seed=12345)
import pandas as pd
import os
import matplotlib.pyplot as plt
import models.snov.kernel_nov_vec as knov_vec  
import models.snov.run_gabor_knov2 as rgk2
import models.snov.gabor_stimuli as gs

### This file contains functions to simulate Gabor novelty with separate novelty models per image patch ###

def define_patches(image_dim,psize,pnum):
    p00 = [int(ii) for ii in np.round(np.linspace(0,image_dim[0]-psize[0],pnum[0]))]    # upper lines of patches
    p01 = [int(ii+psize[0]) for ii in p00]                                              # lower lines of patches
    # p01 = [int(ii) for ii in np.round(np.linspace(psize[0],image_dim[0],pnum[0]))]      
    p10 = [int(ii) for ii in np.round(np.linspace(0,image_dim[1]-psize[1],pnum[1]))]    # left lines of patches
    p11 = [int(ii+psize[1]) for ii in p10]                                          # right lines of patches
    # p11 = [int(ii) for ii in np.round(np.linspace(psize[1],image_dim[1],pnum[1]))]    

    p00.reverse(); p01.reverse()
    pidx = []
    for i in range(len(p00)):
        for j in range(len(p10)):
            pidx.append(np.ix_(np.arange(p00[i],p01[i]),np.arange(p10[j],p11[j])))
    return pidx

####################################################################################################################################################################
# Initialize Gabor novelty and return dictionary with parameters
def init_gabor_knov(pnum=(5,10),psize=None,image_dim=(100,200),gnum=16,k_type='triangle',ksig=1,kcenter=1,cdens=2,seed=12345,rng=None,mask=True,conv=False,parallel=False,adj_w=True,adj_f=False,sampling='basic',contrast='off',softmax_norm=True,eps_k=1,alph_k=0.1,add_empty=False,debug=False,convint=False):
    
    # Define image patches (by indices)
    if psize is None: 
        psize = (int(np.ceil(image_dim[0]/pnum[0])),int(np.ceil(image_dim[1]/pnum[1])))
    pidx = define_patches(image_dim,psize,pnum)

    # Define random generator and stimulus dimensions (for kernels)
    if not rng: rng = np.random.default_rng(seed)
    dim_ranges_gabor = gs.dim_ranges_rad.copy()
    dim_names_gabor = gs.dim_names.copy()
    dim_ranges_gabor[4] = [-psize[1],psize[1]]
    dim_ranges_gabor[5] = [-psize[0],psize[0]]

    # Make gnum Gabor dimensions randomly (kernel centers)
    kcl    = gs.generate_stim(dim_ranges_gabor,gnum,rng,adj_w=adj_w,adj_f=adj_f,sampling=sampling)
    kcl_df = pd.DataFrame(dict(zip(dim_names_gabor,kcl)))
    if debug:
        print(kcl_df)
    kgabor = []
    if conv:
        yw = xw = 10 # define filter size for convolutional kernels
    if mask:
        kmasks = []
    for i in range(len(kcl_df)):
        if conv:
            # Generate conv. kernels directly at the center of the filter patch
            kcl_corr = kcl_df.iloc[i].values
            kcl_corr[4] = 0; kcl_corr[5] = 0
            gi,_ = gs.comp_gabor([-xw,xw],[-yw,yw],kcl_corr.reshape((-1,1)),resolution=2*xw,magn=1,ratio_y_x=1) # add resolution, magn as input parameters (later)
        else:
            # Generate non-convolutional kernels of the same size as the image patch
            gi,_ = gs.comp_gabor(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=psize[0],magn=1,ratio_y_x=psize[1]/psize[0]) # add resolution, magn as input parameters (later)
        kgabor.append(gi)
        if mask and not conv:
            # Generate non-convolutional kernels of the same size as the image patch, with Gaussian mask (only relevant if we use overlapping patches)
            mi,_ = gs.comp_gauss(dim_ranges_gabor[4],dim_ranges_gabor[5],kcl_df.iloc[i].values.reshape((-1,1)),resolution=psize[0],magn=1,ratio_y_x=psize[1]/psize[0]) # add resolution, magn as input parameters (later)
            kmasks.append(mi)
    if add_empty:
        if conv:
            gi = gs.get_empty([-xw,xw],[-yw,yw],resolution=2*xw,ratio_y_x=1)
        else:
            gi = gs.get_empty(dim_ranges_gabor[4],dim_ranges_gabor[5],resolution=psize[0],ratio_y_x=psize[1]/psize[0])
            if mask:
                mi = gs.get_empty(dim_ranges_gabor[4],dim_ranges_gabor[5],resolution=psize[0],ratio_y_x=psize[1]/psize[0],add_eps=1) # Note: we want the mask for the empty filter to be 1 (i.e. no masking)
                kmasks.append(mi.copy())
        kgabor.append(gi)

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
        k = [lambda x,sig,kg=kg: rgk2.k_gabor_fun_conv_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,dens=cdens,parallel=parallel,contrast=contrast,softmax_norm=softmax_norm) for kg in kgabor]
    elif convint:
        k = [lambda x,sig,kg=kg: rgk2.k_gabor_fun_convint_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,dens=cdens,parallel=parallel,contrast=contrast,softmax_norm=softmax_norm) for kg in kgabor]
    elif mask:
        k = [lambda x,sig,kg=kg,km=km: rgk2.k_gabor_fun_mask_vec(x,km,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg,km in zip(kgabor,kmasks)]
    else:
        k = [lambda x,sig,kg=kg: rgk2.k_gabor_fun_vec(x,kg,eval(f'knov.k_{k_type}'),sig,center=kcenter,softmax_norm=softmax_norm) for kg in kgabor]

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
                'ref_gabors_im':kgabor,
                'image_dim':image_dim,
                'pnum':pnum,
                'psize':psize,
                'pidx':pidx
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
def sim_knov_gabor(pidx,ptot,stim,k,ksig0,eps_k,t0_update=0,cdens=2,conv=False,plot=False,figsize=None,rec_time=100,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,flip_idx=None,kmat_seq_flipped=None):
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))

    # Initialize novelty
    kwl = []; kmat_seq = []
    for i in range(len(pidx)):
        sui = np.stack([stim_unique[j][pidx[i]] for j in range(len(stim_unique))],axis=0)
        if conv:
            num_conv = sui[0][::cdens,::cdens].size
            knum = len(k) * num_conv
            kwl_i, _, kmat_seq_i, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=sui,parallel=parallel_k)
        else:
            knum = len(k)
            kwl_i, _, kmat_seq_i, _, _, _ = knov_vec.init_nov(k,ksig0,seq=sui,parallel=parallel_k)
        kwl.append(kwl_i); kmat_seq.append(kmat_seq_i)
    kwl = np.concatenate(kwl,axis=1)
    kmat_seq = np.stack(kmat_seq,axis=2)

    # Check if all stimuli are represented
    represented = (np.sum(kmat_seq,axis=0)!=0).all()
    if not represented:
        print('Simulation will note be executed because at least one input stimulus is not represented by the current kernel model. All recordings are set to NaN. \n')
        klist = []
        plist = np.NaN * np.ones((len(stim_idx),ptot))
        nlist = np.NaN * np.ones((len(stim_idx),ptot))
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
            kk,pk,nk = knov_vec.comp_nov(kwl,kmat_seq[:,stim_idx[t],:])
            klist.append(kk)
            plist.append(pk)
            nlist.append(nk)
            kwlist.append(kwl)

            # Update novelty parameters in response to stimulus
            rk_new    = knov_vec.update_rk_approx1(kwl,kmat_seq[:,stim_idx[t],:])             # Update responsibilities
            # Test
            # np.array([np.round(np.sum(rk_new[:,i]),6)==1 for i in range(rk_new.shape[-1])]).all()
            kwl       = knov_vec.update_nov_approx1(kwl,t+t0_update,rk_new,knum,eps=eps_k)     # Update kernel weights
            # Test
            # np.array([np.round(np.sum(kwl[:,i]),6)==1 for i in range(kwl.shape[-1])]).all()

        kwlist.append(kwl)

    return klist,plist,nlist,kwlist

# Fixed lr
def sim_knov_gabor_flr(pidx,ptot,stim,k,ksig0,alph_k,t0_update=0,cdens=2,conv=False,plot=False,figsize=None,rec_time=100,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,flip_idx=None,kmat_seq_flipped=None):
    if idx:
        stim_unique = stim[0]
        stim_idx    = stim[1]
    else:
        stim_unique = stim
        stim_idx    = list(np.arange(stim.shape[0]))

    # Initialize novelty
    kwl = []; kmat_seq = []
    for i in range(len(pidx)):
        sui = np.stack([stim_unique[j][pidx[i]] for j in range(len(stim_unique))],axis=0)
        if conv:
            num_conv = sui[0][::cdens,::cdens].size
            knum = len(k) * num_conv
            kwl_i, _, kmat_seq_i, _, _, _ = knov_vec.init_nov_conv(k,ksig0,num_conv,seq=sui,parallel=parallel_k)
        else:
            knum = len(k)
            kwl_i, _, kmat_seq_i, _, _, _ = knov_vec.init_nov(k,ksig0,seq=sui,parallel=parallel_k)
        kwl.append(kwl_i); kmat_seq.append(kmat_seq_i)
    kwl = np.concatenate(kwl,axis=1)
    kmat_seq = np.stack(kmat_seq,axis=2)

    # Check if all stimuli are represented (i.e. if at least one patch per image is activated)
    rep1 = np.nansum(kmat_seq,axis=0)
    rep2 = np.sum(rep1!=0,axis=1)/rep1.shape[1]
    represented = (rep2>0).all()
    p_rep = [np.where(rep1[i,:]!=0)[0] for i in range(rep1.shape[0])]
    # represented = (np.sum(kmat_seq,axis=0)!=0).all()
    if not represented:
        print('Simulation will note be executed because at least one input stimulus is not represented by the current kernel model. All recordings are set to NaN. \n')
        klist = []
        plist = np.NaN * np.ones((len(stim_idx),ptot))
        nlist = np.NaN * np.ones((len(stim_idx),ptot))
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
            pt_rep = p_rep[stim_idx[t]]
            kk,pk,nk = knov_vec.comp_nov(kwl,kmat_seq[:,stim_idx[t],:])
            klist.append(kk)
            plist.append(pk)
            nlist.append(nk)
            kwlist.append(kwl)

            # Update novelty parameters in response to stimulus
            rk_new = knov_vec.update_rk_approx1(kwl[:,pt_rep],kmat_seq[:,stim_idx[t],pt_rep])             # Update responsibilities
            # Test
            # np.array([np.round(np.sum(rk_new[:,i]),6)==1 for i in range(rk_new.shape[-1])]).all()
            kwl[:,pt_rep] = knov_vec.update_nov_approx_flr(kwl[:,pt_rep],rk_new,alph=alph_k)     # Update kernel weights
            # Test
            # np.array([np.round(np.sum(kwl[:,i]),6)==1 for i in range(kwl.shape[-1])]).all()
        
        kwlist.append(kwl)

    return klist,plist,nlist,kwlist

####################################################################################################################################################################
# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment
def run_gabor_knov_withparams(stim,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None):
    # Run novelty experiment
    klist_gabor,plist_gabor,nlist_gabor,kwlist_gabor = sim_knov_gabor(pidx=k_params['pidx'],
                                                                      ptot=k_params['pnum'][0]*k_params['pnum'][1],
                                                                        stim=stim,
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
                                                                        kmat_seq_flipped=kmat_seq_flipped)
    # Format data
    pl_gabor = np.stack(plist_gabor,axis=0)
    nl_gabor = np.stack(nlist_gabor,axis=0)
    kl_gabor = np.squeeze(np.stack(klist_gabor)) if len(klist_gabor)>0 else []
    kwl_gabor = np.squeeze(np.stack(kwlist_gabor)) if len(kwlist_gabor) else []
    df_all = pd.DataFrame({'nt':np.nansum(nl_gabor,axis=1),'pt':np.nanprod(pl_gabor,axis=1)})
    return df_all, kl_gabor, kwl_gabor, pl_gabor, nl_gabor

# Simulate Gabor novelty with given stimulus sequence and parameter dictionary for Homann experiment (fixed learning rate)
def run_gabor_knov_withparams_flr(stim,k_params,plot_kernels=False,save_plot=False,save_plot_dir='',save_plot_name='',idx=True,parallel_k=False,flip=False,kmat_seq_flipped=None):
    # Run novelty experiment
    klist_gabor,plist_gabor,nlist_gabor,kwlist_gabor = sim_knov_gabor_flr(pidx=k_params['pidx'],
                                                                          ptot=k_params['pnum'][0]*k_params['pnum'][1],
                                                                            stim=stim,
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
                                                                            kmat_seq_flipped=kmat_seq_flipped)
    # Format data
    pl_gabor = np.stack(plist_gabor,axis=0)
    nl_gabor = np.stack(nlist_gabor,axis=0)
    kl_gabor = np.squeeze(np.stack(klist_gabor)) if len(klist_gabor)>0 else []
    kwl_gabor = np.squeeze(np.stack(kwlist_gabor)) if len(kwlist_gabor)>0 else []
    df_all = pd.DataFrame({'nt':np.nansum(nl_gabor,axis=1),'pt':np.nanprod(pl_gabor,axis=1)})
    return df_all, kl_gabor, kwl_gabor, pl_gabor, nl_gabor

