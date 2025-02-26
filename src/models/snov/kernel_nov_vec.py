import numpy as np
import multiprocessing as mp
import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty')
import src.models.snov.kernel_nov as knov

### Functions to compute simple 1D novelty for vector of k (potentially different) kernel functions ###

######################################################################################################################################################################
# Compute kernel matrix for vector of different k functions (instead of a single one)
def kw(i,k,x): 
    kk = k(x)
    return (i,kk)

def kwsig(i,k,x,ksig): 
    kk = k(x,ksig)
    return (i,kk)

def compute_kmat(k,ksig=[],x=np.array([]),parallel=False): 
    if len(x)>0 and parallel: 
        num_pool = mp.cpu_count()
        pool = mp.Pool(num_pool)
        if len(ksig)==0:
            jobs = [pool.apply_async(kw,args=(i,k[i],x)) for i in range(len(k))]
        else:
            jobs = [pool.apply_async(kwsig,args=(i,k[i],x,ksig[i])) for i in range(len(k))]
        data = [r.get() for r in jobs]
        pool.close()
        pool.join() 
        kmat = np.array(data,axis=1)
    elif len(x)>0 and not parallel:
        if len(ksig)==0:
            kmat = np.array([k[i](x) for i in range(len(k))])
        else:
            kmat = np.array([k[i](x,ksig[i]) for i in range(len(k))])
    else:        
        kmat = None
    return kmat

def compute_kmat_conv(k,ksig=[],x=np.array([]),parallel=False): 
    if len(x)>0 and parallel: 
        num_pool = mp.cpu_count()
        pool = mp.Pool(num_pool)
        if len(ksig)==0:
            jobs = [pool.apply_async(kw,args=(i,k[i],x)) for i in range(len(k))]
        else:
            jobs = [pool.apply_async(kwsig,args=(i,k[i],x,ksig[i])) for i in range(len(k))]
        data = [r.get() for r in jobs]
        data.sort(key=lambda tup: tup[0])
        pool.close()
        pool.join() 
        kmat = np.concatenate(data,axis=1).transpose()
    elif len(x)>0 and not parallel:
        if len(ksig)==0:
            kmat = np.concatenate([k[i](x) for i in range(len(k))],axis=1).transpose()
        else:
            kmat = np.concatenate([k[i](x,ksig[i]) for i in range(len(k))],axis=1).transpose()
    else:        
        kmat = None
    return kmat

# Initialize novelty variables for vector of k functions
def init_nov(k,ksig,x=np.array([]),seq=np.array([]),update_means=False,update_widths=False,full_update=False,parallel=False): 
    # k:kernel function, kc:kernel centers, ksig:kernel widths, seq:sequence of stimuli presented
    kw   = 1/(len(k))*np.ones(len(k)).reshape(-1,1)
    if len(x)>0: kmat = compute_kmat(k,ksig,x,parallel=parallel)
    else:        kmat = None
    if len(seq)>0: kmat_seq = compute_kmat(k,ksig,seq,parallel=parallel) 
    else:          kmat_seq = None

    rksum = None; rkmat = None; kmumat = None                         
    if full_update or update_widths:
        rkmat = 1/(len(k))*np.ones((len(k),len(seq)))  
    return kw, kmat, kmat_seq, rksum, rkmat, kmumat

def init_nov_conv(k,ksig,num_conv,x=np.array([]),seq=np.array([]),update_means=False,update_widths=False,full_update=False,parallel=False):
    # k:kernel function, kc:kernel centers, ksig:kernel widths, seq:sequence of stimuli presented
    kw   = 1/(len(k)*num_conv)*np.ones(len(k)*num_conv).reshape(-1,1)
    if len(x)>0: kmat = compute_kmat_conv(k,ksig,x,parallel=parallel)
    else:        kmat = None
    if len(seq)>0: kmat_seq = compute_kmat_conv(k,ksig,seq,parallel=parallel)
    else:          kmat_seq = None

    rksum = None; rkmat = None; kmumat = None                         
    if full_update or update_widths:
        rkmat = 1/(len(k)*num_conv)*np.ones((len(k)*num_conv,len(seq)))  
    return kw, kmat, kmat_seq, rksum, rkmat, kmumat

def init_convint_nov(k,ksig,x=np.array([]),seq=np.array([]),update_means=False,update_widths=False,full_update=False,parallel=False,pidx=None):
    kw   = 1/(len(k)*len(pidx))*np.ones(len(k)*len(pidx)).reshape(-1,1)
    if len(x)>0: 
        kmat_ll = []
        for i in range(len(pidx)):
            sui = np.stack([x[j][pidx[i]] for j in range(len(x))],axis=0)
            kmat = compute_kmat(k,ksig,sui,parallel=parallel) 
            kmat_ll.append(kmat)
        kmat = np.concatenate(kmat_ll,axis=1)
    else:       
        kmat = None
    if len(seq)>0: 
        kmat_seq_ll = []
        for i in range(len(pidx)):
            sui = np.stack([seq[j][pidx[i]] for j in range(len(seq))],axis=0)
            kmat_seq = compute_kmat(k,ksig,sui,parallel=parallel) 
            kmat_seq_ll.append(kmat_seq)
        kmat_seq = np.stack(kmat_seq_ll,axis=2)
    else:          
        kmat_seq = None
    
    rksum = None; rkmat = None; kmumat = None                         
    if full_update or update_widths:
        rkmat = 1/(len(k)*len(pidx))*np.ones((len(k)*len(pidx),len(seq)))  
    return kw, kmat, kmat_seq, rksum, rkmat, kmumat
    

######################################################################################################################################################################
# Evaluate novelty for array of states
def comp_nov(kw,kmat):
    return knov.comp_nov(kw,kmat)

######################################################################################################################################################################
# Update responsibilities
def update_rk_approx(kw,kmat,t):
    return knov.update_rk_approx(kw,kmat,t)

def update_rkmat_approx(kw,kmat,rkmat,t):
    return knov.update_rkmat_approx(kw,kmat,rkmat,t)

def update_rkmat_full(kw,kmat):
    return knov.update_rkmat_full(kw,kmat)

def update_rk_approx1(kw,kmat):
    return knov.update_rk_approx1(kw,kmat)

######################################################################################################################################################################
# Update novelties
def update_nov_approx_flr(kw,rk,alph):
    return knov.update_nov_approx_flr(kw,rk,alph)

def update_nov_approx(kw,t,rk,knum,eps=1):
    return knov.update_nov_approx(kw,t,rk,knum,eps=1)

def update_nov_full(kw,t,rkmat,rkmat_old,knum,eps=1):
    return knov.update_nov_full(kw,t,rkmat,rkmat_old,knum,eps=1)

def update_nov_approx1(kw,t,rk,knum,eps=1):
    return knov.update_nov_approx1(kw,t,rk,knum,eps=1)

