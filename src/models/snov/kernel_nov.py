import numpy as np

#############################################################################################################################################################################################################
# Compute kernel values for array of states 
def compute_kmat(k,kc,ksig,x=np.array([])):
    if len(x)>0: 
        kmat = np.array([k(x,kc[i],ksig[i]) for i in range(len(kc))])
    else:        
        kmat = None
    return kmat

# Initialize novelty variables
def init_nov(k,kc,ksig,x=np.array([]),seq=np.array([]),update_means=False,update_widths=False,full_update=False): 
    # k:kernel function, kc:kernel centers, ksig:kernel widths, seq:sequence of stimuli presented
    kw   = 1/(len(kc))*np.ones(len(kc)).reshape(-1,1)
    if len(x)>0: kmat = compute_kmat(k,kc,ksig,x)
    else:        kmat = None
    if len(seq)>0: kmat_seq = compute_kmat(k,kc,ksig,seq) 
    else:          kmat_seq = None

    rksum = None; rkmat = None; kmumat = None
    if update_means: 
        rksum = 1/(len(kc))*np.ones((len(kc),1))                            
    if full_update or update_widths:
        rkmat = 1/(len(kc))*np.ones((len(kc),len(seq)))  
    if update_widths: 
        kmumat = kc.reshape((-1,1))*np.ones((len(kc),2)) 
    return kw, kmat, kmat_seq, rksum, rkmat, kmumat

# Evaluate novelty for array of states
def comp_nov(kw,kmat): #,test=False):
    kk = kw*kmat
    # if test: # Tests if de-serialized computation works out correctly (for debugging purposes)
    #     all_true = []
    #     for i in range(kw.shape[-1]):
    #         kki = kw[:,i].reshape((-1,1))*kmat[:,i].reshape((-1,1))
    #         all_true.append((kk[:,i].reshape((-1,1))==kki).all())
    #     print(np.array(all_true).all())
    pk = np.sum(kk,axis=0)
    nk = -np.log(pk)
    return kk,pk,nk

#############################################################################################################################################################################################################
# Update responsibilities
def update_rk_approx(kw,kmat,t):
    kk_s  = kw*kmat[:,t].reshape((-1,1))
    rk    = (kk_s/np.sum(kk_s)).reshape((-1,1))
    return rk

def update_rkmat_approx(kw,kmat,rkmat,t):
    kk_s  = kw*kmat[:,t].reshape((-1,1))
    rk    = (kk_s/np.sum(kk_s)).reshape((-1,1))
    rkmat[:,t] = rk.flatten()
    return rkmat

# for patch processing
def update_rkmat_full(kw,kmat):
    kk_s  = kw*kmat
    rkmat = kk_s/(np.sum(kk_s,axis=0).reshape((1,-1)))
    return rkmat

# for patch processing
def update_rk_approx1(kw,kmat):
    kk_s  = kw*kmat
    rk    = kk_s/(np.sum(kk_s,axis=0).reshape((1,-1)))
    return rk
# Test:
# all_true=[]
# for i in range(kw.shape[-1]):
#     rki    = kk_s[:,i]/np.sum(kk_s[:,i])
#     all_true.append((np.round(rk[:,i],6)==np.round(rki,6)).all())
# print(np.array(all_true).all())

#############################################################################################################################################################################################################
# Update weights (incremental update, fixed learning rate)
def update_nov_approx_flr(kw,rk,alph):
    kw   = kw+alph*(rk-kw)
    return kw

# Update weights (incremental update)
def update_nov_approx(kw,t,rk,knum,eps=1):
    kw   = kw+1/(t+len(kw)*eps)*(rk-kw)
    return kw

# Update weights (incremental update)
def update_nov_approx1(kw,t,rk,knum,eps=1):
    kw   = kw+1/(t+kw.shape[0]*eps)*(rk-kw)
    return kw

# Update weights (full update)
def update_nov_full(kw,t,rkmat,rkmat_old,knum,eps=1):
    kw   = kw+1/(t+knum*eps)*(rkmat[:,t].reshape((-1,1))-kw+np.sum(rkmat[:,:t]-rkmat_old[:,:t],axis=1).reshape((-1,1))) 
    return kw

#############################################################################################################################################################################################################
# Update kernel centers (incremental update, fixed learning rate)
def update_centers_approx_flr(kmu,rk,rksum_trace,alph_trace,alph,st,prec=4):
    rkv = rk/(rk+rksum_trace)
    kmu = kmu + alph*rkv.flatten()*(st*np.ones(np.shape(kmu))-kmu)
    # kmu = [np.round(kmu[i],prec) for i in range(len(kmu))]
    rksum_trace = alph_trace*(rksum_trace + rk)
    return kmu, rksum_trace

# Update kernel centers (incremental update)
def update_centers_approx(kmu,rk,rksum,st,prec=4):
    rkv = rk/(rk+rksum)
    kmu = kmu + rkv.flatten()*(st*np.ones(np.shape(kmu))-kmu)
    # kmu = [np.round(kmu[i],prec) for i in range(len(kmu))]
    rksum = rksum + rk
    return kmu, rksum

# Update kernel centers (full update)
def update_centers_full(rkmat,seq):
    kmu = np.sum(rkmat*seq,axis=1)/np.sum(rkmat,axis=1)
    return kmu

#############################################################################################################################################################################################################
# Update kernel widths (incremental update for fixed centers, fixed learning rate)
def update_widths_approx_flr(ksig,rk,rksum_trace,alph_trace,alph,st,kmu,prec=4):
    rkv = rk/(rk+rksum_trace)
    ksig = ksig + alph*rkv.flatten()*((st*np.ones(np.shape(kmu))-kmu)**2-ksig)
    rksum_trace = alph_trace*(rksum_trace + rk)
    return ksig, rksum_trace

# Update kernel widths (incremental update for fixed centers)
def update_widths_approx(ksig,rk,rksum,st,kmu,prec=4):
    rkv = rk/(rk+rksum)
    ksig = ksig + rkv.flatten()*((st*np.ones(np.shape(kmu))-kmu)**2-ksig)
    rksum = rksum + rk
    return ksig, rksum

# Update kernel widths (incremental/full update)
def update_widths(t,seq,rkmat,kmu_old,prec=4):
    # np.diag(np.matmul(rkmat,(seq.reshape((-1,1))*np.ones((len(seq),len(kmu_old)))-np.ones((len(seq),len(kmu_old)))*kmu_old)))
    ksig = np.sqrt(np.diag(np.matmul(rkmat[:,:t+1],(seq[:t+1].reshape((-1,1))*np.ones((len(seq[:t+1]),len(kmu_old)))-np.ones((len(seq[:t+1]),len(kmu_old)))*kmu_old)**2))/np.sum(rkmat[:,:t+1],axis=1))   
    return ksig

#############################################################################################################################################################################################################
# Define kernels
def k_gauss(s, mu, sig): 
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-1/2*((s-mu)/(sig))**2)

def k_triangle(x,loc,scale): 
    return 1/scale*(1-np.abs(x-loc)/scale)*np.heaviside(1-np.abs(x-loc)/scale,0)

def k_triangle2(x,loc,scale):  # here: loc is the maximum value, scale is the steepness of the triangle
    return scale*x*np.heaviside(x,0)*np.heaviside(loc/scale-x,1) + loc*np.heaviside(x-loc/scale,0)

def k_triangle_eps(x,loc,scale): 
    return 1/scale*(1-np.abs(x-loc)/scale)*np.heaviside(1-np.abs(x-loc)/scale,0)

def k_box(x,loc,scale): 
    return 1/(2*scale)*(np.heaviside(scale+loc-x,1)-np.heaviside(loc-scale-x,1))

# def k_sigmoid(x,scale,shift):
#     return 1/(1+np.exp(-scale*(x-shift)))

def k_sin(x,loc,scale,eps=0):
    kk = None
    if scale>=loc:
        raise ValueError('Scale parameter for k_sigmoid must be smaller than loc parameter value.')
    else:
        if scale<0:
            raise Warning('Scale parameter for k_sigmoid should be positive. Negative values of the scale parameter may lead to unexpected model behavior.')
        a = np.pi/(2*(loc-scale))   # alpha = pi/(2*x0), x0 = 1-ksig
        c = (1-2*eps)/(2-(loc-scale)) # c = (1-2*eps)/(2-x0), x0 = 1-ksig
        kk = eps + c*np.sin(a*x)*np.heaviside(x,eps)*np.heaviside((loc-scale)-x,c) + c*np.heaviside(x-(loc-scale),c)
    return kk

def k_sin_eps(x,loc,scale,eps=0.1):
    return k_sin(x,loc,scale,eps=eps)

def k_sigmoid(x,loc,scale,eps=0):
    kk = None
    if scale>=loc:
        raise ValueError('Scale parameter for k_sigmoid must be smaller than loc parameter value.')
    else:
        if scale<0:
            raise Warning('Scale parameter for k_sigmoid should be positive. Negative values of the scale parameter may lead to unexpected model behavior.')
        a = np.pi/(loc-scale)   # alpha = pi/x0, x0 = 1-ksig
        b = - (loc-scale)/2     # beta = -x0/2, x0 = 1-ksig
        c = (2-4*eps)/(2-(loc-scale)) # c = (2-4*eps)/(2-x0), x0 = 1-ksig
        kk = eps + (c/2*np.sin(a*(x+b)) + c/2)*np.heaviside(x,eps)*np.heaviside((loc-scale)-x,c) + c*np.heaviside(x-(loc-scale),c)
    return kk

def k_sigmoid_eps(x,loc,scale,eps=0.1):
    return k_sigmoid(x,loc,scale,eps=eps)

def fusebounds(f,smin,smax,x,loc,scale): 
    # return (f(x,loc,scale)+f(x,loc-smax,scale)+f(x,smax+loc,scale))*np.heaviside(smax-x,1)*np.heaviside(x-smin,1)
    return (f(x,loc,scale)+f(x,smin-np.abs(smax-loc),scale)+f(x,smax+np.abs(loc-smin),scale))*np.heaviside(smax-x,1)*np.heaviside(x-smin,1)


#############################################################################################################################################################################################################
# Plot kernels
def plot_kernels(ax,knum,k,kmu,ksig,smin,smax,alphs,splot,title=''):
    ax.axhline(y=0,xmin=smin,xmax=smax,c='k')
    for i in range(knum):
        ax.axvline(x=kmu[i],c='k',alpha=alphs[i])
        ki = k(splot,kmu[i],ksig[i])
        ax.plot(splot,ki,c='k',alpha=alphs[i])
    ax.set_xticks(list(kmu)+[smax])
    ax.set_title(title)

# Plot familiarity
def plot_fam(ax,pk,knum,splot,kk=[]):
    if len(kk)>0:
        for i in range(knum):
            ax.plot(splot,kk[i,:],'--',c='orange')
            ax.fill_between(splot,kk[i,:],color='orange',alpha=0.4)
    ax.plot(splot,pk,'-',c='orange',lw=2)
    ax.set_xlabel('State space')
    ax.set_ylabel('Familiarity')