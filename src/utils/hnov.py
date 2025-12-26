import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib import cm
import itertools
import os
import sys
import utils.tree_env as tree
import utils.saveload as sl

# Function to assign Euclidean coordinates to every state in the binary tree env
def assign_coor_eucl(mytree):
    tr = mytree.copy()
    tr['x_eucl']=np.zeros(np.shape(tr.state.values))
    tr['y_eucl']=np.zeros(np.shape(tr.state.values))

    for l in np.unique(tr.level.values):
        if l==-1:
            tr.loc[tr.level==l,'x_eucl']=-1
            tr.loc[tr.level==l,'y_eucl']=0
        elif l==0:
            tr.loc[tr.level==l,'x_eucl']=0
            tr.loc[tr.level==l,'y_eucl']=0
        elif l%2==0: # even levels
            s = tr[tr.level==l].state.values
            sp_x = tr[tr.level==(l-1)].x_eucl.values
            sp_y = tr[tr.level==(l-1)].y_eucl.values
            for i in range(len(sp_x)):
                tr.loc[tr.state==s[2*i],'x_eucl']=sp_x[i]-1/(2**(l/2))
                tr.loc[tr.state==s[2*i],'y_eucl']=sp_y[i]
                tr.loc[tr.state==s[2*i+1],'x_eucl']=sp_x[i]+1/(2**(l/2))
                tr.loc[tr.state==s[2*i+1],'y_eucl']=sp_y[i]
        else: # uneven levels
            s = tr[tr.level==l].state.values
            sp_x = tr[tr.level==(l-1)].x_eucl.values
            sp_y = tr[tr.level==(l-1)].y_eucl.values
            for i in range(len(sp_x)):
                tr.loc[tr.state==s[2*i],'x_eucl']=sp_x[i]
                tr.loc[tr.state==s[2*i],'y_eucl']=sp_y[i]+1/(2**((l+1)/2))
                tr.loc[tr.state==s[2*i+1],'x_eucl']=sp_x[i]
                tr.loc[tr.state==s[2*i+1],'y_eucl']=sp_y[i]-1/(2**((l+1)/2))
    
    mp = tr[['state','x_eucl','y_eucl']]
    mp_mat = mp.to_numpy()
    return tr, mp, mp_mat


# Gets state IDs for all coordinate pairs (x,y) in list_coor
def get_s_from_coor(list_coor,tr):
    s = []
    for i in len(list_coor):
        s.append(tr.loc[[tr.x_eucl==list_coor[i,0] & tr.y_eucl==list_coor[i,1]],'state'].values)
    return s


# Function to assign action vector to every state in the binary tree env
def assign_av(mytree):
    tr = mytree.copy()
    d = len(np.unique(tr.level.values))
    tr['av_0']=np.ones(np.shape(tr.state.values))
    tr['av_0'][0]=-1
    for i in range(1,d-1):
        tr[f'av_{i}']=np.zeros(np.shape(tr.state.values))

    for l in np.unique(tr.level.values):
        if l>0: 
            s = tr[tr.level==l].state.values #all states in level l
            for j in range(1,l): 
                sp_j = tr.loc[tr.level==(l-1),f'av_{j}'].values # get components 1,..,l-1 from parent states
                for i in range(len(sp_j)):
                    tr.loc[tr.state==s[2*i],f'av_{j}']   = sp_j[i]
                    tr.loc[tr.state==s[2*i+1],f'av_{j}'] = sp_j[i]
            for i in range(len(tr[tr.level==l-1].state.values)):
                tr.loc[tr.state==s[2*i],f'av_{l}']   =-1
                tr.loc[tr.state==s[2*i+1],f'av_{l}'] =1
    
    mp_cols1 = ['state']
    mp_cols2 = [f'av_{i}' for i in range(d-1)]
    mp_cols1.extend(mp_cols2)

    mp = tr[mp_cols1]

    mp_matrix = tr[mp_cols2]
    mp_matrix = mp_matrix.to_numpy()

    return tr, mp, mp_matrix

# Make hierarchy with arbitrary number of levels
def make_hierarchy(rtree,levels,join_levels=True,filter_duplicates=False,hnov_type=None,update_type=None,notrace=False,center=False,center_type='box',placefields=False,comp_norm=True,maze_norm=False,eps1=True,uniform_initial=True):                       # levels is list of tree levels whose nodes are centers, e.g. [0,2,4,6], center_type='box' or 'triangle'

    ## Create maze environment ########################################################
    x, P, R, T = tree.tree_df2list(rtree)  
    tr, mp, mp_matrix = assign_av(rtree)                # action vector representation

    ## Assign relative weights of different levels of the hierarchy ###################
    if join_levels:
        w = [1]                                        # weights balancing novelties on different levels of the hierarchy; (list of len(h))
    else:
        w = list(1/len(levels)*np.ones(len(levels)))   # weights balancing novelties on different levels of the hierarchy; (list of len(h))
    
    ## Adjust format of input parameters to multi-level ################################
    comp_spec = dict(zip(['notrace','center','center_type','placefields'],[notrace,center,center_type,placefields])) # dictionary of component specifications
    for k_cs, v_cs in comp_spec.items():
        if not isinstance(v_cs, list):
            comp_spec[k_cs] = [v_cs]*len(levels) # make list of length levels if single value provided

    ## Make components for each level ##################################################
    df_level        = [] # component information for each level
    h_kc_all        = [] # component centers for each level
    h_ki_all        = [] # component widths for each level
    k_list          = [] # component activation matrix for each level
    w_factors_list  = [] # normalization factors for each level
    for ll in range(len(levels)):

        ## Initialize kernel weights, centers and widths (if applicable) ##############
        if comp_spec['placefields'][ll]:  

            h_kc_s = rtree['state'].values       # Centers for width-1 components (all states in the maze, 'small' place fields)
            h_ki_s = [1]*len(h_kc_s)             # Widths for width-1 components

            h_kc_k = rtree.loc[rtree.level==levels[ll],'state'].values   # Centers for width-2 components (only on given level, 'large' place fields)
            h_ki_k = [levels[ll]+1]*len(h_kc_k)                          # Widths for width-2 components
            h_kc_kminus = rtree.loc[rtree.level==levels[ll]+2,'state'].values   # Center for components to subtract (for computation of width-2 components)
            h_ki_kminus = [levels[ll]+3]*len(h_kc_kminus)                            # Widths for components to subtract (for computation of width-2 components)

            h_kc = np.append(h_kc_k,h_kc_s)        # Kernel centers for current level of the hierarchy
            h_ki = np.append(h_ki_k,h_ki_s)        # Kernel widths for current level of the hierarchy
            h_kc_all.append(h_kc)
            h_ki_all.append(h_ki)

            df_level_i = pd.DataFrame({'center': h_kc, \
                                        'width': h_ki, \
                                        'level_id': [ll]*len(h_kc), \
                                        'level': [levels[ll]]*len(h_kc), \
                                        'type': ['placefields']*len(h_kc), \
                                        'small_comp': [False]*len(h_kc_k) + [True]*len(h_kc_s), \
                                        'large_comp': [True]*len(h_kc_k) + [False]*len(h_kc_s)})
            df_level.append(df_level_i)

        elif comp_spec['notrace'][ll]:

            h_kc_k = rtree.loc[rtree.level==levels[ll],'state'].values              # Leaf kernel centers
            h_ki_k = [levels[ll]+1]*len(h_kc_k)                                     # Leaf kernel widths

            if comp_spec['center'][ll]:
                h_kc_s_all = rtree.loc[(rtree.level>=0) & (rtree.level<levels[ll]),'state'].values      # All states outside of the leaf kernels
                h_kc_s     = np.append(np.array([0]),rtree.loc[rtree.level==0,'state'].values)          # Center kernel center
                h_ki_s     = [1] + [levels[ll]]*(len(h_kc_s)-1)                                         # Center kernel width

            else:
                h_kc_s = rtree.loc[rtree.level<levels[ll],'state'].values            # All states outside of the leaf kernels = separate kernels
                h_ki_s = [1]*len(h_kc_s)                                             # Kernel widths = 1 for all

            h_kc = np.append(h_kc_k,h_kc_s)        # Kernel centers for current level of the hierarchy
            h_ki = np.append(h_ki_k,h_ki_s)        # Kernel widths for current level of the hierarchy
            h_kc_all.append(h_kc)
            h_ki_all.append(h_ki)

            df_level_i = pd.DataFrame({'center': h_kc, \
                                        'width': h_ki, \
                                        'level_id': [ll]*len(h_kc), \
                                        'level': [levels[ll]]*len(h_kc), \
                                        'type': ['box']*len(h_kc), \
                                        'leaf_comp': [comp_j in h_kc_k for comp_j in h_kc], \
                                        'center_comp': [comp_j in h_kc_s for comp_j in h_kc]})
            df_level.append(df_level_i)

        else:
            if comp_spec['center'][ll]:
                h_kc_s_all = rtree.loc[(rtree.level>=0) & (rtree.level<levels[ll]),'state'].values            # All states outside of the leaf kernels
                h_kc       = np.append(np.array([1,0]),rtree.loc[rtree.level==levels[ll],'state'].values)     # kernel centers for current level of the hierarchy 
                h_ki       = [[levels[ll],1]+[levels[ll]+1]*(len(h_kc)-2)]
            else:
                h_kc   = np.append(np.array([0]),rtree.loc[rtree.level==levels[ll],'state'].values)           # kernel centers for current level of the hierarchy
                h_ki   = [1]+[levels[ll]+1]*(len(h_kc)-1)                                                     # kernel widths for current level of the hierarchy
                h_kc_all.append(h_kc)
                h_ki_all.append(h_ki)

            df_level_i = pd.DataFrame({'center': h_kc, \
                                    'width': h_ki, \
                                    'level_id': [ll]*len(h_kc), \
                                    'level': [levels[ll]]*len(h_kc), \
                                    'type': ['triangle']*len(h_kc), \
                                    'leaf_comp': [True]*len(h_kc), \
                                    'center_comp': [True]*len(h_kc)})

            if comp_spec['center'][ll]:
                df_level_i.loc[df_level_i.center.isin(h_kc_s_all),'leaf_comp'] = False

            df_level.append(df_level_i)
        
        ## Create kernel matrices for each level of the hierarchy #####################
        k = np.zeros((len(x),len(h_kc))) # Initialize kernel matrix for level i, shape (|S|, number of kernels on level i)

        if comp_spec['placefields'][ll]:
            # Make 'large' placefield components
            if len(h_kc_kminus)==0:
                kminus_all = np.ones((len(x),), dtype=bool) # if no smaller placefields exist, do not subtract anything
            else:
                kminus = []
                for j in range(len((h_kc_kminus))):
                    kmin = 1/(2**(h_ki_kminus[j]))*np.prod(np.abs(mp_matrix[:,:h_ki_kminus[j]]+mp_matrix[h_kc_kminus[j],:h_ki_kminus[j]]),axis=1)
                    kminus.append(kmin<1)
                kminus_all = np.prod(np.array(kminus),axis=0) # shape (|S|), True for states inside any of the smaller placefields to be subtracted

            # Initialize kernel matrix for level i, shape (|S|, number of kernels on level i)
            for j in range(len((h_kc_k))):
                kj1 = 1/(2**(h_ki_k[j]))*np.prod(np.abs(mp_matrix[:,:h_ki_k[j]]+mp_matrix[h_kc_k[j],:h_ki_k[j]]),axis=1) # shape (|S|) -- activations of a single component
                k[:,j] = (kj1>=1) & (kminus_all)  # subtract outer part of box component
            
            # Make 'small' placefield components
            for j in range(len((h_kc_s))):
                k[:,len(h_kc_k)+j] = [1 if x==h_kc_s[j] else 0 for x in x]

        elif comp_spec['notrace'][ll]:
            # Make leaf kernels (without trace, i.e. 'box' kernels)
            for j in range(len((h_kc_k))):
                kj = 1/(2**h_ki_k[j])*np.prod(np.abs(mp_matrix[:,:h_ki_k[j]]+mp_matrix[h_kc_k[j],:h_ki_k[j]]),axis=1) # shape (|S|) -- activations of a single component
                k[:,j] = kj>=1
            # Test: [np.where(k[:,ii]!=0) for ii in range(len(h_kc_k[i]))]

            if comp_spec['center'][ll]:
                # Make home cage kernel 
                k[:,len(h_kc_k)] = [1 if x==0 else 0 for x in x]
                if comp_spec['center_type'][ll]=='triangle':
                    # Make triangle kernels for states outside of the leaf kernels
                    k[:,len(h_kc_k)+1] = 1/2**(np.sum(np.abs(mp_matrix[:]),axis=1)-1) * np.isin(x,h_kc_s_all) 
                else: 
                    # Make box kernels for states outside of the leaf kernels
                    k[:,len(h_kc_k)+1] = np.isin(x,h_kc_s_all) 

            else:
                # Make separate kernels for each state outside of the leaf kernels
                for j in range(len(h_kc_s)):
                    k[:,len(h_kc_k)+j] = [1 if x==h_kc_s[j] else 0 for x in x]

        else:
            if comp_spec['center'][ll]:
                # Make leaf kernels (with trace, i.e. 'triangle' kernels)
                for j in range(1,len(h_kc)):
                    k[:,j] = 1/(2**h_ki[j])*np.prod(np.abs(mp_matrix[:,:h_ki[j]]+mp_matrix[h_kc[j],:h_ki[j]]),axis=1)
                if comp_spec['center_type'][ll]=='triangle':
                    # Make triangle kernels for states outside of the leaf kernels
                    k[:,0] = 1/2**(np.sum(np.abs(mp_matrix[:]),axis=1)-1) * np.isin(x,h_kc_s_all)
                else: 
                    # Make box kernels for states outside of the leaf kernels
                    k[:,0] = np.isin(x,h_kc_s_all) 

            else:
                # Make leaf kernels (with trace, i.e. 'triangle' kernels)
                for j in range(len(h_kc)):
                    k[:,j] = 1/(2**h_ki[j])*np.prod(np.abs(mp_matrix[:,:h_ki[j]]+mp_matrix[h_kc[j],:h_ki[j]]),axis=1)

        # Normalize kernels, so they sum to 1 across all states
        if comp_norm:
            w_factors = np.nansum(k,axis=0,keepdims=True)
            k = k/w_factors
            #print(f"Kernel normalization check: sums of kernels across all states on level {i} are {[np.round(np.sum(k,axis=0),4)]}.\n") 
        else: 
            w_factors = np.ones((1,k.shape[1]))

        if maze_norm:
            k = k/np.nanmean(k)
        
        w_factors_list.append(w_factors)
        k_list.append(k)

    ## Join components levels if specified ############################################ 
    if join_levels:

        # Join lists of kernel centers and widths 
        h_kc = [list(itertools.chain.from_iterable(h_kc_all))] # join all kernel centers of all levels
        h_ki = [list(itertools.chain.from_iterable(h_ki_all))] # join all kernel widths of all levels                                

        # Join dataframes with kernel info of all levels
        df_level = pd.concat(df_level, ignore_index=True)

        # Join kernel matrices of all levels
        k_list         = [np.concatenate(k_list, axis=1)]           # shape (|S|, total number of kernels on all levels)
        w_factors_list = [np.concatenate(w_factors_list, axis=1)]   # shape (1, total number of kernels on all levels)

        # Identify duplicate kernels (same center and width) that appear on different levels of the hierarchy
        valc      = df_level[['center','width']].value_counts()
        valc.name = 'duplicates'
        df_level  = df_level.merge(valc, on=['center','width'], how='left')

        if filter_duplicates:
            keep_idx = np.array(df_level.drop_duplicates(subset=['center','width'], keep='first').index.to_list())
            df_level['keep'] = np.isin(np.arange(len(df_level)),keep_idx)
            h_kc = [[h_kc_cc for idx_cc, h_kc_cc in enumerate(h_kc[0]) if idx_cc in keep_idx]]
            h_ki = [[h_ki_cc for idx_cc, h_ki_cc in enumerate(h_ki[0]) if idx_cc in keep_idx]]
            w_factors_list = [w_factors_list[0][:,keep_idx]]
            k_list = [k_list[0][:,keep_idx]]

        # Recompute weight normalization factors
        sum_per_state = np.sum(k_list[0], axis=1)                   # sum of components per state
        norm_per_state = 1/sum_per_state                            # normalization per state
        w_factors = []
        for j in range(k_list[0].shape[1]):
            states_j = k_list[0][:,j]!=0
            norm_j = norm_per_state[states_j]
            if len(np.unique(norm_j))>1:
                print(f'Warning: component {j} has variable normalization across states: {norm_j}\n')
                w_factor_j = np.nanmean(norm_j)
            else:
                w_factor_j = np.unique(norm_j)[0]
            w_factors.append(w_factor_j)
        w_factors_list = [np.array(w_factors).reshape((1,-1))]
        
        df_level = [df_level]

    ## Set component weights and epsilons #############################################
    if uniform_initial:
        h_w = [(w_factors_list[i]/np.sum(w_factors_list[i])).flatten() for i in range(len(w_factors_list))] # initial component weights are given as w_j^(0) = integral k_j(s) ds / sum_l integral k_l(s) ds
        eps = [len(h_w[i])*h_w[i] for i in range(len(w_factors_list))] # epsilon needs to scale N*w0 the same (for leaky novelty!)
    else:
        h_w = [1/len(h_kc[i])*np.ones(len(h_kc[i])) for i in range(len(w))]  # initial kernel mixture weights for each level of the hierarchy (list of arrays of len(h))
        eps = [len(h_w[i])*h_w[i] for i in range(len(h_w))]
        # if eps1:
        #     eps = [1]*len(h_w)  
        # else: 
        #     eps = h_w.copy()                                                         
        # else:
        #     eps = [1/(len(h_w[i])**2) for i in range(len(h_w))]  
        
    ## Combine all return information into dictionary #################################
    h={'h_k':None,'h_ki':h_ki,'h_kc':h_kc,'h_w':h_w,'mp':mp_matrix,'kmat':k_list,'eps':eps,'k_alph':0.1}

    if hnov_type: h['hnov_type']=hnov_type
    if update_type: h['update_type']=update_type
    
    return w, h, df_level

# Plot maze
def plot_maze(mytree,fig=None,ax=None,plot_walls=True,plot_state=True,plot_reward=True,plot_actions=False,partition=[]):
    # Get Euclidean coordinates for binary tree
    mytree1, mp, mp_mat = assign_coor_eucl(mytree)

    # Define square boxes around tree nodes ('node boxes')
    wx = min(np.abs(mp.loc[mp.x_eucl>0,'x_eucl']))
    wy = min(np.abs(mp.loc[mp.y_eucl>0,'y_eucl']))
    w = min(wx,wy)/2 # width of the nodes boxes
    eps = w/3
    nb = [np.array([mp_mat[i,1]-w,mp_mat[i,2]-w,2*w,2*w]) for i in range(len(mp_mat))] 
    mytree1['node_box'] = nb

    # Define walls and boxes covering the paths between nodes ('path boxes')
    c = (1-w) 
    walls = [[[-c,c],[-c,-c]],[[c,c],[-c,c]],[[-c,c],[c,c]],[[-c,-c],[-c,-w]],[[-c,-c],[w,c]]] # initialized with outer walls: format [[x1,x2],[y1,y2]]
    pb_list = [[np.NaN,np.NaN,np.NaN,np.NaN]]
    arr_list = [[] for i in range(len(mytree1))]
    for s in mytree1.loc[mytree1.level>=0,'state'].values:
        p = mytree1.loc[mytree1.state==s,'parent'].values[0]      # parent state
        pnb = mytree1.loc[mytree1.state==p,'node_box'].values[0]  # parent box
        cnb = mytree1.loc[mytree1.state==s,'node_box'].values[0]  # child box
        cx = mytree1.loc[mytree1.state==s,'x_eucl'].values[0]     # child x
        cy = mytree1.loc[mytree1.state==s,'y_eucl'].values[0]     # child y
        px = mytree1.loc[mytree1.state==p,'x_eucl'].values[0]     # parent x
        py = mytree1.loc[mytree1.state==p,'y_eucl'].values[0]     # parent y
        if cx-px>0 and cy==py: # rightward path
            walls.append([[cnb[0]+cnb[2],cnb[0]+cnb[2]],[cnb[1],cnb[1]+cnb[3]]]) # add right border of child box to walls
            pb = [pnb[0]+pnb[2],pnb[1],cnb[0]-pnb[0]-pnb[2],pnb[3]]      # path box between parent and child (s). ll corner, width : rl corner (pnb), ll corner (cnb)-rl corner (pnb), w (pnb)
            pb_list.append(pb)
            if plot_actions:
                # Rightward path arrows
                x=pb[0]-eps; y=pb[1]+pb[3]/3; dx=pb[2]+2*eps; dy=0               # forward arrow: 0->1
                arr_list[p].append([x,y,dx,dy])
                x=pb[0]+pb[2]+eps; y=pb[1]+2*pb[3]/3; dx=-(pb[2]+2*eps); dy=0    # backward arrow:0<-1
                arr_list[s].append([x,y,dx,dy])
            #MAKE ARROWS:
            #Don't need to check actions, just add two arrows.
            #Save arrows in list of lists: arr_list of length |S| that has list of action arrows for each state.
            #s=1,p=0: Make 0->1 (append to arr_list[0]=arr_list[p]), 0<-1 (append to arr_list[1]=arr_list[s])
            #s=2,p=1: Make 1->2 (append to arr_list[1]=arr_list[p]), 1<-2 (append to arr_list[2]=arr_list[s])
            #...
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[pnb[0]+pnb[2],cnb[0]],[pnb[1],cnb[1]]])               # add lower wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add upper wall of path
            else:
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1],cnb[1]]])               # add lower wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add upper wall of path
        elif cx-px<0 and cy==py: # leftward path (just cnb and pnb exchanged)
            walls.append([[cnb[0],cnb[0]],[cnb[1],cnb[1]+cnb[3]]])        # add left border of child box to walls
            pb = [cnb[0]+cnb[2],cnb[1],pnb[0]-cnb[0]-cnb[2],cnb[3]]      # path box ll corner, width : rl corner (cnb), ll corner (pnb)-rl corner (cnb), w (cnb)
            pb_list.append(pb)
            if plot_actions:
                # Leftward path arrows [y-coordinates of forward/backward arrow changed]:
                x=pb[0]+pb[2]+eps; y=pb[1]+pb[3]/3; dx=-(pb[2]+2*eps); dy=0            # forward arrow: 0->1
                arr_list[p].append([x,y,dx,dy])
                x=pb[0]-eps; y=pb[1]+2*pb[3]/3; dx=pb[2]+2*eps; dy=0     # backward arrow: 0<-1
                arr_list[s].append([x,y,dx,dy])
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[cnb[0]+cnb[2],pnb[0]],[cnb[1],pnb[1]]])               # add lower wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]],[cnb[1]+cnb[3],pnb[1]+pnb[3]]]) # add upper wall of path
            else:
                walls.append([[cnb[0],pnb[0]],[pnb[1],pnb[1]]])               # add lower wall of path
                walls.append([[cnb[0],pnb[0]],[cnb[1]+cnb[3],pnb[1]+pnb[3]]]) # add upper wall of path
        elif cx==px and cy-py>0: # upward path
            walls.append([[cnb[0],cnb[0]+cnb[2]],[cnb[1]+cnb[3],cnb[1]+cnb[3]]]) # add upper border of child box to walls
            pb = [pnb[0],pnb[1]+pnb[3],pnb[2],cnb[1]-pnb[1]-pnb[3]]             # path box ll corner, width : lu corner (pnb), ll corner (cnb)-lu corner (pnb), w (pnb)
            pb_list.append(pb)
            if plot_actions:
                # Upward path arrows:
                x=pb[0]+2*pb[2]/3; y=pb[1]-eps; dx=0; dy=pb[3]+2*eps            # forward arrow: 0->1
                arr_list[p].append([x,y,dx,dy])
                x=pb[0]+pb[2]/3; y=pb[1]+pb[3]+eps; dx=0; dy=-(pb[3]+2*eps)     # backward arrow :0<-1
                arr_list[s].append([x,y,dx,dy])
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[pnb[0],cnb[0]],[pnb[1]+pnb[3],cnb[1]]])               # add left wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]]]) # add right wall of path
            else:
                walls.append([[pnb[0],cnb[0]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]])               # add left wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add right wall of path
        elif cx==px and cy-py<0: # downward path
            walls.append([[cnb[0],cnb[0]+cnb[2]],[cnb[1],cnb[1]]])               # add lower border of child box to walls
            pb = [cnb[0],cnb[1]+cnb[3],cnb[2],pnb[1]-cnb[1]-cnb[3]]             # path box ll corner, width : lu corner (cnb), ll corner (pnb)-lu corner (cnb), w (cnb)
            pb_list.append(pb)
            if plot_actions:
                # Downward path arrows [x-coordinates of forward/backward arrow changed]:
                x=pb[0]+2*pb[2]/3; y=pb[1]+pb[3]+eps; dx=0; dy=-(pb[3]+2*eps)              # forward arrow: 0->1
                arr_list[p].append([x,y,dx,dy])
                x=pb[0]+pb[2]/3; y=pb[1]-eps; dx=0; dy=pb[3]+2*eps            # backward arrow: 0<-1
                arr_list[s].append([x,y,dx,dy])
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[cnb[0],pnb[0]],[cnb[1]+cnb[3],pnb[1]]])               # add left wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]+pnb[2]],[cnb[1]+cnb[3],pnb[1]]]) # add right wall of path
            else:
                walls.append([[cnb[0],cnb[0]],[pnb[1],pnb[1]]])               # add left wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]+pnb[2]],[cnb[1],pnb[1]]]) # add right wall of path
        else: 
            print('Potential problem with Euclidean coordinate assignment. Maze might not be plotted correctly.\n') 
    mytree1['path_box']=pb_list

    # Define partitions (if applicable)
    if len(partition)>0:
        walls_partition = []
        cols_partition = []
        peps = w-0.01
        cmap = plt.cm.get_cmap('plasma')
        cnorm = colors.Normalize(vmin=0, vmax=len(partition[0])-1)
        smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        for i in range(len(partition[0])):
            #s = (partition[:,i]==1).nonzero()[0]
            s = (partition[:,i]==np.max(partition[:,i])).nonzero()[0]
            s_xy = np.take(mp_mat,s,0)
            m_xy = [min(s_xy[:,1])-peps,min(s_xy[:,2])-peps,max(s_xy[:,1])+peps,max(s_xy[:,2])+peps] # min x, min y, max x, max y
            walls_partition.append([[m_xy[0],m_xy[0]],[m_xy[1],m_xy[3]]]) # left wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[2],m_xy[2]],[m_xy[1],m_xy[3]]]) # right wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[0],m_xy[2]],[m_xy[1],m_xy[1]]]) # lower wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[0],m_xy[2]],[m_xy[3],m_xy[3]]]) # upper wall
            cols_partition.append(smap.to_rgba(i))

    ## Basic plot (only the maze)
    if (not fig) and (not ax):
        fig,ax = plt.subplots(figsize=(20,10))
    eps = 0.2
    ax.set_ylim([-(1+eps),1+eps])
    ax.set_xlim([-(1+eps),1+eps])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    bg = ax.add_patch(patches.Rectangle([-(1-w),-(1-w)],2*(1-w),2*(1-w),color='grey')) # grey background patch
    
    # Plot node and path boxes (white)
    s_all = mytree1.state.values
    nb_all = mytree1.node_box.values
    pb_all = mytree1.path_box.values
    patches_nb = []
    patches_pb = []
    for i in range(len(s_all)):
        nb = nb_all[i]
        pb = pb_all[i]
        a1 = ax.add_patch(patches.Rectangle([nb[0],nb[1]],nb[2],nb[3],color='white'))
        a2 = ax.add_patch(patches.Rectangle([pb[0],pb[1]],pb[2],pb[3],color='white'))
        patches_nb.append(a1)
        patches_pb.append(a2)
        
    # Plot walls
    if plot_walls:
        for w in walls:
            ax.plot(w[0],w[1],'k-')
        
    # Plot node numbers
    if plot_state:
        x_all = mytree1.x_eucl.values
        y_all = mytree1.y_eucl.values
        for i in range(len(s_all)): 
            ax.annotate(s_all[i],xy=(x_all[i],y_all[i]),horizontalalignment='center',verticalalignment='center') 
    # plt.show()

    # Plot action arrows
    if plot_actions:
        patches_arr = [[] for i in range(len(arr_list))]
        for i in range(len(s_all)):
            s_arr = arr_list[i]
            for j in range(len(s_arr)):
                a_arr = s_arr[j]
                a = ax.add_patch(patches.Arrow(a_arr[0],a_arr[1],a_arr[2],a_arr[3],width=0.05,color='k'))
                patches_arr[i].append(a)
    else:
        patches_arr=None

    # Plot partitions
    if len(partition)>0:
        for i in range(len(walls_partition)):
            ax.plot(walls_partition[i][0],walls_partition[i][1],'-',c=cols_partition[i],lw=1.5)

    # Plot reward location 
    if plot_reward:
        x_rew = mytree1.loc[mytree1.reward>0,'x_eucl'].values
        y_rew = mytree1.loc[mytree1.reward>0,'y_eucl'].values
        ax.plot(x_rew,y_rew,'Dr',markersize=4)

    return fig, ax, patches_nb, patches_pb, patches_arr


# Plot maze
def plot_maze2(mytree,fig=None,ax=None,plot_walls=True,plot_state=True,plot_reward=True,partition=[]):
    # Get Euclidean coordinates for binary tree
    mytree1, mp, mp_mat = assign_coor_eucl(mytree)

    # Define square boxes around tree nodes ('node boxes')
    wx = min(np.abs(mp.loc[mp.x_eucl>0,'x_eucl']))
    wy = min(np.abs(mp.loc[mp.y_eucl>0,'y_eucl']))
    w = min(wx,wy)/2 # width of the nodes boxes
    nb = [np.array([mp_mat[i,1]-w,mp_mat[i,2]-w,2*w,2*w]) for i in range(len(mp_mat))] 
    mytree1['node_box'] = nb

    # Define walls and boxes covering the paths between nodes ('path boxes')
    c = (1-w)
    walls = [[[-c,c],[-c,-c]],[[c,c],[-c,c]],[[-c,c],[c,c]],[[-c,-c],[-c,-w]],[[-c,-c],[w,c]]] # initialized with outer walls: format [[x1,x2],[y1,y2]]
    pb_list = [[np.NaN,np.NaN,np.NaN,np.NaN]]
    for s in mytree1.loc[mytree1.level>=0,'state'].values:
        p = mytree1.loc[mytree1.state==s,'parent'].values[0]      # parent state
        pnb = mytree1.loc[mytree1.state==p,'node_box'].values[0]  # parent box
        cnb = mytree1.loc[mytree1.state==s,'node_box'].values[0]  # child box
        cx = mytree1.loc[mytree1.state==s,'x_eucl'].values[0]     # child x
        cy = mytree1.loc[mytree1.state==s,'y_eucl'].values[0]     # child y
        px = mytree1.loc[mytree1.state==p,'x_eucl'].values[0]     # parent x
        py = mytree1.loc[mytree1.state==p,'y_eucl'].values[0]     # parent y
        if cx-px>0 and cy==py: # rightward path
            walls.append([[cnb[0]+cnb[2],cnb[0]+cnb[2]],[cnb[1],cnb[1]+cnb[3]]]) # add right border of child box to walls
            pb = [pnb[0]+pnb[2],pnb[1],cnb[0]-pnb[0]-pnb[2],pnb[3]]      # path box ll corner, width : rl corner (pnb), ll corner (cnb)-rl corner (pnb), w (pnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[pnb[0]+pnb[2],cnb[0]],[pnb[1],cnb[1]]])               # add lower wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add upper wall of path
            else:
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1],cnb[1]]])               # add lower wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add upper wall of path
        elif cx-px<0 and cy==py: # leftward path (just cnb and pnb exchanged)
            walls.append([[cnb[0],cnb[0]],[cnb[1],cnb[1]+cnb[3]]])        # add left border of child box to walls
            pb = [cnb[0]+cnb[2],cnb[1],pnb[0]-cnb[0]-cnb[2],cnb[3]]      # path box ll corner, width : rl corner (cnb), ll corner (pnb)-rl corner (cnb), w (cnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[cnb[0]+cnb[2],pnb[0]],[cnb[1],pnb[1]]])               # add lower wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]],[cnb[1]+cnb[3],pnb[1]+pnb[3]]]) # add upper wall of path
            else:
                walls.append([[cnb[0],pnb[0]],[pnb[1],pnb[1]]])               # add lower wall of path
                walls.append([[cnb[0],pnb[0]],[cnb[1]+cnb[3],pnb[1]+pnb[3]]]) # add upper wall of path
        elif cx==px and cy-py>0: # upward path (not checked yet)
            walls.append([[cnb[0],cnb[0]+cnb[2]],[cnb[1]+cnb[3],cnb[1]+cnb[3]]]) # add upper border of child box to walls
            pb = [pnb[0],pnb[1]+pnb[3],pnb[2],cnb[1]-pnb[1]-pnb[3]]             # path box ll corner, width : lu corner (pnb), ll corner (cnb)-lu corner (pnb), w (pnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[pnb[0],cnb[0]],[pnb[1]+pnb[3],cnb[1]]])               # add left wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]]]) # add right wall of path
            else:
                walls.append([[pnb[0],cnb[0]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]])               # add left wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add right wall of path
        elif cx==px and cy-py<0: # downward path
            walls.append([[cnb[0],cnb[0]+cnb[2]],[cnb[1],cnb[1]]])               # add lower border of child box to walls
            pb = [cnb[0],cnb[1]+cnb[3],cnb[2],pnb[1]-cnb[1]-cnb[3]]             # path box ll corner, width : lu corner (cnb), ll corner (pnb)-lu corner (cnb), w (cnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[cnb[0],pnb[0]],[cnb[1]+cnb[3],pnb[1]]])               # add left wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]+pnb[2]],[cnb[1]+cnb[3],pnb[1]]]) # add right wall of path
            else:
                walls.append([[cnb[0],cnb[0]],[pnb[1],pnb[1]]])               # add left wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]+pnb[2]],[cnb[1],pnb[1]]]) # add right wall of path
        else: 
            print('Potential problem with Euclidean coordinate assignment. Maze might not be plotted correctly.\n') 
    mytree1['path_box']=pb_list

    # Define partitions (if applicable)
    if len(partition)>0:
        walls_partition = []
        boxes_partition = []
        cols_partition = []
        peps = w-0.01
        cmap = plt.cm.get_cmap('plasma')
        cnorm = colors.Normalize(vmin=0, vmax=len(partition[0])-1)
        smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        for i in range(len(partition[0])):
            #s = (partition[:,i]==1).nonzero()[0]
            s = (partition[:,i]==np.max(partition[:,i])).nonzero()[0]
            s_xy = np.take(mp_mat,s,0)
            m_xy = [min(s_xy[:,1])-peps,min(s_xy[:,2])-peps,max(s_xy[:,1])+peps,max(s_xy[:,2])+peps] # min x, min y, max x, max y
            walls_partition.append([[m_xy[0],m_xy[0]],[m_xy[1],m_xy[3]]]) # left wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[2],m_xy[2]],[m_xy[1],m_xy[3]]]) # right wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[0],m_xy[2]],[m_xy[1],m_xy[1]]]) # lower wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[0],m_xy[2]],[m_xy[3],m_xy[3]]]) # upper wall
            cols_partition.append(smap.to_rgba(i))
            bpart = [m_xy[0],m_xy[1],m_xy[2]-m_xy[0],m_xy[3]-m_xy[1]]
            boxes_partition.append(bpart)

    ## Basic plot (only the maze)
    if (not fig) and (not ax):
        fig,ax = plt.subplots(figsize=(20,10))
    eps = 0.2
    ax.set_ylim([-(1+eps),1+eps])
    ax.set_xlim([-(1+eps),1+eps])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    bg = ax.add_patch(patches.Rectangle([-(1-w),-(1-w)],2*(1-w),2*(1-w),color='grey')) # grey background patch
    
    # Plot node and path boxes (white)
    s_all = mytree1.state.values
    nb_all = mytree1.node_box.values
    pb_all = mytree1.path_box.values
    patches_nb = []
    patches_pb = []
    for i in range(len(s_all)):
        nb = nb_all[i]
        pb = pb_all[i]
        a1 = ax.add_patch(patches.Rectangle([nb[0],nb[1]],nb[2],nb[3],color='white'))
        a2 = ax.add_patch(patches.Rectangle([pb[0],pb[1]],pb[2],pb[3],color='white'))
        patches_nb.append(a1)
        patches_pb.append(a2)
        
    # Plot walls
    if plot_walls:
        for w in walls:
            ax.plot(w[0],w[1],'k-')
        
    # Plot node numbers
    if plot_state:
        x_all = mytree1.x_eucl.values
        y_all = mytree1.y_eucl.values
        for i in range(len(s_all)): 
            ax.annotate(s_all[i],xy=(x_all[i],y_all[i]),horizontalalignment='center',verticalalignment='center') 
    # plt.show()

    # Plot partitions
    if len(partition)>0:
        for i in range(len(walls_partition)):
            ax.plot(walls_partition[i][0],walls_partition[i][1],'-',c=cols_partition[i],lw=1.5)
        for i in range(len(boxes_partition)):
            bp = boxes_partition[i]
            ax.add_patch(patches.Rectangle([bp[0],bp[1]],bp[2],bp[3],color=cols_partition[4*i],alpha=0.5)) 


    # Plot reward location 
    if plot_reward:
        x_rew = mytree1.loc[mytree1.reward>0,'x_eucl'].values
        y_rew = mytree1.loc[mytree1.reward>0,'y_eucl'].values
        ax.plot(x_rew,y_rew,'Dr',markersize=4)

    return fig, ax, patches_nb, patches_pb


# plot_maze3 - modification of plot_maze2 for final figures
def plot_maze3(mytree,fig=None,ax=None,plot_walls=True,plot_state=True,plot_reward=True,partition=[],plot_type='kernels',figshape=(8,8),save_plot=False,save_name=''):
    # Get Euclidean coordinates for binary tree
    mytree1, mp, mp_mat = assign_coor_eucl(mytree)

    # Define square boxes around tree nodes ('node boxes')
    wx = min(np.abs(mp.loc[mp.x_eucl>0,'x_eucl']))
    wy = min(np.abs(mp.loc[mp.y_eucl>0,'y_eucl']))
    w = min(wx,wy)/2 # width of the nodes boxes
    nb = [np.array([mp_mat[i,1]-w,mp_mat[i,2]-w,2*w,2*w]) for i in range(len(mp_mat))] 
    mytree1['node_box'] = nb

    # Define walls and boxes covering the paths between nodes ('path boxes')
    c = (1-w)
    walls = [[[-c,c],[-c,-c]],[[c,c],[-c,c]],[[-c,c],[c,c]],[[-c,-c],[-c,-w]],[[-c,-c],[w,c]]] # initialized with outer walls: format [[x1,x2],[y1,y2]]
    pb_list = [[np.NaN,np.NaN,np.NaN,np.NaN]]
    for s in mytree1.loc[mytree1.level>=0,'state'].values:
        p = mytree1.loc[mytree1.state==s,'parent'].values[0]      # parent state
        pnb = mytree1.loc[mytree1.state==p,'node_box'].values[0]  # parent box
        cnb = mytree1.loc[mytree1.state==s,'node_box'].values[0]  # child box
        cx = mytree1.loc[mytree1.state==s,'x_eucl'].values[0]     # child x
        cy = mytree1.loc[mytree1.state==s,'y_eucl'].values[0]     # child y
        px = mytree1.loc[mytree1.state==p,'x_eucl'].values[0]     # parent x
        py = mytree1.loc[mytree1.state==p,'y_eucl'].values[0]     # parent y
        if cx-px>0 and cy==py: # rightward path
            walls.append([[cnb[0]+cnb[2],cnb[0]+cnb[2]],[cnb[1],cnb[1]+cnb[3]]]) # add right border of child box to walls
            pb = [pnb[0]+pnb[2],pnb[1],cnb[0]-pnb[0]-pnb[2],pnb[3]]      # path box ll corner, width : rl corner (pnb), ll corner (cnb)-rl corner (pnb), w (pnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[pnb[0]+pnb[2],cnb[0]],[pnb[1],cnb[1]]])               # add lower wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add upper wall of path
            else:
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1],cnb[1]]])               # add lower wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add upper wall of path
        elif cx-px<0 and cy==py: # leftward path (just cnb and pnb exchanged)
            walls.append([[cnb[0],cnb[0]],[cnb[1],cnb[1]+cnb[3]]])        # add left border of child box to walls
            pb = [cnb[0]+cnb[2],cnb[1],pnb[0]-cnb[0]-cnb[2],cnb[3]]      # path box ll corner, width : rl corner (cnb), ll corner (pnb)-rl corner (cnb), w (cnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[cnb[0]+cnb[2],pnb[0]],[cnb[1],pnb[1]]])               # add lower wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]],[cnb[1]+cnb[3],pnb[1]+pnb[3]]]) # add upper wall of path
            else:
                walls.append([[cnb[0],pnb[0]],[pnb[1],pnb[1]]])               # add lower wall of path
                walls.append([[cnb[0],pnb[0]],[cnb[1]+cnb[3],pnb[1]+pnb[3]]]) # add upper wall of path
        elif cx==px and cy-py>0: # upward path (not checked yet)
            walls.append([[cnb[0],cnb[0]+cnb[2]],[cnb[1]+cnb[3],cnb[1]+cnb[3]]]) # add upper border of child box to walls
            pb = [pnb[0],pnb[1]+pnb[3],pnb[2],cnb[1]-pnb[1]-pnb[3]]             # path box ll corner, width : lu corner (pnb), ll corner (cnb)-lu corner (pnb), w (pnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[pnb[0],cnb[0]],[pnb[1]+pnb[3],cnb[1]]])               # add left wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]]]) # add right wall of path
            else:
                walls.append([[pnb[0],cnb[0]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]])               # add left wall of path
                walls.append([[pnb[0]+pnb[2],cnb[0]+cnb[2]],[pnb[1]+pnb[3],cnb[1]+cnb[3]]]) # add right wall of path
        elif cx==px and cy-py<0: # downward path
            walls.append([[cnb[0],cnb[0]+cnb[2]],[cnb[1],cnb[1]]])               # add lower border of child box to walls
            pb = [cnb[0],cnb[1]+cnb[3],cnb[2],pnb[1]-cnb[1]-cnb[3]]             # path box ll corner, width : lu corner (cnb), ll corner (pnb)-lu corner (cnb), w (cnb)
            pb_list.append(pb)
            if mytree1.loc[mytree1.state==s,'nodetype'].values[0]=='branch':
                walls.append([[cnb[0],pnb[0]],[cnb[1]+cnb[3],pnb[1]]])               # add left wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]+pnb[2]],[cnb[1]+cnb[3],pnb[1]]]) # add right wall of path
            else:
                walls.append([[cnb[0],cnb[0]],[pnb[1],pnb[1]]])               # add left wall of path
                walls.append([[cnb[0]+cnb[2],pnb[0]+pnb[2]],[cnb[1],pnb[1]]]) # add right wall of path
        else: 
            print('Potential problem with Euclidean coordinate assignment. Maze might not be plotted correctly.\n') 
    mytree1['path_box']=pb_list

    # Define partitions (if applicable)
    if len(partition)>0:
        walls_partition = []
        boxes_partition = []
        cols_partition = []
        peps = w-0.01
        cmap = plt.cm.get_cmap('plasma')
        cnorm = colors.Normalize(vmin=0, vmax=len(partition[0])-1)
        smap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        for i in range(len(partition[0])):
            #s = (partition[:,i]==1).nonzero()[0]
            s = (partition[:,i]==np.max(partition[:,i])).nonzero()[0]
            s_xy = np.take(mp_mat,s,0)
            m_xy = [min(s_xy[:,1])-peps,min(s_xy[:,2])-peps,max(s_xy[:,1])+peps,max(s_xy[:,2])+peps] # min x, min y, max x, max y
            walls_partition.append([[m_xy[0],m_xy[0]],[m_xy[1],m_xy[3]]]) # left wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[2],m_xy[2]],[m_xy[1],m_xy[3]]]) # right wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[0],m_xy[2]],[m_xy[1],m_xy[1]]]) # lower wall
            cols_partition.append(smap.to_rgba(i))
            walls_partition.append([[m_xy[0],m_xy[2]],[m_xy[3],m_xy[3]]]) # upper wall
            cols_partition.append(smap.to_rgba(i))
            bpart = [m_xy[0],m_xy[1],m_xy[2]-m_xy[0],m_xy[3]-m_xy[1]]
            boxes_partition.append(bpart)

    ## Basic plot (only the maze)
    if (not fig) and (not ax):
        fig,ax = plt.subplots(figsize=figshape)
    elif (not ax):
        ax = fig.subplots(1,1)

    eps=0.1 #eps = 0.2
    ax.set_ylim([-(1+eps),1+eps])
    ax.set_xlim([-(1+eps),1+eps])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    bg = ax.add_patch(patches.Rectangle([-(1-w),-(1-w)],2*(1-w),2*(1-w),color='grey')) # grey background patch
    
    # Plot node and path boxes (white)
    s_all = mytree1.state.values
    nb_all = mytree1.node_box.values
    pb_all = mytree1.path_box.values
    patches_nb = []
    patches_pb = []
    for i in range(len(s_all)):
        nb = nb_all[i]
        pb = pb_all[i]
        a1 = ax.add_patch(patches.Rectangle([nb[0],nb[1]],nb[2],nb[3],color='white'))
        a2 = ax.add_patch(patches.Rectangle([pb[0],pb[1]],pb[2],pb[3],color='white'))
        patches_nb.append(a1)
        patches_pb.append(a2)
        
    # Plot walls
    if plot_walls:
        for w in walls:
            ax.plot(w[0],w[1],'k-')
        
    # Plot node numbers
    if plot_state:
        x_all = mytree1.x_eucl.values
        y_all = mytree1.y_eucl.values
        if plot_type=='labyrinth':  fs = 13
        else:                       fs = 8
        for i in range(len(s_all)): 
            ax.annotate(s_all[i],xy=(x_all[i],y_all[i]),horizontalalignment='center',verticalalignment='center',fontsize=fs) 

    # Plot partitions
    if len(partition)>0:
        for i in range(len(walls_partition)):
            ax.plot(walls_partition[i][0],walls_partition[i][1],'-',c=cols_partition[i],lw=1.5)
        for i in range(len(boxes_partition)):
            bp = boxes_partition[i]
            ax.add_patch(patches.Rectangle([bp[0],bp[1]],bp[2],bp[3],color=cols_partition[4*i],alpha=0.5)) 

    # Plot reward location 
    if plot_reward:
        x_rew = mytree1.loc[mytree1.reward>0,'x_eucl'].values
        y_rew = mytree1.loc[mytree1.reward>0,'y_eucl'].values
        if plot_type=='labyrinth':  
            r = ax.plot(x_rew,y_rew,'Dr',markersize=25)
        else:                       
            ax.plot(x_rew,y_rew,'Dr',markersize=4)

    fig.tight_layout()

    # Save figure
    if save_plot:
        path_save = os.path.join(sl.get_datapath().replace('data','output'),'Figures_Paper/Fig2_maze_env/')
        sl.make_long_dir(path_save)
        plt.savefig(os.path.join(path_save,save_name+'.svg'),bbox_inches='tight')
        plt.savefig(os.path.join(path_save,save_name+'.eps'),bbox_inches='tight')

    return fig, ax, patches_nb, patches_pb

