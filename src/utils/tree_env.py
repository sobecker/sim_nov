import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

## Binary tree without reward
def make_tree_env(num_level):

    # Init dataframe with exit state and first level state
    env = pd.DataFrame({'state':[0,1],'actions':[[1],[2,3,0]],'level':[-1,0],'nodetype':['exit','branch'],'reward':[0,0],'terminal':[1, 0],'parent':[0,0]})

    # Add levels with branching and leaf nodes
    for l in range(1,num_level+1):
        new_states  = list(env['state'].values[-1]+1+np.arange(2**l)) # add two children nodes
        new_parents = np.array([[s,s] for s in env[env.level==l-1]['state'].values]).flatten()
        # new_actions = [[t] for t in np.array([[s,s] for s in env[env.level==l-1]['state'].values]).flatten()] 
        new_actions = [[t] for t in new_parents] # add back actions for children nodes
        if l!=num_level: # case that new nodes are not leaf nodes: add actions to children nodes
            new_actions_b = new_actions.copy()
            new_actions = [list(np.append(arr,s)) for arr,s in zip(np.split(new_states[-1]+1+np.arange(2**(l+1)),len(new_states)),new_actions_b)] 
        add = pd.DataFrame({'state':new_states,'actions':new_actions,'level':list(l*np.ones(len(new_states),dtype='int')),'nodetype':len(new_states)*['branch' if l!=num_level else 'leaf'],'reward':np.zeros(len(new_states),dtype='int'),'terminal':np.zeros(len(new_states),dtype='int'),'parent':new_parents}) # level x should have 2^x states
        env = env.append(add,ignore_index=True)    

    return env
        
## Binary tree with reward in randomly chosen leaf node
def make_rtree_env(num_level,seed=None,rnode=117):
    env = make_tree_env(num_level)

    if not rnode==None:
        if not seed==None:
            print('Both a random seed and fixed reward location provided. Environment is initialized with fixed reward location.')
        env.reward[rnode]=1
    else:
        if seed==None:
            seed = 1234
            print(f'Neither a random seed nor a fixed reward location provided. Environment is initialized with random reward location (seed: {seed}).')
        np.random.seed(seed)
        rleaf = np.random.choice((env.nodetype.values=='leaf').nonzero()[0])
        env.reward[rleaf]=1
    
    return env

## Convert tree into matrices (compatible with previous env input)
def tree_df2list(tree):
    x = tree['state'].values
    R = tree['reward'].values
    T = tree['terminal'].values
    P = []
    P.append([np.NaN, tree['actions'][0][0], np.NaN, np.NaN]) # action order: [back, forward, left, right]
    for i in range(1,len(tree['actions'].values)):
        if len(tree['actions'][i])==3:
            P.append([tree['actions'][i][2],np.NaN,tree['actions'][i][0],tree['actions'][i][1]]) # action order: [back, forward, left, right]
        elif len(tree['actions'][i])==1:
            P.append([tree['actions'][i][0], np.NaN, np.NaN, np.NaN]) # action order: [back, forward, left, right]
    return x, P, R, T

## Get depth of tree from number of states
def get_depth(x):   # x is the state vector that can be extracted from params or via tree_df2list
    i=0
    n=1
    while len(x)>n:
        n+=2**i
        i+=1
    if n==len(x): flag_fit=True
    else: flag_fit=False

    return i-1, flag_fit # i-1 is the depth of the tree; flag_fit is TRUE if the number of states fits exactly to the determined depth and FALSE otherwise

## Get path to goal in tree 
def get_goalpath(tree):
    _, P, R, _ = tree_df2list(tree)
    goal = R.nonzero()[0][0]
    goal_path = [goal]
    while goal_path[-1]>0:
        parent = [i for i in range(len(P)) if goal_path[-1] in P[i]][0]
        goal_path.append(parent)

    return goal_path[::-1]

## Get path that avoids direct goal path as much as possible
def get_nongoalpath(tree,gp):
    s = []
    sl = []
    for i in range(1,max(tree['level']+1)):
        si = tree[tree.level==i]['state'].values
        s.append(np.random.choice(list(set(si)-set(gp))))
        sl.append(i)
    return s, sl

## Create set of plot states (on/off the goal path)
def get_plotstates(tree):
    sp = []
    spl = []

    gp = get_goalpath(tree)
    gpl = [i for i in range(-1,len(gp)-1,1)]

    sp.extend(gp)
    spl.extend(gpl)
    
    ngp, ngpl = get_nongoalpath(tree,gp)
    sp.extend(ngp)
    spl.extend(ngpl)

    return sp, spl
