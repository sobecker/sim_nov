import numpy as np

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.utils.tree_env as tree
import src.utils.hnov as hn
import src.models.mb_agent.mb_surnor as nor

##############################################################################
# Auxiliary functions for parameters                                         #
##############################################################################
def all_zero_x0(trials,epi):
    return np.zeros((trials,epi),dtype=int).tolist()

def seq_per_trial_x0(seq,trials):
    return np.array(seq*trials).reshape((trials,len(seq))).tolist()

def generate_seq(len_epi):
    return None

def auto_seeds(trials):
    return list(range(trials))

##############################################################################
# rAC base params (11-state environment)                                     #
##############################################################################
number_epi = 5
number_trial = 1

base_params_rAC11 = {'sim_name':'rAC11',
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':auto_seeds(number_trial),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':11,                         # number of states in the environment
                    'R':[0,0,0,0,0,0,0,0,0,0,1],    # reward in each state
                    'P':[[0, 1, 7, 8],              # transition matrix: rows = states, cols = actions; each matrix entry P_ij shows the state that results from taking action j in state i
                        [1, 2, 8, 9],
                        [2, 3, 7, 9],
                        [3, 4, 7, 8],
                        [4, 5, 8, 9],
                        [5, 6, 7, 9],
                        [6, 10,7, 8],
                        [7, 8, 9, 0],
                        [8, 7, 9, 0],
                        [9, 7, 8, 0],
                        [10,10,10,10]],
                    'a':4,   
                    'agent_types':['r'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],       # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],     # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],     # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],       # e-trace decay factor of the critic
                    'a_lam':[0.0],       # e-trace decay factor of the actor
                    'temp':[0.01]}       # temperature of the softmax decision, the higher the more probabilistic


##############################################################################
# nAC base params (11-state environment)                                     #
##############################################################################
number_epi = 1
number_trial = 1

base_params_nAC11 = {'sim_name':'nAC11',
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':auto_seeds(number_trial),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':11,                         # number of states in the environment
                    'R':[0,0,0,0,0,0,0,0,0,0,0],    # reward in each state
                    'P':[[0, 1, 7, 8],              # transition matrix: rows = states, cols = actions; each matrix entry P_ij shows the state that results from taking action j in state i
                        [1, 2, 8, 9],
                        [2, 3, 7, 9],
                        [3, 4, 7, 8],
                        [4, 5, 8, 9],
                        [5, 6, 7, 9],
                        [6, 10,7, 8],
                        [7, 8, 9, 0],
                        [8, 7, 9, 0],
                        [9, 7, 8, 0],
                        [10,10,10,10]],
                    'a':4,   
                    'agent_types':['n'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],          # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],        # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],        # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],          # e-trace decay factor of the critic
                    'a_lam':[0.0],          # e-trace decay factor of the actor
                    'temp':[0.01],          # temperature of the softmax decision, the higher the more probabilistic
                    'ntype':'N',            # type of novelty used: 'N-k','-1/N','leaky count' (to be implemented)
                    'k':0}                  #1.5

##############################################################################
# rAC base params (5-state environment)                                      #
##############################################################################
number_epi = 5
number_trial = 1

params_rAC5 = {'sim_name':'rAC5',
                'rec_type':'basic',
                'round_prec':4,
                'number_trials':number_trial,
                'number_epi':number_epi,
                'max_it':1000,
                'seeds':auto_seeds(number_trial),
                'x0':all_zero_x0(number_trial,number_epi),
                'S':5,
                'R':[0,0,0,0,1],
                'P':[[0, 1, 3, 3],
                    [1, 2, 3, 3],
                    [2, 3, 3, 4],
                    [0, 3, 3, 3],
                    [4, 4, 4, 4]],
                'a':4,   
                'agent_types':['r'],
                'decision_weights':[[1]]*number_epi,
                'tauM':[1],
                'RM':[1],
                'c_w0':[0],
                'a_w0':[0],
                'gamma':[0.9],       # discount of previous value estimate during TD error computation, should not be below 0.9
                'c_alph':[0.05],   # critic learning rate, should not be above 0.01 
                'a_alph':[0.05],   # actor learning rate, should not be above 0.01 
                'c_lam':[0.0],    # e-trace decay factor of the critic
                'a_lam':[0.0],    # e-trace decay factor of the actor
                'temp':[0.01]}      # temperature of the softmax decision, the higher the more probabilistic


##############################################################################
# rAC base params (bin-tree environment)                                     #
##############################################################################
number_epi = 5
number_trial = 1
depth = 6
seed = 1234

rtree = tree.make_rtree_env(depth,None,rnode=117)
#rtree = tree.make_rtree_env(depth,seed)
x, P, R, T = tree.tree_df2list(rtree)

base_params_rACtree = {'sim_name':'rAC-rtree',
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000, # 1000 is enough since we expect the good solutions to take significantly below 1000 steps
                    'seeds':auto_seeds(number_trial),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':len(x),                         # number of states in the environment
                    'R':R,    # reward in each state
                    'P':P,
                    'a':4,   
                    'x': x,
                    'T': R, # terminal states
                    'agent_types':['r'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],       # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],     # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],     # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],       # e-trace decay factor of the critic
                    'a_lam':[0.0],       # e-trace decay factor of the actor
                    'temp':[0.01]}       # temperature of the softmax decision, the higher the more probabilistic

##############################################################################
# nAC base params (bin-tree environment)                                     #
##############################################################################
number_epi = 1
number_trial = 1
depth = 6
seed = 1234

rtree = tree.make_rtree_env(depth,None,rnode=117)
#rtree = tree.make_rtree_env(depth,seed)
x, P, R, T = tree.tree_df2list(rtree)

base_params_nACtree = {'sim_name':'nAC-tree',
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':auto_seeds(number_trial),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':len(x),                         # number of states in the environment
                    'R':R,    # reward in each state
                    'P':P,
                    'a':4, 
                    'x':x,
                    'T':R, # terminal states
                    'agent_types':['n'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],          # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],        # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],        # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],          # e-trace decay factor of the critic
                    'a_lam':[0.0],          # e-trace decay factor of the actor
                    'temp':[0.01],          # temperature of the softmax decision, the higher the more probabilistic
                    'ntype':'N',            # type of novelty used: 'N-k','-1/N','leaky count' (to be implemented)
                    'k_alph':1,             # leakiness of counts; 1=not leaky (default)
                    'k':0}                  #1.5

##############################################################################
# mbNoR base params (11-state environment)                                   #
##############################################################################
base_params_mbnortrap_exp = {'sim_name':'mbNoR-trap',
                'rec_type':'basic',
                'number_trials':1,
                'number_epi':5,
                'max_it':1000,
                'seeds':list(range(1)),
                'x0':seq_per_trial_x0([5,8,3,4,7],1), #+[8, 6, 5, 4, 6, 2, 1, 2, 8, 8, 4, 6, 2, 4, 4, 5, 5, 5, 7, 3],12), #all_zero_x0(12,25),
                'S':11,
                'A':4,
                'R':np.array([0,0,0,0,0,0,0,0,0,0,1]), 
                'P':np.array([[0, 1, 7, 8],      # transition matrix: rows = states, cols = actions; each matrix entry P_ij shows the state that results from taking action j in state i
                                [1, 2, 8, 9],
                                [2, 3, 7, 9],
                                [3, 4, 7, 8],
                                [4, 5, 8, 9],
                                [5, 6, 7, 9],
                                [6, 10,7, 8],
                                [7, 8, 9, 0],
                                [8, 7, 9, 0],
                                [9, 7, 8, 0],
                                [10,10,10,10]]),
                } 

##############################################################################
# mbNoR base params (tree environment)                                       #
##############################################################################

number_epi = 1
number_trial = 1
depth = 6
seed = 1234

rtree = tree.make_rtree_env(depth,None,rnode=117)
#rtree = tree.make_rtree_env(depth,seed)
x, P, R, T = tree.tree_df2list(rtree)  

base_params_mbnortree_exp = {'sim_name':'mbNoR-tree',
                'rec_type':'basic',
                'number_trials':number_trial,
                'number_epi':number_epi,
                'max_it':1000,
                'seeds':list(range(number_trial)),
                'x0':all_zero_x0(number_trial,number_epi),
                'S':len(x),  
                'A':4,                       
                'R':R,    
                'P':P,
                'x': x,
                'T': R, 
                'k':0,
                'ntype':'N-k',
                'k_alph': 1 # leakiness of counts; 1=not leaky (default)
                }   


##############################################################################
# H1/2-mbNoR base params (tree environment)                                    #
##############################################################################

def baseparams_h1mbnor(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree) 
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])  

    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'H1-mbNoR-tree_l'+level_str+trace_str+center_str,
                'rec_type':'basic',
                'number_trials':number_trial,
                'number_epi':number_epi,
                'max_it':1000,
                'seeds':list(range(number_trial)),
                'x0':all_zero_x0(number_trial,number_epi),
                'S':len(x),  
                'A':4,                       
                'R':R,    
                'P':P,
                'x': x,
                'T': R, 
                'k':0,
                'ntype':'hN',
                'hnov_type':2,
                'h':h,
                'w':w
        }   

    return bp

def baseparams_h2mbnor(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree) 
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])

    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'H2-mbNoR-tree_l'+level_str+trace_str+center_str,
                'rec_type':'basic',
                'number_trials':number_trial,
                'number_epi':number_epi,
                'max_it':1000,
                'seeds':list(range(number_trial)),
                'x0':all_zero_x0(number_trial,number_epi),
                'S':len(x),  
                'A':4,                       
                'R':R,    
                'P':P,
                'x': x,
                'T': R, 
                'k':0,
                'ntype':'hN',
                'hnov_type':3,
                'h':h,
                'w':w
        }   

    return bp

##############################################################################
# H1/2-mbNoR base params with eps=1 (tree environment)                       #
##############################################################################

def baseparams_h1mbnor_eps1(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree) 
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,eps1=True,update_type=update_type)

    h['hnov_type'] = 2

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])  

    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'H1-mbNoR-tree_l'+level_str+trace_str+center_str,
                'rec_type':'basic',
                'number_trials':number_trial,
                'number_epi':number_epi,
                'max_it':1000,
                'seeds':list(range(number_trial)),
                'x0':all_zero_x0(number_trial,number_epi),
                'S':len(x),  
                'A':4,                       
                'R':R,    
                'P':P,
                'x': x,
                'T': R, 
                'k':0,
                'ntype':'hN',
                'hnov_type':2,
                'h':h,
                'w':w
        }   

    return bp

def baseparams_h2mbnor_eps1(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree) 
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,eps1=True,update_type=update_type)

    h['hnov_type'] = 2

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])
    
    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'H2-mbNoR-tree_l'+level_str+trace_str+center_str,
                'rec_type':'basic',
                'number_trials':number_trial,
                'number_epi':number_epi,
                'max_it':1000,
                'seeds':list(range(number_trial)),
                'x0':all_zero_x0(number_trial,number_epi),
                'S':len(x),  
                'A':4,                       
                'R':R,    
                'P':P,
                'x': x,
                'T': R, 
                'k':0,
                'ntype':'hN',
                'hnov_type':3,
                'h':h,
                'w':w
        }   

    return bp

##############################################################################
# H1/2-nAC base params (tree env)                                            #
##############################################################################

def baseparams_h1nac(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree)
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    h['hnov_type'] = 2

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])

    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'h1nAC-tree_l'+level_str+trace_str+center_str,
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':list(range(number_trial)),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':len(x),                         
                    'R':R,    
                    'P':P,
                    'a':4, 
                    'x': x,
                    'T': R, 
                    'agent_types':['n'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],          # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],        # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],        # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],          # e-trace decay factor of the critic
                    'a_lam':[0.0],          # e-trace decay factor of the actor
                    'temp':[0.01],          # temperature of the softmax decision, the higher the more probabilistic
                    'ntype':'hN',           # type of novelty used: 'N-k','-1/N','leaky count' (to be implemented)
                    'k':0,
                    'h':h,
                    'w':w}                 

    return bp

def baseparams_h2nac(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree)
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    h['hnov_type'] = 3

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])
    
    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str + 'h2nAC-tree_l'+level_str+trace_str+center_str,
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':list(range(number_trial)),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':len(x),                         
                    'R':R,    
                    'P':P,
                    'a':4, 
                    'x': x,
                    'T': R, 
                    'agent_types':['n'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],          # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],        # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],        # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],          # e-trace decay factor of the critic
                    'a_lam':[0.0],          # e-trace decay factor of the actor
                    'temp':[0.01],          # temperature of the softmax decision, the higher the more probabilistic
                    'ntype':'hN',           # type of novelty used: 'N-k','-1/N','leaky count' (to be implemented)
                    'k':0,
                    'h':h,
                    'w':w}                 

    return bp

##############################################################################
# H1/2-nAC base params with eps=1 (tree env)                                 #
##############################################################################

def baseparams_h1nac_eps1(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree)
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,eps1=True,update_type=update_type)

    h['hnov_type'] = 2

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])
    
    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'H1-nAC-tree_l'+level_str+trace_str+center_str,
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':list(range(number_trial)),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':len(x),                         
                    'R':R,    
                    'P':P,
                    'a':4, 
                    'x': x,
                    'T': R, 
                    'agent_types':['n'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],          # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],        # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],        # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],          # e-trace decay factor of the critic
                    'a_lam':[0.0],          # e-trace decay factor of the actor
                    'temp':[0.01],          # temperature of the softmax decision, the higher the more probabilistic
                    'ntype':'hN',           # type of novelty used: 'N-k','-1/N','leaky count' (to be implemented)
                    'k':0,
                    'h':h,
                    'w':w}                 

    return bp

def baseparams_h2nac_eps1(levels,notrace=False,center=False,center_type='box',update_type=None):
    number_epi = 1
    number_trial = 1
    depth = 6
    seed = 1234

    rtree = tree.make_rtree_env(depth,None,rnode=117)
    #rtree = tree.make_rtree_env(depth,seed)
    x, P, R, T = tree.tree_df2list(rtree)
    w,h = hn.make_hierarchy(rtree,levels,notrace=notrace,center=center,center_type=center_type,eps1=True,update_type=update_type)

    h['hnov_type'] = 3

    level_str = str(levels[0])
    for i in range(len(levels)-1):
        level_str += str(levels[i])

    trace_str = '_notrace' if notrace else ''
    center_str = f'_center-{center_type}' if center else ''
    leaky_str = 'leaky_' if update_type=='leaky' else ''

    bp = {'sim_name':leaky_str+'H2-nAC-tree_l'+level_str+trace_str+center_str,
                    'rec_type':'basic',
                    'round_prec':4,
                    'number_trials':number_trial,
                    'number_epi':number_epi,
                    'max_it':1000,
                    'seeds':list(range(number_trial)),
                    'x0':all_zero_x0(number_trial,number_epi),
                    'S':len(x),                         
                    'R':R,    
                    'P':P,
                    'a':4, 
                    'x': x,
                    'T': R, 
                    'agent_types':['n'],
                    'decision_weights':[[1]]*number_epi,
                    'tauM':[1],
                    'RM':[1],
                    'c_w0':[0],
                    'a_w0':[0],
                    'gamma':[0.9],          # discount of previous value estimate during TD error computation, should not be below 0.9
                    'c_alph':[0.05],        # critic learning rate, should not be above 0.01 
                    'a_alph':[0.05],        # actor learning rate, should not be above 0.01 
                    'c_lam':[0.0],          # e-trace decay factor of the critic
                    'a_lam':[0.0],          # e-trace decay factor of the actor
                    'temp':[0.01],          # temperature of the softmax decision, the higher the more probabilistic
                    'ntype':'hN',           # type of novelty used: 'N-k','-1/N','leaky count' (to be implemented)
                    'k':0,
                    'h':h,
                    'w':w}                 

    return bp

# Generates baseparams (sim) for 'hnac-gn' # 'hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
def get_baseparams_all_hnac(alg_type,levels,eps1=True,hnov_type=2,notrace=False,center=False,center_type='box',update_type=None):
    if eps1:
        if hnov_type==1:    bp = baseparams_h1nac_eps1(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
        elif hnov_type==2:  bp = baseparams_h2nac_eps1(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
    else:
        if hnov_type==1:    bp = baseparams_h1nac(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
        elif hnov_type==2:  bp = baseparams_h2nac(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    if 'gv' in alg_type: 
        bp['agent_types'] = ['gn']
    if 'goi' in alg_type:
        bp['ntype'] = 'hN-k'

    return bp

# Generates baseparams (sim) for 'hnac-gn' # 'hnor','hnac-gn','hnac-gn-gv','hnac-gn-goi','hnac-gn-gv-goi'
def get_baseparams_all_hnor(alg_type,levels,eps1=True,hnov_type=2,notrace=False,center=False,center_type='box',update_type=None):
    if eps1:
        if hnov_type==1:    bp = baseparams_h1mbnor_eps1(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
        elif hnov_type==2:  bp = baseparams_h2mbnor_eps1(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
    else:
        if hnov_type==1:    bp = baseparams_h1mbnor(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
        elif hnov_type==2:  bp = baseparams_h2mbnor(levels,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    return bp

# Method returns three dictionaries of hybrid parameters
def baseparams_hybrid():
    params_exp = base_params_mbnortree_exp.copy()
    params_hybrid = {'w_mf':0.5, 'w_mb':0.5,'sim_name':'hybrid_balanced'}
    params_exp.update(params_hybrid)
    params_mb  = nor.import_exploration_params_surnor()
    params_mf  = base_params_nACtree.copy()
    return params_exp,params_mb,params_mf

# Method combines three dictionaries of hybrid parameters into single parameter dictionary
def comb_params(p_exp,p_mb,p_mf):
    p_mb.update(p_exp)
    overlap = ['rec_type','k','ntype','k_alph','h','w']
    for i in range(len(overlap)):
        if overlap[i] in p_mf.keys(): p_mf[f'mf_{overlap[i]}'] = p_mf.pop(overlap[i])
        if overlap[i] in p_mb.keys(): p_mb[f'mb_{overlap[i]}'] = p_mb.pop(overlap[i])
    p_mf.update(p_mb)
    return p_mf

# Method returns single parameter dictionary
def baseparams_hybrid_comb():
    p_exp, p_mb, p_mf = baseparams_hybrid()
    p_mf = comb_params(p_exp,p_mb,p_mf)
    return p_mf

# Method returns three dictionaries with params for granular hybrid agent
def baseparams_all_hhybrid(mb_alg_type,mf_alg_type,levels,eps1=True,hnov_type=2,notrace=False,center=False,center_type='box',path_surnor='/lcncluster/becker/RL_reward_novelty/src/mbnor/',update_type=None):
    params_exp  = get_baseparams_all_hnor(mb_alg_type,levels,eps1=eps1,hnov_type=hnov_type,notrace=notrace,center=center,center_type=center_type,update_type=update_type)
    params_mb   = nor.import_exploration_params_surnor(path=path_surnor)
    params_mf   = get_baseparams_all_hnac(mf_alg_type,levels,eps1=eps1,hnov_type=hnov_type,notrace=notrace,center=center,center_type=center_type,update_type=update_type)

    params_hybrid = {'w_mf':0.5, 'w_mb':0.5,'sim_name':'g-hybrid_balanced'}
    params_exp.update(params_hybrid)

    return params_exp,params_mb,params_mf

# Method returns single dictionary with params for granular hybrid agent
def baseparams_all_hhybrid_comb(mb_alg_type,mf_alg_type,levels,eps1=True,hnov_type=2,notrace=False,center=False,center_type='box',path_surnor='/lcncluster/becker/RL_reward_novelty/src/mbnor/',update_type=None):
    p_exp, p_mb, p_mf = baseparams_all_hhybrid(mb_alg_type,mf_alg_type,levels,eps1=eps1,hnov_type=hnov_type,notrace=notrace,center=center,center_type=center_type,path_surnor=path_surnor,update_type=update_type)
    p_mf = comb_params(p_exp,p_mb,p_mf)
    return p_mf


    


