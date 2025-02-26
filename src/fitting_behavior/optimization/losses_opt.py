from skopt.utils import use_named_args

import base_params_opt as bpo
import opt_params_opt as opo

import sys
sys.path.append('/Users/sbecker/Projects/RL_reward_novelty/')
sys.path.append('/lcncluster/becker/RL_reward_novelty/')

import src.models.mf_agent.experiment as e

##############################################################################
# rAC: single-sim, no OI, loss = steps in epi 2-5                            #
##############################################################################
@use_named_args(dimensions=opo.dim_noOI)
def loss_rew_single(gamma,c_alph,a_alph,c_lam,a_lam,temp):
    # Load simulation parameters
    sim_params      = bpo.base_params_rAC11

    # Update parameters that are being optimized
    sim_params['gamma']     = [gamma]
    sim_params['c_alph']    = [c_alph]
    sim_params['a_alph']    = [a_alph]
    sim_params['c_lam']     = [c_lam]
    sim_params['a_lam']     = [a_lam]
    sim_params['temp']      = [temp]
    
    # Run simulation
    exp = e.experiment(sim_params,flag_saveData=False)
    all_data, _, _, _, _ = exp.runExperiment()

    # Compute loss: total steps in episode 2-5   
    all_data = all_data[all_data['epi']!=0]                           # filter data: only episode 2-5
    l = all_data[['it','subID','epi']].groupby(['subID']).count()     # count total steps across all episodes

    return l['it'].item()

##############################################################################
# rAC: multi-sim, no OI, loss = steps in epi 2-5                             #
##############################################################################
@use_named_args(dimensions=opo.dim_noOI)
def loss_rew_multi(gamma,c_alph,a_alph,c_lam,a_lam,temp):
    # Load simulation parameters
    sim_params      = bpo.base_params_rAC11

    # Update parameters that are being optimized
    sim_params['gamma']     = [gamma]
    sim_params['c_alph']    = [c_alph]
    sim_params['a_alph']    = [a_alph]
    sim_params['c_lam']     = [c_lam]
    sim_params['a_lam']     = [a_lam]
    sim_params['temp']      = [temp]
    
    # Run simulation
    exp = e.experiment(sim_params,flag_saveData=False)
    all_data, _, _, _, _ = exp.runExperiment()

    # Compute loss: total steps in episode 2-5   
    all_data = all_data[all_data['epi']!=0]                           # filter data: only episode 2-5
    l = all_data[['it','subID','epi']].groupby(['subID']).count()     # count total steps across all episodes

    return l['it'].item()


