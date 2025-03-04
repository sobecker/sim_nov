#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:51:25 2021

@author: sbecker
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import timeit
import os
import csv
import pickle
from models.mf_agent.ac import *
import utils.saveload as sl 


class recorder():
    
    def __init__(self,params,rec_type='basic'):
        self.rec_type   = rec_type
        self.data_basic = []
        self.cols_basic = ['subID','epi','it','state','action','next_state','reward','terminal','reward active']
        
        if self.rec_type=='advanced1' or self.rec_type=='advanced2':
            for i in range(len(params['agent_types'])):
                self.cols_basic.extend([f'mod-{i}: M'])
            
            for i in range(len(params['agent_types'])):
                self.cols_basic.extend([f'mod-{i}: TD'])
            
            for i in range(len(params['agent_types'])):
                if params['agent_types'][i]=='n' and params['ntype']=='hN':
                    for j in range(len(params['w'])):
                        self.cols_basic.extend([f'mod-{i}: Mh_{j}'])

            for i in range(len(params['agent_types'])):
                if params['agent_types'][i]=='n' and params['ntype']=='hN':
                    for j in range(len(params['w'])):
                        self.cols_basic.extend([f'mod-{i}: Mhw_{j}'])

            for i in range(len(params['agent_types'])):
                if params['agent_types'][i]=='n' and params['ntype']=='hN':
                    for j in range(len(params['w'])):
                        self.cols_basic.extend([f'mod-{i}: Mhg_{j}'])
            
            for i in range(len(params['agent_types'])):
                if params['agent_types'][i]=='n':
                    self.cols_basic.extend([f'mod-{i}: nov_post'])
        
        if self.rec_type=='advanced2':
            self.cols_wc = []
            self.wc = []
            self.cols_ec = []
            self.ec = []
            self.cols_wa = []
            self.wa = []
            self.cols_ea = []
            self.ea = []

            for i in range(len(params['agent_types'])):
                if params['agent_types'][i]=='gn' and len(params['h'])>0 and 'kmat' in params['h'].keys():
                    range1 = len(params['h']['kmat'][0][0,:])
                else: 
                    range1 = params['S']
                for j in range(range1):
                    self.cols_wc.extend([f'mod-{i}: WC_{j}'])
                    
            for i in range(len(params['agent_types'])):
                if params['agent_types'][i]=='gn' and len(params['h'])>0 and 'kmat' in params['h'].keys():
                    range1 = len(params['h']['kmat'][0][0,:])
                else: 
                    range1 = params['S']
                for j in range(range1):
                    self.cols_ec.extend([f'mod-{i}: EC_{j}'])
            
            for i in range(len(params['agent_types'])):
                range1 = params['S']
                for j in range(range1):    # rows of the weight matrix
                    # for k in range(np.sum(~np.isnan(params['P'][j]))):
                    # for k in range(params['a']):    #columns of the weight matrix
                    for k in range(len(params['P'][j])):
                        if not np.isnan(params['P'][j][k]):
                            self.cols_wa.extend([f'mod-{i}: WA_{j}-{k}'])
                        
            for i in range(len(params['agent_types'])):
                range1 = params['S']
                for j in range(range1):
                    #for k in range(np.sum(~np.isnan(params['P'][j]))):
                    #for k in range(params['a']):
                    for k in range(len(params['P'][j])):
                        if not np.isnan(params['P'][j][k]):
                            self.cols_ea.extend([f'mod-{i}: EA_{j}-{k}'])
                
        # here: make empty dictionary with fields: trial, epi, it, s, a, foundGoal
        # and conditional fields (according to params): Mi, TDi and WCi_jk, WAi_jk, ECi_jk, EAi_jk for i=1,2,...
        
    def recordData(self,d_basic,wc=None,ec=None,wa=None,ea=None):
        self.data_basic.append(d_basic.copy())
        if self.rec_type=='advanced2':
            self.wc.append(wc.copy())
            self.ec.append(ec.copy())
            self.wa.append(wa.copy())
            self.ea.append(ea.copy())
        
    def readoutData_basic(self):
        return pd.DataFrame(self.data_basic,columns=self.cols_basic)

    def readoutData_advanced2(self):
        df_wc = pd.DataFrame(self.wc,columns=self.cols_wc)
        df_ec = pd.DataFrame(self.ec,columns=self.cols_ec)
        df_wa = pd.DataFrame(self.wa,columns=self.cols_wa)
        df_ea = pd.DataFrame(self.ea,columns=self.cols_ea)
        return df_wc,df_ec,df_wa,df_ea
    
    def saveData(self,dir_data,format_data='df',data_name='data_basic'):
        if not os.path.isdir(dir_data):
            os.mkdir(dir_data)
        
        start_save = timeit.default_timer()
        
        data_basic = self.readoutData_basic()
        if format_data=='df':
            data_basic.to_pickle(dir_data / f'{data_name}.pickle')  
        elif format_data=='csv':
            data_basic.to_csv(dir_data / f'{data_name}.csv',sep='\t')

        if self.rec_type=='advanced2':
            wc,ec,wa,ea = self.readoutData_advanced2()
            if format_data=='df':
                wc.to_pickle(dir_data / 'wc.pickle')
                ec.to_pickle(dir_data / 'ec.pickle')  
                wa.to_pickle(dir_data / 'wa.pickle')  
                ea.to_pickle(dir_data / 'ea.pickle')    
            elif format_data=='csv':
                wc.to_csv(dir_data / 'wc.csv',sep='\t')
                ec.to_csv(dir_data / 'ec.csv',sep='\t')
                wa.to_csv(dir_data / 'wa.csv',sep='\t')
                ea.to_csv(dir_data / 'ea.csv',sep='\t')

        end_save = timeit.default_timer()
        return (end_save-start_save)
        
    def saveParams(self,params,dir_params,format_params='dict',params_name='params'):
        if not os.path.isdir(dir_params):
            os.mkdir(dir_params)
        
        if format_params=='dict':
            save_params = params.copy()
            # if 'h' in save_params.keys():# and save_params['h']['h_k']: 
            #     del save_params['h']
            with open(dir_params / f'{params_name}.pickle', 'wb') as f:
                pickle.dump(save_params,f)   
        elif format_params=='csv':
            with open(dir_params / f'{params_name}.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=params.keys())
                writer.writeheader()
                writer.writerow(params)
        elif format_params=='txt':
            with open(dir_params / f'{params_name}.txt', 'w') as f: 
                for key, value in params.items(): 
                    f.write('%s:%s\n' % (key, value))

class experiment():
    
    def __init__(self,params,verbose=False,flag_saveData=False,dataFolder='',flag_returnData=False):
        # Upon creation of an experiment, set experimental parameters and save timestamp
        self.params         = params   
        self.rec            = recorder(self.params,self.params['rec_type'])  
        self.timestamp      = datetime.datetime.now()
        self.verbose        = verbose
        self.saveData       = flag_saveData
        self.returnData     = flag_returnData
        self.dataFolder     = dataFolder
        self.hierarchical   = ('n' in params['agent_types'] and params['ntype']=='hN')

        if self.saveData:
            if not self.dataFolder:
                self.dataFolder = sl.get_rootpath() / 'data' / 'auxiliary_simulations'
            if verbose: 
                print(f"Start making folder to save data.\n")
            self.dataFolder = self.dataFolder / f'{self.timestamp.strftime("%Y_%m_%d_%H-%M-%S")}_{self.params["sim_name"]}'
            sl.make_long_dir(self.dataFolder)
            if verbose:
                print(f"Simulation data will be saved in: \n{self.dataFolder}\n")

        if verbose: 
            print(f"Created new experiment.\n")

    
    def runExperiment(self):
        if self.verbose: print(f"Start running experiment.\n")
        
        start_exp = timeit.default_timer()
        
        # Create environment (fixed across subjects) 
        T = (self.params['T'] if 'T' in self.params.keys() else np.array([]))
        t_deact = (self.params['t_deact'] if 't_deact' in self.params.keys() else 0)

        self.env        = env(self.params['S'],self.params['P'],self.params['R'],T,t_deact)
        self.goal       = self.env.getGoal()
        self.terminal   = self.env.getTerminal()
        self.t_deact    = self.env.getTDeact()
        
        # Run trials
        for trial in range(self.params['number_trials']):
            self.runTrial(trial)
            
        end_exp = timeit.default_timer()
        exp_duration = end_exp - start_exp

        if self.saveData:
            self.rec.saveParams(self.params,self.dataFolder,'dict')
            self.rec.saveParams(self.params,self.dataFolder,'csv')
            sl.saveCodeVersion(self.dataFolder)
            
            if self.verbose: print(f"Start saving data into folder {self.dataFolder}.\n")
            sdur1 = self.rec.saveData(self.dataFolder,'df')
            sdur2 = self.rec.saveData(self.dataFolder,'csv')

            if self.verbose: print(f'Done saving data, time elapsed: {sdur1 + sdur2} sec.\n') 

        if self.verbose: print("Start reading out data.\n")
        
        if self.returnData:
            data_basic = self.rec.readoutData_basic()
            exp_data = [data_basic]
            if self.rec.rec_type=='advanced2':
                wc,ec,wa,ea = self.rec.readoutData_advanced2()
                exp_data.append(wc)
                exp_data.append(ec)            
                exp_data.append(wa)
                exp_data.append(ea)    
        else:
            exp_data = None
            
        return exp_data, self.params, self.timestamp, exp_duration, self.dataFolder
            
            
    def runTrial(self,trial):
        if self.verbose: print(f"Start running trial {trial}.\n")
        
        # Set random seed for the trial
        np.random.seed(self.params['seeds'][trial])
        
        # Create agent (fixed across trials)
        self.agent  = agent(self.params)
        
        # Run episodes
        for epi in range(self.params['number_epi']):
            self.runEpisode(trial,epi)
        
        if self.verbose: print(f"Finished running trial {trial}.\n")
            
            
    def runEpisode(self,trial,epi):
        if self.verbose: print(f"Start running trial {trial}, episode {epi}.\n")

        # Initialize flags and counters
        flag_foundTerminal = False
        it = 1 ### WAS 0 BEFORE

        # Reset goal states (to active) and initialize list of deactivated goal states and deactivation counters
        for s in self.goal:
            self.env.activateGoal(s)
        g_deact = []
        it_deact = []
        
        # Set initial state of agent
        self.env.setAgentLoc(self.params['x0'][trial][epi])
        self.agent.giveStateInput(self.params['x0'][trial][epi])
        self.agent.updateMod(self.params['x0'][trial][epi])
            
        # Initialize eligibility traces
        self.agent.resetTraces()
        
        # Record initial agent state: 
            #a = np.nan 
            #x = x0 (initial state)
            #m = r(x0) / nov(x0)
            #td = np.nan
            #w = w0 (initial weights)
            #e = e0 (initial traces, i.e. zero)
        # rec_list = [trial,epi,it,self.env.agentLoc,-1,flag_foundGoal]
        # if self.rec.rec_type == 'advanced1' or self.rec.rec_type == 'advanced2':
        #     rec_list = rec_list + list(np.nan*np.zeros(len(self.params['agent_types']))) + list(np.nan*np.zeros(len(self.params['agent_types'])))
        # if self.rec.rec_type == 'advanced2':
        #     rec_list = rec_list #+ list(np.nan*np.zeros(len(self.params[])) new_wcs.flatten() + new_ecs.flatten() + new_was.flatten() + new_eas.flatten()
        # self.rec.recordData(rec_list)
        
        #it += 1
        
        # Simulate steps until goal found or maximum number of steps reached
        while not flag_foundTerminal and it<self.params['max_it']:
            #if self.verbose: print(f"Step {it}:\n")

            # Active goal states if their deactivation time is over
            it_act = (((np.array(it_deact)-self.t_deact)>0).nonzero()[0])
            for i in it_act:
                self.env.activateGoal(g_deact)
                g_deact.pop(i)
                it_deact.pop(i)
            
            flag_foundTerminal,flag_foundGoal = self.runStep(trial,epi,it,flag_foundTerminal)

            # Deactivate reward for some time
            if flag_foundGoal:
                self.env.deactivateGoal(self.env.agentLoc)
                g_deact.append(self.env.agentLoc)
                it_deact.append(0)
            it += 1
            it_deact = [it_deact[i]+1 for i in range(len(it_deact))]
        
       
    def runStep(self,trial,epi,it,flag_foundTerminal):
        s_current      = self.env.agentLoc
        a              = self.agent.act(s_current,epi)          # decide on action
        s_next, r      = self.env.evalAction(a,s_current)   # take action in environment
        m, mh, mw, mg  = self.agent.evalMod(s_next,r)       # compute modulator signal 
        self.agent.updateMod(s_next)                        # update state of modulator (e.g. state count, time count for novelty)
        
        tds, new_wcs, new_ecs, new_was, new_eas = self.agent.learn(s_current,s_next,a,m) # learn from modulator signal

        nov_post = self.agent.evalModAll()

        # Check whether agent has reached the goal / terminal state
        flag_foundGoal=False
        if s_next in self.env.R.nonzero()[0]:
            flag_foundGoal = True
            if self.verbose: print(f"Found goal state after {it} iterations.\n")
            
        if s_next == self.terminal:
            flag_foundTerminal = True
            if self.verbose: print(f"Episode ended in terminal state after {it} iterations.\n")
            
        # Record current step k = 1,...,N
            #a = a(k) (k-th action, will lead from x(k-1) to x(k)) 
            #x = x(k) (k-th state after initial state)
            #m = r(x(k)) / nov(x(k))
            #td = td(k) (td error resulting from taking action a(k) in state x(k-1))
            #w = w(k) (weights after k updates)
            #e = e(k) (traces after one update, i.e. remembering state x0 or the action a0)
        rec_list = [trial,epi,it,s_current,a,s_next,flag_foundGoal,flag_foundTerminal,(r!=0)]
        if self.rec.rec_type == 'advanced1' or self.rec.rec_type == 'advanced2':
            if self.hierarchical:
                rec_list = rec_list + m + tds + mh + mw + mg + nov_post
            else:
                rec_list = rec_list + m + tds + nov_post
        if self.rec.rec_type == 'advanced2':
            wc_notnan = (~np.isnan(new_wcs)).nonzero()[0]
            wa_notnan = (~np.isnan(new_was)).nonzero()[0]
            rec_wc = [new_wcs[i] for i in wc_notnan]
            rec_ec = [new_ecs[i] for i in wc_notnan] 
            rec_wa = [new_was[i] for i in wa_notnan] 
            rec_ea = [new_eas[i] for i in wa_notnan]
            self.rec.recordData(rec_list,rec_wc,rec_ec,rec_wa,rec_ea)
        else:
            self.rec.recordData(rec_list)
        
        return flag_foundTerminal, flag_foundGoal


def run_ac_exp(params,verbose=False,saveData=False,dirData='',returnData=False):
    # Create folder to save data
    dataFolder = None
    timestamp  = None
    if saveData:
        if len(dirData)==0:
            dirData = sl.get_datapath()
            print(f"Directory to save data not specified. Data is saved in current directory:\n{dirData}\n")
        if verbose: print(f"Start making folder to save data.\n")
        timestamp = datetime.datetime.now()
        dataFolder = sl.make_long_dir(dirData,timestamp.strftime('%Y_%m_%d_%H-%M-%S')+f'_{params["sim_name"]}')
        print(f"Simulation data will be saved in: \n{dataFolder}\n")

    # Start timer
    if verbose: print(f"Start running experiment.\n")
    start_exp = timeit.default_timer()

    # Create environment
    if verbose: print('Creating environment.\n')
    S = params['S']; P = params['P']; R = params['R']
    T       = (params['T'] if 'T' in params.keys() else np.array([]))   # Set terminal states
    t_deact = (params['t_deact'] if 't_deact' in params.keys() else 0)  # Set reward deactivation
    ac_env     = env(S,list(P),list(R),T,t_deact)
    sg      = ac_env.getGoal()
    t_deact = ac_env.getTDeact()
    term    = ac_env.getTerminal()
    hierarchical = ('n' in params['agent_types'] and params['ntype']=='hN')

    # Creating recorder
    if verbose: print('Creating recorder.\n')
    rec = recorder(params,params['rec_type'])  

    # Run trials
    for trial in range(params['number_trials']):
        if verbose: print(f"Start running trial {trial}.\n")
        np.random.seed(params['seeds'][trial]) # Set random seed
        start_trial = timeit.default_timer()   # Start timer
        ac_agent       = agent(params)            # Create agent
        
        # Run episodes
        for e in range(params['number_epi']):
            if verbose: print(f"Start running episode {e}.\n")
            start_epi = timeit.default_timer()

            # Reset goal states (to active) and initialize list of deactivated goal states and deactivation counters
            for s in sg:
                ac_env.activateGoal(s)
            g_deact = []
            it_deact = []
            
            # Initialize the state variables
            Tmax          = params['max_it']
            foundTerminal = False
            it            = 1 # start with 0 or 1?
            
            # Set initial state of agent
            ac_env.setAgentLoc(params['x0'][trial][e])
            ac_agent.giveStateInput(params['x0'][trial][e])
            ac_agent.updateMod(params['x0'][trial][e])
            ac_agent.resetTraces()
        
            # Simulate steps until goal found or maximum number of steps reached
            while not foundTerminal and it<Tmax:
                # Active goal states if their deactivation time is over
                it_act = (((np.array(it_deact)-t_deact)>0).nonzero()[0])
                for i in it_act:
                    ac_env.activateGoal(g_deact)
                    g_deact.pop(i)
                    it_deact.pop(i)
            
                # Take next action + update values
                s_current      = ac_env.agentLoc
                a              = ac_agent.act(s_current,e)          # decide on action
                s_next, r      = ac_env.evalAction(a,s_current)     # take action in environment
                m, mh, mw, mg  = ac_agent.evalMod(s_next,r)         # compute modulator signal 
                ac_agent.updateMod(s_next)                          # update state of modulator (e.g. state count, time count for novelty)
        
                tds, new_wcs, new_ecs, new_was, new_eas = ac_agent.learn(s_current,s_next,a,m) # learn from modulator signal

                # Check whether agent has reached the goal / terminal state
                foundGoal = False
                if s_next in np.nonzero(ac_env.R)[0]:
                    foundGoal = True
                    if verbose: print(f"Found goal state after {it} iterations.\n") 
                if s_next == term:
                    foundTerminal = True
                    if verbose: print(f"Episode ended in terminal state after {it} iterations.\n")
                
                # Record step
                rec_list = [trial,e,it,s_current,a,s_next,foundGoal,foundTerminal,(r!=0)]
                if rec.rec_type == 'advanced1' or rec.rec_type == 'advanced2':
                    if hierarchical:    rec_list = rec_list + m + tds + mh + mw + mg
                    else:               rec_list = rec_list + m + tds 
                if rec.rec_type == 'advanced2':
                    wc_notnan = (~np.isnan(new_wcs)).nonzero()[0]
                    wa_notnan = (~np.isnan(new_was)).nonzero()[0]
                    rec_wc = [new_wcs[i] for i in wc_notnan]
                    rec_ec = [new_ecs[i] for i in wc_notnan] 
                    rec_wa = [new_was[i] for i in wa_notnan] 
                    rec_ea = [new_eas[i] for i in wa_notnan]
                    rec.recordData(rec_list,rec_wc,rec_ec,rec_wa,rec_ea)
                else:
                    rec.recordData(rec_list)

                # Deactivate reward for some time
                if foundGoal:
                    ac_env.deactivateGoal(ac_env.agentLoc)
                    g_deact.append(ac_env.agentLoc)
                    it_deact.append(0)
                it += 1
                it_deact = [it_deact[i]+1 for i in range(len(it_deact))]

            end_epi = timeit.default_timer()
            if verbose: print(f"Simulated episode {e} in {end_epi-start_epi} s.\n")

        end_trial = timeit.default_timer()
        if verbose: print(f"Simulated agent {trial} in {end_trial-start_trial} s.\n")

    end_exp = timeit.default_timer()
    exp_duration = end_exp - start_exp
    if verbose: print(f"Simulated agent {trial} in {exp_duration} s.\n")

    if saveData:
        if verbose: print(f"Start saving data into folder {dataFolder}.\n")
        start_data = timeit.default_timer()  
        rec.saveParams(params,dataFolder,'dict')
        rec.saveParams(params,dataFolder,'csv')
        sl.saveCodeVersion(dataFolder)
        rec.saveData(dataFolder,'df')
        rec.saveData(dataFolder,'csv')
        end_data = timeit.default_timer()
        if verbose: print(f'Done saving data, time elapsed: {end_data-start_data} sec.\n') 
        
    if returnData:
        if verbose: print("Start reading out data.\n")
        start_data = timeit.default_timer()  
        data_basic = rec.readoutData_basic()
        exp_data = [data_basic]
        if rec.rec_type=='advanced2':
            wc,ec,wa,ea = rec.readoutData_advanced2()
            exp_data.append(wc)
            exp_data.append(ec)            
            exp_data.append(wa)
            exp_data.append(ea) 
        end_data = timeit.default_timer()
        if verbose: print(f'Done reading data, time elapsed: {end_data-start_data} sec.\n') 
           
    else:
        exp_data = None
            
    return exp_data, params, timestamp, exp_duration, dataFolder
        
            
         

        
        
        
        
        
        
        
        
        
