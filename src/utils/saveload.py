import numpy as np
import pandas as pd
import os
import pickle
import csv
import subprocess
from pathlib import Path

##############################################################################
#       PATHS / DIRECTORIES                                                  #
##############################################################################
def get_rootpath(target_name='sim_nov', start_path=Path(__file__).resolve()):
    """Get the root path of a project by searching for a target directory name.

    Args:
        target_name (str): The name of the target directory.
        start_path (Path): The starting path to search from.

    Returns:
        parent (Path): The root path of the project (if it exists, else None).

    """
    for parent in start_path.parents:
        if parent.name==target_name:
            return parent
    return None

# def get_datapath():
#     path  = os.getcwd()
#     spath = path.split('/')
#     lpath = ''
#     for i in range(1,len(spath)):
#         if 'RL_reward_novelty' in spath[-i]:
#             break
#         else:
#             lpath = lpath+'../'
#     if not lpath:
#         lpath='./'
#     lpath = lpath+'data/'
#     return lpath

# def make_dir(dir,folder=''):
#     if not folder:
#         full_dir = dir
#     else:
#         if dir[-1]=='/': full_dir = dir+folder
#         else: full_dir = dir+'/'+folder

#     if not os.path.isdir(full_dir):
#         os.mkdir(full_dir) 
#     return full_dir

def make_long_dir(path):
    """
    Create a directory and all its parent directories if they do not exist.
    """
    max_depth = 10
    c = 0
    made_path = None
    path_obj = Path(path)

    # Check for valid initial path
    if not path_obj.parts[0] == '/' and not path_obj.parts[0]: 
        raise ValueError('Invalid path.')

    # Iterate over path parts, starting from the second part
    for i in range(2, len(path_obj.parts) + 1):
        sub_path = path_obj.parts[:i]
        current_path = Path(*sub_path)

        if not current_path.exists():
            made_path = os.mkdir(current_path)
            c += 1
        
        if c >= max_depth:
            break

    return made_path

##############################################################################
#       DATA SAVE / LOAD                                                     #
##############################################################################
def load_sim_data(dir_data,file_data='data_basic.pickle'):
    """
    Load simulation data from a given directory (for fitting).
    """
    path_data = dir_data / file_data
     
    flag_load = True
    if not os.path.exists(path_data):
        print(f'Data file {file_data} does not exist in {path_data}. Trying to load file all_data (old format) instead.')
        path_data = Path(str(path_data).replace(file_data,'all_data.pickle'))
        if not os.path.exists(path_data):
            print(f'Data file {file_data} does not exist in {path_data}. Please specify a valid data path and file.')
            flag_load = False                   
    if flag_load:
        type_data = str(path_data).split('.')[-1]
        if type_data=='pickle' or type_data=='pkl':
            data = pd.read_pickle(path_data)
        elif type_data=='csv': 
            data = pd.read_csv(path_data,index_col=0)
        elif type_data=='npy': 
            data = np.load(path_data)
    return data

def load_sim_params(dir_params,file_params='params.pickle',auto_path=False):
    if auto_path:
        if 'data' in dir_params:
            path_params = dir_params / file_params
        elif 'src/' in os.getcwd():
            path_params = os.path.join('..','..','data',dir_params,file_params)
        else: 
            path_params = os.path.join('.','data',dir_params,file_params)
    else:
        if dir_params[-1]=='/': path_params = dir_params+file_params
        else: path_params = dir_params+'/'+file_params 

    path_params = dir_params / file_params
    type_data = file_params.split('.')[-1]
    if type_data=='pickle' or type_data=='pkl':
        params = pd.read_pickle(path_params)
    elif type_data=='csv': 
        dd = pd.read_csv(path_params)
        string_fields = ['sim_name','rec_type','mb_rec_type','mf_rec_type','ntype','mb_ntype','mf_ntype',
                            'round_prec','number_trials','number_epi','max_it','S','a','k','mb_k','mf_k','simID']
        params = {}
        for l in list(dd.columns):
            if not l in string_fields:
                params[l] = eval(dd[l].values[0])
            else:
                params[l] = dd[l].values[0]

    # with open(path_params, 'rb') as f:
    #     params = pickle.load(f)
    return params

def load_df_stateseq(dir='',file='load_df_stateseq.pickle'):
    if not dir:
        if 'src/' in os.getcwd():
            path = os.path.join('..','..','ext_data','Rosenberg2021',file)
        else: 
            path = os.path.join('.','ext_data','Rosenberg2021',file)
    else:
        path = dir / file
    
    with open(path,'rb') as f:
        df_stateseq = pickle.load(f)
    return df_stateseq

def load_human_data(dir_data='ext_data',file_data='raw_data_behav.csv',map_epi={1:0,2:1,3:2,4:3,5:4},
                    map_states={1:10,2:0,3:7,4:2,5:4,6:1,7:8,8:9,9:3,10:5,11:6}):
    ## Load data
    all_data = pd.read_csv('./'+dir_data+'/'+file_data)      # import behavioural data as dataframe
    all_data = all_data[all_data['env']==3]                         # exclude all data from second block (surprise block)
    all_data['epi'].replace(map_epi,inplace=True)                   # adjust episode data labels
    all_data['state'].replace(map_states,inplace=True)              # adjust state data labels
    all_data['next_state'].replace(map_states,inplace=True)  
    return all_data

def excludeFailedTrials(data):
    data1 = data.groupby(['subID','epi']).last()
    #data2 = data1['reward'].groupby(['subID']).all()
    data2 = data1['reward'].groupby(['subID']).first()
    if not data2.dtype=='bool':
        data2 = data2.apply(lambda x: bool(x))
    subID_noFail    = data2[data2].index.values
    data_noFail     = data[data['subID'].isin(subID_noFail)]
    return data_noFail

##############################################################################
#       CONVERSION OF DATA STRUCTURES                                        #
##############################################################################
def convert_dict_to_csv(path,file='params.pickle'):
    params = load_sim_params(path,file)
    with open((path+'/'+file).replace('.pickle','.csv'),'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=params.keys())
        writer.writeheader()
        writer.writerow(params)
            
def convert_df_to_csv(path,file='all_data.pickle'):
    all_data = load_sim_data(path,file)
    all_data.to_csv((path+'/'+file).replace('.pickle','.csv'),sep='\t')

def save_dict_as_csv(mydict,path):
    with open(path,'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=mydict.keys())
        writer.writeheader()
        writer.writerow(mydict)

def load_dict_from_csv(path):
    mydict = pd.read_csv(path).transpose().to_dict()[0]
    # mydict = []
    # with open(path) as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         mydict.append(dict(row))

    # with open(path, mode='r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     mydict = dict((rows[0],rows[1]) for rows in reader)
    return mydict

# Convert q-values nac->nor
def qvals_nac2nor(all_data,qvals,params,dir_save=''):
    P = params['P']
    Pnn = (~np.isnan(P)).nonzero()
    Pnn_conv = (Pnn[0],np.array([P[i][j] for i,j in zip(Pnn[0],Pnn[1])]))
    # Pnn_conv = (Pnn[0],map(int,np.array(P)[Pnn]))
    qvals_np = qvals.to_numpy()
    l_qi = []
    for i in range(qvals_np.shape[0]):
        qi      = np.empty((len(P),len(P)))
        qi[:]   = np.nan
        qi[Pnn_conv] = qvals_np[i,:]
        l_qi.append(qi)
    qvals_conv = all_data[['subID','epi','it']].copy()
    qvals_conv['qvals'] = l_qi.copy()
    if len(dir_save)>0:
        file_save = dir_save+('' if dir_save[-1]=='/' else '/')+'qvals'
        qvals_conv.to_pickle(file_save+'.pickle')
        qvals_conv.to_csv(file_save+'.csv')
    return qvals_conv

# Convert q-values nor->nac
def qvals_nor2nac(qvals,params,dir_save=''):
    Pnn = (~np.isnan(params['P'])).nonzero()
    cq = [f'mod-0: WA_{i}-{j}' for i,j in zip(Pnn[0],Pnn[1])]
    # Reshape and extract not-nan qvalues
    q = np.array(list(map(lambda x: x.flatten(),qvals['qvals'].values)))
    qnn = (~np.isnan(q[0,:])).nonzero()
    q1 = q[:,qnn[0]]
    q1 = q1.reshape((q1.shape[0],q1.shape[-1]))
    # Write not-nan qvalues into correct columns
    #qvals_conv = qvals[['subID','epi','it']].copy()
    qvals_conv = pd.DataFrame(columns=cq)
    qvals_conv[cq] = q1
    # Save
    if len(dir_save)>0:
        file_save = dir_save+('' if dir_save[-1]=='/' else '/')+'wa'
        qvals_conv.to_pickle(file_save+'.pickle')
        qvals_conv.to_csv(file_save+'.csv')
    return qvals_conv

# Convert old data format (all_data) to new data format (data_basic, qvals, beliefs)
def convert_old2new(dir_data,file='all_data.pickle'):
    all_data = load_sim_data(dir_data,file)
    data_basic = all_data[['subID','epi','it','state','action','next_state','reward','novelty','foundGoal']]
    data_basic.to_csv(dir_data+'/data_basic.csv')
    data_basic.to_pickle(dir_data+'/data_basic.pickle')
    if 'q-vals' in all_data.columns():
        qvals = all_data[['subID','epi','it','q-vals']]
        qvals.to_csv(dir_data+'/qvals.csv')
        qvals.to_pickle(dir_data+'/qvals.pickle')
    if 'beliefs' in all_data.columns():
        beliefs = all_data[['subID','epi','it','beliefs']]
        beliefs.to_csv(dir_data+'/beliefs.csv')
        beliefs.to_pickle(dir_data+'/beliefs.pickle')

##############################################################################
#       LOGGING                                                              #
##############################################################################
def write_log(log_file,*args):
    line = ' '.join([str(a) for a in args])
    log_file.write(line+'\n')
    print(line)

def saveCodeVersion(dir_cv,dir_git='',file_cv='code_version.txt'):
    if not os.path.isdir(dir_cv):
        os.mkdir(dir_cv)
    
    os.chdir(dir_git)
        
    with open(dir_cv / file_cv,'w') as f:
        cv = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        f.write(cv)

##############################################################################
#      BASIC FUNCTIONS                                                       #
##############################################################################
def get_random_seed(length,n,init_seed=None):
    if not init_seed: 
        np.random.seed()
    else:
        np.random.seed(init_seed)
    min = 10**(length-1)
    max = 9*min + (min-1)
    return np.random.randint(min, max, n)