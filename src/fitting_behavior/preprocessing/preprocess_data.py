import pandas as pd
import pickle
import os
import sys
sys.path.append('/Users/sbecker/Projects/sim_nov/')
import src.utils.micedata as mice

### Script to preprocess mouse data by Rosenberg et al. (2021) ###

# Requirements: 
# Before running this script, clone into Rosenberg-2021-Repository: https://github.com/markusmeister/Rosenberg-2021-Repository.
# Set dir_load below to the path where you extracted the Rosenberg repository. 

dir_load = '/Users/sbecker/Projects/Rosenberg-2021-Repository/outdata' 
dir_save = '/Users/sbecker/Projects/sim_nov/ext_data/Rosenberg2021'

# Extract full paths for all mice
mice.get_AllMice_until_maxit(savedata=True,dir_load=dir_load,dir_save=dir_save)

# Extract paths until first goal state encounter for all mice [basis for fitting!]
df_unrew ,failed_unrew = mice.get_UnrewExp_until_goal(savedata=True,excludefailed=True,dir_load=dir_load,dir_save=dir_save)
df_rew, failed_rew     = mice.get_RewExp_until_goal(savedata=True,excludefailed=True,dir_load=dir_load,dir_save=dir_save)
df_all                 = pd.concat([df_unrew,df_rew])
with open(os.path.join(dir_save,'df_stateseq_AllMiceUntilG.pickle'),'wb') as f:
    pickle.dump(df_all,f)
df_all.to_csv(os.path.join(dir_save,'df_statseq_AllMiceUntilG.csv'),sep='\t')