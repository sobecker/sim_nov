from pathlib import Path
import pandas as pd
import pickle
import utils.micedata as mice
import utils.saveload as sl
import sys

### Script to preprocess mouse data by Rosenberg et al. (2021) ###

# Requirements: 
# Before running this script, clone into Rosenberg-2021-Repository: https://github.com/markusmeister/Rosenberg-2021-Repository.
# Set dir_load below to the path where you extracted the Rosenberg repository. 

# Set paths
project_root = sl.get_rootpath()
dir_save = project_root / 'ext_data' / 'Rosenberg2021'
dir_load = Path('/Users/sbecker/Projects/Rosenberg-2021-Repository/outdata')
sys.path.append(str(dir_load))

# Extract full paths for all mice
mice.get_AllMice_until_maxit(savedata=True,dir_load=dir_load,dir_save=dir_save)

# Extract paths until first goal state encounter for all mice [basis for fitting!]
df_unrew ,failed_unrew = mice.get_UnrewExp_until_goal(savedata=True,excludefailed=True,dir_load=dir_load,dir_save=dir_save)
df_rew, failed_rew     = mice.get_RewExp_until_goal(savedata=True,excludefailed=True,dir_load=dir_load,dir_save=dir_save)
df_all                 = pd.concat([df_unrew,df_rew])
with open(dir_save / 'df_stateseq_AllMiceUntilG.pickle','wb') as f:
    pickle.dump(df_all,f)
df_all.to_csv(dir_save / 'df_statseq_AllMiceUntilG.csv',sep='\t')