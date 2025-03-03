from pathlib import Path
import os

### Run once before using this repo to install all packages from the repo. ###

path = Path('/Users/sbecker/Projects/sim_nov/src')
path = path.resolve() 
os.chdir(path)

os.system('pip install -e .')