'''
 used for 5-fold cross validation
 copy training data to path9, testing data to path35
 code: 11 used for 41 projects, 22 used for 1k projects
'''

import sys, os, shutil
import PATH_VARIABLES
import numpy as np

# working mode and fold number (1, 2, 3, 4, 5)
WORK_MODE = int(sys.argv[1])
FOLD_NUMBER = int(sys.argv[2]) # 1, 2, 3, 4, 5

if WORK_MODE == 11:
    # for 41
    path9 = PATH_VARIABLES.path9
    path35 = PATH_VARIABLES.path35
    path_total = PATH_VARIABLES.path40_pls
    path_f = PATH_VARIABLES.name_40_shuffled
elif WORK_MODE == 22:
    # for 1k
    path9 = PATH_VARIABLES.path9
    path35 = PATH_VARIABLES.path35
    path_total = PATH_VARIABLES.path1k_pls
    path_f = PATH_VARIABLES.name_1k_shuffled
    
with open(path_f) as f:
    contents = f.readlines()
    contents = [cnts.strip() for cnts in contents]
    
names_all = list(contents)
name_folds = np.array_split(np.array(names_all), 5)
name_folds = [list(fld) for fld in name_folds]

# use name_folds
# clear train and test holder

flist_1 = os.listdir(path9)
for fn1 in flist_1:
    if os.path.isfile(path9 + '/' + fn1):
        os.remove(path9 + '/' + fn1)
    
flist_2 = os.listdir(path35)
for fn2 in flist_2:
    if os.path.isfile(path35 + '/' + fn2):
        os.remove(path35 + '/' + fn2)

files_to_small = []
files_to_large = []

for ii in xrange(len(name_folds)):
    if ii==FOLD_NUMBER-1:
        files_to_small += name_folds[ii]
    else:
        files_to_large += name_folds[ii]
        
files_to_small = [path_total+'/'+fl for fl in files_to_small]
files_to_large = [path_total+'/'+fl for fl in files_to_large]

for fl in files_to_small:
    shutil.copy2(fl, path35)
    
for fl in files_to_large:
    shutil.copy2(fl, path9)
    

