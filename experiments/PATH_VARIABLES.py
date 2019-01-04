'''
 specify paths
'''

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
dir_prj_root = cur_dir[:cur_dir.rfind('/')]
dir_docs = dir_prj_root + '/docs'
dir_expresults = dir_prj_root + '/expresults'
dir_experiments = dir_prj_root + '/experiments'

# training, testing
path9 = "PATH_TO_DS_Train"
path35 = "PATH_TO_DS_Test"
path40_pls = ""
path1k_pls = "PATH_TO_DS_All"

name_40_shuffled = dir_docs + ''
name_1k_shuffled = dir_docs + '/name_1kprjs_shuffled.txt'

test_ScaleWithLog_ScaleNoLog = dir_expresults + '/test_ScaleWithLog_ScaleNoLog.csv'
test_NonDeepModel_ScaleOrNot = dir_expresults + "/test_NonDeepModel_ScaleOrNot.csv"
test_Models_ScaleOrNot = dir_expresults + '/test_Models_ScaleOrNot.csv'
test_Models_SWorNSW = dir_experiments + '/test_Models_SWorNSW.csv'
parameter_tuning_result_holder = dir_expresults + "/parameter_tuning_results_holder.csv"
test_MLP_best_activationType = dir_expresults + '/test_MLP_best_activationType.csv'
test_MLP_best_OPT_type = dir_expresults + '/test_MLP_best_OPT_type.csv'
test_MLP_best_hlayers = dir_expresults + '/test_MLP_best_hlayers.csv'
test_CNN_best_activationType = dir_expresults + '/test_CNN_best_activationType.csv'
test_CNN_best_OPT_type = dir_expresults + '/test_CNN_best_OPT_type.csv'
test_CNN_best_filterSize = dir_expresults + '/test_CNN_best_filterSize.csv'
test_gcF_on35 = dir_expresults + '/test_gcF_on35.csv'

#hold 5FoldCVResults
dir_5foldcvr = dir_expresults + '/5FoldCVResults'
cv_5fold_repo_RF = dir_5foldcvr + '/cv_5fold_repo_RF.csv'
cv_5fold_repo_RFf = dir_5foldcvr + '/cv_5fold_repo_RFf.csv'
cv_5fold_repo_ET = dir_5foldcvr + '/cv_5fold_repo_ET.csv'
cv_5fold_repo_DS = dir_5foldcvr + '/cv_5fold_repo_DS.csv'
cv_5fold_repo_DT = dir_5foldcvr + '/cv_5fold_repo_DT.csv'
cv_5fold_repo_KN = dir_5foldcvr + '/cv_5fold_repo_KN.csv'
cv_5fold_repo_NB = dir_5foldcvr + '/cv_5fold_repo_NB.csv'
cv_5fold_repo_SV = dir_5foldcvr + '/cv_5fold_repo_SV.csv'
cv_5fold_repo_NN = dir_5foldcvr + '/cv_5fold_repo_NN.csv'
cv_5fold_repo_MLP = dir_5foldcvr + '/cv_5fold_repo_MLP.csv'
cv_5fold_repo_MLPs = dir_5foldcvr + '/cv_5fold_repo_MLPs.csv'
cv_5fold_repo_CNN = dir_5foldcvr + '/cv_5fold_repo_CNN.csv'
cv_5fold_repo_CNNs = dir_5foldcvr + '/cv_5fold_repo_CNNs.csv'
cv_5fold_repo_caF = dir_5foldcvr + '/cv_5fold_repo_caF.csv'
cv_5fold_repo_caFs = dir_5foldcvr + '/cv_5fold_repo_caFs.csv'
cv_5fold_repo_gcF = dir_5foldcvr + '/cv_5fold_repo_gcF.csv'