'''
 using RandomForest Model from sklearn
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from experiments import parse_DataSet_weights, PATH_VARIABLES

def get_m_RandomForest(n_estimators = 150):
    """
    get an instantiation of random forest classifier
    """
    my_clf = RandomForestClassifier(n_estimators)
    return my_clf

def test_ScaleWithLog_ScaleNoLog():
    """
    test if log helps improve performance when do scaling
    using 5-fold CV
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS_SWL = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, log_or_not=True, scale_or_not=True)
    myDS_SNL = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, log_or_not=False, scale_or_not=True)
    [X_train_SWL, y_train_SWL, _] = myDS_SWL.prepare_DS_for_training_rf()
    [X_train_SNL, y_train_SNL, _] = myDS_SNL.prepare_DS_for_training_rf()
    MDL_SWL = get_m_RandomForest()
    MDL_SNL = get_m_RandomForest()
    MDL_SWL.fit(X_train_SWL, y_train_SWL)
    MDL_SNL.fit(X_train_SNL, y_train_SNL)
    cname_to_acc = {}
    for i in xrange(len(myDS_SWL.p35_list)):
        [X_test_SWL, y_test_SWL, cpath_SWL, _] = myDS_SWL.get_ith_test_data_rf(i)
        y_pred_SWL = MDL_SWL.predict(X_test_SWL)
        acc_SWL = accuracy_score(y_test_SWL, y_pred_SWL)
        cname_to_acc[cpath_SWL] = [acc_SWL]
    for i in xrange(len(myDS_SNL.p35_list)):
        [X_test_SNL, y_test_SNL, cpath_SNL, _] = myDS_SNL.get_ith_test_data_rf(i)
        y_pred_SNL = MDL_SNL.predict(X_test_SNL)
        acc_SNL = accuracy_score(y_test_SNL, y_pred_SNL)
        cur_list = cname_to_acc[cpath_SNL]
        cur_list.append(acc_SNL)
        cname_to_acc[cpath_SNL] = cur_list
        
    with open(PATH_VARIABLES.test_ScaleWithLog_ScaleNoLog, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            cur_list = [str(ele) for ele in cur_list]
            add_here += cur_list
            f.write(','.join(add_here) + '\n')
            
#test_ScaleWithLog_ScaleNoLog()

def test_find_best_tree_number(estim_list):
    """
    try different n_n_estimators, get the best one
    try [10, 50, 100, 150, 200]
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=True, scale_or_not=False, log_or_not=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    cname_to_acc = {}
    for n_e in estim_list:
        MDL = get_m_RandomForest(n_e)
        MDL.fit(X_train, y_train)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            y_pred = MDL.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    # write cname_to_acc
    with open(PATH_VARIABLES.parameter_tuning_result_holder, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            
# call the parameter tuning process
#test_find_best_tree_number([10, 50, 100, 150, 200])

def run_m_RandomForest(is_subset=True):
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=False, log_or_not=False, subset=is_subset)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    MDL = get_m_RandomForest()
    if is_subset:
        mdl_name = "RandomForest"
        rst_f_path = PATH_VARIABLES.cv_5fold_repo_RF
    else:
        mdl_name = "RandomForestfull"
        rst_f_path = PATH_VARIABLES.cv_5fold_repo_RFf
    #training time
    t_start = time.time()
    MDL.fit(X_train, y_train)
    t_end = time.time()
    t_time = t_end - t_start
    cname_to_perfs = {}
    for i in xrange(len(myDS.p35_list)):
        [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
        y_pred = MDL.predict(X_test)
        ###
        # added to include more
        prj_name = cpath.split('/')[-1]
        prj_name = prj_name.split('.csv')[0] # remove if do not need
        y_pred_prob = MDL.predict_proba(X_test).tolist()
        y_test_ls = list(y_test)
        for i in xrange(len(y_pred_prob)):
            y_pred_prob[i].append(y_test_ls[i])
        for i in xrange(len(y_pred_prob)):
            with open("/home/lab/probrst/" + "prj_name", 'a') as f:
                cur_ls = [str(item) for item in y_pred_prob[i]]
                cur_ls = ','.join(cur_ls)
                f.write(cur_ls + '\n')
        ###
        #accuracy
        cur_accuracy = accuracy_score(y_test, y_pred)
        #error
        cur_error = 1 - cur_accuracy
        #precision
        cur_precision = precision_score(y_test, y_pred)
        #recall
        cur_recall = recall_score(y_test, y_pred)
        #f-measure
        cur_f1 = f1_score(y_test, y_pred)
        #AUROC
        y_score = MDL.predict_proba(X_test)
        y_score = [float(ys[1]) for ys in y_score]
        if sum(y_test)==0 or sum(y_test)==len(y_test):
            cur_auroc = "NA"
        else:
            cur_auroc = roc_auc_score(y_test, y_score)
        #
        prj_name = cpath.split('/')[-1]
        prj_name = prj_name.split('.csv')[0]
        cname_to_perfs[prj_name] = [mdl_name, t_time, cur_accuracy, cur_error, cur_precision, cur_recall, cur_f1, cur_auroc]
        
    with open(rst_f_path, 'a') as f:
        for prj in cname_to_perfs:
            add_here = [prj]
            cur_list = cname_to_perfs[prj]
            cur_list = [str(ele) for ele in cur_list]
            add_here += cur_list
            f.write(','.join(add_here) + '\n')
    
#run_m_RandomForest()
if __name__=='__main__':
    #test_find_best_tree_number([10, 50, 100, 150, 200])
    run_m_RandomForest(is_subset=True)
    run_m_RandomForest(is_subset=False)