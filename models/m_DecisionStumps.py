'''
 AdaBoostClassifier with weak learners as decision stumps
 n_estimaters can be choose to get better performance
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from experiments import parse_DataSet_weights, PATH_VARIABLES

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_m_DecisionStumps(n_estimators = 100):
    """
    choose n_estimators to get better performance
    50 is default value
    """
    my_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                                n_estimators=n_estimators, 
                                algorithm="SAMME")
    return my_clf

def test_find_best_tree_number(estim_list):
    """
    @param estim_list: tree number list
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=False, log_or_not=False, subset=True)
    [X_train, y_train, sweights] = myDS.prepare_DS_for_training_rf()
    cname_to_acc = {}
    for n_e in estim_list:
        MDL = get_m_DecisionStumps(n_e)
        MDL.fit(X_train, y_train, sweights)
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

def run_m_DecisionStumps():
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=False, log_or_not=False, subset=True)
    [X_train, y_train, sweights] = myDS.prepare_DS_for_training_rf()
    MDL = get_m_DecisionStumps()
    #training time
    t_start = time.time()
    MDL.fit(X_train, y_train, sweights)
    t_end = time.time()
    t_time = t_end - t_start
    cname_to_perfs = {}
    for i in xrange(len(myDS.p35_list)):
        [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
        y_pred = MDL.predict(X_test)
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
        cname_to_perfs[prj_name] = ['DecisionStumps', t_time, cur_accuracy, cur_error, cur_precision, cur_recall, cur_f1, cur_auroc]
        
    with open(PATH_VARIABLES.cv_5fold_repo_DS, 'a') as f:
        for prj in cname_to_perfs:
            add_here = [prj]
            cur_list = cname_to_perfs[prj]
            cur_list = [str(ele) for ele in cur_list]
            add_here += cur_list
            f.write(','.join(add_here) + '\n')
            
#run_m_DecisionStumps()
if __name__ == '__main__':
    #test_find_best_tree_number([10, 50, 100, 150, 200])
    run_m_DecisionStumps()