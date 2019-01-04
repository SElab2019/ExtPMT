'''
 implementation of gcForest, refer to https://github.com/kingfengji/gcForest
 two settings, gc and ca
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import PATH_VARIABLES
from experiments import parse_DataSet_weights

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import time

from gcforest.gcforest import GCForest

def get_m_gcForest(mtype="ca"):
    """
    @param mtype: "ca" or "gc"
    @param n_esti: n_estimators param in get_ca_config
    """
    if mtype=="ca":
        config = get_ca_config()
        gc = GCForest(config)
        return gc
    if mtype=="gc":
        config = get_gc_config()
        gc = GCForest(config)
        return gc

# cascading config
def get_ca_config():
    """
    @param n_esti: n_estimators for tree classifier
    """
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", 
                                    "n_estimators": 50, "max_depth": None, "n_jobs": -1})
    #ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", 
    #                                "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", 
                                    "n_estimators": 50, "max_depth": None, "n_jobs": -1})
    #ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", 
    #                                "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config

# multi-grained scanning
def get_gc_config():
    """
    fix all the parameters
    """
    config = {}
    ne_config = {}
    ca_config = {}
    ne_config["outputs"] = ["pool1/24/ets", "pool1/24/rf", 
                            "pool1/12/ets", "pool1/12/rf", 
                            "pool1/6/ets", "pool1/6/rf"]
    ne_config["layers"] = []
    ne_config["layers"].append({"type": "FGWinLayer", "name": "win1/24", "bottoms": ["X","y"], 
                                "tops": ["win1/24/ets", "win1/24/rf"], "n_classes": 2,
                                "estimators": [{"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10},
                                               {"n_folds":3,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10}
                                               ], 
                                "win_x":1, "win_y":24,
                                })
    ne_config["layers"].append({"type": "FGWinLayer", "name": "win1/12", "bottoms": ["X","y"], 
                                "tops": ["win1/12/ets", "win1/12/rf"], "n_classes": 2,
                                "estimators": [{"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10},
                                               {"n_folds":3,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10}
                                               ], 
                                "win_x":1, "win_y":12,
                                })
    ne_config["layers"].append({"type": "FGWinLayer", "name": "win1/6", "bottoms": ["X","y"], 
                                "tops": ["win1/6/ets", "win1/6/rf"], "n_classes": 2,
                                "estimators": [{"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10},
                                               {"n_folds":3,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10}
                                               ], 
                                "win_x":1, "win_y":6,
                                })
    ne_config["layers"].append({"type": "FGPoolLayer", "name": "pool1", 
                                "bottoms": ["win1/24/ets", "win1/24/rf", "win1/12/ets", "win1/12/rf", "win1/6/ets", "win1/6/rf"], 
                                "tops": ["pool1/24/ets", "pool1/24/rf", "pool1/12/ets", "pool1/12/rf", "pool1/6/ets", "pool1/6/rf"], 
                                "pool_method": "avg", 
                                "win_x": 2, "win_y": 2
                                })
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["look_indexs_cycle"] = [[0, 1], [2, 3], [4, 5]]
    ca_config["n_classes"] = 2
    ca_config["estimators"] = [{"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":100,"max_depth":None,"n_jobs":-1,"max_features":1}, 
                               {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":100,"max_depth":None,"n_jobs":-1,"max_features":1}, 
                               {"n_folds":3,"type":"RandomForestClassifier","n_estimators":100,"max_depth":None,"n_jobs":-1}, 
                               {"n_folds":3,"type":"RandomForestClassifier","n_estimators":100,"max_depth":None,"n_jobs":-1}
                               ]
    config["net"] = ne_config
    config["cascade"] = ca_config
    return config

def test_gcF_on35(m_type="ca"):
    """
    4X100, 2X100, 2X50
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    if m_type=="ca":
        myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=False, scale_or_not=False, log_or_not=False)
    if m_type=="gc":
        myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=False, scale_or_not=True, log_or_not=True)
    MDL = get_m_gcForest(mtype=m_type)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    if m_type=="gc":
        X_train = X_train[:, np.newaxis, :, np.newaxis,]
    MDL.fit_transform(X_train, y_train)
    cname_to_acc = {}
    for i in xrange(len(myDS.p35_list)):
        [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
        if m_type=="gc":
            X_test = X_test[:, np.newaxis, :, np.newaxis,]
        y_pred = MDL.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if cpath in cname_to_acc:
            cur_list = cname_to_acc[cpath]
            cur_list.append(acc)
            cname_to_acc[cpath] = cur_list
        else:
            cname_to_acc[cpath] = [acc]
            
    with open(PATH_VARIABLES.test_gcF_on35, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            

def run_m_caForest(is_subset=False):
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=is_subset, scale_or_not=False, log_or_not=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    MDL = get_m_gcForest("ca")
    if is_subset==False:
        clr_name = "caForest"
        rst_file_path = PATH_VARIABLES.cv_5fold_repo_caF
    else:
        clr_name = "caForestsubset"
        rst_file_path = PATH_VARIABLES.cv_5fold_repo_caFs
    #training time
    t_start = time.time()
    MDL.fit_transform(X_train, y_train)
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
        cname_to_perfs[prj_name] = [clr_name, t_time, cur_accuracy, cur_error, cur_precision, cur_recall, cur_f1, cur_auroc]
        
    with open(rst_file_path, 'a') as f:
        for prj in cname_to_perfs:
            add_here = [prj]
            cur_list = cname_to_perfs[prj]
            cur_list = [str(ele) for ele in cur_list]
            add_here += cur_list
            f.write(','.join(add_here) + '\n')
            
def run_m_gcForest():
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=False, scale_or_not=True, log_or_not=True)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    X_train = X_train[:, np.newaxis, :, np.newaxis,] ########
    MDL = get_m_gcForest("gc")
    #training time
    t_start = time.time()
    MDL.fit_transform(X_train, y_train)
    t_end = time.time()
    t_time = t_end - t_start
    cname_to_perfs = {}
    for i in xrange(len(myDS.p35_list)):
        [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
        X_test = X_test[:, np.newaxis, :, np.newaxis,] ########
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
        cname_to_perfs[prj_name] = ['gcForest', t_time, cur_accuracy, cur_error, cur_precision, cur_recall, cur_f1, cur_auroc]
        
    with open(PATH_VARIABLES.cv_5fold_repo_gcF, 'a') as f:
        for prj in cname_to_perfs:
            add_here = [prj]
            cur_list = cname_to_perfs[prj]
            cur_list = [str(ele) for ele in cur_list]
            add_here += cur_list
            f.write(','.join(add_here) + '\n')
    
if __name__=='__main__':
    #test_gcF_on35(m_type="ca")
    run_m_caForest(is_subset=False)
    run_m_caForest(is_subset=True)
    run_m_gcForest()