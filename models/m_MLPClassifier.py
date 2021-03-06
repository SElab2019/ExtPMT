'''
 using neural network implementation from sklearn
 limite hidden layer to 2
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from experiments import parse_DataSet_weights, PATH_VARIABLES

def get_m_MLPClassifier(hidden_layer_sizes = (5, 4)):
    """
    choose hidden_layer_sizes to get better performance
    The default solver 'adam' works pretty well on relatively large datasets
    """
    my_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    return my_clf

def test_find_best_hidden_layers(hyr_list):
    """
    @param hyr_list: list of hidden layer size settings
    first layer: [5, 15, 25], also used if there is only one layer
    second layer: [2, 4, 8]
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=True, scale_or_not=True, log_or_not=True)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    cname_to_acc = {}
    for hyr in hyr_list:
        print hyr
        MDL = get_m_MLPClassifier(hyr)
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
            
# find best layer settings
candidate_layer_settings = [(5), 
                            (15), 
                            (25), 
                            (5, 2), 
                            (5, 4), 
                            (15, 2), 
                            (15, 4), 
                            (15, 8), 
                            (25, 2), 
                            (25, 4), 
                            (25, 8)
                            ]
#test_find_best_hidden_layers(candidate_layer_settings)

def run_m_MLPClassifier():
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=True, scale_or_not=True, log_or_not=True)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    MDL = get_m_MLPClassifier()
    #training time
    t_start = time.time()
    MDL.fit(X_train, y_train)
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
        cname_to_perfs[prj_name] = ['MLPClassifier', t_time, cur_accuracy, cur_error, cur_precision, cur_recall, cur_f1, cur_auroc]
        
    with open(PATH_VARIABLES.cv_5fold_repo_NN, 'a') as f:
        for prj in cname_to_perfs:
            add_here = [prj]
            cur_list = cname_to_perfs[prj]
            cur_list = [str(ele) for ele in cur_list]
            add_here += cur_list
            f.write(','.join(add_here) + '\n')
            
#run_m_MLPClassifier()
if __name__=='__main__':
    #test_find_best_hidden_layers(candidate_layer_settings)
    run_m_MLPClassifier()