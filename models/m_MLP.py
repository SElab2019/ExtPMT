'''
 deep MLP model, implemented with keras & TensorFlow
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import itertools
import time

from experiments import PATH_VARIABLES
from experiments import parse_DataSet_weights

def get_m_MLP(input_size=95, hlayers=[95, 45, 45], activation_type='sigmoid', DP_rate=0.4, opt_type = "Adagrad"):
    """
    @param input_size: input size to MLP
    @param hlayers: list of hidden layer size
    @param activation_type: activation type
    @param DP_rate: dropout parameters, here is fixed for each hidden layers
    @return: a MLP model, input-hlayers[0]-...-hlayers[-1]-output
    default: [95, 45, 45] & [24, 24, 24]
    """
    # input
    input_img = Input(shape=(input_size,))
    # hidden
    encoded = Dense(hlayers[0], activation=activation_type)(input_img)
    encoded = Dropout(DP_rate)(encoded)
    for i in xrange(1, len(hlayers)):
        encoded = Dense(hlayers[i], activation=activation_type)(encoded)
        encoded = Dropout(DP_rate)(encoded)
    # output
    decoded = Dense(2, activation='softmax')(encoded)
    m_MLP = Model(input_img, decoded)
    # compile
    m_MLP.compile(optimizer=opt_type, loss='categorical_crossentropy', metrics=['accuracy'])
    return m_MLP

def test_MLP_best_activationType(acti_list=['sigmoid', 'relu', 'tanh', 'linear']):
    """
    @param acti_list: activation type
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=True, log_or_not=True, subset=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    y_train = keras.utils.to_categorical(y_train, 2)
    cname_to_acc = {}
    for acti in acti_list:
        MDL = get_m_MLP(activation_type=acti)
        MDL.fit(X_train, y_train, 
                batch_size=128, 
                epochs=120, 
                verbose=2, 
                validation_split=0.3)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            y_test = keras.utils.to_categorical(y_test, 2)
            acc = MDL.evaluate(X_test, y_test, verbose=0)[1]
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    with open(PATH_VARIABLES.test_MLP_best_activationType, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            
# get best activation type
#test_MLP_best_activationType()

def test_MLP_best_OPT_type(opt_list=["RMSprop", "Adagrad", "Adadelta", "Adam"]):
    """
    @param opt_list: list of optimizers, ["RMSprop", "Adagrad", "Adadelta", "Adam"]
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=True, log_or_not=True, subset=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    y_train = keras.utils.to_categorical(y_train, 2)
    cname_to_acc = {}
    for opttype in opt_list:
        MDL = get_m_MLP(opt_type=opttype)
        MDL.fit(X_train, y_train, 
                batch_size=128, 
                epochs=120, 
                verbose=2, 
                validation_split=0.3)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            y_test = keras.utils.to_categorical(y_test, 2)
            acc = MDL.evaluate(X_test, y_test, verbose=0)[1]
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    with open(PATH_VARIABLES.test_MLP_best_OPT_type, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')

# 95, 45
candidates_layer_settings = []
ls2 = [list(l) for l in list(itertools.product([95, 45], repeat=2))]
ls3 = [list(l) for l in list(itertools.product([95, 45], repeat=3))]
candidates_layer_settings += ls2 + ls3
# [[95, 95], [95, 45], [45, 95], [45, 45], [95, 95, 95], [95, 95, 45], [95, 45, 95], [95, 45, 45], [45, 95, 95], [45, 95, 45], [45, 45, 95], [45, 45, 45]]

# 23, 12
candidates_layer_settings_subset = []
ls2_subset = [list(l) for l in list(itertools.product([24, 12], repeat=2))]
ls3_subset = [list(l) for l in list(itertools.product([24, 12], repeat=3))]
candidates_layer_settings_subset += ls2_subset + ls3_subset
# [[24, 24], [24, 12], [12, 24], [12, 12], [24, 24, 24], [24, 24, 12], [24, 12, 24], [24, 12, 12], [12, 24, 24], [12, 24, 12], [12, 12, 24], [12, 12, 12]]

def test_MLP_best_hlayer_setting(hlyr_list=candidates_layer_settings, is_subset=False):
    """
    @param hlyr_list: list of hidden layer size, ranges from 2 to 3 layers
    can be used to reduce sizes
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=True, log_or_not=True, subset=is_subset)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    y_train = keras.utils.to_categorical(y_train, 2)
    cname_to_acc = {}
    for hlyr in hlyr_list:
        MDL = get_m_MLP(hlayers=hlyr)
        if is_subset:
            MDL = get_m_MLP(input_size=12, hlayers=hlyr)
            # used for subset setting
        MDL.fit(X_train, y_train, 
                batch_size=128, 
                epochs=120, 
                verbose=2, 
                validation_split=0.3)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            y_test = keras.utils.to_categorical(y_test, 2)
            acc = MDL.evaluate(X_test, y_test, verbose=0)[1]
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    with open(PATH_VARIABLES.test_MLP_best_hlayers, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            
# test to get best hidden layer setting
#test_MLP_best_hlayer_setting()

def run_m_MLP(is_subset=False):
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=is_subset, scale_or_not=True, log_or_not=True)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    y_train = keras.utils.to_categorical(y_train, 2)
    if is_subset==True:
        clr_name = "MLPsubset"
        rst_file_path = PATH_VARIABLES.cv_5fold_repo_MLPs
        MDL = get_m_MLP(input_size=12, hlayers=[24, 24, 24])
    else:
        clr_name = "MLP"
        rst_file_path = PATH_VARIABLES.cv_5fold_repo_MLP
        MDL = get_m_MLP()
    #training time
    t_start = time.time()
    MDL.fit(X_train, y_train, 
            batch_size=128, 
            epochs=50, 
            verbose=2, 
            validation_split=0.3)
    t_end = time.time()
    t_time = t_end - t_start
    cname_to_perfs = {}
    for i in xrange(len(myDS.p35_list)):
        [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
        y_pred_prob = MDL.predict(X_test)
        y_pred = []
        y_score = []
        for prob in y_pred_prob:
            if prob[0]>prob[1]:
                y_pred.append(0)
            else:
                y_pred.append(1)
            y_score.append(prob[1])
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
            
        
#run_m_MLP()
if __name__=='__main__':
    #test_MLP_best_OPT_type()
    #test_MLP_best_hlayer_setting(hlyr_list=candidates_layer_settings, is_subset=False)
    #test_MLP_best_hlayer_setting(hlyr_list=candidates_layer_settings_subset, is_subset=True)
    run_m_MLP(is_subset=False)
    run_m_MLP(is_subset=True)