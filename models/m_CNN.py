'''
 implementation of CNN with keras & TensorFlow
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten

import time
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from experiments import PATH_VARIABLES
from experiments import parse_DataSet_weights

def get_m_CNN(input_size=95, layerSize=[36, 36], kernelSize=6, acti='tanh', polSize=6, opt_type="Adadelta", is_subset=False):
    """
    @param layerSize: #filters chooses for each convolutional layer
    @param kernelSize: 1d kernel size
    @param acti: activation type
    @param polSize: max pooling size  
    layerSize: 95->[36, 36], 12->[36, 36]
    """
    m_model = Sequential()
    
    m_model.add(Conv1D(filters=layerSize[0], kernel_size=(kernelSize), 
                       activation=acti, input_shape=(input_size, 1)))
    # if using full feature list, using MaxPooling
    if is_subset==False:
        m_model.add(MaxPool1D(pool_size=polSize))
    m_model.add(Dropout(0.4))
    
    for ls in layerSize[1:]:
        m_model.add(Conv1D(filters=ls, kernel_size=(kernelSize), 
                           activation=acti))
        if is_subset==False:
            m_model.add(MaxPool1D(pool_size=polSize))
        m_model.add(Dropout(0.4))
        
    m_model.add(Flatten())
    m_model.add(Dense(32, activation='sigmoid'))
    m_model.add(Dropout(0.4))
    m_model.add(Dense(2, activation='softmax'))
    m_model.compile(optimizer=opt_type, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return m_model

def test_CNN_best_activationType(acti_list=['sigmoid', 'relu', 'tanh', 'linear']):
    """
    @param acti_list: list of activation type
      results shows that both performs similar
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=False, scale_or_not=False, log_or_not=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    X_train = np.expand_dims(X_train, axis=2)
    y_train = keras.utils.to_categorical(y_train, 2)
    cname_to_acc = {}
    for acti in acti_list:
        MDL = get_m_CNN(acti=acti)
        MDL.fit(X_train, y_train, 
                batch_size=128, 
                epochs=12, 
                verbose=2, 
                validation_split=0.3)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            X_test = np.expand_dims(X_test, axis=2)
            y_test = keras.utils.to_categorical(y_test, 2)
            acc = MDL.evaluate(X_test, y_test, verbose=0)[1]
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    with open(PATH_VARIABLES.test_CNN_best_activationType, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            
# test to get best activation type
#test_CNN_best_activationType()

def test_CNN_best_OPT_type(opt_list=["RMSprop", "Adagrad", "Adadelta", "Adam"]):
    """
    @param opt_list: list of optimizers, ["RMSprop", "Adagrad", "Adadelta", "Adam"]
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, scale_or_not=False, log_or_not=False, subset=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    X_train = np.expand_dims(X_train, axis=2)
    y_train = keras.utils.to_categorical(y_train, 2)
    cname_to_acc = {}
    for opttype in opt_list:
        MDL = get_m_CNN(opt_type=opttype)
        MDL.fit(X_train, y_train, 
                batch_size=128, 
                epochs=12, 
                verbose=2, 
                validation_split=0.3)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            X_test = np.expand_dims(X_test, axis=2)
            y_test = keras.utils.to_categorical(y_test, 2)
            acc = MDL.evaluate(X_test, y_test, verbose=0)[1]
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    with open(PATH_VARIABLES.test_CNN_best_OPT_type, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            

filter_size_candidate = [list(l) for l in list(itertools.product([12, 24, 36], repeat=2))]
#[[12, 12], [12, 24], [12, 36], [24, 12], [24, 24], [24, 36], [36, 12], [36, 24], [36, 36]]

def test_CNN_best_hlayers(flr_list=filter_size_candidate, is_subset=False):
    """
    @param flr_list: list of filter sizes
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=is_subset, scale_or_not=False, log_or_not=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    X_train = np.expand_dims(X_train, axis=2)
    y_train = keras.utils.to_categorical(y_train, 2)
    cname_to_acc = {}
    for flr in flr_list:
        MDL = get_m_CNN(layerSize=flr)
        if is_subset==True:
            MDL = get_m_CNN(input_size=12, is_subset=True, layerSize=flr)
            # case for using subset feature, test using same filter size list
        MDL.fit(X_train, y_train, 
                batch_size=128, 
                epochs=16, 
                verbose=2, 
                validation_split=0.3)
        for i in xrange(len(myDS.p35_list)):
            [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
            X_test = np.expand_dims(X_test, axis=2)
            y_test = keras.utils.to_categorical(y_test, 2)
            acc = MDL.evaluate(X_test, y_test, verbose=0)[1]
            if cpath in cname_to_acc:
                cur_list = cname_to_acc[cpath]
                cur_list.append(acc)
                cname_to_acc[cpath] = cur_list
            else:
                cname_to_acc[cpath] = [acc]
                
    print cname_to_acc
    with open(PATH_VARIABLES.test_CNN_best_filterSize, 'a') as f:
        for prj in cname_to_acc:
            prj_name = prj.split('/')[-1]
            prj_name = prj_name.split('.csv')[0]
            add_here = [prj_name]
            cur_list = cname_to_acc[prj]
            for acc in cur_list:
                add_here.append(str(acc))
            
            add_here = ','.join(add_here)
            f.write(add_here + '\n')
            
#report the accuracy for filter layer size
#test_CNN_best_hlayers()

def run_m_CNN(is_subset=False):
    """
    training & prediction
    record:
      classifier name, training time, accuracy, error, precision, recall, F-measure, AUROC
    """
    path_train = PATH_VARIABLES.path9
    path_test = PATH_VARIABLES.path35
    myDS = parse_DataSet_weights.parse_DataSet_weights(path_train, path_test, subset=is_subset, scale_or_not=False, log_or_not=False)
    [X_train, y_train, _] = myDS.prepare_DS_for_training_rf()
    X_train = np.expand_dims(X_train, axis=2)
    y_train = keras.utils.to_categorical(y_train, 2)
    if is_subset==False:
        clr_name = "CNN"
        rst_file_path = PATH_VARIABLES.cv_5fold_repo_CNN
        MDL = get_m_CNN()
    else:
        clr_name = "CNNsubset"
        rst_file_path = PATH_VARIABLES.cv_5fold_repo_CNNs
        MDL = get_m_CNN(input_size=12, is_subset=True)
    #training time
    t_start = time.time()
    MDL.fit(X_train, y_train, 
            batch_size=128, 
            epochs=30, 
            verbose=2, 
            #sample_weight=sweights, 
            validation_split=0.3)
    t_end = time.time()
    t_time = t_end - t_start
    cname_to_perfs = {}
    for i in xrange(len(myDS.p35_list)):
        [X_test, y_test, cpath, _] = myDS.get_ith_test_data_rf(i)
        X_test = np.expand_dims(X_test, axis=2)
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

#run_m_CNN()
if __name__=='__main__':
    #test_CNN_best_hlayers(is_subset=True)
    run_m_CNN(is_subset=False)
    run_m_CNN(is_subset=True)