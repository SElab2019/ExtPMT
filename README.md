# ExtPMT #

ML code and Data for "An Extensive Study on Cross-project Predictive Mutation Testing"

### Requirements ###

* python 2.7
* keras 2.2.4
* tensorflow 1.10.0
* scikit-learn 0.20.1
* for dependencies required for running Deep Forest, please refer to [gcForest](https://github.com/kingfengji/gcForest)

### Data ###

All data used in the experiments can be found in */docs/Data.tar.gz*

### Run ###

You can run 5-fold cross validation on 654 subjects using preferred ml model(s) through following steps:

* Extract */docs/Data.tar.gz* to a directory (name it as *DS_All*), specify the path to *DS_All* in */experiments/PATH_VARIABLES.py* line 17 (*path1k_pls*).
* Create two empty directories to put training set (*DS_Train*) and testing set (*DS_Test*) during run time, specify path to *DS_train* and *DS_Test* in */experiments/PATH_VARIABLES.py* line 14 (*path9*) and line 15 (*path35*).
* In line 9 of */experiments/cv5fold.sh*, specify the path to */experiments/cv5fold_file_copy.py*
* In line 11 of */experiments/cv5fold.sh*, specify the ml model(s) to run cross validation, available models in our implementation can be found in */models*.
* Then cd to */experiments*, run 5-fold cross validation by *sh cv5fold.sh*.

### Report ###
After running 5-fold cross validation, report can be found in */expresults* with the following format.  

| subject_name | model_name | training_time (s) | accuracy | error | precision | recall | F1-score | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixture-factory | CNN | 8357.3336 | 0.7834 | 0.2166 | 0.7635 |0.9950 | 0.8640 | 0.6947 |
| pengyifan-commons | CNN | 8357.3336 | 0.8992 | 0.1008 | 0.7869 |0.9971 | 0.8796 | 0.9386 |
