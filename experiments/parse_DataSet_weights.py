'''
 training with weights, mainly used for non-deep models
 "rf" stands for randomforest in the orignal design
'''

import os
from math import log10
from sklearn.preprocessing import minmax_scale
import numpy as np

SUBSET_ORIGINAL_PMT = [2, 3, 5, 4, 0, 8, 23, 15, 13, 17, 7, 6]

class parse_DataSet_weights(object):
    def __init__(self, path9, path35, log_or_not = False, scale_or_not = False, subset = True):
        self.path9 = path9 # training set directory
        self.path35 = path35 # testing set directory
        self.log_or_not = log_or_not # if true, take log to runtime feature
        self.scale_or_not = scale_or_not # True, then do minmaxscale
        self.subset = subset # True, use a subset of features
        self.p9_list = []
        self.p35_list = []
        #
        self.rtype_set = set() # record for train add another on
        self.mutator_set = set()
        #for train, test can be done on the fly
        self.data_sets_train = []
        self.data_sizes_train = []
        self.check_killed_no_cov_train = []
        self.DS_sum_info_train = []
        #for test, one by one
        #run
        self.run()
        return
    
    def get_paths(self):
        #training
        fnames = os.listdir(self.path9)
        for i in xrange(len(fnames)):
            fnames[i] = self.path9 + '/' + fnames[i]
        self.p9_list = list(fnames)
        #testing
        fnames = os.listdir(self.path35)
        for i in xrange(len(fnames)):
            fnames[i] = self.path35 + '/' + fnames[i]
        self.p35_list = list(fnames)
        return
    
    #case 2, deep
    def parse_data_sets_train_rf(self):
        for i in xrange(len(self.p9_list)):
            cpath = self.p9_list[i]
            cdata = []
            ds = [0, 0, 0]
            ck = 0
            with open(cpath) as f:
                for ln in f:
                    linfo = ln.strip().split(',')
                    if linfo[0]!='Detected':
                        for i in xrange(3, len(linfo)):
                            #if linfo[i]!='?':
                            #    linfo[i] = float(linfo[i])
                            if self.log_or_not:
                                if linfo[i]!='?':
                                    if (i==3 or i==4 or i==5 or i==6) and float(linfo[i])>1.0:
                                    #if (i==3) and float(linfo[i])>1.0:
                                        #pass
                                        linfo[i] = log10(float(linfo[i])) / log10(1.2)
                                    else:
                                        linfo[i] = float(linfo[i])
                            else:
                                if linfo[i]!='?':
                                    linfo[i] = float(linfo[i])
                                
                        cdata.append(linfo)
                        self.rtype_set.add(linfo[2])
                        self.mutator_set.add(linfo[1])
                        if linfo[0]=='true' and linfo[3]==0: ck += 1
                        if linfo[0]=='false':
                            ds[0] += 1
                        else:
                            ds[1] += 1
                        ds[2] += 1
                        
            self.data_sets_train.append(cdata)
            self.data_sizes_train.append(ds)
            self.check_killed_no_cov_train.append(ck)
            #self.rtype_set.add('NORECORD') # add one for all other type
        return
    
    # case 1 and case 2
    def category_to_numeric_train(self):
        '''
        category to integer, except '?'
        '''
        rtypes = list(self.rtype_set)
        mutators = list(self.mutator_set)
        for i in xrange(len(self.data_sets_train)):
            sum_info = [[0, 0, 0] for _ in xrange(len(self.data_sets_train[0][0]))]
            for j in xrange(len(self.data_sets_train[i])):
                for k in xrange(len(self.data_sets_train[i][j])):
                    if k==0:
                        if self.data_sets_train[i][j][k]=='true':
                            self.data_sets_train[i][j][k] = 1
                        else:
                            self.data_sets_train[i][j][k] = 0
                    elif k==1:
                        mt_ind = mutators.index(self.data_sets_train[i][j][k])
                        self.data_sets_train[i][j][k] = mt_ind
                    elif k==2:
                        rt_ind = rtypes.index(self.data_sets_train[i][j][k])
                        self.data_sets_train[i][j][k] = rt_ind
                    else:
                        if self.data_sets_train[i][j][k]!='?':
                            self.data_sets_train[i][j][k] = float(self.data_sets_train[i][j][k])
                    
                    if self.data_sets_train[i][j][k]!='?':
                        if self.data_sets_train[i][j][0]==0:
                            sum_info[k][0] += self.data_sets_train[i][j][k]
                        else:
                            sum_info[k][1] += self.data_sets_train[i][j][k]
                        sum_info[k][2] += self.data_sets_train[i][j][k]
                    
            if self.data_sizes_train[i][0]==0:
                sum_info = [[0, float(aa[1])/self.data_sizes_train[i][1], 
                             float(aa[2])/self.data_sizes_train[i][2]] for aa in sum_info]
            elif self.data_sizes_train[i][1]==0:
                sum_info = [[float(aa[0])/self.data_sizes_train[i][0], 0, 
                             float(aa[2])/self.data_sizes_train[i][2]] for aa in sum_info]
            else:
                sum_info = [[float(aa[0])/self.data_sizes_train[i][0], float(aa[1])/self.data_sizes_train[i][1], 
                             float(aa[2])/self.data_sizes_train[i][2]] for aa in sum_info]
            #sum_info = [[float(aa[0])/self.data_sizes_train[i][0], float(aa[1])/self.data_sizes_train[i][1], 
            #             float(aa[2])/self.data_sizes_train[i][2]] for aa in sum_info]
            self.DS_sum_info_train.append(sum_info)
        return
    
    def prepare_DS_for_training_rf(self):
        x_train = []
        y_train = []
        train_lens = []
        for ii in xrange(len(self.data_sets_train)):
            dsii = self.get_train_set(self.data_sets_train[ii], ii, self.scale_or_not)
            for idx in xrange(len(dsii[0])):
                #2, 3, 0, 66, 43, 84, 59, 75, 19, 1, 4, 5
                if self.subset == True:
                    x_added = []
                    for kk in xrange(len(SUBSET_ORIGINAL_PMT)):
                        x_added.append(dsii[0][idx][SUBSET_ORIGINAL_PMT[kk]])
                    """
                    here_CA = x_added[5]
                    here_CE = x_added[6]
                    if here_CE != 0:
                        here_Instability = float(here_CA) / here_CE
                    else:
                        here_Instability = 0
                    x_added.append(here_Instability)
                    """
                if self.subset == False:
                    x_added = list(dsii[0][idx])
                x_train.append(x_added)
                y_train.append(dsii[1][idx])
                
            train_lens.append(len(dsii[0]))
            
        sweights = self.get_sample_weight(train_lens)
        x_train = np.array(x_train)
        y_train = np.array(y_train).T
        sweights = np.array(sweights)
        data_seqs = [x_train, y_train, sweights]
        return data_seqs
    
    def get_train_set(self, DS, iNd, scale):
        ds = list(DS)
        tx = []
        ty = []
        for i in xrange(len(ds)):
            for j in xrange(len(ds[i])):
                if ds[i][j]=='?':
                    ds[i][j] = self.DS_sum_info_train[iNd][j][0] if ds[i][0]==0 else self.DS_sum_info_train[iNd][j][1]
            tx.append(ds[i][1:])
            ty.append(ds[i][0])
        #scale used for deep model
        if scale==True:
            tx0 = [t[:2] for t in tx]
            tx1 = [t[2:] for t in tx]
            tx1 = minmax_scale(tx1, (-1, 1)).tolist()
            tx = []
            for i in xrange(len(tx0)):
                tx.append(tx0[i]+tx1[i])
            
        return (tx, ty)
    
    def get_sample_weight(self, train_lens):
        sm = sum(train_lens)
        invs = [float(sm)/tl for tl in train_lens]
        rtw = []
        for i in xrange(len(train_lens)):
            advec = [invs[i]/train_lens[i] for _ in xrange(train_lens[i])]
            rtw += advec
        return rtw
    
    
    '''
    test part, get one by one
    '''
    def get_ith_test_data_rf(self, ith):
        [cpath, cdata, ds, ck] = self.parse_ith_test_data_rf(ith)
        [cdata, sum_info] = self.category_to_numeric_test(cdata, ds)
        [x_test, y_test] = self.get_test_set(cdata, sum_info, self.scale_or_not)
        for ii in xrange(len(x_test)):
            #2, 3, 0, 66, 43, 84, 59, 75, 19, 1, 4, 5
            if self.subset == True:
                rpl = []
                for kk in xrange(len(SUBSET_ORIGINAL_PMT)):
                    rpl.append(x_test[ii][SUBSET_ORIGINAL_PMT[kk]])
                """
                here_CA = rpl[5]
                here_CE = rpl[6]
                if here_CE != 0:
                    here_Instability = float(here_CA) / here_CE
                else:
                    here_Instability = 0
                rpl.append(here_Instability)
                """
            if self.subset == False:
                rpl = list(x_test[ii])
            x_test[ii] = rpl
        
        print len(y_test)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        return [x_test, y_test, cpath, ds]
    
    def parse_ith_test_data_rf(self, ith):
        for _ in xrange(len([1])):
            cpath = self.p35_list[ith]
            cdata = []
            ds = [0, 0, 0]
            ck = 0
            with open(cpath) as f:
                for ln in f:
                    linfo = ln.strip().split(',')
                    if linfo[0]!='Detected':
                        for i in xrange(3, len(linfo)):
                            #if linfo[i]!='?':
                            #    linfo[i] = float(linfo[i])
                            if self.log_or_not:
                                if linfo[i]!='?':
                                    if (i==3 or i==4 or i==5 or i==6) and float(linfo[i])>1.0:
                                    #if (i==3) and float(linfo[i])>1.0:
                                        linfo[i] = log10(float(linfo[i])) / log10(1.2)
                                    else:
                                        linfo[i] = float(linfo[i])
                            else:
                                if linfo[i]!='?':
                                    linfo[i] = float(linfo[i])
                                
                        cdata.append(linfo)
                        if linfo[0]=='true' and linfo[3]==0: ck += 1
                        if linfo[0]=='false':
                            ds[0] += 1
                        else:
                            ds[1] += 1
                        ds[2] += 1
                        
        return [cpath, cdata, ds, ck]
    
    def category_to_numeric_test(self, cdata1, ds):
        '''
        category to integer, except '?'
        '''
        cdata = list(cdata1)
        rtypes = list(self.rtype_set)
        mutators = list(self.mutator_set)
        for _ in xrange(len([1])):
            sum_info = [[0, 0, 0] for _ in xrange(len(cdata[0]))]
            for j in xrange(len(cdata)):
                for k in xrange(len(cdata[j])):
                    if k==0:
                        if cdata[j][k]=='true':
                            cdata[j][k] = 1
                        else:
                            cdata[j][k] = 0
                    elif k==1:
                        if cdata[j][k] in mutators:
                            mt_ind = mutators.index(cdata[j][k])
                            cdata[j][k] = mt_ind
                        else:
                            cdata[j][k] = len(mutators)
                    elif k==2:
                        if cdata[j][k] in rtypes:
                            rt_ind = rtypes.index(cdata[j][k])
                            cdata[j][k] = rt_ind
                        else:
                            cdata[j][k] = len(rtypes)
                    else:
                        if cdata[j][k]!='?':
                            cdata[j][k] = float(cdata[j][k])
                    
                    if cdata[j][k]!='?':
                        if cdata[j][0]==0:
                            sum_info[k][0] += cdata[j][k]
                        else:
                            sum_info[k][1] += cdata[j][k]
                        sum_info[k][2] += cdata[j][k]
                    
            if ds[0]==0:
                sum_info = [[0, float(aa[1])/ds[1], 
                             float(aa[2])/ds[2]] for aa in sum_info]
            elif ds[1]==0:
                sum_info = [[float(aa[0])/ds[0], 0, 
                             float(aa[2])/ds[2]] for aa in sum_info]
            else:
                sum_info = [[float(aa[0])/ds[0], float(aa[1])/ds[1], 
                             float(aa[2])/ds[2]] for aa in sum_info]
            #sum_info = [[float(aa[0])/ds[0], float(aa[1])/ds[1], 
            #             float(aa[2])/ds[2]] for aa in sum_info]
            #self.DS_sum_info_train.append(sum_info)
        return [cdata, sum_info]
    
    def get_test_set(self, cdata, sum_info, scale):
        dataset = list(cdata)
        tx = []
        ty = []
        for i in xrange(len(dataset)):
            for j in xrange(len(dataset[i])):
                if dataset[i][j]=='?':
                    dataset[i][j] = sum_info[j][2]
            tx.append(dataset[i][1:])
            ty.append(dataset[i][0])
            
        if scale:
            tx0 = [t[:2] for t in tx]
            tx1 = [t[2:] for t in tx]
            tx1 = minmax_scale(tx1, (-1, 1)).tolist()
            tx = []
            for i in xrange(len(tx0)):
                tx.append(tx0[i]+tx1[i])
            
        return (tx, ty)
    
    
    def run(self):
        self.get_paths()
        self.parse_data_sets_train_rf()
        self.category_to_numeric_train()
        #self.prepare_DS_for_training_rf()
    
if __name__=='__main__':
    path9 = "/home/mdy/9BaseRaw"
    path35 = "/home/mdy/35PrjRaw"
    #Deep
    t2 = parse_DataSet_weights(path9, path35, scale_or_not=True)
    t2.get_ith_test_data_rf(0)
    