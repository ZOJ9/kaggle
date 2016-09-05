#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

try:
  import cPickle as pickle
except ImportError:
  import pickle


def load_data(train_src_path,train_sp_path):
    
    print('Read train and test')
    train = pd.read_csv(train_src_path, dtype={'device_id': np.str})
    label = train["group"]
    train.drop(['age','gender'], axis=1,inplace=True)
    train_label = train["group"]
    lable_group = LabelEncoder().fit(train_label)
    train_label = lable_group.transform(train_label)
    nclasses = len(lable_group.classes_)
    
    trf = open(train_sp_path, 'rb')
    train_sp = pickle.load(trf)
    trf.close()
    
    train.drop(['device_id','group'], axis=1,inplace=True)
 
    return train_sp,train_label,nclasses,label

def load_test_data(test_src_path,test_sp_path):

    print('Read train and test')
    test = pd.read_csv(test_src_path, dtype={'device_id': np.str})

    ttf = open(test_sp_path, 'rb')
    test_sp = pickle.load(ttf)
    ttf.close()

    return test_sp,test

def score(Xtrain,y,clf, random_state = 0):
    kf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest,:] = clf.predict_proba(Xte)
        # Downsize to one fold only for kernels
        return log_loss(yte, pred[itest, :])
    return log_loss(y, pred)

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print "train_no_path train_path test_no_path test_path cross(1)_pred(2)"
        sys.exit()
    
    train_no_sp_path = sys.argv[1]
    train_sp_path = sys.argv[2]
    test_no_sp_path = sys.argv[3]
    test_sp_path = sys.argv[4]
    s = sys.argv[5]
 
    train_no_src_path = "./data/train_no_events.csv"
    #train_src_path = "./data/train_events.csv"
    train_src_path = "./data/gender_age_train.csv"
    test_no_src_path =  "./data/test_no_events.csv"
    test_src_path = "./data/test_events.csv"

    Xtrain_no,y_no,nclasses_no,label_no = load_data(train_no_src_path,train_no_sp_path)
    Xtrain,y,nclasses,label = load_data(train_src_path,train_sp_path)
    print "train_no shape:"
    print Xtrain_no.shape
    print "train shape:"
    print Xtrain.shape

    Xtest_no,test_no = load_test_data(test_no_src_path,test_no_sp_path)
    Xtest,test = load_test_data(test_src_path,test_sp_path)

    if s == "1":
        print "10 n_folds...."
        loss_no = score(Xtrain_no,y_no,LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))
        loss = score(Xtrain,y,LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))
        print "10 n_foldes no events result : %f"%(loss_no)
        print "10 n_foldes events result : %f"%(loss)
    elif s == "2":
        ######### 1.pred no events
        print "test_no shape: (%d,%d)"%(test_no.shape[0],test_no.shape[1])
        print "test_no_sp shape: (%d,%d)"%(Xtest_no.shape[0],Xtest_no.shape[1])
        device_id_no = test_no.device_id
        targetencoder_no = LabelEncoder().fit(label_no)
        clf_no = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
        clf_no.fit(Xtrain_no, y_no)
        y_pred_no = clf_no.predict_proba(Xtest_no)
        pred_no = pd.DataFrame(y_pred_no, index = test_no.device_id, columns=targetencoder_no.classes_)
        print pred_no
        pred_no.to_csv("./sklearn_train/data/lr_pred_no_events.csv",index=True)

        ######### 2.pred events
        print "test shape: (%d,%d)"%(test.shape[0],test.shape[1])
        print "test_sp shape: (%d,%d)"%(Xtest.shape[0],Xtest.shape[1])
        targetencoder = LabelEncoder().fit(label)
        clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
        clf.fit(Xtrain, y)
        y_pred = clf.predict_proba(Xtest)
        pred = pd.DataFrame(y_pred, index = test.device_id, columns=targetencoder.classes_)
        pred.to_csv("./sklearn_train/data/lr_pred_events.csv",index=True)
        print pred
        submmit = pd.concat((pred_no,pred),axis=0)
        print submmit
        print submmit.shape
        submmit.to_csv("./submmit_0822.csv", index=True)
        
    else:
        print "error cross(1)_pred(2)" 
        sys.exit(0)




