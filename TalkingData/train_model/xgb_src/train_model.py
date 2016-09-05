#!/usr/bin/env python

import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import sys
import numpy as np

np.random.seed(1234)

import pandas as pd
from pandas import DataFrame
import time
from scipy import sparse, io

import xgboost as xgb
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

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

def cross(Xtrain,y):

    params = {
            "objective": "multi:softprob",
            "num_class": 12,
            "booster": "gblinear",
            "eval_metric": "mlogloss",
            "eta": 0.05,
            "silent": 1,
            "lambda" : 3,
            "alpha": 2,
    }

    X_train, X_val, y_train, y_val = train_test_split(Xtrain,y, train_size=.90, random_state=10)
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val) 

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, 1000, evals=watchlist,early_stopping_rounds=10, verbose_eval=True) 

def train(Xtrain,y,test,Xtest,num_boost_round):

    Xtrain = Xtrain.toarray()
    Xtest = Xtest.toarray()

    params = {
            "objective": "multi:softprob",
            "num_class": 12,
            "booster": "gblinear",
            "eval_metric": "mlogloss",
            "eta": 0.05,
            "silent": 1,
            "lambda" : 3,
            "alpha": 2,
    }
 
    dtrain = xgb.DMatrix(Xtrain, y)
    dtest = xgb.DMatrix(Xtest)

    watchlist = [(dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,verbose_eval=True) 

    print("Predict test set...")
    test_prediction = gbm.predict(dtest)
    return test_prediction

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
    train_src_path = "./data/train_events.csv"
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
        print "no events cross"
        cross(Xtrain_no.toarray(),y_no)
        print "events cross"
        cross(Xtrain.toarray(),y)
    elif s == "2":
        ######### 1.pred no events
        print "test_no shape: (%d,%d)"%(test_no.shape[0],test_no.shape[1])
        print "test_no_sp shape: (%d,%d)"%(Xtest_no.shape[0],Xtest_no.shape[1])

        lable_group_no = LabelEncoder().fit(label_no)
        train_label_no = lable_group_no.fit_transform(label_no)

        y_pred_no = train(Xtrain_no,y_no,test_no,Xtest_no,111)        

        pred_no = pd.DataFrame(y_pred_no, index = test_no.device_id, columns=lable_group_no.classes_)
        print pred_no
        pred_no.to_csv("./xgb_src/data/xgb_no_events.csv",index=True)

        ######### 2.pred events
        print "test shape: (%d,%d)"%(test.shape[0],test.shape[1])
        print "test_sp shape: (%d,%d)"%(Xtest.shape[0],Xtest.shape[1])
        lable_group = LabelEncoder().fit(label)
        train_label = lable_group.fit_transform(label)

        y_pred = train(Xtrain,y,test,Xtest,31)

        pred = pd.DataFrame(y_pred, index = test.device_id, columns=lable_group.classes_)
        print pred
        pred.to_csv("./xgb_src/data/xgb_events",index=True) 

        submmit = pd.concat((pred_no,pred),axis=0)
        print submmit
        print submmit.shape
        submmit.to_csv("./../submmit_data/submmit_0822_2.csv", index=True)
    else:
        print "error cross(1)_pred(2)" 
        sys.exit(0)




