#!/usr/bin/env python

import sys
import numpy as np

np.random.seed(1234)

import pandas as pd
from pandas import DataFrame
import time
from scipy import sparse, io
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

try:
  import cPickle as pickle
except ImportError:
  import pickle

def cross(train_path,test_path,select_feat):
    
    print('Read train and test')
    train = pd.read_csv("./data/train_no_events.csv", dtype={'device_id': np.str})
    train.drop(['age','gender'], axis=1,inplace=True)
    train_label = train["group"]
    lable_group = LabelEncoder()
    train_label = lable_group.fit_transform(train_label)

    test = pd.read_csv("./data/test_no_events.csv", dtype={'device_id': np.str})
    test["group"] = np.nan
    
    trf = open(train_path, 'rb')
    train_sp = pickle.load(trf)
    trf.close()
    
    ttf = open(test_path, 'rb')
    test_sp = pickle.load(ttf)
    ttf.close()

    train_sp = train_sp.toarray()

    if select_feat == "1":
        X_train, X_val, y_train, y_val = train_test_split(train_sp,train_label, train_size=.90, random_state=10)
        print X_train.shape
        print train.shape
        print X_val.shape
        print("# Feature Selection")
        selector = SelectPercentile(f_classif, percentile=100)
    
        selector.fit(X_train, y_train)
    
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        print X_train.shape
        print X_val.shape
        train_sp = selector.transform(train_sp)
        test_sp = selector.transform(test_sp)

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_val, y_val)
        #dtrain = xgb.DMatrix(train_sp, train_label)
        
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

        params2 = { 
             "objective": "multi:softprob",
             "num_class": 12,
             "booster" : "gbtree",
             "eval_metric": "mlogloss",
             "eta": 0.05,
             "max_depth": 6,
             "subsample": 0.7,
             "colsample_bytree": 0.7,
             "num_parallel_tree":1,
             "seed": 114,
             "silent":1,
        }

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params2, dtrain, 1000, evals=watchlist,
                early_stopping_rounds=10, verbose_eval=True)

    else:
        selector = SelectPercentile(f_classif, percentile=100)

        selector.fit(train_sp, train_label)

        #X_train = selector.transform(X_train)
        #X_val = selector.transform(X_val)

        train_sp = selector.transform(train_sp)
        test_sp = selector.transform(test_sp)

        #dtrain = xgb.DMatrix(X_train, y_train)
        #dvalid = xgb.DMatrix(X_val, y_val)
        dtrain = xgb.DMatrix(train_sp,train_label)
        dtest = xgb.DMatrix(test_sp)

        params = {
            "objective": "multi:softprob",
            "num_class": 12,
            "booster" : "gbtree",
            "eval_metric": "mlogloss",
            "eta": 0.05,
            "max_depth": 8,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "num_parallel_tree":1,
            "seed": 114,
            "silent":1,
        } 

        params2 = {
            "objective": "multi:softprob",
            "num_class": 12,
            "booster": "gblinear",
            "max_depth": 6,
            "eval_metric": "mlogloss",
            "eta": 0.05,
            "silent": 1,
            "lambda" : 3,
            "alpha": 2,
        } 

        res = xgb.cv(params2, dtrain, num_boost_round=700, nfold=5,callbacks=[xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(3)])
        print (res)

if __name__ == "__main__":
 
    if len(sys.argv) != 4:
        print "select train_path test_path,select_feat(1 is select feat other no)"
        sys.exit(0)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    select_feat = sys.argv[3]

    cross(train_path,test_path,select_feat)
    








