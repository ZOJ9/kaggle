#!/usr/bin/env python

import os
import sys
import numpy as np

np.random.seed(1234)

import pandas as pd
from pandas import DataFrame
import time
from scipy import sparse, io
from sklearn.preprocessing import LabelEncoder

import datetime

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

try:
  import cPickle as pickle
except ImportError:
  import pickle

def pro_data(train_path,test_path,outfile,valfile):
    
    print('Read train and test')
    train = pd.read_csv("./data/train_events.csv", dtype={'device_id': np.str})
    train.drop(['age','gender'], axis=1,inplace=True)
    train_label = train["group"]
    lable_group = LabelEncoder()
    train_label = lable_group.fit_transform(train_label)

    #test = pd.read_csv("/home/wangzhe/data/gender_age_test.csv", dtype={'device_id': np.str})
    #test["group"] = np.nan
    
    trf = open(train_path, 'rb')
    train_sp = pickle.load(trf)
    trf.close()
    
    #ttf = open(test_path, 'rb')
    #test_sp = pickle.load(ttf)
    #ttf.close()

    X_train, X_val, y_train, y_val = train_test_split(train_sp,train_label, train_size=.90, random_state=2016)

    print "train data : %d %d"%(X_train.shape[0],X_train.shape[1])
    print "val data : %d %d"%(X_val.shape[0],X_val.shape[1])    
 
    vlfw = open(valfile,'w')
    trfw = open(outfile,'w')

    for idx,feats in enumerate(X_train):
        if idx % 10000 == 0:
            print "running : %d"%(idx)
        cols = sorted(feats.nonzero()[1])
        cols = map(lambda x:str(x+1)+":1",cols)
        label = y_train[idx]
        outline = str(label) + " " + " ".join(cols)
        trfw.write(outline)
        trfw.write("\n")

    for idx,feats in enumerate(X_val):
        cols = sorted(feats.nonzero()[1])
        cols = map(lambda x:str(x+1)+":1",cols)
        label = y_val[idx]
        outline = str(label) + " " + " ".join(cols)
        vlfw.write(outline)
        vlfw.write("\n")
    print "run done!"

def train_model_val(train_file,val_file):
    
    model_file = "./data/model"
    pred_file = "./data/pred"

    cmd = "./lib/liblinear-multicore-2.11-1/train -s 0 -B 1 -n 12 -e 1e-5 -c 1.5 %s %s"%(train_file,model_file)
    print("train model")
    ret = os.system(cmd)
    if ret != 0:
        print "train model error"
        sys.exit(0)
    
    pred_cmd = "./lib/liblinear-multicore-2.11-1/predict -b 1 %s %s %s"%(val_file,model_file,pred_file)
    print("pred model")
    ret = os.system(pred_cmd)
    if ret != 0:
        print "pred error"
        sys.exit(0)
    return pred_file

def metric(pred):

    fr = open(pred)
    
    label = []
    pred_vals = []

    for idx,line in enumerate(fr):
        if idx == 0:
            print line
            continue
        items = line.strip().split(" ")
        label.append(float(items[0]))
        p = map(float,items[1:])
        pred_vals.append(p)

    ####################################
    print "metric"
    y_true = np.array(label)
    y_pred = np.array(pred_vals)
    eps=1e-15

    predictions = np.clip(y_pred, eps, 1 - eps)

    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    multi_log_loss =  -1.0 / rows * vsota
    print "loss : %f"%(multi_log_loss)
    


if __name__ == "__main__":
 
    if len(sys.argv) != 5:
        print "select train_path test_path, outfile valfile"
        sys.exit(0)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    outfile = sys.argv[3]
    valfile = sys.argv[4]

    print "pro data ........"
    pro_data(train_path,test_path,outfile,valfile)
    
    print "train and pred val data ....."
    pred_file = train_model_val(outfile,valfile)

    print "metric pred result ...."
    metric(pred_file)    








