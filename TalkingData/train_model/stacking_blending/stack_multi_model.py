#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from pandas import DataFrame
import time
from scipy import sparse, io

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

import xgboost as xgb
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

try:
  import cPickle as pickle
except ImportError:
  import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop,Nadam
from keras.utils import np_utils


def nnet_train(X_train,y_train,X_test,nb_epoch):
    
    nb_classes = 12
    
    model = Sequential()
    model.add(Dense(500, input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    #model.add(Dense(200, init='uniform'))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))

    model.add(Dense(12, init='uniform'))
    model.add(Activation('softmax')) 
    nadam = Nadam(lr=1e-4)   
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, class_mode='categorical')

    model.fit(X_train.toarray(), y_train, nb_epoch=nb_epoch, batch_size=150)  

    y_preds = model.predict(X_test.toarray())
    
    return y_preds

def xgb_train(Xtrain,y,Xtest,num_boost_round):

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
    xgb_preds = gbm.predict(dtest)
    return xgb_preds

def load_data(train_src_path,train_sp_path):
    
    print('Read train and test')
    train = pd.read_csv(train_src_path, dtype={'device_id': np.str})
    label = train["group"]
    train.drop(['age','gender'], axis=1,inplace=True)
    train_label = train["group"]
    lable_group = LabelEncoder().fit(train_label)
    train_label = lable_group.transform(train_label)

    train["value"] = np.ones(len(train.values))
    nnet_label = pd.pivot_table(train,index=train.index, columns='group',
                                    values='value',fill_value=0)

    nclasses = len(lable_group.classes_)
    
    trf = open(train_sp_path, 'rb')
    train_sp = pickle.load(trf)
    trf.close()
    
    train.drop(['device_id','group'], axis=1,inplace=True)
 
    return train_sp,train_label,nnet_label,nclasses,label

def load_test_data(test_src_path,test_sp_path):

    print('Read train and test')
    test = pd.read_csv(test_src_path, dtype={'device_id': np.str})

    ttf = open(test_sp_path, 'rb')
    test_sp = pickle.load(ttf)
    ttf.close()

    return test_sp,test

def cross(Xtrain,y,y_nnet,random_state = 0):

    kf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=random_state)
    #lr_pred = np.zeros((y.shape[0],nclasses))
    for itrain, itest in kf:
        lr_pred = np.zeros((y.shape[0],nclasses))
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        ytr_nnet = y_nnet[itrain,:]
        
        #lr
        print "#################### cross lr ##############################"
        clf = LogisticRegression(C=0.15, multi_class='multinomial',solver='lbfgs')
        clf.fit(Xtr, ytr)
        lr_pred[itest,:] = clf.predict_proba(Xte)
        lr_loss = log_loss(yte, lr_pred[itest, :])

        #xgb
        print "#################### cross xgb #############################"
        xgb_pred = xgb_train(Xtr,ytr,Xte,106)
        xgb_loss = log_loss(yte, xgb_pred)

        #nnet
        print "################### cross nnet ##############################"
        nnet_pred = nnet_train(Xtr,ytr_nnet,Xte,15)
        nnet_loss = log_loss(yte, nnet_pred)

        #average
        print lr_pred[itest, :].shape
        print xgb_pred.shape
        print nnet_pred.shape
        aver_pred = np.add(lr_pred[itest, :]*0.3,np.add(xgb_pred*0.2,nnet_pred*0.5))
        #aver_pred = np.add(lr_pred[itest, :],np.add(xgb_pred,nnet_pred))/3.0
        lr_nnet_pred = np.add(lr_pred[itest, :],nnet_pred)/2.0
        xgb_nnet_pred = np.add(xgb_pred,nnet_pred)/2.0

        print aver_pred.shape
        all_loss = log_loss(yte, aver_pred)
        lr_nnet_loss = log_loss(yte, lr_nnet_pred)
        xgb_nnet_loss = log_loss(yte, xgb_nnet_pred)
     
        
        print "lr loss : %f"%(lr_loss)
        print "xgb loss : %f"%(xgb_loss)
        print "nnet loss : %f"%(nnet_loss)
        print "lr nnet loss :%f"%(lr_nnet_loss)
        print "xgb nnet loss :%f"%(xgb_nnet_loss)
        print "all aver loss : %f"%(all_loss)
        return ""
        
    #return log_loss(y, pred)

def train_(Xtrain,y,y_nnet,Xtest,random_state = 0):

    #lr
    print "#################### cross lr ##############################"
    clf = LogisticRegression(C=0.15, multi_class='multinomial',solver='lbfgs')
    clf.fit(Xtrain, y)
    lr_pred = clf.predict_proba(Xtest)

    #xgb
    print "#################### cross xgb #############################"
    xgb_pred = xgb_train(Xtrain,y,Xtest,106)

    #nnet
    print "################### cross nnet ##############################"
    nnet_pred = nnet_train(Xtrain,y_nnet,Xtest,15)

    #average
    print lr_pred.shape
    print xgb_pred.shape
    print nnet_pred.shape
    
    aver_pred = np.add(lr_pred*0.2,np.add(xgb_pred*0.2,nnet_pred*0.6))
    #aver_pred = np.add(lr_pred,np.add(xgb_pred,nnet_pred))/3.0
    lr_nnet_pred = np.add(lr_pred,nnet_pred)/2.0
    xgb_nnet_pred = np.add(xgb_pred,nnet_pred)/2.0
    print aver_pred.shape
    return aver_pred,lr_nnet_pred,xgb_nnet_pred

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print "train_no_path test_no_path cross(1)_pred(2)"
        sys.exit()
    
    train_no_sp_path = sys.argv[1]
    test_no_sp_path = sys.argv[2]
    s = sys.argv[3]
 
    #train_no_src_path = "./data/gender_age_train.csv"
    train_no_src_path = "./data/train_no_events.csv"
    test_no_src_path =  "./data/test_no_events.csv"


    Xtrain_no,y_no,y_nnet,nclasses,label_no = load_data(train_no_src_path,train_no_sp_path)
    print "train_no shape:"
    print Xtrain_no.shape

    Xtest_no,test_no = load_test_data(test_no_src_path,test_no_sp_path)

    if s == "1":
        cross(Xtrain_no,y_no,y_nnet.values,random_state = 0)
    elif s == "2":
        ######### 1.pred no events
        print "test_no shape: (%d,%d)"%(test_no.shape[0],test_no.shape[1])
        print "test_no_sp shape: (%d,%d)"%(Xtest_no.shape[0],Xtest_no.shape[1])
        device_id_no = test_no.device_id
        targetencoder_no = LabelEncoder().fit(label_no)
        aver_pred,lr_nnet_pred,xgb_nnet_pred = train_(Xtrain_no,y_no,y_nnet.values,Xtest_no,random_state = 0)
        aver_preds = pd.DataFrame(aver_pred, index = test_no.device_id, columns=targetencoder_no.classes_)
        lr_nnet_preds = pd.DataFrame(lr_nnet_pred, index = test_no.device_id, columns=targetencoder_no.classes_)
        xgb_nnet_preds = pd.DataFrame(xgb_nnet_pred, index = test_no.device_id, columns=targetencoder_no.classes_)
        
        aver_preds.to_csv("./stack_model/data/aver_all_pred.csv",index=True)
        lr_nnet_preds.to_csv("./stack_model/data/lr_nnet_pred.csv",index=True)
        xgb_nnet_preds.to_csv("./stack_model/data/xgb_nnet_pred.csv",index=True)


        submmit = pd.concat((pred_no,pred),axis=0)
        print submmit
        print submmit.shape
        submmit.to_csv("./../submmit_data/submmit_0822.csv", index=True)
        
    else:
        print "error cross(1)_pred(2)" 
        sys.exit(0)




