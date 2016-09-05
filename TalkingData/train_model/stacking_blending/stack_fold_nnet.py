#!/usr/bin/env python

import time
import sys
import numpy as np

np.random.seed(234)  

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop,Nadam,Adagrad
from keras.utils import np_utils


try:
  import cPickle as pickle
except ImportError:
  import pickle

def load_data_and_train_model():
    
    print('Read lr data')
    lr_events_train = pd.read_csv("./sklearn_train/data/lr_train_all_all.csv",
                          dtype={'device_id': np.str})

    lr_events_test = pd.read_csv("./sklearn_train/data/lr_test_all.csv",
                       dtype={'device_id': np.str})
    lr_events_train_no = pd.read_csv("./sklearn_train/data/lr_train_no.csv",
                         dtype={'device_id': np.str})

    lr_events_test_no = pd.read_csv("./sklearn_train/data/lr_test_no.csv",
                       dtype={'device_id': np.str})    
   

    print('Read xgb data')
    xgb_events_train = pd.read_csv("./xgb_src/data/xgb_train_events_time.csv",
                       dtype={'device_id': np.str})

    xgb_events_test = pd.read_csv("./xgb_src/data/xgb_test_events_time.csv",
                       dtype={'device_id': np.str})

    nnet_events_train = pd.read_csv("./nnet/data/nnet_train_events_time.csv",
                       dtype={'device_id': np.str})

    nnet_events_test = pd.read_csv("./nnet/data/nnet_test_events_time.csv",
                       dtype={'device_id': np.str})

    print "lr_events shape : (%d,%d)"%(lr_events_train.shape[0],lr_events_train.shape[1])
    print "xgb_events shape : (%d,%d)"%(xgb_events_train.shape[0],xgb_events_train.shape[1])
    print "nnet_events shape : (%d,%d)"%(nnet_events_train.shape[0],nnet_events_train.shape[1])
    #xgb_events_train.drop('label', axis=1,inplace=True)
    #nnet_events_train.drop('label', axis=1,inplace=True)
    #sub_train = pd.merge(lr_events_train,xgb_events_train,how='left',on='device_id')
    #train = pd.merge(sub_train,nnet_events_train,how='left',on='device_id')
    ##train = pd.merge(sub_train,nnet_events_train,how='left',on='device_id')
    train = lr_events_train
    test = pd.merge(lr_events_test,xgb_events_test,how='left',on='device_id')
    train.drop('device_id', axis=1,inplace=True)
    
    train_label = train.label
    lable_group = LabelEncoder().fit(train_label)
    train_label = lable_group.transform(train_label)

    train["value"] = np.ones(len(train.values))
    nnet_label = pd.pivot_table(train,index=train.index, columns='label',
                                    values='value',fill_value=0)
    train.drop(['label','value'], axis=1,inplace=True)
    Xtrain = train.values

    X_train, X_val, y_train, y_val = train_test_split(Xtrain, nnet_label.values, train_size=.90, random_state=30)
    print("# Feature Selection")
    selector = SelectPercentile(f_classif, percentile=100)
    #selector = SelectPercentile(f_classif, percentile=40)
    y = map(lambda x : list(x).index(1),list(y_train))
    selector.fit(X_train,np.array(y))
    
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)

    nb_classes = 12
    nb_epoch = 100
    
    model = Sequential()
    model.add(Dense(500, input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(200, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(12, init='uniform'))
    model.add(Activation('softmax')) 
    
    nadam = Nadam(lr=1e-4)
    adagrad = Adagrad(lr=0.01, epsilon=1e-6)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')
    model.compile(loss='categorical_crossentropy', optimizer=nadam, class_mode='categorical')
    #model.compile(loss='categorical_crossentropy', optimizer=adagrad, class_mode='categorical')
    model.fit(X_train, y_train, nb_epoch=1000, batch_size=150,validation_data=[X_val,y_val])  

    score = model.evaluate(X_val, y_val, batch_size=150,show_accuracy=True, verbose=1)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
 
    load_data_and_train_model()
    








