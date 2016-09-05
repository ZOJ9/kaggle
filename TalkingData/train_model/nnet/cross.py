#!/usr/bin/env python

import time
import sys
import numpy as np
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

def load_data_and_train_model(train_path,test_path):
    
    print('Read train and test')
    train = pd.read_csv("./data/gender_age_train.csv", dtype={'device_id': np.str})
    train.drop(['device_id','age','gender'], axis=1,inplace=True)
    train["value"] = np.ones(len(train.values))
    train_label = pd.pivot_table(train,index=train.index, columns='group',
                                    values='value',fill_value=0)
    train.drop(['value','group'], axis=1,inplace=True)

    test = pd.read_csv("./data/gender_age_test.csv", dtype={'device_id': np.str})
    test["group"] = np.nan
    
    trf = open(train_path, 'rb')
    train_sp = pickle.load(trf)
    trf.close()
    
    ttf = open(test_path, 'rb')
    test_sp = pickle.load(ttf)
    ttf.close()

    X_train, X_val, y_train, y_val = train_test_split(train_sp,train_label.values, train_size=.90, random_state=10)
    del train_sp,train_label,test_sp
    print("# Feature Selection")
    selector = SelectPercentile(f_classif, percentile=100)
    y = map(lambda x : list(x).index(1),list(y_train))
    selector.fit(X_train,np.array(y))
    
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)

    #train_sp = selector.transform(train_sp)
    #test_sp = selector.transform(test_sp)
    #train model
    nb_classes = 12
    nb_epoch = 100
    
    ##PReLU  LeakyReLU
    model = Sequential()
    model.add(Dense(500, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(12))
    model.add(Activation('softmax')) 

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    adagrad = Adagrad(lr=0.01, epsilon=1e-6)    
    nadam = Nadam(lr=1e-4)       
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')
    model.compile(loss='categorical_crossentropy', optimizer=nadam, class_mode='categorical')
    model.fit(X_train.toarray(), y_train, nb_epoch=100, batch_size=150,validation_data=[X_val.toarray(),y_val])  

    score = model.evaluate(X_val.toarray(), y_val, batch_size=150,show_accuracy=True, verbose=1)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
 
    if len(sys.argv) != 3:
        print "select train_path test_path"
        sys.exit(0)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    load_data_and_train_model(train_path,test_path)
    








