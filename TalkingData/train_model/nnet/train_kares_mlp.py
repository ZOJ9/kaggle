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
from keras.optimizers import SGD, Adam, RMSprop,Nadam
from keras.utils import np_utils


try:
  import cPickle as pickle
except ImportError:
  import pickle

def load_data_and_train_model(train_path,test_path,out_file):
    
    print('Read train and test')
    train = pd.read_csv("./data/gender_age_train.csv", dtype={'device_id': np.str})
    train.drop(['device_id','age','gender'], axis=1,inplace=True)
    #train_label = train["group"]
    #lable_group = LabelEncoder()
    #train_label = lable_group.fit_transform(train_label)
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
    print("# Feature Selection")
    selector = SelectPercentile(f_classif, percentile=25)
    y = map(lambda x : list(x).index(1),list(y_train))
    selector.fit(X_train,np.array(y))
    
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)

    train_sp = selector.transform(train_sp)
    test_sp = selector.transform(test_sp)
    #train model
    nb_classes = 12
    nb_epoch = 60
    
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
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, class_mode='categorical')

    model.fit(train_sp.toarray(), train_label.values, nb_epoch=nb_epoch, batch_size=150)  

    y_preds = model.predict(test_sp.toarray())
    
    device_id = test['device_id']
    submission = pd.DataFrame(y_preds, columns=train_label.columns)
    submission["device_id"] = device_id
    submission = submission.set_index("device_id")
    submission.to_csv(out_file, index=True, index_label='device_id') 
    

if __name__ == "__main__":
 
    if len(sys.argv) != 4:
        print "select train_path test_path out_file"
        sys.exit(0)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    out_path = sys.argv[3]

    load_data_and_train_model(train_path,test_path,out_path)
    








