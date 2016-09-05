#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from multiprocessing import Pool

import xgboost as xgb

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop,Nadam,Adagrad
from keras.utils import np_utils

try:
    import cPickle as pickle
except ImportError:
    import pickle

def load_data(train_src_path,train_sp_path,test_src_path,test_sp_path):
    train = pd.read_csv(train_src_path,dtype={'device_id':np.str})
    test = pd.read_csv(test_src_path,dtype={'device_id':np.str}) 
    
    trf = open(train_sp_path, 'rb')
    train_sp = pickle.load(trf)
    trf.close()

    ttf = open(test_sp_path, 'rb')
    test_sp = pickle.load(ttf)
    ttf.close()

    print "train shape : (%d,%d)"%(train.shape[0],train.shape[1])
    print "test shape : (%d,%d)"%(test.shape[0],test.shape[1])
    print "train sp shape : (%d,%d)"%(train_sp.shape[0],train_sp.shape[1])
    print "test sp shape : (%d,%d)"%(test_sp.shape[0],test_sp.shape[1])

    return train,test,train_sp,test_sp
 
def nnet_train(Xtr,ytr,Xte,yte,Xtest,nb_epoch):
    
    selector = SelectPercentile(f_classif, percentile=40)
    y = map(lambda x : list(x).index(1),list(ytr))
    selector.fit(Xtr,np.array(y))

    Xtr = selector.transform(Xtr)
    Xtest = selector.transform(Xtest)
    Xte = selector.transform(Xte)

    #train model
    nb_classes = 12
    
    model = Sequential()
    model.add(Dense(500, input_dim=Xtr.shape[1], init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(200, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(12, init='glorot_uniform'))
    model.add(Activation('softmax')) 
    nadam = Nadam(lr=1e-4)   
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, class_mode='categorical')

    model.fit(Xtr.toarray(),ytr, nb_epoch=nb_epoch, batch_size=150,validation_data=[Xte.toarray(),yte])  
    test_pred = model.predict(Xtest.toarray())
    print test_pred    
    return test_pred

def split_data(train,test,train_sp,test_sp):

    datas = []
    pool = Pool(6)
    Xtest = test_sp

    label = train["group"]
    label_encod = LabelEncoder().fit(label)
    label = label_encod.transform(label)
    y = label

    
    train["value"] = np.ones(len(train.values))
    y_nnet = pd.pivot_table(train,index=train.index, columns='group',values='value',fill_value=0)
    y_nnet = y_nnet.values

    kf = StratifiedKFold(label, n_folds=10, shuffle=True, random_state=823)    

    data_num = 0
    for itrain, itest in kf:
        data_num += 1
        Xtr, Xte = train_sp[itrain, :], train_sp[itest, :]
        ytr, yte = y_nnet[itrain,:], y_nnet[itest,:]
        
        #nnet_train(Xtr,ytr,Xte,yte,Xtest,10)
        datas.append(pool.apply_async(nnet_train,args=(Xtr,ytr,Xte,yte,Xtest,8,)))
    pool.close()
    pool.join()

    datas = map(lambda x : x.get(),datas)

    #new_train = reduce(lambda x,y:np.concatenate((x,y)),train_pred)
    new_test = reduce(lambda x,y:x+y,datas)
    new_test = new_test/float(len(datas))

    pred_test = pd.DataFrame(new_test,index = test.device_id, columns=label_encod.classes_)
    pred_test.to_csv("./data/average_pred_10fold_relu_r823.csv",index=True)

if __name__ == "__main__":

    train_src_path = "./data/gender_age_train.csv"
    test_src_path = "./data/gender_age_test.csv"
    train_sp_path = "./sparse_data/train_sp8.data"
    test_sp_path = "./sparse_data/test_sp8.data"
 
    train,test,train_sp,test_sp = load_data(train_src_path,train_sp_path,test_src_path,test_sp_path)
    split_data(train,test,train_sp,test_sp)




