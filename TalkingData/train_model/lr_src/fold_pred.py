#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

from multiprocessing import Pool

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

def lr_train(Xtr,ytr,Xte,yte,Xtest,tr_device):

    clf = LogisticRegression(C=0.15,multi_class='multinomial',solver='lbfgs')
    clf.fit(Xtr, ytr)
    pred = clf.predict_proba(Xte)
    test_pred = clf.predict_proba(Xtest)
    return tr_device,yte,pred,test_pred
 
def split_data(train,test,train_sp,test_sp):

    datas = []
    pool = Pool(5)
    Xtrain = train_sp
    Xtest = test_sp

    label = train["group"]
    label_encod = LabelEncoder().fit(label)
    label = label_encod.transform(label)
    y = label

    kf = StratifiedKFold(label, n_folds=10, shuffle=True, random_state=0)
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        tr_device = train.device_id[itest]
        datas.append(pool.apply_async(lr_train,args=(Xtr,ytr,Xte,yte,Xtest,tr_device))) 
    pool.close()
    pool.join()
    datas = map(lambda x : x.get(),datas)
    
    devices = map(lambda x:x[0],datas)
    new_label = map(lambda x:x[1],datas)
    train_pred = map(lambda x:x[2],datas)
    test_pred = map(lambda x:x[3],datas)
    
    new_train = reduce(lambda x,y:np.concatenate((x,y)),train_pred)
    new_test = reduce(lambda x,y:x+y,test_pred)
    new_test = new_test/float(len(datas))
    new_label = reduce(lambda x,y:np.concatenate((x,y)),new_label)
    new_device = reduce(lambda x,y:np.concatenate((x,y)),devices)
    
    pred_train = pd.DataFrame(new_train,index = range(train.shape[0]), columns=label_encod.classes_)
    pred_train['label'] = new_label
    pred_train['device_id'] = new_device
    pred_test = pd.DataFrame(new_test,index = test.device_id, columns=label_encod.classes_)
    pred_train.to_csv("./data/lr_train_no.csv",index=False)
    pred_test.to_csv("./data/lr_test_no.csv",index=True)

if __name__ == "__main__":

    train_src_path = "./data/train_no_events.csv"
    test_src_path = "./data/test_no_events.csv"
    train_sp_path = "./sparse_data/train_sp_no_events.data"
    test_sp_path = "./sparse_data/test_sp_no_events.data"
   
    #train_src_path = "./data/train_events.csv"
    #test_src_path = "./data/test_events.csv"
    #train_sp_path = "./sparse_data/train_sp_events_time.data"
    #test_sp_path = "./sparse_data/test_sp_events_time.data"
 
    train,test,train_sp,test_sp = load_data(train_src_path,train_sp_path,test_src_path,test_sp_path)
    split_data(train,test,train_sp,test_sp)




