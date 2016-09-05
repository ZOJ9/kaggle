#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series
import time
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

try:
    import cPickle as pickle
except ImportError:
    import pickle

def read_train_test():

    #Read train and test
    print('Read train and test')
    train = pd.read_csv("./data/gender_age_train.csv", dtype={'device_id': np.str})
    #train = pd.read_csv("/home/wangzhe/data/train_no_events.csv", dtype={'device_id': np.str})
    train.drop(['age','gender'], axis=1,inplace=True)
    train_label = train["group"]
    lable_group = LabelEncoder()
    train_label = lable_group.fit_transform(train_label)
    test = pd.read_csv("./data/test_no_events.csv", dtype={'device_id': np.str})
    test["group"] = np.nan
    #tr_tt = pd.concat((train,test),axis=0, ignore_index=True)
    tr_tt = pd.concat((test,train),axis=0, ignore_index=True)
    print tr_tt.shape
 
    #read phone info 
    print('Read phone info')
    pbd = pd.read_csv("./data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    tr_tt_phone = pd.merge(tr_tt, pbd, how="left", on="device_id")
    del tr_tt,pbd
    tr_tt_phone['phone_brand'] = tr_tt_phone['phone_brand'].apply(lambda x:"phone_brand:"+str(x))
    tr_tt_phone['device_model'] = tr_tt_phone['device_model'].apply(lambda x:"device_model:"+str(x))
    tr_tt_phone['device_brand_model'] = tr_tt_phone.phone_brand.str.cat(tr_tt_phone.device_model)
    
    device_brand = tr_tt_phone[['device_id','phone_brand']]
    device_device = tr_tt_phone[['device_id','device_model']]
    device_brand_model = tr_tt_phone[['device_id','device_brand_model']]
    device_brand.columns.values[1] = "feature"
    device_device.columns.values[1] = "feature"
    device_brand_model.columns.values[1] = "feature"

    feats = pd.concat((device_brand,device_brand_model),
                       axis=0, ignore_index=True)
    feats.drop_duplicates(['device_id','feature'],
                                        keep='first', inplace=True)
    print tr_tt_phone
    del tr_tt_phone

    # creat sparse matrix
    print("creat sparse matrix")
    feats_row = feats["device_id"].unique()
    feats_col = feats["feature"].unique()
 
    le = LabelEncoder().fit(feats["device_id"])
    data = np.ones(len(feats))
    row = le.transform(feats["device_id"])
    col = LabelEncoder().fit_transform(feats["feature"])
    sparse_matrix = sparse.csr_matrix(
        (data, (row, col)), shape=(len(feats_row), len(feats_col)))
 
    sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]
     
    
    # get train and test
    print("output train and test")
    train_row = le.transform(train["device_id"])
    train_sparse = sparse_matrix[train_row, :]
 
    test_row = le.transform(test["device_id"])
    test_sparse = sparse_matrix[test_row, :]

    print train_sparse.shape
    print test_sparse.shape

    #pickle save
    print("Save data")
    trf = open('./sparse_data/train_sp_no_events_all.data', 'wb')
    pickle.dump(train_sparse, trf)
    trf.close()
 
    ttf = open('./sparse_data/test_sp_no_events.data', 'wb')
    pickle.dump(test_sparse, ttf)
    ttf.close()

if __name__ == "__main__":

    print("*******start******")
    read_train_test()
    print("*******end********")
