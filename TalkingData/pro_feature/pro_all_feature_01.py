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
    train.drop(['age','gender'], axis=1,inplace=True)
    train_label = train["group"]
    lable_group = LabelEncoder()
    train_label = lable_group.fit_transform(train_label)
    test = pd.read_csv("/home/wangzhe/data/gender_age_test.csv", dtype={'device_id': np.str})
    test["group"] = np.nan
    tr_tt = pd.concat((train,test),axis=0, ignore_index=True)
 
    #Read phone info 
    print('Read phone info')
    pbd = pd.read_csv("./data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    tr_tt_phone = pd.merge(tr_tt, pbd, how="left", on="device_id")
    del tr_tt,pbd
    tr_tt_phone['phone_brand'] = tr_tt_phone['phone_brand'].apply(lambda x:"phone_brand:"+str(x))
    tr_tt_phone['device_model'] = tr_tt_phone['device_model'].apply(lambda x:"device_model:"+str(x))
    
    device_brand = tr_tt_phone[['device_id','phone_brand']]
    device_device = tr_tt_phone[['device_id','device_model']]
    device_brand.columns.values[1] = "feature"
    device_device.columns.values[1] = "feature"
    device_phone_feats = pd.concat((device_brand, device_device),
                       axis=0, ignore_index=True)
    device_phone_feats.drop_duplicates(['device_id','feature'],
                                        keep='first', inplace=True)
    del tr_tt_phone

    #Read Events
    print('Read events')
    events = pd.read_csv("./data/events.csv",dtype={'device_id': np.str})

    # pro time and local
    print("pro time and local")
    events["times"] = events["timestamp"]

    f = lambda x:time.strftime("%m_%d",time.strptime(x,"%Y-%m-%d %H:%M:%S"))
    events["month_day"]=events["times"].map(f)

    f = lambda x:time.strftime("%H_",time.strptime(x,"%Y-%m-%d %H:%M:%S"))
    events["hour"]=events["times"].map(f)

    f = lambda x:time.strftime("%w",time.strptime(x,"%Y-%m-%d %H:%M:%S"))
    events["week"]=events["times"].map(f)

    f = lambda x:time.strftime("%w_%H",time.strptime(x,"%Y-%m-%d %H:%M:%S"))
    events["week_hr"]=events["times"].map(f)

    f = lambda x:time.strftime("%m_%d_%H",time.strptime(x,"%Y-%m-%d %H:%M:%S"))
    events["month_day_hour"]=events["times"].map(f)


    events["local_id"] = np.sqrt((events["longitude"]*events["longitude"]) + (events["latitude"]*events["latitude"]))
    events["local_id"] = events["local_id"].map(lambda x:"local_id:"+str(int(x)))
    events.drop(['timestamp','longitude','latitude'], axis=1,inplace=True)

    m_d_h_feats  = events[["device_id","month_day_hour"]]
    week_feats = events[["device_id","week"]]
    md_feats = events[["device_id","month_day"]]
    hour_feats = events[["device_id","hour"]]
    wh_feats = events[["device_id","week_hr"]]
    lc_feats = events[["device_id","local_id"]]

    m_d_h_feats.columns.values[1] = "feature"
    week_feats.columns.values[1] = "feature"
    wh_feats.columns.values[1] = "feature"
    lc_feats.columns.values[1] = "feature"
    md_feats.columns.values[1] = "feature"
    hour_feats.columns.values[1] = "feature"
        
    wh_lc = pd.concat((wh_feats, lc_feats),
                       axis=0, ignore_index=True)
    wh_lc.drop_duplicates(['device_id','feature'], keep='first', inplace=True) 
    device_event = events[["device_id","event_id"]]
    del events,wh_feats,lc_feats
     
    #app_label 
    print("pro app label")
    app_label = pd.read_csv("./data/app_labels.csv",
                            dtype={'label_id':np.str,'app_id':np.str})
    app_label["new_labels"] = app_label.groupby(['app_id'])["label_id"].transform(lambda x: " ".join(set("label_id:" + str(s) for s in x)))
    app_label = app_label[["app_id","new_labels"]]
    app_label["app_id"] = app_label["app_id"].map(lambda x:"app_id:"+str(x))
    app_label.drop_duplicates('app_id', keep='first', inplace=True)

    #Read app_events
    print("pro app_events")
    app_events = pd.read_csv("./data/app_events.csv",
                                dtype={'app_id':np.str})
    app_events['apps'] = app_events.groupby(['event_id'])["app_id"].transform(
                          lambda x: " ".join(set("app_id:" + str(s) for s in x)))
    app_events.drop_duplicates('event_id', keep='first', inplace=True)
    events_app = pd.merge(device_event,app_events,how='left',on='event_id')
    del app_events,device_event
     
    #device and app_id
    print("pro device_id and app_id")
    #events_app = events_app[events_app["is_active"] == 1]
    device_app = events_app[["device_id","apps"]]
    device_app.dropna(axis=0,inplace=True)
    device_app['new_apps'] = device_app.groupby(['device_id'])['apps'].transform(
                            lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" ")))) 
    device_app.drop_duplicates('device_id', keep='first', inplace=True)
    device_app.drop('apps', axis=1,inplace=True)
  
    device_app = pd.concat([Series(row['device_id'], row['new_apps'].split(' '))
                    for _, row in device_app.iterrows()]).reset_index()
    device_app.columns = ['app_id', 'device_id']
    device_app = device_app[['device_id','app_id']]
    del events_app
    
    #device_id and label_id
    device_label = pd.merge(device_app,app_label,how='left',on='app_id')
    del app_label
    device_label = device_label[["device_id","new_labels"]]
    device_label.dropna(axis=0,inplace=True)
    device_label['label_id'] = device_label.groupby(['device_id'])['new_labels'].transform(lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" ")))) 
    device_label.drop_duplicates('device_id', keep='first', inplace=True)
    device_label.drop('new_labels', axis=1,inplace=True)
    device_label = pd.concat([Series(row['device_id'], row['label_id'].split(' '))
                    for _, row in device_label.iterrows()]).reset_index()
    device_label.columns = ['label_id', 'device_id']
    device_label = device_label[['device_id','label_id']]
  
    device_app.columns.values[1] = "feature"
    device_label.columns.values[1] = "feature"
    
    feats = pd.concat((device_brand, device_device,device_app, device_label,wh_lc),
                       axis=0, ignore_index=True)
    del device_brand, device_device,device_app, device_label
    
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
    print sparse_matrix
     
        # get train and test
    print("output train and test")
    train_row = le.transform(train["device_id"])
    train_sparse = sparse_matrix[train_row, :]
 
    test_row = le.transform(test["device_id"])
    test_sparse = sparse_matrix[test_row, :]
 
    #pickle save
    print("Save data")
    trf = open('/home/wangzhe/src/sparse_data/train_sp4.data', 'wb')
    pickle.dump(train_sparse, trf)
    trf.close()
 
    ttf = open('/home/wangzhe/src/sparse_data/test_sp4.data', 'wb')
    pickle.dump(test_sparse, ttf)
    ttf.close()
    
if __name__ == "__main__":

    print("*******start******")
    read_train_test()
    print("*******end********")
