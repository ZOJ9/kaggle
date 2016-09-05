#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series
import time

def read_train_test():

    #Read train and test
    print('Read train and test')
    train = pd.read_csv("./data/gender_age_train.csv", dtype={'device_id': np.str})
    test = pd.read_csv("./data/gender_age_test.csv", dtype={'device_id': np.str})
    
    print 'read events'
    events = pd.read_csv("./data/events.csv", dtype={'device_id': np.str})
    events = events[['device_id','event_id']]
    events.drop_duplicates('device_id', keep='first', inplace=True)
 
    print 'merge data'
    train_events = pd.merge(train,events,how='left',on='device_id',left_index=True)
    test_events = pd.merge(test,events,how='left',on='device_id',left_index=True)
    train_events.fillna(-1, inplace=True)   
    test_events.fillna(-1, inplace=True)

    train_no_events = train_events[train_events.event_id == -1]
    train_events = train_events[train_events.event_id != -1]
    train_no_events.drop('event_id',axis=1, inplace=True)
    train_events.drop('event_id',axis=1, inplace=True)

    test_no_events = test_events[test_events.event_id == -1]
    test_events = test_events[test_events.event_id != -1]
    test_no_events.drop('event_id',axis=1, inplace=True)
    test_events.drop('event_id',axis=1, inplace=True)
 
    train_no_events.to_csv("./data/train_no_events.csv",index=False) 
    train_events.to_csv("./data/train_events.csv",index=False)

    test_no_events.to_csv("./data/test_no_events.csv",index=False)        
    test_events.to_csv("./data/test_events.csv",index=False)

    print "train data shape : %d %d"%(train.shape[0],train.shape[1])
    print "train no events shape : %d %d"%(train_no_events.shape[0],train_no_events.shape[1])
    print "train with events shape : %d %d"%(train_events.shape[0],train_events.shape[1])

    print "test data shape : %d %d"%(test.shape[0],test.shape[1])
    print "test no events shape : %d %d"%(test_no_events.shape[0],test_no_events.shape[1])
    print "test with events shape : %d %d"%(test_events.shape[0],test_events.shape[1])

if __name__ == "__main__":

    print("*******start******")
    read_train_test()
    print("*******end********")
