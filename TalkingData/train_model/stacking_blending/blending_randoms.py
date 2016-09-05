#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from pandas import DataFrame

def blending():

    file1 = "./data/average_pred_10fold_relu_r1234.csv"
    data1 = pd.read_csv(file1, dtype={'device_id': np.str})
    device_id = data1.device_id
    data1.drop('device_id', axis=1,inplace=True)
    val1 = data1.values

    file2 = "./data/average_pred_10fold_relu_r2234.csv"
    data2 = pd.read_csv(file2, dtype={'device_id': np.str})
    data2.drop('device_id', axis=1,inplace=True)
    val2 = data2.values

    file3 = "./data/average_pred_10fold_relu_init.csv"
    data3 = pd.read_csv(file3, dtype={'device_id': np.str})
    data3.drop('device_id', axis=1,inplace=True)
    val3 = data3.values   

    file4 = "./data/average_pred_10fold_relu_r3234.csv"
    data4 = pd.read_csv(file4, dtype={'device_id': np.str})
    data4.drop('device_id', axis=1,inplace=True)
    val4 = data4.values 

    file5 = "./data/average_pred_10fold_relu_r4234.csv"
    data5 = pd.read_csv(file5, dtype={'device_id': np.str})
    data5.drop('device_id', axis=1,inplace=True)
    val5 = data5.values
    
    file6 = "./data/average_pred_10fold_relu_r5234.csv"
    data6 = pd.read_csv(file6, dtype={'device_id': np.str})
    data6.drop('device_id', axis=1,inplace=True)
    val6 = data6.values

    file7 = "./data/average_pred_10fold_relu_r623.csv"
    data7 = pd.read_csv(file7, dtype={'device_id': np.str})
    data7.drop('device_id', axis=1,inplace=True)
    val7 = data7.values

    file8 = "./data/average_pred_10fold_relu_r723.csv"
    data8 = pd.read_csv(file8, dtype={'device_id': np.str})
    data8.drop('device_id', axis=1,inplace=True)
    val8 = data8.values

    file9 = "./data/average_pred_10fold_relu_r823.csv"
    data9 = pd.read_csv(file9, dtype={'device_id': np.str})
    data9.drop('device_id', axis=1,inplace=True)
    val9 = data9.values

    file10 = "./data/average_pred_10fold_relu_r923.csv"
    data10 = pd.read_csv(file10, dtype={'device_id': np.str})
    data10.drop('device_id', axis=1,inplace=True)
    val10 = data10.values

    val = val1+val2+val3+val4+val5+val6+val7+val8+val9+val10
    val = val/10.0
    preds = pd.DataFrame(val, index = device_id, columns=data1.columns)

    preds.to_csv("./blending_data/blending_10.csv",index=True)

if __name__ == "__main__":

    blending()

