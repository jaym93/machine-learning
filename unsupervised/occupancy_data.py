# File: occupancy_data.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:02 PM 
# Project: machine-learning
# Description: 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

df_train = pd.read_csv(os.path.join('data', 'occupancy', 'datatrain_cleaned.csv')).convert_objects(convert_numeric=True)
df_test = pd.read_csv(os.path.join('data', 'occupancy', 'datatest_cleaned.csv')).convert_objects(convert_numeric=True)
y_train = df_train['Occupancy'].values.reshape(-1,1)
y_test = df_test['Occupancy'].values.reshape(-1,1)

### 2D DATA FOR VISUALIZATION
# X_train = df_train[['Light', 'CO2']].values
# X_test = df_test[['Light', 'CO2']].values

#### COMPLETE DATA
# find how data was cleaned in data/auto/cleaning_process.txt
col_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']

trainfile = open(os.path.join('data','occupancy','datatrain_cleaned.csv'))
trainfile.readline() # removes the header line
X_train = np.loadtxt(trainfile, delimiter=",")[:, 0:-1]

testfile = open(os.path.join('data','occupancy','datatest_cleaned.csv'))
testfile.readline() # removes the header line
X_test = np.loadtxt(testfile, delimiter=",")[:, 0:-1]

# print(X_train.shape, X_test.shape)

# Standardize the data for better results
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(y_test.reshape(-1,1).shape, y_train.reshape(-1,1).shape)