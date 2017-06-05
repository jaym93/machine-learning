# File: auto_data.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:02 PM 
# Project: machine-learning
# Description: 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

#### COMPLETE DATA
file = open(os.path.join('data','auto','auto_cleaned.csv'))
# find how data was cleaned in data/auto/cleaning_process.txt
file.readline() # removes the header line
col_names = ['fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
complex = np.loadtxt(file, delimiter=",")
X = complex[:, 0:-1]
y = complex[:, -1]

### 1D DATA FOR VISUALIZATION
# df_auto = pd.read_csv(os.path.join('data','auto','auto_cleaned.csv'))
# we are only regressing on the highway mpg and horsepower features - this is only to visualize the data in a 2D graph
# adding attributes to X with np.reshape() can enable working with more features, as shown in X_complex
# y = df_auto['price'].values
# X = df_auto[['horsepower']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# Standardize the data for better results
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
