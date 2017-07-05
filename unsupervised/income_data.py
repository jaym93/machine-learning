# File: occupancy_data.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:02 PM 
# Project: machine-learning
# Description: 

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import functools
import io
import numpy as np
import sys
import pandas as pd

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
            #print(df)
    return df

## TRAINING DATA
df_train = pd.read_csv(os.path.join('data', 'income', 'income_train.csv'))
df_train.convert_objects(convert_numeric=True)
df_train.fillna(0, inplace=True)
df_train = handle_non_numerical_data(df_train)
# df.to_csv(os.path.join('data', 'income', 'income_train_new.csv'))

arr = np.array(df_train).astype(float)
df_train = pd.DataFrame(arr)
# df.to_csv(os.path.join('data', 'income', 'income_train_norm.csv'))

## TESTING DATA
df_test = pd.read_csv(os.path.join('data', 'income', 'income_test.csv'))
df_test.convert_objects(convert_numeric=True)
df_test.fillna(0, inplace=True)
df_test = handle_non_numerical_data(df_test)
# df.to_csv(os.path.join('data', 'income', 'income_train_new.csv'))

arr = np.array(df_test).astype(float)
df_test = pd.DataFrame(arr)
# df.to_csv(os.path.join('data', 'income', 'income_train_norm.csv'))

#########################

X_train = df_train.values
X_test = df_test.values

sc = preprocessing.StandardScaler()
sc.fit(X_train)
sc.fit(X_test)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
