# File: auto_knn.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM
# Project: machine-learning
# Description:

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import timeit
import numpy as np
import os, sys
import time, datetime

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

sys.stdout = open(os.path.join("images","knn","auto","rawdata.txt"),'a')
file = open(os.path.join("images","knn","auto",'data.csv'), mode='a')
file.write("neighbours, fit time, train time, test time, train mse, test mse, train mae, test mae\n")

from auto_data import X_train, X_test, y_train, y_test


def doKnn(nei, i):
    print("_______________________________________________________")
    log("Neighbours:", nei)
    print("Time:", datetime.datetime.now(), "Neighbours:", nei)

    knn = KNeighborsRegressor(n_neighbors=nei, algorithm='auto')
    start = time.time()
    knn.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = knn.predict(X_train)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = knn.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    file.write(str(nei)+","+str(train_time)+","+str(fit_time)+","+str(test_time)+","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+"\n")

    # Setup graph data
    # plt.scatter(X_test, y_test, label='test data')
    # plt.plot(X_train, y_pred_train, label='k='+str(nei), linewidth=0.5)
    # plt.xlabel('horsepower')
    # plt.ylabel('price')
    # plt.legend()
    # plt.title('auto_knn_'+'k='+str(nei))
    # plt.gcf().savefig(os.path.join("images","knn","auto",str(i)+'_auto_knn_'+'k'+str(nei)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

# Regressing in 1D
# comment all plt.* for real regression
i = 1
for k in range(1, 21):
    doKnn(k, i)
    sys.stdout.flush()
    file.flush()
    i += 1
sys.stdout.close()
file.close()

