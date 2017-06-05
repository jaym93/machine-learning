# File: auto_svm.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM 
# Project: machine-learning
# Description:

from __future__ import print_function
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import timeit
import numpy as np
import os, sys
import time, datetime

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

sys.stdout = open(os.path.join("images","svm","auto","raw2.txt"),'a')
file = open(os.path.join("images","svm","auto",'data2.csv'), mode='a')
file.write("kernel, penalty param, gamma, fit time, train time, test time, train mse, test mse, train mae, test mae\n")

from auto_data import X_train, X_test, y_train, y_test

def runSvm(ker,cee,gam, i):
    print("_______________________________________________________")
    log("Kernel:", ker, "C:", cee, "Gamma:", gam)
    print("Time:", datetime.datetime.now(), "Kernel:", ker, "C:", cee, "Gamma:", gam)

    svm = SVR(kernel=ker, C=cee, gamma=gam, max_iter=1000000000, degree=2)
    start = time.time()
    svm.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = svm.predict(X_train)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = svm.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    file.write(ker+","+str(c)+","+str(gam)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+"\n")

    # setup data for plotting
    # plt.plot(X_test, y_pred_test, label='c='+str(cee)+' g='+str(gam), linewidth=0.5)
    # plt.xlabel('horsepower')
    # plt.ylabel('price')
    # plt.scatter(X_test, y_test, label='test data')
    # plt.legend()
    # plt.title('auto_svm kernel='+ker+' deg=2 max_iter=10e9')
    # plt.gcf().savefig(os.path.join("images", "svm", "auto", str(i)+'_auto_svm_'+ker+'2_c'+str(cee)+'_gam'+str(gam)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

kernels = ['linear', 'rbf', 'poly']
c_variations= [1,2,4,8,16,32,64,128,256,512,1024]
gamma_variations = [0.1,0.01,0.001,0.0001,0.00001]

# Regressing in 1D
# comment all plt.* for real regression
i = 1
for k in kernels:
    for c in c_variations:
        for gam in gamma_variations:
            runSvm(k, c, gam, i)
    #     runSvm(k, c, 0.01, i)
    #     sys.stdout.flush()
    #     file.flush()
    #     i += 1
    # print("___________________")
    # for gam in gamma_variations:
    #     runSvm(k, 2, gam, i)
    #     sys.stdout.flush()
    #     file.flush()
    #     i += 1
sys.stdout.close()
file.close()
