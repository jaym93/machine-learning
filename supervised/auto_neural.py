# File: auto_neural.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM 
# Project: machine-learning
# Description:

from __future__ import print_function
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time
import numpy as np
import os, sys
import datetime

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

sys.stdout = open(os.path.join("images","neuralnet","auto","rawdata.txt"),'a')
file = open(os.path.join("images","neuralnet","auto",'data.csv'), mode='a')
file.write("layer config, batch size, alpha, iterations, train score, test score, fit time, train time, test time, train mse, test mse, train mae, test mae\n")

from auto_data import X_train, X_test, y_train, y_test

def doNn(layers, alp, b_size, iter):
    print("_______________________________________________________")
    log("Network:", str(layers), "alpha:", alp, "batch_size:", b_size)
    print("Time:", datetime.datetime.now(), "Layers:", layers, "alpha:", alp, "batch_size:", b_size)

    nn = MLPRegressor(hidden_layer_sizes=layers, alpha=alp, batch_size=b_size, max_iter=10000)
    start = time.time()
    nn.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = nn.predict(X_train)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = nn.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_score = nn.score(X_train, y_pred_train)
    test_score = nn.score(X_test, y_pred_test)
    file.write(str(layers).replace(',', '-')+","+str(b_size)+","+str(alp)+","+str(nn.n_iter_)+","+str(fit_time)+","+str(train_score)+","+str(test_score)+","+str(train_time)+","+str(test_time)+","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+"\n")

    # setup data for plotting
    plt.plot(X_test, y_pred_test, label='b_size='+str(b_size), linewidth=0.5)
    plt.legend()
    plt.title('auto_neural dep='+str(layers)+" alpha="+str(alp)+" max_iter=10e4")

layer_opts = [(5,), (10,), (15,), (5, 5), (5, 10), (10, 5), (10, 10), (5,), (5,5), (5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5, 5), (10, 5, 5), (5, 10, 5), (5, 5, 10), (5, 10, 15, 20), (20, 15, 10, 5)]
alp_opts = [0.1, 0.01, 0.001, 0.0001]
batch_opts = [10,20,30,40,50,60,70,80,90,100]

i = 1
# Regressing in 1D
# comment all plt.* for real regression
# for l in layer_opts:
#     plt.xlabel('horsepower')
#     plt.ylabel('price')
#     plt.scatter(X_test, y_test, label='test data')
#     for a in alp_opts:
#         doNn(layers=l, alp=a, b_size=100, iter=i)
#         sys.stdout.flush()
#         file.flush()
#     plt.gcf().savefig(os.path.join("images", "neuralnet", "auto", str(i)+'_occu_neural_lyr'+str(l)+'_bat'+str(100)+'.png'), bbox_inches='tight')
#     plt.cla()
#     plt.clf()
#     i += 1
# sys.stdout.close()
# file.close()
#
# for l in layer_opts:
#     plt.xlabel('horsepower')
#     plt.ylabel('price')
#     plt.scatter(X_test, y_test, label='test data')
#     for b in batch_opts:
#         doNn(layers=l, alp=0.001, b_size=b, iter=i)
#         sys.stdout.flush()
#         file.flush()
#     plt.gcf().savefig(os.path.join("images", "neuralnet", "auto", str(i)+'_occu_neural_lyr'+str(l)+'_bat'+str(30)+'.png'), bbox_inches='tight')
#     plt.cla()
#     plt.clf()
#     i += 1
# sys.stdout.close()
# file.close()

for l in layer_opts:
    for a in alp_opts:
        for b in batch_opts:
            doNn(layers=l, alp=a, b_size=b, iter=i)
sys.stdout.close()
file.close()