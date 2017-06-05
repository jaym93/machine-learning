# File: auto_adaboost.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:21 PM
# Project: machine-learning
# Description:

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import os, sys

sys.stdout = open(os.path.join("images", "boosting", "auto", "rawdata.txt"), 'a')
file = open(os.path.join("images", "boosting", "auto", 'data.csv'), mode='a')
file.write("algorithm, tree depth, learning rate, no of estimators, fit time, train time, test time, train mse, test mse, train mae, test mae\n")

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# import our training and test data
from auto_data import X_train, X_test, y_train, y_test

def doBoost(treedepth, learn_rate, n_est, i):
    print("_______________________________________________________")
    log("DTree Depth:", treedepth)
    print("Time:", datetime.datetime.now(), "Algorithm: Decision Tree Depth:", treedepth)

    tree = DecisionTreeRegressor(criterion='mse', max_depth=treedepth)
    start = time.time()
    tree.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = tree.predict(X_train)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = tree.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    file.write("Decision Tree,"+str(treedepth)+","+str(learn_rate)+","+str(n_est)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+"\n")

    print("_______________________________________________________")
    log("Adaboost Depth:", treedepth)
    print("Time:", datetime.datetime.now(), "Algorithm: Adaboost Depth:", treedepth, "Learning rate:", learn_rate, "No. of estimators:", n_est)

    ada = AdaBoostRegressor(base_estimator=tree, n_estimators=n_est, learning_rate=learn_rate, random_state=0)
    start = time.time()
    ada.fit(X_train, y_train)
    end = time.time()
    ada_fit_time = end-start
    start = time.time()
    y_pred_train = ada.predict(X_train)
    end = time.time()
    ada_train_time = end-start
    start = time.time()
    y_pred_test = ada.predict(X_test)
    end = time.time()
    ada_test_time = end-start

    ada_train_mse = mean_squared_error(y_train, y_pred_train)
    ada_test_mse = mean_squared_error(y_test, y_pred_test)
    ada_train_mae = mean_absolute_error(y_train, y_pred_train)
    ada_test_mae = mean_absolute_error(y_test, y_pred_test)

    file.write("Adaboost Tree,"+str(treedepth)+","+str(learn_rate)+","+str(n_est)+","+str(ada_fit_time)+","+str(ada_train_time)+","+str(ada_test_time)+","+str(ada_train_mse)+","+str(ada_test_mse)+","+str(ada_train_mae)+","+str(ada_test_mae)+"\n")

    # setup either ada plot or tree plot
    # plt.plot(X_test, y_pred_test, label='est='+str(n_est), linewidth=0.5)
    # plt.title('auto_adaboost dep='+str(treedepth)+' lrate='+str(learn_rate))

n_estimators = [10, 20, 30, 40, 50]
learn_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Regressing in 1D
# comment all plt.* for real regression
i = 1
for dep in range(1,6):
    for l in learn_rates:
        # plt.xlabel('horsepower')
        # plt.ylabel('price')
        # plt.scatter(X_test, y_test, label='test data')
        for e in n_estimators:
            doBoost(dep, l, e, i)
        # plt.legend()
        # plt.gcf().savefig(os.path.join("images", "boosting", "auto", str(i)+'_auto_boosting_dep'+str(dep)+'_est'+str(e)+'_lr'+str(l).replace(".", "-")+'.png'), bbox_inches='tight')
        i += 1
        # plt.cla()
        # plt.clf()
        sys.stdout.flush()
        file.flush()
sys.stdout.close()
file.close()