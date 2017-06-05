# File: auto_decision.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:21 PM 
# Project: machine-learning
# Description: 

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import timeit
import numpy as np
import time, datetime
import os, sys

sys.stdout = open(os.path.join("images","decision tree","auto","rawdata.txt"),'a')
file = open(os.path.join("images","decision tree","auto",'data.csv'), mode='a')
file.write("tree depth, tree width, fit time, train time, test time, train mse, test mse, train mae, test mae\n")

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from auto_data import X_train, X_test, y_train, y_test

def doDecision(dep,wid):
    print("_______________________________________________________")
    log("Depth:", dep)
    print("Time:", datetime.datetime.now(), "Depth:", dep, "Split:", wid)
    tree = DecisionTreeRegressor(criterion='mse', max_depth=dep, min_samples_split=wid)

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

    export_graphviz(tree, out_file=os.path.join("images", "decision tree", "auto", 'auto_tree_entropy_depth'+str(dep)+'split'+str(wid)+'.dot'))

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print('Decision Tree train/test error: %.3f/%.3f' % (train_mse, test_mse))
    file.write(str(dep)+","+str(wid)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+"\n")

    # set up data for plotting
    # plt.plot(X_test, y_pred_test, label=' wid'+str(wid), linewidth=0.5)

# Regressing in 1D
# comment all plt.* for real regression
i = 1
for dep in range(1, 11):
    # plt.xlabel('horsepower')
    # plt.ylabel('price')
    # plt.scatter(X_test, y_test, label='test data')
    for split in range (2, 11):
        doDecision(dep, split)
        sys.stdout.flush()
    # plt.legend()
    # plt.title('auto_tree_mse depth='+str(dep))
    # plt.gcf().savefig(os.path.join("images","decision tree","auto",str(i)+'_auto_tree_mse_depth'+str(dep)+'split'+str(split)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()
    i += 1
sys.stdout.close()
file.close()
