# File: occupancy_adaboost.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM
# Project: machine-learning
# Description:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import os, sys

sys.stdout = open(os.path.join("images","boosting","occupancy","rawdata.txt"),'a')
file = open(os.path.join("images","boosting","occupancy",'data.csv'), mode='a')
file.write("algorithm, tree depth, learning rate, no of estimators, train score, test score, fit time, train time, test time, inaccurate, train accuracy, test accuracy, train precision, test precision, train recall, test recall, train f1, test f1\n")

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# import our training and test data
from occupancy_data import X_train_std, X_test_std, y_train, y_test

def doBoost(treedepth, learn_rate, n_est, i):
    print("_______________________________________________________")
    log("DTree Depth:", treedepth)
    print("Time:", datetime.datetime.now(), "Algorithm: Decision Tree Depth:", treedepth)

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=treedepth)
    start = time.time()
    tree.fit(X_train_std, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = tree.predict(X_train_std)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = tree.predict(X_test_std)
    end = time.time()
    test_time = end-start

    inaccurate = (y_test != y_pred_test).sum()
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train)
    test_precision = precision_score(y_test, y_pred_test)
    train_recall = recall_score(y_train, y_pred_train)
    test_recall = recall_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)
    train_score = tree.score(X_train_std, y_pred_train)
    test_score = tree.score(X_test_std, y_pred_test)

    print("On training set:\n", classification_report(y_train, y_pred_train, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("........................................................")
    print("On testing set:\n", classification_report(y_test, y_pred_test, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    file.write("Decision Tree,"+str(treedepth)+","+str(learn_rate)+","+str(n_est)+","+str(train_score)+","+str(test_score)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(inaccurate)+","+str(train_accuracy)+","+str(test_accuracy)+","+str(train_precision)+","+str(test_precision)+","+str(train_recall)+","+str(test_recall)+","+str(train_f1)+","+str(test_f1)+"\n")

    print("_______________________________________________________")
    log("AdaTree Depth:", treedepth, "Learning rate:", learn_rate, "No. of estimators:", n_est)
    print("Time:", datetime.datetime.now(), "Algorithm: Adaboost Depth:", treedepth, "Learning rate:", learn_rate, "No. of estimators:", n_est)

    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=n_est, learning_rate=learn_rate, random_state=0)
    start = time.time()
    ada.fit(X_train_std, y_train)
    end = time.time()
    ada_fit_time = end-start
    start = time.time()
    y_pred_train = ada.predict(X_train_std)
    end = time.time()
    ada_train_time = end-start
    start = time.time()
    y_pred_test = ada.predict(X_test_std)
    end = time.time()
    ada_test_time = end-start

    ada_inaccurate = (y_test != y_pred_test).sum()
    ada_train_accuracy = accuracy_score(y_train, y_pred_train)
    ada_test_accuracy = accuracy_score(y_test, y_pred_test)
    ada_train_precision = precision_score(y_train, y_pred_train)
    ada_test_precision = precision_score(y_test, y_pred_test)
    ada_train_recall = recall_score(y_train, y_pred_train)
    ada_test_recall = recall_score(y_test, y_pred_test)
    ada_train_f1 = f1_score(y_train, y_pred_train)
    ada_test_f1 = f1_score(y_test, y_pred_test)
    ada_train_score = ada.score(X_train_std, y_pred_train)
    ada_test_score = ada.score(X_test_std, y_pred_test)

    print("On training set:\n", classification_report(y_train, y_pred_train, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("........................................................")
    print("On testing set:\n", classification_report(y_test, y_pred_test, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    file.write("Adaboost Tree,"+str(treedepth)+","+str(learn_rate)+","+str(n_est)+","+str(ada_train_score)+","+str(ada_test_score)+","+str(ada_fit_time)+","+str(ada_train_time)+","+str(ada_test_time)+","+str(ada_inaccurate)+","+str(ada_train_accuracy)+","+str(ada_test_accuracy)+","+str(ada_train_precision)+","+str(ada_test_precision)+","+str(ada_train_recall)+","+str(ada_test_recall)+","+str(ada_train_f1)+","+str(ada_test_f1)+"\n")

    # setup either ada plot or tree plot
    # plt.xlabel('Light[std]')
    # plt.ylabel('CO2[std]')
    # plot_decision_regions(X_test_std, y_test, clf=tree, res=0.5, legend=2)
    # plt.title('occu_adaboost dep='+str(treedepth)+' estimators='+str(n_est)+' lrate='+str(learn_rate))
    # plt.gcf().savefig(os.path.join("images", "boosting", "occupancy", str(i)+'_occu_boosting_dep'+str(treedepth)+'_est'+str(n_est)+'_lr'+str(learn_rate)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90]
learn_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# Classifying in 2D
# comment all plt.* for real regression
i = 1
# for dep in range(1,6):
#     for e in n_estimators:
#         doBoost(dep, 0.001, e, i)
#         sys.stdout.flush()
#         file.flush()
#         i += 1
#     for l in learn_rates:
#         doBoost(dep, l, 30, i)
#         sys.stdout.flush()
#         file.flush()
#         i += 1

for dep in range(1, 11):
    for e in n_estimators:
        for l in learn_rates:
            doBoost(dep, l, e, i)
sys.stdout.close()
file.close()
