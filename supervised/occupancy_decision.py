# File: occupancy_decision.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM
# Project: machine-learning
# Description:

from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import os, sys

sys.stdout = open(os.path.join("images","decision tree","occupancy","raw.txt"),'a')
file = open(os.path.join("images","decision tree","occupancy",'data.csv'), mode='a')
file.write("tree depth, tree width, fit time, train time, test time, inaccurate, train accuracy, test accuracy, train precision, test precision, train recall, test recall, train f1, test f1\n")

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from occupancy_data import X_train_std, X_test_std, y_train, y_test

def doDecision(len,wid, i):
    print("_______________________________________________________")
    log("Depth:", dep)
    print("Time:", datetime.datetime.now(), "Depth:", dep, "Split:", wid)

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=dep, random_state=0)
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
    file.write(str(dep)+","+str(wid)+","+str(train_score)+","+str(test_score)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(inaccurate)+","+str(train_accuracy)+","+str(test_accuracy)+","+str(train_precision)+","+str(test_precision)+","+str(train_recall)+","+str(test_recall)+","+str(train_f1)+","+str(test_f1)+"\n")

    # set up data for plotting
    # plt.xlabel('Light[std]')
    # plt.ylabel('CO2[std]')
    # plt.title('occu_tree dep='+str(dep)+' wid='+str(wid))
    # plot_decision_regions(X_test_std, y_test, clf=tree, res=0.5, legend=2)
    # plt.gcf().savefig(os.path.join("images","decision tree","occupancy",str(i)+'_occu_tree_dep'+str(dep)+'_wid'+str(wid)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

# Classifying in 2D
# comment all plt.* for real regression
i = 1
for dep in range(1,11):
    # width is always 2 because this is a binary classification
    doDecision(dep, 2, i)
    sys.stdout.flush()
    file.flush()
    i += 1
sys.stdout.close()
file.close()
