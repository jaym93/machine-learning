# File: occupancy_neural.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:21 PM
# Project: machine-learning
# Description:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import os, sys

sys.stdout = open(os.path.join("images","neuralnet","occupancy","rawdata.txt"),'a')
file = open(os.path.join("images","neuralnet","occupancy",'data.csv'), mode='a')
file.write("layer config, batch size, alpha, iterations, train score, test score, fit time, train time, test time, inaccurate, train accuracy, test accuracy, train precision, test precision, train recall, test recall, train f1, test f1\n")

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# import our training and test data
from occupancy_data import X_train_std, X_test_std, y_train, y_test

def doNn(layers,alp, b_size, iter):
    print("_______________________________________________________")
    log("Network:", str(layers), "alpha:", alp, "batch_size:", b_size)
    print("Time:", datetime.datetime.now(), "Layers:", layers,  "alpha:", alp, "batch_size:", b_size)

    nn = MLPClassifier(hidden_layer_sizes=layers, alpha=alp, batch_size=b_size, max_iter=1000000)
    start = time.time()
    nn.fit(X_train_std, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = nn.predict(X_train_std)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = nn.predict(X_test_std)
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
    train_score = nn.score(X_train_std, y_pred_train)
    test_score = nn.score(X_test_std, y_pred_test)

    print("On training set:\n", classification_report(y_train, y_pred_train, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("........................................................")
    print("On testing set:\n", classification_report(y_test, y_pred_test, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    file.write(str(layers).replace(',', '-')+","+str(b_size)+","+str(alp)+","+str(nn.n_iter_)+","+str(train_score)+","+str(test_score)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(inaccurate)+","+str(train_accuracy)+","+str(test_accuracy)+","+str(train_precision)+","+str(test_precision)+","+str(train_recall)+","+str(test_recall)+","+str(train_f1)+","+str(test_f1)+"\n")

    # setup either ada plot or tree plot
    # plt.xlabel('Light[std]')
    # plt.ylabel('CO2[std]')
    # plot_decision_regions(X_test_std, y_test, clf=nn, res=0.5, legend=2)
    # plt.title('occu_neural dep='+str(layers)+"\nalpha="+str(alp)+" batch_size="+str(b_size))
    # plt.gcf().savefig(os.path.join("images", "neuralnet", "occupancy", str(iter)+'_occu_neural_lyr'+str(layers)+'_alp'+str(alp)+'_bat'+str(b_size)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

layer_opts = [(5,), (10,), (5, 5), (5, 10), (10, 5), (10, 10), (5, 5, 5), (10, 5, 5), (5, 10, 5), (5, 5, 10), (5, 10, 15), (15, 10, 5), (5, 5, 5, 5), (5, 5, 5, 5, 5)]
alp_opts = [0.1, 0.01, 0.001, 0.0001]
batch_opts = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

# Classifying in 2D
# comment all plt.* for real regression
i = 1
# for l in layer_opts:
#    doNn(layers=l, alp=0.0001, b_size=100, iter=i)
#    i += 1
#    sys.stdout.flush()
#    file.flush()
# for b in batch_opts:
#     doNn(layers=(5,5,5), alp=0.0001, b_size=b, iter=i)
#     i += 1
# sys.stdout.flush()
# file.flush()
# for a in alp_opts:
#     doNn(layers=(5,5,5), alp=a, b_size=100, iter=i)
#     i += 1
#     sys.stdout.flush()
#     file.flush()
for l in layer_opts:
    for b in batch_opts:
        for a in alp_opts:
            doNn(layers=l, alp=a, b_size=b, iter=i)
sys.stdout.close()
file.close()
