# File: occupancy_knn.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM
# Project: machine-learning
# Description:

from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time
import os, sys
import datetime

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

sys.stdout = open(os.path.join("images","knn","occupancy","raw.txt"),'a')
file = open(os.path.join("images","knn","occupancy",'data.csv'), mode='a')
file.write("neighbours, inaccurate, train score, test score, fit time, train time, test time, train accuracy, test accuracy, train precision, test precision, train recall, test recall, train f1, test f1\n")

from occupancy_data import X_train_std, X_test_std, y_train, y_test


def doKnn(nei, i):
    print("_______________________________________________________")
    log("Neighbours:", nei)
    print("Time:", datetime.datetime.now(), "Neighbours:", nei)

    knn = KNeighborsClassifier(n_neighbors=nei, algorithm='auto')
    start = time.time()
    knn.fit(X_train_std, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = knn.predict(X_train_std)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = knn.predict(X_test_std)
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
    train_score = knn.score(X_train_std, y_pred_train)
    test_score = knn.score(X_test_std, y_pred_test)

    print("On training set:\n", classification_report(y_train, y_pred_train, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("........................................................")
    print("On testing set:\n", classification_report(y_test, y_pred_test, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    file.write(str(nei)+","+str(inaccurate)+","+str(train_score)+","+str(test_score)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(train_accuracy)+","+str(test_accuracy)+","+str(train_precision)+","+str(test_precision)+","+str(train_recall)+","+str(test_recall)+","+str(train_f1)+","+str(test_f1)+"\n")

    # Setup graph data
    # plot_decision_regions(X_test_std, y_test, clf=knn, res=2.5, legend=2)
    # plt.xlabel('Light[std]')
    # plt.ylabel('CO2[std]')
    # plt.title('occu_knn k='+str(nei))
    # plt.gcf().savefig(os.path.join("images","knn","occupancy", str(i)+'_occu_knn_'+'k'+str(nei)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

# Classifying in 2D
# comment all plt.* for real regression
i = 1
for k in range(1, 21):
    doKnn(k, i)
    sys.stdout.flush()
    file.flush()
    i += 1
sys.stdout.close()
file.close()

