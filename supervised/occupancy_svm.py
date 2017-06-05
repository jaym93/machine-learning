# File: occupancy_svm.py
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 01-06-2017 09:07 PM 
# Project: machine-learning
# Description:

from __future__ import print_function
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import time
import numpy as np
import os, sys
import datetime

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
sys.stdout = open(os.path.join("images","svm","occupancy","rawdata.txt"),'a')
file = open(os.path.join("images","svm","occupancy",'data.csv'), mode='a')
file.write("kernel, penalty param, gamma, train score, test score, fit time, train time, test time, inaccurate, train accuracy, test accuracy, train precision, test precision, train recall, test recall, train f1, test f1\n")

from occupancy_data import X_train_std, X_test_std, y_train, y_test

def runSvm(ker,cee,gam, iter):
    print("_______________________________________________________")
    log("Kernel:", ker, "C:", cee, "Gamma:", gam)
    print("Time:", datetime.datetime.now(), "Kernel:", ker, "C:", cee, "Gamma:", gam)

    svm = SVC(kernel=ker, C=cee, random_state=0, gamma=gam, max_iter=1000000)
    start = time.time()
    svm.fit(X_train_std, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = svm.predict(X_train_std)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = svm.predict(X_test_std)
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
    train_score = svm.score(X_train_std, y_pred_train)
    test_score = svm.score(X_test_std, y_pred_test)

    print("On training set:\n", classification_report(y_train, y_pred_train, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("........................................................")
    print("On testing set:\n", classification_report(y_test, y_pred_test, labels=None, target_names=None, sample_weight=None, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    file.write(ker+","+str(cee)+","+str(gam)+","+str(train_score)+","+str(test_score)+","+str(fit_time)+","+str(train_time)+","+str(test_time)+","+str(inaccurate)+","+str(train_accuracy)+","+str(test_accuracy)+","+str(train_precision)+","+str(test_precision)+","+str(train_recall)+","+str(test_recall)+","+str(train_f1)+","+str(test_f1)+"\n")
    
    #setup data for plotting
    # plot_decision_regions(X_test_std, y_test, clf=svm, res=2.5, legend=2)
    # plt.xlabel('Light[std]')
    # plt.ylabel('CO2[std]')
    # plt.title('occu_svm kernel='+ker+' c='+str(cee)+' gamma='+str(gam))
    # plt.gcf().savefig(os.path.join("images","svm","occupancy",str(i)+'_occu_svm_'+ker+'_c'+str(cee)+'_gam'+str(gam)+'.png'), bbox_inches='tight')
    # plt.cla()
    # plt.clf()

kernels = ['linear','rbf', 'poly']
c_variations= [1,2,4,8,16,32,64,128,256,512,1024]
gamma_variations = [0.1,0.01,0.001,0.0001,0.00001]

# Classifying in 2D
# comment all plt.* for real regression
i = 1
# for k in kernels:
#     for c in c_variations:
#         runSvm(k, c, 0.01, i)
#         sys.stdout.flush()
#         file.flush()
#         i += 1
#     print("___________________")
#     for gam in gamma_variations:
#         runSvm(k, 64, gam, i)
#         sys.stdout.flush()
#         file.flush()
#         i += 1

for k in kernels:
    for c in c_variations:
        for gam in gamma_variations:
            runSvm(k, c, gam, i)
sys.stdout.close()
file.close()
