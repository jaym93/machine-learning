# File: clustering 
# Author: Jayanth M (jayanth6@gatech.edu)
# Created: 6/27/2017 10:46 PM 
# Project: machine-learning
# Description:

from __future__ import print_function
import numpy as np
import os, sys
from matplotlib import pyplot as plt
import time

from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GaussianMixture as EM
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from income_data import X_train_std, X_test_std, y_train, y_test
file = open(os.path.join('income_telemetry.csv'), "w")
sys.stdout = open(os.path.join('income_telemetry.txt'), 'a')
def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %f sec' % (method.__name__, args, kw, te-ts))
        return result

    return timed

def plot(axes, values, x_label, y_label, title, name):
    plt.clf()
    plt.plot(*values)
    plt.axis(axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name+".png", dpi=500)
    plt.clf()

@timeit
def pca(tx, ty, rx, ry):
    compressor = PCA(n_components = tx[1].size//2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="Inc_PCA", times=50)
    km(newtx, ty, newrx, ry, add="Inc_PCA", times=50)
    #nn(newtx, ty, newrx, ry, add="Inc_PCA")

@timeit
def ica(tx, ty, rx, ry):
    compressor = ICA(whiten=False)  # for some people, whiten needs to be off
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)

    em(newtx, ty, newrx, ry, add="Inc_ICA", times=50)
    km(newtx, ty, newrx, ry, add="Inc_ICA", times=50)
    #nn(newtx, ty, newrx, ry, add="Inc_ICA")

@timeit
def rca(tx, ty, rx, ry):
    compressor = RandomProjection(tx[1].size)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="Inc_RCA", times=50)
    km(newtx, ty, newrx, ry, add="Inc_RCA", times=50)
    #nn(newtx, ty, newrx, ry, add="Inc_RCA")

@timeit
def em(tx, ty, rx, ry, add="", times=5):
    errs = []

    # this is what we will compare to
    checker = EM(n_components=2)
    checker.fit(ry.reshape(-1, 1))
    truth = checker.predict(ry.reshape(-1, 1))

    # so we do this a bunch of times
    for i in range(2, times):
        clusters = {x: [] for x in range(i)}
        print(i)
        # create a clusterer
        clf = EM(n_components=i)
        clf.fit(tx)  # fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set

        # here we make the arguably awful assumption that for a given cluster,
        # all values in tha cluster "should" in a perfect world, belong in one
        # class or the other, meaning that say, cluster "3" should really be
        # all 0s in our truth, or all 1s there
        #
        # So clusters is a dict of lists, where each list contains all items
        # in a single cluster
        for index, val in enumerate(result):
            clusters[val].append(index)

        # then we take each cluster, find the sum of that clusters counterparts
        # in our "truth" and round that to find out if that cluster should be
        # a 1 or a 0
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}

        # the processed list holds the results of this, so if cluster 3 was
        # found to be of value 1,
        # for each value in clusters[3], processed[value] == 1 would hold
        processed = [mapper[val] for val in result]
        print(truth)
        print(processed)
        errs.append(sum((processed-truth)**2) / float(len(ty)))
    plot([0, times, min(errs)-.1, max(errs)+.1], [range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "Expectation Maximization Error", "EM"+add)

    # dank magic, wrap an array cuz reasons
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    #nn(newtx, ty, newrx, ry, add="EM_"+add)

@timeit
def km(tx, ty, rx, ry, add="", times=5):
    #this does the exact same thing as the above
    errs = []

    checker = KM(n_clusters=2)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2, times):
        print(i)
        clusters = {x: [] for x in range(i)}
        clf = KM(n_clusters=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set
        for index, val in enumerate(result):
            clusters[val].append(index)
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ty)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "KMeans clustering error", "KM"+add)

    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    #nn(newtx, ty, newrx, ry, add="KM_"+add)

@timeit
def nn(tx, ty, rx, ry, add="", iterations=1):
    """
    trains and plots a neural network on the data we have
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(tx[1].size, 5, 1, bias=True)
    ds = ClassificationDataSet(tx[1].size, 1)
    for i in range(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.01)
    train = zip(tx, ty)
    test = zip(rx, ry)
    for i in positions:
        trainer.train()
        resultst.append(sum(np.array([(round(network.activate(t_x)) - t_y)**2 for t_x, t_y in train])/float(len(train))))
        resultsr.append(sum(np.array([(round(network.activate(t_x)) - t_y)**2 for t_x, t_y in test])/float(len(test))))
        # resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print(str(i) + "," + str(resultst[-1]) + "," + str(resultsr[-1]))
        log(str(i) + "," + str(resultst[-1]) + "," + str(resultsr[-1]))
    plot([0, iterations, 0, 1], (positions, resultst, "ro", positions, resultsr, "bo"), "Network Epoch", "Percent Error", "Neural Network Error", "NN"+add)
    sys.stdout.flush()

if __name__=="__main__":
    name = "income"
    # train = name+".data"
    # test = name+".test"
    train_x, train_y, test_x, test_y = X_train_std, y_train, X_test_std, y_test
    #nn(train_x, train_y, test_x, test_y); print('nn done\a')
    em(train_x, train_y, test_x, test_y, times=50); log('em done\a')
    km(train_x, train_y, test_x, test_y, times=50); log('km done\a')
    pca(train_x, train_y, test_x, test_y); log('pca done\a')
    ica(train_x, train_y, test_x, test_y); log('ica done\a')
    rca(train_x, train_y, test_x, test_y); log('randproj done\a')
    file.close()
