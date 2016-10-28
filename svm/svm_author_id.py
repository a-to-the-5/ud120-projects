#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.svm import SVC
from time import time
from sklearn.metrics import accuracy_score
import numpy as np


def test_svm(kernel, train_data_slice, f_train, l_train, f_test, l_test, kernel_params=None):
    if kernel_params is None:
        kernel_params = {}
    clf = SVC(kernel=kernel, **kernel_params)
    t = time()
    clf.fit(f_train[:len(f_train) / train_data_slice],
            l_train[:len(l_train) / train_data_slice])
    t2 = time()
    pred = clf.predict(f_test)
    t3 = time()
    print "\n"
    print "%i %% of data with %s kernel (kernel parameters %s):" % (100 / train_data_slice, kernel, str(kernel_params))
    print "training time: ", t2 - t
    print "prediction time: ", t3 - t2
    print accuracy_score(l_test, pred)
    return pred


# test_svm('linear', 1, features_train, labels_train, features_test, labels_test)
#test_svm('linear', 100, features_train, labels_train, features_test, labels_test)
#for C in [10.0, 100.0, 1000.0, 10000.0]:
#    test_svm('rbf', 100, features_train, labels_train, features_test, labels_test, {'C': C})
#test_svm('rbf', 1, features_train, labels_train, features_test, labels_test, {'C': 10000.0})
#pred = test_svm('rbf', 100, features_train, labels_train, features_test, labels_test, {'C': 10000.0})
#print pred[[10,26,50]]
pred = test_svm('rbf', 1, features_train, labels_train, features_test, labels_test, {'C': 10000.0})
print np.count_nonzero(pred)
#########################################################
