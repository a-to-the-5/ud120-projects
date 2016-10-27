#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

print "training data: ", type(features_train), " : ", features_train.shape
print "training labels: ", type(labels_train), " : ", len(labels_train)

print "test data: ", type(features_test), " : ", features_test.shape
print "test labels: ", type(labels_test), " : ", len(labels_test)


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)